"""Cosmology-free clustering in observed coordinates (n̂, z).

Everything here works in *observed* coordinates — angular separation Δθ on the
sky and redshift separation Δz — with no fiducial cosmology, no comoving
distances, and no peculiar-velocity model. Only measured correlations enter,
so downstream tasks can infer cosmology without the bias a fiducial cosmology
would imprint.

Two public entry points:

- ``measure_xi_theta_z``: Landy-Szalay ξ(Δθ, Δz) from pair counts binned in
  angular and redshift separation. A single 4-D ``query_pairs`` over the
  tagged union of data and (window-drawn) randoms yields DD, DR and RR at
  once.

- ``sample_catalogs_lgcp_observed``: posterior-predictive catalogs from a
  log-Gaussian Cox process whose covariance is the *anisotropic* observed
  kernel K(Δθ, Δz) = ln(1 + ξ(Δθ, Δz)). The GraphGP field is generated on
  points embedded as (n̂, α·z) using the anisotropic-covariance fork, then
  log-normal/Poisson-sampled over window candidates. Catalogs come out as
  (RA, Dec, z).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .quaia import make_random_from_selection_function


def _radec_to_nhat(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=1)


def measure_xi_theta_z(
    catalog,
    *,
    theta_max_deg: float = 2.5,
    theta_min_deg: float = 0.02,
    dz_max: float = 0.03,
    n_theta: int = 16,
    n_z: int = 10,
    theta_log: bool = True,
    n_data: int = 90_000,
    n_rand_factor: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Landy-Szalay ξ(Δθ, Δz) in observed coordinates — no cosmology.

    Pairs are binned in angular separation Δθ (degrees) and redshift
    separation Δz. Randoms are drawn from the separable observed window
    ``sel_map(n̂) · n(z)`` (still cosmology-free). DD, DR and RR come from one
    ``query_pairs`` over the data∪random union embedded as (n̂, β·z).

    Δθ is **log-spaced** (``theta_log=True``) from ``theta_min_deg`` to
    ``theta_max_deg`` with an extra ``[0, theta_min_deg]`` first bin: galaxy
    angular clustering is power-law, so a linear grid (first usable bin ~0.2°)
    leaves the steep core below 0.2° unresolved and forces the kernel to
    *extrapolate* it — the dominant source of the old "core ~40% low" residual.
    The radial correlation is steep (most signal at Δz ≲ 0.01), so Δz is
    binned linearly but finely near zero.

    Returns ``(theta_edges, z_edges, xi_grid)`` with ``xi_grid`` of shape
    ``(n_theta, n_z)``.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    ra_d = np.asarray(catalog.ra_data, dtype=np.float64)
    dec_d = np.asarray(catalog.dec_data, dtype=np.float64)
    z_d = np.asarray(catalog.z_data, dtype=np.float64)
    nd = min(n_data, len(ra_d))
    di = rng.choice(len(ra_d), nd, replace=False)
    ra_d, dec_d, z_d = ra_d[di], dec_d[di], z_d[di]

    nr = n_rand_factor * nd
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=catalog.sel_map, n_random=nr,
        z_data=np.asarray(catalog.z_data), nside=catalog.nside, rng=rng,
    )
    ra_r = np.asarray(ra_r, np.float64); dec_r = np.asarray(dec_r, np.float64)
    z_r = np.asarray(z_r, np.float64)

    # bins — Δθ log-spaced (resolve the power-law core) with a [0, θ_min] first
    # bin; Δz linear. (A finer-near-zero Δz spacing over-resolves the first bin
    # into a shot-noise-dominated, σ²-inflating sliver.)
    if theta_log:
        theta_edges = np.concatenate(
            [[0.0], np.geomspace(theta_min_deg, theta_max_deg, n_theta)])
    else:
        theta_edges = np.linspace(0.0, theta_max_deg, n_theta + 1)
    z_edges = np.linspace(0.0, dz_max, n_z + 1)
    chord_max = 2.0 * np.sin(np.radians(theta_max_deg) / 2.0)

    # 4-D embedding: (n̂, β·z), β chosen so Δz_max maps to chord_max
    beta = chord_max / dz_max
    nhat_d = _radec_to_nhat(ra_d, dec_d)
    nhat_r = _radec_to_nhat(ra_r, dec_r)
    P = np.vstack([
        np.hstack([nhat_d, (beta * z_d)[:, None]]),
        np.hstack([nhat_r, (beta * z_r)[:, None]]),
    ])
    tag = np.concatenate([np.zeros(nd, bool), np.ones(nr, bool)])  # False=data
    R = np.sqrt(chord_max ** 2 + (beta * dz_max) ** 2)

    tree = cKDTree(P)
    pairs = tree.query_pairs(R, output_type="ndarray")   # (Npair, 2)
    i, j = pairs[:, 0], pairs[:, 1]

    # angular + redshift separation for each pair (from the embedding)
    chord = np.linalg.norm(P[i, :3] - P[j, :3], axis=1)
    dtheta = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1)))
    dz = np.abs(P[i, 3] - P[j, 3]) / beta

    ti, tj = tag[i], tag[j]
    is_dd = (~ti) & (~tj)
    is_rr = ti & tj
    is_dr = ti ^ tj

    def hist(mask):
        return np.histogram2d(dtheta[mask], dz[mask],
                              bins=[theta_edges, z_edges])[0]

    DD = hist(is_dd); RR = hist(is_rr); DR = hist(is_dr)

    # Landy-Szalay with count normalisations (undirected pairs)
    nDD = 0.5 * nd * (nd - 1)
    nRR = 0.5 * nr * (nr - 1)
    nDR = float(nd) * float(nr)
    dd = DD / nDD; rr = RR / nRR; dr = DR / nDR
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(rr > 0, (dd - 2.0 * dr + rr) / rr, 0.0)
    return theta_edges, z_edges, xi


def build_observed_kernel(
    theta_edges: np.ndarray,
    z_edges: np.ndarray,
    xi_grid: np.ndarray,
    *,
    alpha: Optional[float] = None,
    jitter: float = 0.02,
    matern_p: int = 1,
    n_bins: int = 600,
):
    """Anisotropic Matérn kernel fit to observed ξ(Δθ, Δz) — PSD by construction.

    A directly-tabulated 2-D kernel (free-form or separable) is not guaranteed
    to be positive-definite on the real clustered candidate distribution, and a
    single non-PSD Vecchia block makes graphGP return an all-NaN field. Instead
    we use a genuine covariance: a Matérn of the *embedded* distance

        d = √(chord² + (α·Δz)²),

    evaluated as an ordinary isotropic Matérn on points embedded as (n̂, α·z).
    This is PSD for any α, and is anisotropic — its angular correlation length
    is ℓ and its radial length ℓ/α. The hyperparameters are read from the data:
    the amplitude σ² = ln(1+ξ(0,0)) (log-normal link), the angular scale from
    the half-drop of ln(1+ξ(Δθ,0)), and α from the ratio of the angular and
    radial half-drop scales.

    Returns ``(covariance, alpha)`` where ``covariance`` is a standard graphGP
    ``(cov_bins, cov_vals)`` tuple to be used with points embedded by ``α``.
    """
    import graphgp as gp

    theta_c = 0.5 * (theta_edges[1:] + theta_edges[:-1])
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])
    chord_c = 2.0 * np.sin(np.radians(theta_c) / 2.0)

    from scipy.optimize import curve_fit

    xi = np.clip(np.asarray(xi_grid), 0.0, None)
    KG_theta = np.minimum.accumulate(np.log1p(xi[:, 0]))   # angular profile
    KG_z = np.minimum.accumulate(np.log1p(xi[0, :]))       # radial profile

    def matern1(d, ell):
        u = np.sqrt(3.0) * np.asarray(d) / ell             # ν = 3/2
        return (1.0 + u) * np.exp(-u)

    # Two-component (narrow + broad) Matérn fit to the angular profile. A single
    # Matérn cannot match the power-law-like shape of galaxy angular clustering
    # (steep core, broad wings); a sum of two Matérns can, and stays PSD.
    def sum2(d, A1, l1, A2, l2):
        return A1 * matern1(d, l1) + A2 * matern1(d, l2)

    c_lo, c_hi = float(chord_c[0]), float(chord_c[-1])
    p0 = [KG_theta[0] * 0.6, c_hi * 0.15, KG_theta[0] * 0.4, c_hi * 0.6]
    try:
        popt, _ = curve_fit(sum2, chord_c, KG_theta, p0=p0,
                            bounds=([1e-3, c_lo * 0.3, 1e-3, c_lo],
                                    [1e3, c_hi, 1e3, 10.0 * c_hi]), maxfev=20000)
        A1, l1, A2, l2 = (float(x) for x in popt)
    except Exception:
        A1, l1, A2, l2 = KG_theta[0], c_hi * 0.2, 0.0, c_hi

    # angular/radial anisotropy α from the half-amplitude scales of each profile
    def half_scale(prof, coords):
        below = np.where(prof < 0.5 * prof[0])[0]
        return float(coords[below[0]]) if len(below) else float(coords[-1])
    if alpha is None:
        alpha = float(half_scale(KG_theta, chord_c) / max(half_scale(KG_z, z_c), 1e-9))

    # tabulate the summed kernel on a shared bin grid (PSD; sum of two Matérns)
    r_max = max(2.0, 6.0 * max(l1, l2))
    cb, v1 = gp.extras.matern_kernel(p=1, variance=A1, cutoff=l1,
                                     r_min=1e-6, r_max=r_max, n_bins=n_bins, jitter=jitter)
    _, v2 = gp.extras.matern_kernel(p=1, variance=A2, cutoff=l2,
                                    r_min=1e-6, r_max=r_max, n_bins=n_bins, jitter=jitter)
    cov = (cb, v1 + v2)
    return cov, alpha


def _matern1(d, ell):
    """Matérn ν=3/2 correlation function, (1 + √3 d/ℓ) exp(−√3 d/ℓ)."""
    u = np.sqrt(3.0) * np.asarray(d, dtype=np.float64) / ell
    return (1.0 + u) * np.exp(-u)


def build_observed_kernel_2d(
    theta_edges: np.ndarray,
    z_edges: np.ndarray,
    xi_grid: np.ndarray,
    *,
    alpha: Optional[float] = None,
    jitter: float = 0.02,
    n_ltheta: int = 8,
    n_lz: int = 6,
    n_s: int = 512,
    n_zg: int = 256,
):
    """Genuine 2-D PSD anisotropic kernel fit to observed ξ(Δθ, Δz).

    The old kernel was an *embedded isotropic* Matérn of d = √(chord²+(αΔz)²):
    a single global anisotropy ratio α at all scales. Galaxy clustering in
    observed coordinates is not elliptical — redshift-space distortions make the
    Δθ/Δz aspect ratio scale-dependent — so a single-shape kernel cannot match
    the power-law w(θ) (it came out core-low / wing-high simultaneously).

    Here we fit the full 2-D log-kernel K_G(Δθ, Δz) = ln(1+ξ(Δθ, Δz)) with a
    **sum of tensor-product Matérns**

        K_G(Δθ, Δz) ≈ Σ_k A_k · M(chord(Δθ); ℓθ_k) · M(Δz; ℓz_k),   A_k ≥ 0,

    over a fixed bank of (ℓθ_k, ℓz_k) scale pairs. Each term is a product of two
    1-D Matérns, so its covariance matrix is the Hadamard product of two PSD
    matrices — PSD by the Schur product theorem — and a non-negative-weighted
    sum of PSD kernels is PSD. The weights A_k are solved by non-negative least
    squares (convex, robust), guaranteeing positive-definiteness without the
    NaN-field failure of a directly-tabulated 2-D kernel. The flexible bank of
    scales captures the steep core, the broad wings, *and* a scale-dependent
    Δθ/Δz aspect ratio simultaneously.

    Returns ``(AnisotropicCovariance, alpha)`` for use with points embedded as
    (n̂, α·z) via ``graphgp.embed_points``.
    """
    from scipy.optimize import nnls
    import graphgp as gp

    # bin centres (geometric for the log-θ axis, arithmetic for the [0,·] bin)
    theta_c = np.empty(len(theta_edges) - 1)
    theta_c[0] = 0.5 * theta_edges[1]
    theta_c[1:] = np.sqrt(theta_edges[1:-1] * theta_edges[2:])
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])
    chord_c = 2.0 * np.sin(np.radians(theta_c) / 2.0)

    xi = np.clip(np.asarray(xi_grid, dtype=np.float64), 0.0, None)
    KG = np.log1p(xi)                      # (n_theta, n_z) target log-kernel

    # bank of length scales spanning the measured range (and a little beyond)
    lthetas = np.geomspace(0.7 * chord_c[0], 1.5 * chord_c[-1], n_ltheta)
    lzs = np.geomspace(0.7 * max(z_c[0], 1e-4), 1.5 * z_c[-1], n_lz)

    # design matrix: one column per (ℓθ, ℓz) tensor-Matérn basis function,
    # evaluated on the (Δθ, Δz) measurement grid and flattened.
    cols, scales = [], []
    for lt in lthetas:
        mt = _matern1(chord_c, lt)                  # (n_theta,)
        for lz in lzs:
            mz = _matern1(z_c, lz)                  # (n_z,)
            cols.append(np.outer(mt, mz).ravel())
            scales.append((lt, lz))
    A_mat = np.stack(cols, axis=1)                  # (n_theta*n_z, n_basis)
    coeffs, _ = nnls(A_mat, KG.ravel())

    # evaluation grid for the AnisotropicCovariance (dense; covers a little
    # beyond the measured range, where the Matérns have decayed toward 0).
    sb = np.concatenate([[0.0], np.geomspace(1e-5, 1.5 * chord_c[-1], n_s - 1)])
    zb = np.concatenate([[0.0], np.geomspace(1e-5, 2.0 * z_c[-1], n_zg - 1)])

    # assemble the fitted kernel on the (sb, zb) evaluation grid
    grid = np.zeros((n_s, n_zg))
    kg_theta0 = np.zeros(n_s)   # K_G(Δθ, 0) profile, for α
    kg_z0 = np.zeros(n_zg)      # K_G(0, Δz) profile, for α
    for (lt, lz), a in zip(scales, coeffs):
        if a <= 0:
            continue
        mt = _matern1(sb, lt)
        mz = _matern1(zb, lz)
        grid += a * np.outer(mt, mz)
        kg_theta0 += a * mt * _matern1(0.0, lz)
        kg_z0 += a * _matern1(0.0, lt) * mz

    if alpha is None:
        def half_scale(prof, coords):
            below = np.where(prof < 0.5 * prof[0])[0]
            return float(coords[below[0]]) if len(below) else float(coords[-1])
        alpha = float(half_scale(kg_theta0, sb) / max(half_scale(kg_z0, zb), 1e-9))

    cov = gp.build_anisotropic_covariance(sb, zb, grid, float(alpha), jitter=jitter)
    return cov, alpha


def _nnls_bands(theta_edges, z_edges, xi_grid, n_ltheta=8, n_lz=6):
    """Shared NNLS fit of K_G(Δθ,Δz) → (scales, coeffs, sb, zb, bin centres)."""
    from scipy.optimize import nnls

    theta_c = np.empty(len(theta_edges) - 1)
    theta_c[0] = 0.5 * theta_edges[1]
    theta_c[1:] = np.sqrt(theta_edges[1:-1] * theta_edges[2:])
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])
    chord_c = 2.0 * np.sin(np.radians(theta_c) / 2.0)
    KG = np.log1p(np.clip(np.asarray(xi_grid, np.float64), 0.0, None))

    lthetas = np.geomspace(0.7 * chord_c[0], 1.5 * chord_c[-1], n_ltheta)
    lzs = np.geomspace(0.7 * max(z_c[0], 1e-4), 1.5 * z_c[-1], n_lz)
    cols, scales = [], []
    for lt in lthetas:
        mt = _matern1(chord_c, lt)
        for lz in lzs:
            cols.append(np.outer(mt, _matern1(z_c, lz)).ravel())
            scales.append((lt, lz))
    coeffs, _ = nnls(np.stack(cols, axis=1), KG.ravel())
    return scales, coeffs, chord_c, z_c


def build_observed_bands(
    theta_edges: np.ndarray,
    z_edges: np.ndarray,
    xi_grid: np.ndarray,
    *,
    n_cand: int,
    theta_split_deg: float = 0.35,
    alpha_fine: float = 2.0,
    jitter: float = 0.02,
    n_s: int = 512,
    n_zg: int = 256,
    coarse_ref_deg: float = 0.35,
    coarse_min: int = 3_000,
):
    """Decompose K_G into additive bands for multi-resolution synthesis.

    The NNLS tensor-Matérn bands are split by angular scale: all bands with
    ℓθ ≤ ``theta_split_deg`` are summed into one *fine* band generated at full
    candidate density (the dense graph realizes them well), while each broader
    band becomes its own band generated on a sub-sample matched to its angular
    scale and kriged up. Each band carries its own embedding ``alpha`` (the
    broad bands use the isotropising α_b = ℓθ/ℓz so a point's neighbours span
    the band in both axes; α cancels from the kernel value). The band fields are
    independent, so their covariances sum back to the full K_G.

    Returns ``(bands, sigma2)`` where ``bands`` is a list of
    ``{"cov", "alpha", "n_coarse"}`` dicts for ``graphgp.generate_additive``.
    """
    import graphgp as gp

    scales, coeffs, chord_c, z_c = _nnls_bands(theta_edges, z_edges, xi_grid)
    sb = np.concatenate([[0.0], np.geomspace(1e-5, 1.5 * chord_c[-1], n_s - 1)])
    zb = np.concatenate([[0.0], np.geomspace(1e-5, 2.0 * z_c[-1], n_zg - 1)])
    chord_split = 2.0 * np.sin(np.radians(theta_split_deg) / 2.0)
    chord_ref = 2.0 * np.sin(np.radians(coarse_ref_deg) / 2.0)

    def grid_of(items):
        g = np.zeros((len(sb), len(zb)))
        for (lt, lz), a in items:
            g += a * np.outer(_matern1(sb, lt), _matern1(zb, lz))
        return g

    active = [((lt, lz), a) for (lt, lz), a in zip(scales, coeffs) if a > 1e-3]
    sigma2 = float(sum(a for _, a in active))

    # fine band: all ℓθ ≤ split, summed, full density
    fine = [it for it in active if it[0][0] <= chord_split]
    bands = []
    if fine:
        gf = grid_of(fine)
        bands.append({"cov": gp.build_anisotropic_covariance(sb, zb, gf, float(alpha_fine),
                                                             jitter=jitter),
                      "alpha": float(alpha_fine), "n_coarse": None})
    # broad bands: one per (ℓθ, ℓz), sub-sampled to match the angular scale
    for (lt, lz), a in [it for it in active if it[0][0] > chord_split]:
        gb = grid_of([((lt, lz), a)])
        alpha_b = float(lt / max(lz, 1e-9))
        n_coarse = int(np.clip(n_cand * (chord_ref / lt) ** 2, coarse_min, n_cand))
        bands.append({"cov": gp.build_anisotropic_covariance(sb, zb, gb, alpha_b,
                                                             jitter=jitter),
                      "alpha": alpha_b, "n_coarse": n_coarse})
    return bands, sigma2


def sample_catalogs_lgcp_observed(
    catalog,
    *,
    n_samples: int = 8,
    seed: int = 0,
    w_completeness: Optional[np.ndarray] = None,
    n_cand_factor: int = 20,
    n0: int = 256,
    k: int = 30,
    embed_alpha: Optional[float] = 2.0,
    additive: bool = False,
    chunk_size: Optional[int] = 50_000,
    measure_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic observed-space LGCP catalogs — no cosmology.

    1. Measure ξ(Δθ, Δz) and build the anisotropic kernel K_G = ln(1+ξ).
    2. Draw window candidates (RA, Dec, z) from sel_map × n(z); embed as
       (n̂, α·z); build the GraphGP graph once.
    3. For each draw: generate the anisotropic field (chunked, GPU), form the
       log-normal intensity 1+δ = exp(f − σ²/2), and inhomogeneous-Poisson
       sample to (RA, Dec, z).

    Returns ``(catalogs, theta_edges, z_edges, xi_grid)``.
    """
    import jax
    import jax.numpy as jnp
    import graphgp as gp

    jax.config.update("jax_enable_x64", True)

    # ``embed_alpha`` is the radial embedding scale used ONLY for the graph's
    # neighbour selection — it cancels out of the AnisotropicCovariance *value*
    # (aniso_evaluate divides Δz by alpha after the embedding multiplies by it),
    # so the kernel is identical for any alpha. A larger alpha pushes
    # different-redshift candidates apart in the embedded metric, so each
    # point's k neighbours are drawn from more nearly the same redshift and span
    # a wider *angular* range — which is what the Vecchia field needs to realize
    # the kernel's broad angular wings (alpha≈2 markedly outperforms the
    # correlation-isotropising alpha≈0.3 on w(θ) at θ≳0.5°).
    te, ze, xi = measure_xi_theta_z(catalog, seed=seed, **(measure_kwargs or {}))

    nd = catalog.N_data
    if w_completeness is None:
        w_completeness = np.ones(nd)
    w_sum = float(np.asarray(w_completeness).sum())
    n_cand = int(n_cand_factor * nd)

    # Kernel. ``additive`` decomposes K_G into independent tensor-Matérn bands
    # generated each at its own resolution (broad/long-range bands on a coarse
    # sub-sample, kriged up) and summed — the multi-resolution synthesis that
    # avoids the dense graph screening out broad angular power. Otherwise a
    # single AnisotropicCovariance is generated on one full-density graph.
    if additive:
        bands, sigma2 = build_observed_bands(te, ze, xi, n_cand=n_cand,
                                             alpha_fine=embed_alpha or 2.0)
        cov, alpha = None, embed_alpha
        if verbose:
            ncs = [b["n_coarse"] for b in bands]
            print(f"[obs-lgcp] additive: {len(bands)} bands, n_coarse={ncs}  σ²={sigma2:.3f}")
    else:
        cov, alpha = build_observed_kernel_2d(te, ze, xi, alpha=embed_alpha)
        # σ² is the kernel's zero-separation amplitude: cov_vals[0] for a 1-D
        # (cov_bins, cov_vals) kernel, or grid[0,0] for an AnisotropicCovariance.
        sigma2 = (float(np.asarray(cov[1])[0]) if isinstance(cov, tuple)
                  else float(np.asarray(cov.grid)[0, 0]))
        if verbose:
            print(f"[obs-lgcp] anisotropic kernel: alpha={alpha:.3f}  σ²={sigma2:.3f}")

    rng0 = np.random.default_rng(seed)
    ra_c, dec_c, z_c = make_random_from_selection_function(
        sel_map=catalog.sel_map, n_random=n_cand,
        z_data=np.asarray(catalog.z_data), nside=catalog.nside, rng=rng0,
    )
    ra_c = np.asarray(ra_c, np.float64); dec_c = np.asarray(dec_c, np.float64)
    z_c = np.asarray(z_c, np.float64)
    nhat_c = _radec_to_nhat(ra_c, dec_c)

    def embed(a):
        return jnp.asarray(np.hstack([nhat_c, (a * z_c)[:, None]]), dtype=jnp.float64)

    if verbose:
        print(f"[obs-lgcp] {n_cand:,} window candidates; building graph ...")
    if not additive:
        points = embed(alpha)
        graph = gp.build_graph(points, n0=min(n0, n_cand // 2), k=min(k, n_cand - 1))

    # Candidates already carry continuous positions from the window sampler;
    # multiply-occupied candidates land at Δθ=Δz=0 (below the smallest measured
    # bin, an unresolved 1-halo) so need no jitter.  A *large* jitter would move
    # the catalog off the nside pixel structure that the comparison randoms
    # share, producing a gross footprint mismatch — so we apply only a tiny tie
    # break far below the smallest measured separation.
    jit_ang = 1e-3 * float(te[1])    # deg, ~0.1% of first Δθ bin
    jit_z = 1e-3 * float(ze[1])

    out = []
    for s in range(n_samples):
        if additive:
            f = np.asarray(gp.generate_additive(
                embed(embed_alpha or 2.0), bands, k=k, n0=n0,
                seed=seed + 1 + s, chunk_size=chunk_size, embed=embed,
                verbose=verbose and s == 0))
        else:
            eps = np.random.default_rng(seed + 1 + s).standard_normal(n_cand)
            try:
                f = np.asarray(gp.generate(graph, cov, jnp.asarray(eps, dtype=jnp.float64),
                                           chunk_size=chunk_size))
            except TypeError:
                f = np.asarray(gp.generate(graph, cov, jnp.asarray(eps, dtype=jnp.float64)))
        # guard against rare near-singular Vecchia blocks (near-coincident
        # candidates) producing field outliers: clip to ±8σ and zero any NaN.
        sig = np.sqrt(max(sigma2, 1e-12))
        f = np.where(np.isfinite(f), f, 0.0)
        f = np.clip(f, -8.0 * sig, 8.0 * sig)
        opd = np.exp(f - 0.5 * sigma2)
        opd_sum = float(opd.sum())
        alpha_thin = w_sum / opd_sum if opd_sum > 0 else 0.0
        rng = np.random.default_rng(1000 + seed + s)
        counts = rng.poisson(alpha_thin * opd)
        occ = counts > 0
        idx = np.repeat(np.where(occ)[0], counts[occ])
        ra_g = ra_c[idx] + rng.normal(0.0, jit_ang, size=len(idx))
        dec_g = dec_c[idx] + rng.normal(0.0, jit_ang, size=len(idx))
        z_g = z_c[idx] + rng.normal(0.0, jit_z, size=len(idx))
        out.append({
            "ra": ra_g.astype(np.float32),
            "dec": dec_g.astype(np.float32),
            "z": z_g.astype(np.float32),
            "N_galaxies": int(len(idx)),
            "multi_frac": float(np.mean(counts[occ] > 1)),
        })
        if verbose:
            print(f"[obs-lgcp] sample {s+1}/{n_samples}: N={out[-1]['N_galaxies']:,} "
                  f"multi_frac={out[-1]['multi_frac']:.4f}")
    return out, te, ze, xi

