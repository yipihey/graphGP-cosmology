"""Window/weight-corrected 2D clustering kernel via Landy-Szalay pair counting.

This is the measurement-first pipeline: a single, reusable, FKP×completeness
*weighted* Landy-Szalay estimator of the observed-space correlation
ξ(Δθ, Δz) measured against the analytic randoms (sel_map × n(z)). It is used
**identically** to

  1. measure the data kernel K_in(Δθ, Δz) from BOSS (weighted; then the survey
     window is deconvolved to the true clustering K), which is reused directly
     as the GraphGP generation covariance, and
  2. re-measure K_out(Δθ, Δz) from each generated catalog,

so the window, weights and estimator cancel between input and output by
construction — the honest closure test is K_out ≈ K_in across the whole plane
(plus the w(θ) projection). No parametric kernel fit; the measured K is the
source of truth.

Everything is in observed coordinates (Δθ in degrees, Δz) — no fiducial
cosmology, no comoving distances.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import perf
from .observed import _radec_to_nhat
from .quaia import make_random_from_selection_function


def measure_K2d(
    ra_d, dec_d, z_d, w_d,
    ra_r, dec_r, z_r, w_r,
    *,
    theta_edges: np.ndarray,
    z_edges: np.ndarray,
    return_counts: bool = False,
    precomp_rr: dict = None,
):
    """Weighted Landy-Szalay ξ(Δθ, Δz) from one 4-D ``query_pairs``.

    Points (data ∪ randoms) are embedded as (n̂, β·z) with β chosen so the Δz
    window maps to the angular chord window; a single ``query_pairs`` over the
    union yields the weighted DD, DR, RR pair histograms binned in (Δθ, Δz).

    Pair weights are the products ``w_i · w_j``; the Landy-Szalay normalisations
    use the weighted counts ``W=Σw`` and ``W2=Σw²`` so the estimator is unbiased
    under the supplied (FKP×completeness) weights.

    Returns ``(theta_edges, z_edges, xi)`` or, with ``return_counts``, also a
    dict of the normalised ``dd, dr, rr`` and raw weighted ``DD, DR, RR``.
    """
    from scipy.spatial import cKDTree
    from . import perf

    ra_d = np.asarray(ra_d, np.float64); dec_d = np.asarray(dec_d, np.float64)
    z_d = np.asarray(z_d, np.float64); w_d = np.asarray(w_d, np.float64)
    ra_r = np.asarray(ra_r, np.float64); dec_r = np.asarray(dec_r, np.float64)
    z_r = np.asarray(z_r, np.float64); w_r = np.asarray(w_r, np.float64)
    nd, nr = len(ra_d), len(ra_r)

    theta_max = float(theta_edges[-1]); dz_max = float(z_edges[-1])
    chord_max = 2.0 * np.sin(np.radians(theta_max) / 2.0)
    beta = chord_max / dz_max
    R = np.sqrt(chord_max ** 2 + (beta * dz_max) ** 2)
    Pd = np.hstack([_radec_to_nhat(ra_d, dec_d), (beta * z_d)[:, None]])
    Pr = np.hstack([_radec_to_nhat(ra_r, dec_r), (beta * z_r)[:, None]])

    def hist_pairs(Pa, Pb, wa, wb, i, j):
        chord = np.linalg.norm(Pa[i, :3] - Pb[j, :3], axis=1)
        dth = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
        dz = np.abs(Pa[i, 3] - Pb[j, 3]) / beta
        return np.histogram2d(dth, dz, bins=[theta_edges, z_edges], weights=wa[i] * wb[j])[0]

    Wd, W2d = w_d.sum(), (w_d ** 2).sum()
    nDD = 0.5 * (Wd ** 2 - W2d); nDR = Wd * (w_r.sum())

    with perf.timer("measure_K2d"):
        with perf.timer("measure_K2d.DD"):
            pdd = cKDTree(Pd).query_pairs(R, output_type="ndarray")
            perf.count("pairs.DD", len(pdd))
            DD = hist_pairs(Pd, Pd, w_d, w_d, pdd[:, 0], pdd[:, 1])
        # DR: data-vs-random cross pairs (always needed)
        with perf.timer("measure_K2d.DR"):
            nbr = cKDTree(Pd).query_ball_tree(cKDTree(Pr), R)
            di = np.repeat(np.arange(nd), [len(x) for x in nbr])
            rj = np.fromiter((k for x in nbr for k in x), dtype=np.int64,
                             count=sum(len(x) for x in nbr))
            perf.count("pairs.DR", len(di))
            DR = hist_pairs(Pd, Pr, w_d, w_r, di, rj)
        if precomp_rr is not None:                     # cached RR (randoms fixed)
            rr = precomp_rr["rr"]; nRR = precomp_rr["nRR"]
        else:
            with perf.timer("measure_K2d.RR"):
                prr = cKDTree(Pr).query_pairs(R, output_type="ndarray")
                perf.count("pairs.RR", len(prr))
                RR = hist_pairs(Pr, Pr, w_r, w_r, prr[:, 0], prr[:, 1])
            nRR = 0.5 * (w_r.sum() ** 2 - (w_r ** 2).sum())
            rr = RR / nRR

    dd = DD / nDD; dr = DR / nDR
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(rr > 0, (dd - 2.0 * dr + rr) / rr, 0.0)

    if return_counts:
        return theta_edges, z_edges, xi, {"dd": dd, "dr": dr, "rr": rr, "nRR": nRR}
    return theta_edges, z_edges, xi


def compute_rr(ra_r, dec_r, z_r, w_r, *, theta_edges, z_edges):
    """Precompute the normalised random-random RR(Δθ,Δz) and its normalisation,
    to be reused across many ``measure_K2d`` calls against the SAME randoms
    (``precomp_rr=`` argument) — skips the dominant random-random pair count."""
    from scipy.spatial import cKDTree
    from . import perf
    ra_r = np.asarray(ra_r, np.float64); dec_r = np.asarray(dec_r, np.float64)
    z_r = np.asarray(z_r, np.float64); w_r = np.asarray(w_r, np.float64)
    theta_max = float(theta_edges[-1]); dz_max = float(z_edges[-1])
    chord_max = 2.0 * np.sin(np.radians(theta_max) / 2.0); beta = chord_max / dz_max
    R = np.sqrt(chord_max ** 2 + (beta * dz_max) ** 2)
    Pr = np.hstack([_radec_to_nhat(ra_r, dec_r), (beta * z_r)[:, None]])
    with perf.timer("compute_rr"):
        prr = cKDTree(Pr).query_pairs(R, output_type="ndarray")
        perf.count("pairs.RR", len(prr))
        chord = np.linalg.norm(Pr[prr[:, 0], :3] - Pr[prr[:, 1], :3], axis=1)
        dth = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
        dz = np.abs(Pr[prr[:, 0], 3] - Pr[prr[:, 1], 3]) / beta
        RR = np.histogram2d(dth, dz, bins=[theta_edges, z_edges],
                            weights=w_r[prr[:, 0]] * w_r[prr[:, 1]])[0]
    nRR = 0.5 * (w_r.sum() ** 2 - (w_r ** 2).sum())
    return {"rr": RR / nRR, "nRR": nRR}


def measure_close_pair_dz(catalog, collision_scale_deg: float = 62.0 / 3600.0):
    """Empirical signed Δz of *observed* angular close pairs (≤ collision scale).

    Surviving close pairs (both redshifts measured — e.g. tile overlaps that
    escaped fiber collision) sample the true redshift-separation distribution of
    collided pairs (collisions are imposed by tiling, not physics). Their Δz
    carries the clustered (Δz≈0, true 1-halo pairs) + background (broad, chance
    projections) mixture *data-drivenly*, so the missing partner's redshift can
    be drawn as z_host + Δz without a parametric clustered/background fraction.
    Returned symmetrised (±Δz).
    """
    from scipy.spatial import cKDTree
    nhat = _radec_to_nhat(np.asarray(catalog.ra_data), np.asarray(catalog.dec_data))
    z = np.asarray(catalog.z_data, np.float64)
    chord = 2.0 * np.sin(np.radians(collision_scale_deg) / 2.0)
    pairs = cKDTree(nhat).query_pairs(chord, output_type="ndarray")
    dz = z[pairs[:, 1]] - z[pairs[:, 0]]
    return np.concatenate([dz, -dz])


@perf.timed("complete_catalog")
def complete_catalog(
    catalog,
    *,
    seed: int = 0,
    collision_scale_deg: float = 62.0 / 3600.0,
    count: str = "round",
    z_assign: str = "data",
    dz_pool=None,
    verbose: bool = False,
):
    """One equal-weight realization of the systematics-corrected catalog.

    Replaces the FKP×completeness *weighting* with an explicit *completion*: keep
    the observed galaxies and add the ones the completeness weights say are
    missing (fiber collisions w_cp, redshift failures w_noz, imaging systematics
    w_systot — **not** FKP, which is an estimator weight). Each galaxy is
    realized ``n_i`` times with E[n_i] = w_c,i = w_systot·(w_cp+w_noz−1) (the
    BOSS completeness weight), so the **equal-weight** catalog reproduces the
    **w_c-weighted** clustering at resolved separations by construction
    (Σ nᵢnⱼ → Σ w_c,i w_c,j).

    Every missing galaxy is a *local* addition (collisions, failures, and the
    imaging systematic alike: the systematic-missing galaxy is clustered like
    the local field, NOT scattered over the global n(z) — drawing it from the
    global n(z) would dilute the radial clustering). It is placed within the
    unresolved collision scale of the host (preserving angular clustering); its
    redshift is set by ``z_assign``:

    - ``'host'``: z_host — the nearest-neighbour assumption the BOSS weights
      themselves make; reproduces the w_c-weighted clustering exactly.
    - ``'data'`` (recommended): z_host + Δz with Δz drawn from the measured
      close-pair distribution — relaxes the NN assumption using the observed
      mix of true close pairs (Δz≈0) and chance projections (broad Δz), giving
      more realistic small-scale *radial* structure.
    - ``'nz'``: global n(z) (background); ``'mix'``: half host / half n(z).

    ``count='poisson'`` makes the integer counts stochastic (realizations also
    span the missing-number shot noise; w_systot<1 over-dense regions are thinned
    when n_i=0); ``count='round'`` is deterministic.

    Returns ``dict(ra, dec, z, N)`` — an equal-weight catalog.
    """
    rng = np.random.default_rng(seed)
    ra = np.asarray(catalog.ra_data, np.float64)
    dec = np.asarray(catalog.dec_data, np.float64)
    z = np.asarray(catalog.z_data, np.float64)
    one = np.ones(len(ra))
    wsys = np.asarray(catalog.w_sys_data if catalog.w_sys_data is not None else one)
    wcp = np.asarray(catalog.w_cp_data if catalog.w_cp_data is not None else one)
    wnoz = np.asarray(catalog.w_noz_data if catalog.w_noz_data is not None else one)
    w_c = wsys * (wcp + wnoz - 1.0)                       # completeness weight

    n = (rng.poisson(w_c) if count == "poisson"
         else np.floor(w_c + rng.random(len(w_c))).astype(int))  # randomized round
    n_extra = np.maximum(n - 1, 0)
    keep = n > 0                                          # base copy kept iff n≥1
    if z_assign == "data" and dz_pool is None:
        dz_pool = measure_close_pair_dz(catalog, collision_scale_deg)

    ra_out = [ra[keep]]; dec_out = [dec[keep]]; z_out = [z[keep]]
    host = np.repeat(np.arange(len(ra)), n_extra)        # host index per extra copy
    m = len(host)
    if m:
        # angular: jitter within the collision scale (≪ smallest measured bin)
        s = np.radians(collision_scale_deg) / 3.0
        dra = np.degrees(rng.normal(0, s, m) / np.cos(np.radians(dec[host])))
        ddec = np.degrees(rng.normal(0, s, m))
        ra_e = ra[host] + dra; dec_e = dec[host] + ddec
        zc = z[host]
        if z_assign == "data":
            z_e = zc + rng.choice(dz_pool, m)
        elif z_assign == "nz":
            z_e = rng.choice(z, m)
        elif z_assign == "mix":
            z_e = np.where(rng.random(m) < 0.5, zc, rng.choice(z, m))
        else:  # 'host'
            z_e = zc
        ra_out.append(ra_e); dec_out.append(dec_e); z_out.append(z_e)

    ra_f = np.concatenate(ra_out); dec_f = np.concatenate(dec_out); z_f = np.concatenate(z_out)
    if verbose:
        print(f"[complete] N_obs={len(ra):,} -> N_eq={len(ra_f):,} "
              f"(+{100*(len(ra_f)/len(ra)-1):.1f}%, {m:,} added, z_assign={z_assign})")
    return {"ra": ra_f.astype(np.float32), "dec": dec_f.astype(np.float32),
            "z": z_f.astype(np.float32), "N": len(ra_f)}


def _clpair_density(dz_pool, n_bins: int = 121, dz_max: float = 0.06):
    """Empirical p(Δz) of observed close pairs → a callable density on Δz.

    Built from ``measure_close_pair_dz`` (symmetrised signed Δz). Returns a
    function evaluating the normalised histogram density at arbitrary Δz (0
    outside the range), used as the clustering prior that pulls a collided
    partner's redshift toward its host's when the pair is physical.
    """
    dz = np.asarray(dz_pool, np.float64)
    edges = np.linspace(-dz_max, dz_max, n_bins)
    h, _ = np.histogram(np.clip(dz, -dz_max, dz_max), bins=edges, density=True)
    cen = 0.5 * (edges[1:] + edges[:-1])
    return lambda x: np.interp(np.abs(x), np.abs(cen[cen >= 0]),
                               h[cen >= 0], left=h[cen >= 0][0], right=0.0)


# provenance codes for completed-catalog galaxies (per object)
PROV = {"observed": 0, "collided": 1, "zfail": 2, "systot": 3, "zhost": 4, "inpaint": 5}
PROV_NAME = {v: k for k, v in PROV.items()}


def _systot_restore_extras(base_ra, base_dec, base_z, src, rng, jitter_arcsec=1.0):
    """Restore ``len(src)`` WEIGHT_SYSTOT-implied galaxies at the survivor scale.

    WEIGHT_SYSTOT is a *smooth* (degree-scale) imaging-systematic density boost,
    so the restored galaxies must trace the field at the **survivor's position**
    (``src[e]``), not be clustered onto an individual neighbour — placing them at
    the local nearest-neighbour scale (~arcmin) injects spurious small-scale
    power. We therefore restore each extra at its source position displaced by a
    Gaussian of ``jitter_arcsec`` (~1″, far below the BOSS fiber scale and any
    analysis scale), carrying the source redshift. This reproduces w(θ)/ξ at all
    resolved scales (identically to exact duplication) while removing the
    unphysical Δθ=0 delta-spike that corrupts kNN / coincident-point statistics.
    Returns ``(ra, dec, z)`` of the extras."""
    if len(src) == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0))
    sig = jitter_arcsec / 3600.0
    cd = np.cos(np.radians(base_dec[src]))
    ra = base_ra[src] + rng.normal(0, 1, len(src)) * sig / np.maximum(cd, 1e-3)
    dec = base_dec[src] + rng.normal(0, 1, len(src)) * sig
    return ra % 360.0, dec, base_z[src]


def build_gp_field(catalog, *, n_samples=8, seed=0, verbose=False, **kwargs):
    """Convenience builder for ``z_mode='graphgp'``: the conditional GP posterior
    density field (graphGP / Matheron). Build ONCE and pass as ``gp_field=`` to
    :func:`complete_catalog_photoz` to amortise the (expensive) field solve across
    an ensemble of realizations. Thin wrapper over
    :func:`density_field.sample_posterior_density_field` (``kwargs``: nside,
    n_z_bins, r_edges, …)."""
    from .density_field import sample_posterior_density_field
    return sample_posterior_density_field(catalog, n_samples=n_samples, seed=seed,
                                          verbose=verbose, **kwargs)


def _graphgp_zmiss(targets, photoz, dz_pool, gp_field, draw_index, z_o, z_host, miss_kind, rng):
    """Missing-galaxy redshifts from ONE conditional GP field draw, evaluated along
    each missing galaxy's sightline:
        p(z | n̂, colours) ∝ (1+δ_GP(n̂,z)) · n̄(z) · p_photoz(z)   (× close-pair prior)
    i.e. z_mode='field' with the graphGP posterior field replacing the KNN-KDE local
    density (correlated across missing galaxies via the shared draw). Returns
    ``(z_miss, zhost_fallback)``."""
    import healpy as hp
    from .photoz import photoz_features
    ra_m = np.asarray(targets.ra, np.float64); dec_m = np.asarray(targets.dec, np.float64)
    host = np.asarray(targets.host_index); coll = (miss_kind == "collided") & (host >= 0)
    feat = photoz_features(targets.colors, targets.mags); zk, wk = photoz.posterior(feat)
    pcl = _clpair_density(dz_pool)
    nside = gp_field.nside; zc = 0.5 * (gp_field.z_edges[1:] + gp_field.z_edges[:-1])
    zgrid = np.linspace(z_o.min(), z_o.max(), 256)
    nbar_z = np.interp(zgrid, zc, np.histogram(z_o, bins=gp_field.z_edges)[0].astype(float),
                       left=0.0, right=0.0)
    dl = gp_field.delta_lightcone[draw_index % gp_field.n_samples]      # (n_z, N_pix) = 1+δ
    pix = hp.ang2pix(nside, np.radians(90.0 - dec_m), np.radians(ra_m % 360.0))
    bw_p = 0.02; M = len(ra_m)
    z_miss = np.empty(M); fb = np.zeros(M, bool)
    for i in range(M):
        pf = np.interp(zgrid, zc, dl[:, pix[i]], left=0.0, right=0.0) * nbar_z   # GP local density × n̄(z)
        w = wk[i]; ok = np.isfinite(w) & (w > 0)
        pp = ((w[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
              if ok.any() else np.ones_like(zgrid))
        p = pf * pp
        if coll[i]:
            p = p * pcl(zgrid - z_host[i])
        s = p.sum()
        if s > 0:
            z_miss[i] = rng.choice(zgrid, p=p / s)
        else:
            z_miss[i] = z_host[i] if np.isfinite(z_host[i]) else float(np.median(z_o))
            fb[i] = True
    return z_miss, fb


@perf.timed("complete_catalog_photoz")
def complete_catalog_photoz(
    catalog, targets, photoz,
    *,
    seed: int = 0,
    clustering_prior: str = "data",
    dz_pool=None,
    count: str = "round",
    systot_mode: str = "analog",
    z_mode: str = "field",
    gp_field=None,
    gp_kwargs=None,
    verbose: bool = False,
):
    """Equal-weight completion using REAL imaging positions + photo-z redshifts.

    The missing galaxies (``targets``: fiber collisions + redshift failures) are
    real photometric detections — known positions, only the redshift uncertain.
    So we (1) keep every observed galaxy, (2) add every missing target at its
    KNOWN position with a redshift sampled from its photo-z posterior
    p(z|colours) — for collided objects reweighted by the close-pair clustering
    prior p(Δz) (a physical pair is near the host's z; a projection is not), and
    (3) realize the imaging systematic w_systot.

    ``systot_mode`` controls (3):

    - ``'analog'`` (default): the integer **excess** ``n_i−1`` of each object is
      restored at the survivor scale (:func:`_systot_restore_extras`) — a
      sub-arcsec jitter of the source position carrying its redshift — which
      reproduces w(θ)/ξ at all resolved scales while removing the unphysical
      Δθ=0 delta-spike that exact duplication creates and that corrupts kNN /
      coincident-point statistics. One un-jittered copy of each object with
      ``n_i≥1`` is kept; ``n_i=0`` thins it.
    - ``'duplicate'`` (legacy, for A/B tests): ``n_i`` exact copies via
      ``np.repeat`` (creates Δθ=0 duplicates; biases small-scale/higher-order).

    Either way E[count per host group] = w_systot·(w_cp+w_noz−1) = w_c, so the
    weighted clustering is reproduced in the mean. Cosmology-free throughout.
    Returns ``dict(ra, dec, z, N, prov)`` where ``prov`` is a per-object
    provenance code (see :data:`PROV`).

    ``z_mode`` selects the missing-galaxy redshift engine:

    - ``'field'`` (default): a local-density (KNN) KDE of the K nearest observed
      spec-z along the sightline × p_photoz × close-pair prior. Fast, cosmology-
      free, compresses to the shareable inverse-CDF package.
    - ``'graphgp'``: the **conditional anisotropic GP posterior** density field
      (graphGP / Matheron, :func:`density_field.sample_posterior_density_field`)
      evaluated along each missing sightline — correlated across missing galaxies,
      the more flexible engine other surveys may need. Pass a precomputed
      ``gp_field`` (a ``DensityFieldResult``; realization ``seed % n_samples`` is
      used) to amortise the field solve across an ensemble; if ``None`` it is built
      from ``catalog`` with ``n_samples=1`` (one solve per call — pass ``gp_field``
      for ensembles). ``gp_kwargs`` (dict) overrides the build (nside, n_z_bins,
      r_edges, …). The fiducial cosmology in the GP prior is a gauge/unit choice
      (validated cosmology-invariant to <0.1%), so this stays data-driven.
    - ``'nn'``: nearest-neighbour host z (+ close-pair Δz for collisions). Sharp.
    - ``'photoz'``: per-object p(z|colours) only (LOS-smeared).
    """
    from .photoz import photoz_features

    rng = np.random.default_rng(seed)
    ra_o = np.asarray(catalog.ra_data, np.float64)
    dec_o = np.asarray(catalog.dec_data, np.float64)
    z_o = np.asarray(catalog.z_data, np.float64)
    wsys_o = np.asarray(catalog.w_sys_data if catalog.w_sys_data is not None
                        else np.ones(len(ra_o)))

    # ---- redshift of each missing target ----
    host = np.asarray(targets.host_index)
    z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
    miss_kind = np.asarray(targets.miss_kind)
    if dz_pool is None:
        dz_pool = measure_close_pair_dz(catalog)
    dz_pool = np.asarray(dz_pool, np.float64)
    M = len(host)
    z_miss = np.empty(M); zhost_fallback = np.zeros(M, bool)

    if z_mode == "nn":
        # SHARP, clustering-faithful assignment (BOSS w_cp/w_noz convention; Guo+ 2012):
        # broad photo-z (sigma_z~0.03 ~ 90 Mpc/h) smears the line of sight and destroys
        # 3-D redshift-space clustering, so for objects with a host we assign the
        # nearest-neighbour redshift — collisions get host z + a measured close-pair Δz
        # (true close pairs sit at ~the host z), redshift failures get the host z.
        coll = (miss_kind == "collided") & (host >= 0)
        zf = (miss_kind != "collided") & (host >= 0)
        z_miss[coll] = z_host[coll] + rng.choice(dz_pool, int(coll.sum()))
        z_miss[zf] = z_host[zf]
        nohost = host < 0
        if nohost.any():                                   # rare: no host -> photo-z / global
            feat = photoz_features(np.asarray(targets.colors)[nohost],
                                   None if targets.mags is None else np.asarray(targets.mags)[nohost])
            zk, wk = photoz.posterior(feat)
            for a, i in enumerate(np.where(nohost)[0]):
                w = wk[a]; ok = np.isfinite(w) & (w > 0)
                z_miss[i] = rng.choice(zk[a][ok], p=w[ok] / w[ok].sum()) if ok.any() else rng.choice(z_o)
                zhost_fallback[i] = True
    elif z_mode == "field":
        # PRINCIPLED, local-density (KNN) redshift estimate. A missing galaxy at angular
        # position n̂ is drawn from its LINE-OF-SIGHT density posterior
        #   p(z | n̂, colours) ∝ (1+δ_g(n̂,z)) · n̄(z) · p_photoz(z|colours)
        # where (1+δ_g)·n̄ is estimated nonparametrically as a KDE of the redshifts of
        # the K nearest observed (spec-z) galaxies — a fast, cosmology-free KNN
        # *approximation* to the conditional GP field along the sightline (the actual
        # graphGP Matheron posterior is the separate engine in density_field.py; on real
        # CMASS it recovers the same clustering, so this KNN proxy is the default).
        # p_photoz is the colour likelihood, and collisions add the
        # close-pair prior about the host. This places each galaxy on a REAL, colour-
        # consistent local structure (sharp where structure is) instead of a delta at
        # one neighbour (NN) or a broad LOS-smearing photo-z — recovering 3-D clustering.
        from scipy.spatial import cKDTree
        feat = photoz_features(targets.colors, targets.mags)
        zk, wk = photoz.posterior(feat)
        K = min(150, len(z_o))
        _, nn = cKDTree(_radec_to_nhat(ra_o, dec_o)).query(
            _radec_to_nhat(np.asarray(targets.ra), np.asarray(targets.dec)), k=K, workers=-1)
        zgrid = np.linspace(z_o.min(), z_o.max(), 256)
        bw_f, bw_p = 0.004, 0.02                            # field / photo-z KDE bandwidths
        pcl = _clpair_density(dz_pool)
        coll_i = (miss_kind == "collided") & (host >= 0)
        for i in range(M):
            znb = z_o[nn[i]]
            pf = np.exp(-0.5 * ((zgrid[:, None] - znb[None, :]) / bw_f) ** 2).sum(1)   # local field
            w = wk[i]; ok = np.isfinite(w) & (w > 0)
            pp = ((w[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
                  if ok.any() else np.ones_like(zgrid))
            p = pf * pp
            if coll_i[i]:                                   # collisions: sharpen toward the host
                p = p * pcl(zgrid - z_host[i])
            s = p.sum()
            if s > 0:
                z_miss[i] = rng.choice(zgrid, p=p / s) + rng.normal(0, bw_f * 0.5)
            else:
                z_miss[i] = z_host[i] if np.isfinite(z_host[i]) else rng.choice(z_o)
                zhost_fallback[i] = True
    elif z_mode == "graphgp":
        # CONDITIONAL ANISOTROPIC GP posterior field (graphGP / Matheron) along each
        # missing sightline — the principled, correlated version of 'field' (the more
        # flexible engine for other surveys). Pass a precomputed gp_field to amortise
        # the solve across an ensemble (realization = seed % n_samples); else build one.
        if gp_field is None:
            gp_field = build_gp_field(catalog, n_samples=1, seed=seed, **(gp_kwargs or {}))
            draw_index = 0
        else:
            draw_index = seed % gp_field.n_samples
        z_miss, zhost_fallback = _graphgp_zmiss(targets, photoz, dz_pool, gp_field, draw_index,
                                                z_o, z_host, miss_kind, rng)
    else:
        # 'photoz': per-object redshift from p(z|colours) × close-pair prior (more
        # realistic per-object z, but LOS-smeared — degrades 3-D redshift-space clustering).
        feat = photoz_features(targets.colors, targets.mags)
        zk, wk = photoz.posterior(feat)
        if clustering_prior == "data":
            pcl = _clpair_density(dz_pool)
            coll = (miss_kind == "collided") & (host >= 0)
            wk = wk.copy(); wk[coll] *= pcl(zk[coll] - z_host[coll, None])
        for i in range(len(zk)):
            w = wk[i]; ok = np.isfinite(w) & (w > 0)
            if ok.any():
                z_miss[i] = rng.choice(zk[i][ok], p=w[ok] / w[ok].sum())
            else:
                z_miss[i] = z_host[i] if np.isfinite(z_host[i]) else rng.choice(z_o)
                zhost_fallback[i] = True

    # ---- base equal-weight set: observed (spec-z) + missing (photo-z) ----
    base_ra = np.concatenate([ra_o, np.asarray(targets.ra, np.float64)])
    base_dec = np.concatenate([dec_o, np.asarray(targets.dec, np.float64)])
    base_z = np.concatenate([z_o, z_miss])
    base_wsys = np.concatenate([wsys_o, wsys_o[np.clip(host, 0, len(z_o) - 1)]])
    # base provenance: observed spec-z, then each missing target by kind (zhost if fell back)
    miss_prov = np.where(zhost_fallback, PROV["zhost"],
                         np.where(np.asarray(targets.miss_kind) == "collided",
                                  PROV["collided"], PROV["zfail"]))
    base_prov = np.concatenate([np.full(len(ra_o), PROV["observed"]), miss_prov])

    # ---- imaging-systematic completion ----
    if systot_mode == "duplicate":                         # legacy: exact duplicates (Δθ=0)
        n = (rng.poisson(base_wsys) if count == "poisson"
             else np.floor(base_wsys + rng.random(len(base_wsys))).astype(int))
        idx = np.repeat(np.arange(len(base_ra)), n)
        out_ra, out_dec, out_z, out_prov = base_ra[idx], base_dec[idx], base_z[idx], base_prov[idx]
    else:                                                  # 'analog': KEEP ALL real galaxies,
        # add only the WEIGHT_SYSTOT EXCESS as local analogs. Never drop a real detection
        # (so all correlation functions are preserved and no observed galaxy is discarded);
        # E[n_i] = 1 + max(w_systot-1,0). The excess restores the imaging-systematic deficit.
        n_extra = np.floor(np.maximum(base_wsys - 1.0, 0.0) + rng.random(len(base_wsys))).astype(int)
        src = np.repeat(np.arange(len(base_ra)), n_extra)
        ex_ra, ex_dec, ex_z = _systot_restore_extras(base_ra, base_dec, base_z, src, rng)
        out_ra = np.concatenate([base_ra, ex_ra])
        out_dec = np.concatenate([base_dec, ex_dec])
        out_z = np.concatenate([base_z, ex_z])
        out_prov = np.concatenate([base_prov, np.full(len(ex_ra), PROV["systot"])])
    if verbose:
        print(f"[complete-photoz] N_obs={len(ra_o):,} + {targets.N:,} missing "
              f"-> N_eq={len(out_ra):,} (+{100*(len(out_ra)/len(ra_o)-1):.1f}%), "
              f"mode={systot_mode}, zhost-fallback={int(zhost_fallback.sum())}")
    return {"ra": out_ra.astype(np.float32), "dec": out_dec.astype(np.float32),
            "z": out_z.astype(np.float32), "N": len(out_ra),
            "prov": out_prov.astype(np.int8)}


@perf.timed("generate_catalogs_from_kernel")
def generate_catalogs_from_kernel(
    catalog, cov, sigma2,
    *,
    alpha: float = 2.0,
    n_samples: int = 5,
    seed: int = 0,
    w_completeness=None,
    n_cand_factor: int = 20,
    n0: int = 256,
    k: int = 30,
    sampling: str = "poisson",
    chunk_size: Optional[int] = 50_000,
    verbose: bool = False,
):
    """LGCP catalogs from a *prebuilt* anisotropic kernel ``cov`` (σ²=``sigma2``).

    The generation path of the measurement-first pipeline: draw window
    candidates (sel_map × n(z)), embed as (n̂, α·z), build the GraphGP graph,
    and for each draw form the log-normal intensity exp(f − σ²/2) and
    inhomogeneous-Poisson sample to (RA, Dec, z). The window enters through the
    candidates, so a field with the *true* (deconvolved) covariance produces
    catalogs whose LS re-measurement carries the window back.
    """
    import jax
    import jax.numpy as jnp
    import graphgp as gp

    jax.config.update("jax_enable_x64", True)
    nd = catalog.N_data
    if w_completeness is None:
        w_completeness = np.ones(nd)
    w_sum = float(np.asarray(w_completeness).sum())
    n_cand = int(n_cand_factor * nd)

    rng0 = np.random.default_rng(seed)
    ra_c, dec_c, z_c = make_random_from_selection_function(
        sel_map=catalog.sel_map, n_random=n_cand,
        z_data=np.asarray(catalog.z_data), nside=catalog.nside, rng=rng0)
    ra_c = np.asarray(ra_c, np.float64); dec_c = np.asarray(dec_c, np.float64)
    z_c = np.asarray(z_c, np.float64)
    nhat_c = _radec_to_nhat(ra_c, dec_c)
    points = jnp.asarray(np.hstack([nhat_c, (alpha * z_c)[:, None]]), dtype=jnp.float64)
    if verbose:
        print(f"[K2d-gen] {n_cand:,} candidates; building graph (α={alpha}) ...")
    graph = gp.build_graph(points, n0=min(n0, n_cand // 2), k=min(k, n_cand - 1))
    sig = np.sqrt(max(sigma2, 1e-12))

    out = []
    for s in range(n_samples):
        eps = np.random.default_rng(seed + 1 + s).standard_normal(n_cand)
        f = np.asarray(gp.generate(graph, cov, jnp.asarray(eps, dtype=jnp.float64),
                                   chunk_size=chunk_size))
        f = np.where(np.isfinite(f), f, 0.0)
        f = np.clip(f, -8.0 * sig, 8.0 * sig)
        opd = np.exp(f - 0.5 * sigma2)
        opd_sum = float(opd.sum())
        a_thin = w_sum / opd_sum if opd_sum > 0 else 0.0
        rng = np.random.default_rng(1000 + seed + s)
        if sampling == "bernoulli":
            # at most one galaxy per candidate — removes the unphysical Δθ=0
            # multi-occupancy spike. Valid when the candidate density oversamples
            # the field (p<1); peaks above 1 are clipped (rare once σ² is capped).
            p = np.clip(a_thin * opd, 0.0, 1.0)
            counts = (rng.random(n_cand) < p).astype(int)
        else:
            counts = rng.poisson(a_thin * opd)
        idx = np.repeat(np.where(counts > 0)[0], counts[counts > 0])
        out.append({"ra": ra_c[idx].astype(np.float32),
                    "dec": dec_c[idx].astype(np.float32),
                    "z": z_c[idx].astype(np.float32),
                    "N_galaxies": int(len(idx)),
                    "multi_frac": float(np.mean(counts[counts > 0] > 1))})
        if verbose:
            print(f"[K2d-gen] sample {s+1}/{n_samples}: N={out[-1]['N_galaxies']:,} "
                  f"multi_frac={out[-1]['multi_frac']:.3f}")
    return out


def fkp_weight_of_z(z_query, z_data, w_fkp_data, n_bins: int = 80):
    """Smooth FKP weight as a function of redshift, learned from the data.

    The FKP weight is a deterministic function of n(z); we recover w_fkp(z) by
    binning the data's per-object ``WEIGHT_FKP`` against z and interpolating, so
    the analytic randoms can be assigned matching FKP weights.
    """
    z_data = np.asarray(z_data, np.float64)
    w_fkp_data = np.asarray(w_fkp_data, np.float64)
    edges = np.linspace(z_data.min(), z_data.max(), n_bins + 1)
    which = np.clip(np.digitize(z_data, edges) - 1, 0, n_bins - 1)
    num = np.bincount(which, weights=w_fkp_data, minlength=n_bins)
    den = np.bincount(which, minlength=n_bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ok = den > 0
    prof = np.interp(centres, centres[ok], num[ok] / den[ok])
    return np.interp(np.asarray(z_query, np.float64), centres, prof)


def measure_K2d_data(
    catalog,
    *,
    theta_edges: np.ndarray,
    z_edges: np.ndarray,
    n_data: Optional[int] = None,
    n_rand_factor: int = 4,
    seed: int = 0,
    return_counts: bool = False,
):
    """Weighted LS ξ(Δθ, Δz) of the BOSS data vs analytic randoms.

    Data carry the full FKP×completeness weight (``catalog.w_data``); the
    analytic randoms (sel_map × n(z)) are assigned FKP weights via
    :func:`fkp_weight_of_z`. ``n_data`` optionally subsamples the data (pair
    counts scale steeply with N — use the full set only for the final K_in).
    Returns the same as :func:`measure_K2d`.
    """
    rng = np.random.default_rng(seed)
    z_all = np.asarray(catalog.z_data)          # full n(z) and FKP(z) profile
    ra_d = np.asarray(catalog.ra_data); dec_d = np.asarray(catalog.dec_data)
    z_d = z_all; w_d = np.asarray(catalog.w_data)
    if n_data is not None and n_data < len(ra_d):
        sel = rng.choice(len(ra_d), n_data, replace=False)
        ra_d, dec_d, z_d, w_d = ra_d[sel], dec_d[sel], z_d[sel], w_d[sel]
    nr = n_rand_factor * len(ra_d)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=catalog.sel_map, n_random=nr, z_data=z_all, nside=catalog.nside, rng=rng)
    if catalog.w_fkp_data is not None:
        w_r = fkp_weight_of_z(z_r, z_all, catalog.w_fkp_data)
    else:
        w_r = np.ones(len(ra_r))
    return measure_K2d(ra_d, dec_d, z_d, w_d, ra_r, dec_r, z_r, w_r,
                       theta_edges=theta_edges, z_edges=z_edges,
                       return_counts=return_counts)


def kernel_from_K2d(
    theta_edges, z_edges, xi_true,
    *,
    alpha: float = 2.0,
    jitter: float = 0.02,
    theta_cap_deg: float = 0.0,
    n_ltheta: int = 12,
    n_lz: int = 8,
    n_s: int = 512,
    n_zg: int = 256,
):
    """PSD ``AnisotropicCovariance`` that reproduces the measured 2D K.

    The target is the measured/deconvolved log-kernel K = ln(1+ξ_true) on the
    (Δθ, Δz) grid. We represent it with a **dense** non-negative bank of
    tensor-product Matérns fit by NNLS — PSD by the Schur product theorem (so no
    NaN-field failure), and rich enough (``n_ltheta × n_lz`` components) to track
    the measured K closely rather than impose a smooth parametric shape. The
    grid is evaluated on a fine (chord, Δz) mesh for GraphGP.

    ``theta_cap_deg`` (off by default) floors the narrowest Matérn scale in the
    basis bank, mildly bounding σ². NOTE: it only reduces σ² modestly (the
    zero-lag σ²=ΣA_k is forced up by the measured K≈2.8 at the smallest bins),
    because a log-normal field fundamentally needs σ² ≥ K(smallest reproduced
    scale). A *hard* flatten of the core would cut σ² more but is incompatible
    with random candidates — most points have a neighbour inside the flat core,
    making those Vecchia blocks degenerate and collapsing the field.

    ``alpha`` is only the graph embedding scale (it cancels from the kernel
    value). Returns ``(AnisotropicCovariance, sigma2)``.
    """
    from scipy.optimize import nnls
    import graphgp as gp

    theta_c = np.empty(len(theta_edges) - 1)
    theta_c[0] = 0.5 * theta_edges[1]
    theta_c[1:] = np.sqrt(theta_edges[1:-1] * theta_edges[2:])
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])
    chord_c = 2.0 * np.sin(np.radians(theta_c) / 2.0)
    KG = np.log1p(np.clip(np.asarray(xi_true, np.float64), 0.0, None))

    lt_min = 0.5 * chord_c[0]
    if theta_cap_deg:
        lt_min = max(lt_min, 2.0 * np.sin(np.radians(theta_cap_deg) / 2.0))
    lthetas = np.geomspace(lt_min, 2.0 * chord_c[-1], n_ltheta)
    lzs = np.geomspace(0.5 * max(z_c[0], 1e-4), 2.0 * z_c[-1], n_lz)
    cols, scales = [], []
    for lt in lthetas:
        mt = _matern1(chord_c, lt)
        for lz in lzs:
            cols.append(np.outer(mt, _matern1(z_c, lz)).ravel())
            scales.append((lt, lz))
    coeffs, _ = nnls(np.stack(cols, axis=1), KG.ravel())

    sb = np.concatenate([[0.0], np.geomspace(1e-5, 1.5 * chord_c[-1], n_s - 1)])
    zb = np.concatenate([[0.0], np.geomspace(1e-5, 2.0 * z_c[-1], n_zg - 1)])
    grid = np.zeros((len(sb), len(zb)))
    for (lt, lz), a in zip(scales, coeffs):
        if a > 0:
            grid += a * np.outer(_matern1(sb, lt), _matern1(zb, lz))
    cov = gp.build_anisotropic_covariance(sb, zb, grid, float(alpha), jitter=jitter)
    return cov, float(grid[0, 0] * (1.0 + jitter))


def _matern1(d, ell):
    """Matérn ν=3/2 correlation, (1 + √3 d/ℓ) exp(−√3 d/ℓ)."""
    u = np.sqrt(3.0) * np.asarray(d, np.float64) / ell
    return (1.0 + u) * np.exp(-u)


def deconvolve_window(xi, rr_norm):
    """Integral-constraint deconvolution of the LS ξ to the true clustering.

    A finite survey cannot constrain the mean density, so the Landy-Szalay
    estimator is biased low by the integral constraint — a single constant
    offset (window mode-coupling beyond this is negligible at θ ≲ 2°, far below
    the footprint scale):

        ξ_LS(s) = ξ_true(s) − IC,   IC = Σ_all-s RR_norm(s) ξ_true(s),

    where ``RR_norm`` is the random-random count **normalised by the total
    number of random pairs** (so it sums to 1 over the *whole* footprint, not
    just the measured θ-range). Because ξ→0 beyond the measured range, the sum
    is carried by the measured bins; to first order ξ_true ≈ ξ_LS there:

        IC ≈ Σ_measured RR_norm(s) ξ_LS(s),    ξ_true = ξ_LS + IC.

    Normalising by the *total* pairs (≈0.5·W_r²) — not by Σ over the measured
    bins — is essential: over a ~3000 deg² footprint the true IC is ~1e-3, i.e.
    LS already recovers the window-corrected clustering at θ ≲ 2°. (Dividing by
    the measured-range RR instead overestimates IC by the ratio of the footprint
    area to the measured area.) Pass the normalised ``rr`` from ``measure_K2d``.

    Returns ``(xi_true, ic)``.
    """
    rr = np.asarray(rr_norm, np.float64); xi = np.asarray(xi, np.float64)
    ic = float((rr * xi).sum())
    return xi + ic, ic
