"""DESI Y1 QSO vs Quaia G<20: side-by-side kNN-CDF comparison HTML.

Loads BOTH sets of pipeline artifacts (Quaia and DESI, both run on the
matched z range and theta grid) and renders comparison panels in a
single tabbed HTML at ``output/knn_cdf_desi_quaia_presentation.html``.

Conventions:
- Quaia: blue, solid circles
- DESI:  orange, solid squares
- Curves overlaid where the comparison is direct (sigma^2, xi).
- Side-by-side columns where overlay is too busy (CIC PMF, P(N=k)).

Inputs (must exist):
- output/quaia_full_dd.npz, _rd_1x_kmax.npz, _rd_0p2x.npz, _rr.npz
- output/desi_full_dd.npz, _rd_1x_kmax.npz, _rd_0p2x.npz, _rr.npz
"""

from __future__ import annotations

import base64
import io
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import poisson

from twopt_density.distance import DistanceCosmo, comoving_distance
from twopt_density.knn_cdf import KnnCdfResult
from twopt_density.knn_derived import (
    sigma2_clust, cic_pmf, xi_dp, xi_ls, xi_ls_annular, xi_ls_cross,
    mean_count,
)
from twopt_density.jackknife import jackknife_region_labels


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
HTML = os.path.join(OUTPUT_DIR, "knn_cdf_desi_quaia_presentation.html")


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _result_from_artifact(art, prefix, theta_key, flavor):
    sum_n = art[f"{prefix}_sum_n"]
    H_key = f"{prefix}_H_geq_k"
    H = art[H_key] if H_key in art.files else np.zeros(
        sum_n.shape + (1,), dtype=np.int64,
    )
    H_pr_key = f"{prefix}_H_geq_k_per_region"
    H_pr = art[H_pr_key] if H_pr_key in art.files else None
    # Detect diagonal-only layout from cube shape:
    #   full sum_n is 3D (n_theta, n_z_q, n_z_n); diagonal is 2D (n_theta, n_z).
    is_diagonal = (sum_n.ndim == 2)

    def _opt(name):
        """Read optional field if present (back-compat with older
        artifacts that don't carry sum_n3/sum_n4)."""
        return art[name] if name in art.files else None

    return KnnCdfResult(
        H_geq_k=H, sum_n=sum_n, sum_n2=art[f"{prefix}_sum_n2"],
        N_q=art[f"{prefix}_N_q"], theta_radii_rad=art[theta_key],
        z_q_edges=art["z_q_edges"], z_n_edges=art["z_n_edges"],
        flavor=flavor, backend_used="numba",
        area_per_cap=2.0 * np.pi * (1.0 - np.cos(art[theta_key])),
        sum_n_per_region=art[f"{prefix}_sum_n_per_region"],
        sum_n2_per_region=art[f"{prefix}_sum_n2_per_region"],
        N_q_per_region=art[f"{prefix}_N_q_per_region"],
        H_geq_k_per_region=H_pr,
        is_diagonal=is_diagonal,
        sum_n3=_opt(f"{prefix}_sum_n3"),
        sum_n4=_opt(f"{prefix}_sum_n4"),
        sum_n3_per_region=_opt(f"{prefix}_sum_n3_per_region"),
        sum_n4_per_region=_opt(f"{prefix}_sum_n4_per_region"),
    )


def _diag(arr, iq):
    """Extract z_q=z_n=iq diagonal from a full or diagonal cube along
    the (z_q, z_n) plane. Full cubes have ndim >= 3 with shape
    (n_theta, n_z_q, n_z_n, ...); diagonal cubes have ndim >= 2 with
    shape (n_theta, n_z, ...). Returns array with the (z_q, z_n) (or
    z) axes collapsed to a scalar."""
    if arr.ndim >= 3 and arr.shape[1] == arr.shape[2]:
        return arr[:, iq, iq]
    return arr[:, iq]


def _diag_pmf(arr, iq, k):
    """Index a PMF-like cube at the z_q=z_n=iq diagonal slice and the
    ``k`` axis. ``k`` may be a single int or a slice. Handles full
    cubes (n_theta, n_z_q, n_z_n, k_max+1) and diagonal cubes
    (n_theta, n_z, k_max+1)."""
    if arr.ndim == 4:
        return arr[:, iq, iq, k]
    return arr[:, iq, k]


def _full_diag_2d(arr):
    """Return a (n_z, n_theta) 2D array of the diagonal z_q=z_n slice
    suitable for `pcolormesh`. Accepts full cubes (n_theta, n_z_q,
    n_z_n[, ...]) or diagonal cubes (n_theta, n_z[, ...]). Trailing
    axes (e.g. k) are not supported here.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
        # full (n_theta, n_z, n_z) → take diagonal
        n_z = arr.shape[1]
        out = np.empty((n_z, arr.shape[0]), dtype=arr.dtype)
        for i in range(n_z):
            out[i, :] = arr[:, i, i]
        return out
    if arr.ndim == 2:
        # diagonal (n_theta, n_z) → transpose to (n_z, n_theta)
        return arr.T
    raise ValueError(
        f"_full_diag_2d: unsupported shape {arr.shape}")


def _theta_edges_for_pcolormesh(theta_deg_arr):
    """Compute log-mid theta bin edges for `pcolormesh(... shading='flat')`."""
    log_th = np.log10(theta_deg_arr)
    log_edges = np.concatenate([
        [log_th[0] - 0.5 * (log_th[1] - log_th[0])],
        0.5 * (log_th[:-1] + log_th[1:]),
        [log_th[-1] + 0.5 * (log_th[-1] - log_th[-2])],
    ])
    return 10.0 ** log_edges


def _heatmap_panel_diag(ax, arr, theta_deg_arr, z_edges_arr,
                         cmap="viridis", vmin=None, vmax=None,
                         logscale=False, label=None, title=None):
    """Draw a (z, θ) heatmap of the diagonal z_q=z_n slice of `arr`.

    `arr` may be a full cube (n_theta, n_z, n_z[, ...]) or diagonal
    cube (n_theta, n_z[, ...]). `z_edges_arr` is `(n_z + 1,)` shell
    edges; `theta_deg_arr` is `(n_theta,)` shell midpoints in
    degrees. Returns the QuadMesh for further colorbar work.
    """
    data = _full_diag_2d(arr)  # (n_z, n_theta)
    if logscale:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.log10(np.where(data > 0, data, np.nan))
    th_edges = _theta_edges_for_pcolormesh(theta_deg_arr)
    pc = ax.pcolormesh(th_edges, z_edges_arr, data,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="flat")
    ax.set_xscale("log")
    ax.set_xlabel("θ [deg]")
    if label is not None:
        ax.set_ylabel(label)
    if title is not None:
        ax.set_title(title, fontsize=10)
    return pc


def _per_region_sigma2_samples(res):
    """Per-region jackknife samples of σ²_clust. Handles both full
    cubes (sum_n_per_region.ndim == 4) and diagonal cubes (ndim == 3).
    Returns shape (n_regions, n_theta, n_z_q, n_z_n) for full,
    (n_regions, n_theta, n_z) for diagonal."""
    pr = res.sum_n_per_region
    n_regions = pr.shape[-1]
    is_diag = (pr.ndim == 3)  # (n_theta, n_z, n_regions)
    samples = np.zeros((n_regions,) + pr.shape[:-1], dtype=np.float64)
    for k in range(n_regions):
        keep = np.ones(n_regions, dtype=bool); keep[k] = False
        sum_n_k = res.sum_n_per_region[..., keep].sum(axis=-1)
        sum_n2_k = res.sum_n2_per_region[..., keep].sum(axis=-1)
        N_q_k = res.N_q_per_region[..., keep].sum(axis=-1)
        Nq = N_q_k.astype(np.float64)
        safe = np.where(Nq > 0, Nq, np.inf)
        if is_diag:
            mu = sum_n_k / safe[None, :]
            var = sum_n2_k / safe[None, :] - mu * mu
        else:
            mu = sum_n_k / safe[None, :, None]
            var = sum_n2_k / safe[None, :, None] - mu * mu
        samples[k] = np.where(mu > 0, var / mu**2 - 1.0/mu, np.nan)
    return samples


def _per_region_xi_dp_samples(res_dd, res_rd):
    """Per-region jackknife samples of xi_DP. Shape-aware for both
    full and diagonal cubes."""
    pr = res_dd.sum_n_per_region
    n_regions = pr.shape[-1]
    is_diag = (pr.ndim == 3)
    samples = np.zeros((n_regions,) + pr.shape[:-1], dtype=np.float64)
    for k in range(n_regions):
        keep = np.ones(n_regions, dtype=bool); keep[k] = False
        sd = res_dd.sum_n_per_region[..., keep].sum(axis=-1)
        sr = res_rd.sum_n_per_region[..., keep].sum(axis=-1)
        Nd = res_dd.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        Nr = res_rd.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        if is_diag:
            nbar_dd = sd / np.where(Nd > 0, Nd, np.inf)[None, :]
            nbar_rd = sr / np.where(Nr > 0, Nr, np.inf)[None, :]
        else:
            nbar_dd = sd / np.where(Nd > 0, Nd, np.inf)[None, :, None]
            nbar_rd = sr / np.where(Nr > 0, Nr, np.inf)[None, :, None]
        samples[k] = nbar_dd / np.where(nbar_rd > 0, nbar_rd, np.inf) - 1.0
    return samples


def sigma2_poisson_window(
    sel_maps: dict,        # {region: hp_map}; pass a single-region dict for
                           # surveys without per-region maps
    region_of_cap: np.ndarray,   # length-N array of region labels per cap
    ra_caps: np.ndarray, dec_caps: np.ndarray,
    theta_radii_rad: np.ndarray,
    nside_mask: int,
    n_subsample: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """σ²(θ) for an unbiased Poisson process sampling the SAME window
    function (and same cap-centre selection), with NO random catalog.

    Derivation: a compound Poisson source population with rate
    λ × W(Ω) × n(z) gives, in a cap of solid angle V_cap centred on q,

        ⟨N | q⟩ = λ · A_w(q, θ),     A_w(q,θ) ≡ ∫_cap(q,θ) W(Ω) dΩ
        Var(N | q) = λ · A_w(q, θ)        (Poisson)

    Averaging over cap centres q,

        ⟨N⟩      = λ · ⟨A_w⟩_q
        Var(N)   = λ · ⟨A_w⟩_q + λ² · Var_q(A_w)

    so σ²_clust(θ) = Var(N)/⟨N⟩² − 1/⟨N⟩
                    = Var_q(A_w) / ⟨A_w⟩_q²

    — purely a window-function quantity. λ cancels; randoms aren't
    needed. Returns a length-n_theta array.

    The expensive bit is the per-cap pixel sum; we subsample
    ``n_subsample`` cap centres (drawn uniformly at random from the
    inputs) for a stable Var/⟨⟩² estimate. The result is an
    unconditional window quantity, so the subsample doesn't bias it.
    """
    import healpy as hp

    rng = np.random.default_rng(seed)
    n_caps_total = ra_caps.size
    if n_caps_total == 0:
        return np.full(theta_radii_rad.size, np.nan)
    take = (rng.choice(n_caps_total,
                       min(n_subsample, n_caps_total), replace=False)
            if n_caps_total > n_subsample else np.arange(n_caps_total))
    ra_s = ra_caps[take]; dec_s = dec_caps[take]
    region_s = region_of_cap[take]

    pix_area = hp.nside2pixarea(nside_mask)
    theta_q = np.deg2rad(90.0 - dec_s)
    phi_q = np.deg2rad(ra_s % 360.0)
    vecs_q = hp.ang2vec(theta_q, phi_q)
    theta_max = float(theta_radii_rad.max())

    n_t = theta_radii_rad.size
    A_w = np.zeros((take.size, n_t), dtype=np.float64)
    for i in range(take.size):
        ipix = hp.query_disc(nside_mask, vecs_q[i], theta_max,
                              inclusive=True)
        if ipix.size == 0:
            continue
        sel = sel_maps[region_s[i]]
        theta_pix, phi_pix = hp.pix2ang(nside_mask, ipix)
        # great-circle separation
        cos_sep = (np.sin(theta_q[i]) * np.sin(theta_pix)
                   * np.cos(phi_q[i] - phi_pix)
                   + np.cos(theta_q[i]) * np.cos(theta_pix))
        sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
        w_pix = sel[ipix]
        # cumulative sum over sorted separations -> A_w(theta) for free
        order = np.argsort(sep)
        sep_s = sep[order]; w_s = w_pix[order]
        cum = np.cumsum(w_s) * pix_area
        # for each requested theta, pick the cumulative value at the
        # last pixel with sep <= theta
        idx = np.searchsorted(sep_s, theta_radii_rad, side="right") - 1
        for t in range(n_t):
            A_w[i, t] = cum[idx[t]] if idx[t] >= 0 else 0.0

    mean_A = A_w.mean(axis=0)
    var_A = A_w.var(axis=0)
    return np.where(mean_A > 0, var_A / mean_A ** 2, np.nan)


def _per_shell_weight_sum(z_data: np.ndarray, w_data: np.ndarray,
                          z_edges: np.ndarray) -> np.ndarray:
    """Sum of per-object weights of ``z_data`` falling into each shell
    defined by ``z_edges``."""
    sums = np.zeros(z_edges.size - 1, dtype=np.float64)
    for iz in range(z_edges.size - 1):
        mask = (z_data >= z_edges[iz]) & (z_data <= z_edges[iz + 1])
        sums[iz] = float(w_data[mask].sum())
    return sums


def _per_region_xi_ls_samples(res_dd, res_dr, res_rr,
                              n_neigh_dd_per_zn, n_neigh_dr_per_zn,
                              n_neigh_rr_per_zn):
    """Per-region jackknife samples of σ²_LS = ⟨ξ⟩_cap. Shape-aware."""
    pr = res_dd.sum_n_per_region
    n_regions = pr.shape[-1]
    is_diag = (pr.ndim == 3)
    safe_dd_n = np.where(n_neigh_dd_per_zn > 0,
                         n_neigh_dd_per_zn, np.inf).astype(np.float64)
    safe_dr_n = np.where(n_neigh_dr_per_zn > 0,
                         n_neigh_dr_per_zn, np.inf).astype(np.float64)
    safe_rr_n = np.where(n_neigh_rr_per_zn > 0,
                         n_neigh_rr_per_zn, np.inf).astype(np.float64)
    samples = np.zeros((n_regions,) + pr.shape[:-1], dtype=np.float64)
    for k in range(n_regions):
        keep = np.ones(n_regions, dtype=bool); keep[k] = False
        sd = res_dd.sum_n_per_region[..., keep].sum(axis=-1)
        sr = res_dr.sum_n_per_region[..., keep].sum(axis=-1)
        srr = res_rr.sum_n_per_region[..., keep].sum(axis=-1)
        Nd = res_dd.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        Nr = res_dr.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        Nrr = res_rr.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        if is_diag:
            nbar_dd = sd / np.where(Nd > 0, Nd, np.inf)[None, :]
            nbar_dr = sr / np.where(Nr > 0, Nr, np.inf)[None, :]
            nbar_rr = srr / np.where(Nrr > 0, Nrr, np.inf)[None, :]
            mu_dd = nbar_dd / safe_dd_n[None, :]
            mu_dr = nbar_dr / safe_dr_n[None, :]
            mu_rr = nbar_rr / safe_rr_n[None, :]
        else:
            nbar_dd = sd / np.where(Nd > 0, Nd, np.inf)[None, :, None]
            nbar_dr = sr / np.where(Nr > 0, Nr, np.inf)[None, :, None]
            nbar_rr = srr / np.where(Nrr > 0, Nrr, np.inf)[None, :, None]
            mu_dd = nbar_dd / safe_dd_n[None, None, :]
            mu_dr = nbar_dr / safe_dr_n[None, None, :]
            mu_rr = nbar_rr / safe_rr_n[None, None, :]
        denom = np.where(mu_rr > 0, mu_rr, np.inf)
        samples[k] = (mu_dd - 2.0 * mu_dr + mu_rr) / denom
    return samples


def _per_region_xi_ls_annular_samples(res_dd, res_dr, res_rr,
                                       n_neigh_dd_per_zn,
                                       n_neigh_dr_per_zn,
                                       n_neigh_rr_per_zn):
    """Per-region jackknife samples of the TRUE differential ξ_LS(θ).
    Same as ``_per_region_xi_ls_samples`` but applies LS to per-annulus
    pair counts (np.diff along θ with a zero pre-pad). Shape-aware for
    full and diagonal cubes."""
    pr = res_dd.sum_n_per_region
    n_regions = pr.shape[-1]
    is_diag = (pr.ndim == 3)
    safe_dd_n = np.where(n_neigh_dd_per_zn > 0,
                         n_neigh_dd_per_zn, np.inf).astype(np.float64)
    safe_dr_n = np.where(n_neigh_dr_per_zn > 0,
                         n_neigh_dr_per_zn, np.inf).astype(np.float64)
    safe_rr_n = np.where(n_neigh_rr_per_zn > 0,
                         n_neigh_rr_per_zn, np.inf).astype(np.float64)

    def _ann(arr):
        # arr shape (n_theta, n_z_q, n_z_n) or (n_theta, n_z); diff
        # along axis 0 with a zero pre-pad so output[0] = arr[0].
        pad = np.concatenate([np.zeros((1,) + arr.shape[1:]), arr], axis=0)
        return np.diff(pad, axis=0)

    samples = np.zeros((n_regions,) + pr.shape[:-1], dtype=np.float64)
    for k in range(n_regions):
        keep = np.ones(n_regions, dtype=bool); keep[k] = False
        sd = res_dd.sum_n_per_region[..., keep].sum(axis=-1)
        sr = res_dr.sum_n_per_region[..., keep].sum(axis=-1)
        srr = res_rr.sum_n_per_region[..., keep].sum(axis=-1)
        Nd = res_dd.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        Nr = res_dr.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        Nrr = res_rr.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
        if is_diag:
            nbar_dd = _ann(sd) / np.where(Nd > 0, Nd, np.inf)[None, :]
            nbar_dr = _ann(sr) / np.where(Nr > 0, Nr, np.inf)[None, :]
            nbar_rr = _ann(srr) / np.where(Nrr > 0, Nrr, np.inf)[None, :]
            mu_dd = nbar_dd / safe_dd_n[None, :]
            mu_dr = nbar_dr / safe_dr_n[None, :]
            mu_rr = nbar_rr / safe_rr_n[None, :]
        else:
            nbar_dd = _ann(sd) / np.where(Nd > 0, Nd, np.inf)[None, :, None]
            nbar_dr = _ann(sr) / np.where(Nr > 0, Nr, np.inf)[None, :, None]
            nbar_rr = _ann(srr) / np.where(Nrr > 0, Nrr, np.inf)[None, :, None]
            mu_dd = nbar_dd / safe_dd_n[None, None, :]
            mu_dr = nbar_dr / safe_dr_n[None, None, :]
            mu_rr = nbar_rr / safe_rr_n[None, None, :]
        denom = np.where(mu_rr > 0, mu_rr, np.inf)
        samples[k] = (mu_dd - 2.0 * mu_dr + mu_rr) / denom
    return samples


def _se(samples):
    """Per-cell jackknife standard error.

    With fine z-binning some cells (high-z, low-density) have NaN
    samples in some regions because the relevant shell was empty
    after dropping that region. Per-cell finite-sample averaging
    handles this gracefully; cells with fewer than 2 finite samples
    return NaN. Symbol shape:
        samples: (n_regions, n_theta, n_z, ...) or
                 (n_regions, n_theta, n_z_q, n_z_n, ...)
        returns: shape of samples without the first axis.
    """
    finite = np.isfinite(samples)
    n_used = finite.sum(axis=0).astype(np.float64)
    masked = np.where(finite, samples, 0.0)
    mean = masked.sum(axis=0) / np.maximum(n_used, 1.0)
    diff = np.where(finite, samples - mean, 0.0)
    factor = np.where(n_used > 1, (n_used - 1.0) / n_used, np.nan)
    var = factor * (diff ** 2).sum(axis=0)
    return np.sqrt(np.maximum(var, 0.0))


def load_catalog(prefix, label):
    """Load DD + RD + RR (+ optional DR, offdiag) + chi(z) for one
    catalog. Returns dict of cubes + derived arrays.

    DR and off-diagonal artifacts are optional (note v4_1 §2-5 add-ons);
    if missing the corresponding `res_dr` / `offdiag_artifact` slots
    are None and downstream panels gracefully degrade.
    """
    dd = np.load(os.path.join(OUTPUT_DIR, f"{prefix}_full_dd.npz"))
    rd = np.load(os.path.join(OUTPUT_DIR, f"{prefix}_full_rd_1x_kmax.npz"))
    rr_path = os.path.join(OUTPUT_DIR, f"{prefix}_full_rr.npz")
    rr = np.load(rr_path) if os.path.exists(rr_path) else None
    dr_path = os.path.join(OUTPUT_DIR, f"{prefix}_full_dr.npz")
    dr = np.load(dr_path) if os.path.exists(dr_path) else None
    off_path = os.path.join(OUTPUT_DIR, f"{prefix}_full_offdiag.npz")
    offdiag = np.load(off_path) if os.path.exists(off_path) else None

    res_dd = _result_from_artifact(dd, "dd_bao", "theta_radii_bao_rad", "DD")
    res_rd = _result_from_artifact(rd, "rd", "theta_radii_rad", "RD")
    res_rr = _result_from_artifact(rr, "rr", "theta_radii_rad", "DD") if rr is not None else None
    res_dr = _result_from_artifact(dr, "dr", "theta_radii_rad", "DR") if dr is not None else None

    z_q_edges = dd["z_q_edges"]
    z_q_mid = 0.5 * (z_q_edges[:-1] + z_q_edges[1:])
    theta = dd["theta_radii_bao_rad"]

    fid = DistanceCosmo(Om=0.31, h=0.68)
    import jax.numpy as jnp
    chi_at_z = np.asarray(comoving_distance(jnp.asarray(z_q_mid), fid))

    print(f"  {label}: theta {np.degrees(theta).min():.2f}-"
          f"{np.degrees(theta).max():.2f} deg, "
          f"{theta.size} bins, z shells {np.round(z_q_edges, 3)}, "
          f"N_data={int(dd['n_d'])}"
          + (f"  [DR loaded]" if dr is not None else "")
          + (f"  [offdiag n_z={offdiag['z_q_edges'].size - 1}]"
             if offdiag is not None else ""))

    return dict(
        label=label, prefix=prefix,
        dd_artifact=dd, rd_artifact=rd, rr_artifact=rr,
        dr_artifact=dr, offdiag_artifact=offdiag,
        res_dd=res_dd, res_rd=res_rd, res_rr=res_rr, res_dr=res_dr,
        theta=theta, z_q_edges=z_q_edges, z_q_mid=z_q_mid,
        chi_at_z=chi_at_z, n_z_q=z_q_mid.size, n_d=int(dd["n_d"]),
        n_regions=int(dd["n_regions"]),
        k_max=int(dd["k_max"]),
    )


def derive_xi_panel(cat, data_weight_sum_per_zn=None):
    """Compute xi_DP and xi_LS plus jackknife errors for one catalog.

    Parameters
    ----------
    cat
        Dict from ``load_catalog``.
    data_weight_sum_per_zn
        Optional ``(n_z_n,)`` per-shell SUM of per-object data weights
        in the data catalog. Required for any catalog whose DD/DR pass
        was run with ``weights_neigh != None`` (e.g. DESI's
        WEIGHT*WEIGHT_FKP), otherwise ``xi_LS`` picks up a constant
        offset of ``1 - <w>`` — the data side carries weights but the
        normaliser doesn't, so DD and DR are inflated relative to RR.
        For an unweighted catalog (Quaia) leave ``None`` to fall back
        to the per-shell count from ``res_dd.N_q``.
    """
    res_dd, res_rd, res_rr = cat["res_dd"], cat["res_rd"], cat["res_rr"]

    xi_dp_cube = xi_dp(res_dd, res_rd)
    xi_dp_samples = _per_region_xi_dp_samples(res_dd, res_rd)
    se_xi_dp = _se(xi_dp_samples)

    if res_rr is None:
        return dict(xi_dp=xi_dp_cube, se_xi_dp=se_xi_dp,
                    xi_ls=None, se_xi_ls=None)

    # Per-shell normalisation for LS density. For the DD and DR passes,
    # which were run with weights_neigh = w_data (DESI) or no weights
    # (Quaia), the right normaliser is the sum of per-object weights in
    # the corresponding z_n shell. RR was run unweighted, so its
    # normaliser is the unweighted count (== res_rr.N_q since queries
    # == neighbors).
    if data_weight_sum_per_zn is not None:
        n_neigh_dd = np.asarray(data_weight_sum_per_zn, dtype=np.float64)
    else:
        n_neigh_dd = res_dd.N_q.astype(np.float64)
    n_neigh_dr = n_neigh_dd               # DR neighbor catalog == data
    n_neigh_rr = res_rr.N_q.astype(np.float64)

    xi_ls_cube = xi_ls(res_dd, res_rd, res_rr,
                       n_neigh_dd, n_neigh_dr, n_neigh_rr)
    xi_ls_samples = _per_region_xi_ls_samples(
        res_dd, res_rd, res_rr,
        n_neigh_dd, n_neigh_dr, n_neigh_rr,
    )
    se_xi_ls = _se(xi_ls_samples)

    # NOTE: True differential ξ_LS(θ) is computed inline in main() on a
    # coarsened θ grid (the 90-bin grid is too fine for per-annulus
    # pair counts; the 12-edge coarse grid gives ~11 bins with usable
    # signal-to-noise per bin).

    return dict(xi_dp=xi_dp_cube, se_xi_dp=se_xi_dp,
                xi_ls=xi_ls_cube, se_xi_ls=se_xi_ls)


def _load_cross_artifact(path):
    """Load one quaia_x_desi_*.npz pass into a KnnCdfResult-shaped object."""
    art = np.load(path, allow_pickle=True)
    sum_n = art["sum_n"]
    return KnnCdfResult(
        H_geq_k=np.zeros(sum_n.shape + (1,), dtype=np.int64),
        sum_n=sum_n, sum_n2=art["sum_n2"], N_q=art["N_q"],
        theta_radii_rad=art["theta_radii_rad"],
        z_q_edges=art["z_q_edges"], z_n_edges=art["z_n_edges"],
        flavor=str(art["label"]), backend_used="numba",
        area_per_cap=2.0 * np.pi * (1.0 - np.cos(art["theta_radii_rad"])),
    )


def load_cross_artifacts():
    """Load the four quaia_x_desi_* cross artifacts. Returns dict or
    None if any are missing."""
    paths = {k: os.path.join(OUTPUT_DIR, f"quaia_x_desi_{k}.npz")
             for k in ("dd", "dr", "rd", "rr")}
    if not all(os.path.exists(p) for p in paths.values()):
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        print(f"  cross artifacts missing: {missing}; "
              "run demos/quaia_x_desi_pipeline.py")
        return None
    res = {k: _load_cross_artifact(p) for k, p in paths.items()}
    print(f"  cross: theta {np.degrees(res['dd'].theta_radii_rad).min():.2f}"
          f"-{np.degrees(res['dd'].theta_radii_rad).max():.2f} deg, "
          f"{res['dd'].theta_radii_rad.size} bins; "
          f"DD shape {res['dd'].sum_n.shape}")
    return res


def main():
    print("loading Quaia + DESI artifacts ...")
    quaia = load_catalog("quaia", "Quaia G<20")
    desi = load_catalog("desi", "DESI Y1 QSO")
    print("loading DESI x Quaia cross artifacts ...")
    cross = load_cross_artifacts()

    # Sanity: matching grid for direct comparison
    assert np.array_equal(quaia["theta"], desi["theta"]), \
        "theta grids must match for side-by-side"
    assert np.array_equal(quaia["z_q_edges"], desi["z_q_edges"]), \
        "z grids must match"
    theta_deg = np.degrees(quaia["theta"])
    z_q_mid = quaia["z_q_mid"]
    n_z_q = quaia["n_z_q"]
    chi_at_z = quaia["chi_at_z"]   # same cosmology -> same chi

    # When n_z_q is large (e.g. 64 in diagonal-only mode), per-shell
    # column grids become unreadable. Pick 4 representative z indices
    # (quartile midpoints in shell space) for the column-per-shell
    # panels; the full z range is shown in the heatmap variants.
    N_Z_PLOT = min(4, n_z_q)
    if n_z_q <= N_Z_PLOT:
        Z_PLOT_IDX = list(range(n_z_q))
    else:
        Z_PLOT_IDX = np.round(
            (np.arange(N_Z_PLOT) + 0.5) * n_z_q / N_Z_PLOT
        ).astype(int).tolist()
    n_z_plot = len(Z_PLOT_IDX)
    z_plot_mid = [z_q_mid[i] for i in Z_PLOT_IDX]
    chi_plot = [chi_at_z[i] for i in Z_PLOT_IDX]
    print(f"  z-shell plot indices: {Z_PLOT_IDX} "
          f"(z = {[f'{z:.2f}' for z in z_plot_mid]})")

    # Load raw catalogs early — we need DESI per-shell weight sums to
    # normalise xi_LS for DESI's weighted DD/DR (otherwise the LS sits
    # at a constant offset of 1 - <w> ≈ 0.13–0.17 at large theta).
    print("\nloading raw catalog positions ...")
    from twopt_density.distance import DistanceCosmo
    from twopt_density.quaia import load_quaia, load_selection_function
    from twopt_density.desi import load_desi_qso
    fid = DistanceCosmo(Om=0.31, h=0.68)
    q_cat = load_quaia(
        catalog_path="/Users/tabel/Research/data/quaia/quaia_G20.0.fits",
        selection_path="/Users/tabel/Research/data/quaia/selection_function_NSIDE64_G20.0.fits",
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    md_q = (q_cat.z_data >= 0.8) & (q_cat.z_data <= 2.1)
    q_ra, q_dec, q_z = q_cat.ra_data[md_q], q_cat.dec_data[md_q], q_cat.z_data[md_q]
    d_cat = load_desi_qso(
        catalog_paths=[os.path.join(REPO_ROOT, "data/desi/QSO_NGC_clustering.dat.fits"),
                       os.path.join(REPO_ROOT, "data/desi/QSO_SGC_clustering.dat.fits")],
        randoms_paths=None, fid_cosmo=fid, z_min=0.8, z_max=2.1, with_weight_fkp=True,
    )
    desi_w_sum_per_zn = _per_shell_weight_sum(
        d_cat.z_data, d_cat.w_data, desi["z_q_edges"])
    print(f"  DESI per-shell <w>: "
          f"{[f'{(s/n):.3f}' for s, n in zip(desi_w_sum_per_zn, desi['res_dd'].N_q)]}")
    print(f"  -> LS large-theta plateau without this fix would be 1-<w>")

    print("\nderiving sigma^2_clust ...")
    s2_q = sigma2_clust(quaia["res_dd"])
    s2_d = sigma2_clust(desi["res_dd"])
    se_s2_q = _se(_per_region_sigma2_samples(quaia["res_dd"]))
    se_s2_d = _se(_per_region_sigma2_samples(desi["res_dd"]))

    print("deriving xi (DP + LS) ...")
    xi_q = derive_xi_panel(quaia)                       # Quaia unweighted
    xi_d = derive_xi_panel(desi, data_weight_sum_per_zn=desi_w_sum_per_zn)

    print("deriving CIC PMF ...")
    pmf_q = cic_pmf(quaia["res_dd"])
    pmf_d = cic_pmf(desi["res_dd"])

    print("\nrendering figures ...")
    figs = {}

    # ---- Catalog: sky maps + n(z) overlay --------------------------

    rng = np.random.default_rng(0)
    q_show = rng.choice(q_ra.size, min(50000, q_ra.size), replace=False)
    d_show = rng.choice(d_cat.ra_data.size, min(50000, d_cat.ra_data.size), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].scatter(q_ra[q_show] % 360, q_dec[q_show], s=0.3, alpha=0.4, color="#1f77b4")
    axes[0].set_xlim(0, 360); axes[0].set_ylim(-90, 90)
    axes[0].set_xlabel("RA [deg]"); axes[0].set_ylabel("Dec [deg]")
    axes[0].set_title(f"Quaia G<20  N={q_ra.size:,}")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(d_cat.ra_data[d_show] % 360, d_cat.dec_data[d_show],
                    s=0.3, alpha=0.4, color="#ff7f0e")
    axes[1].set_xlim(0, 360); axes[1].set_ylim(-90, 90)
    axes[1].set_xlabel("RA [deg]"); axes[1].set_ylabel("Dec [deg]")
    axes[1].set_title(f"DESI Y1 QSO  N={d_cat.ra_data.size:,}")
    axes[1].grid(alpha=0.3)

    axes[2].hist(q_z, bins=50, alpha=0.6, color="#1f77b4",
                 label=f"Quaia (N={q_ra.size:,})", density=True)
    axes[2].hist(d_cat.z_data, bins=50, alpha=0.6, color="#ff7f0e",
                 label=f"DESI Y1 QSO (N={d_cat.ra_data.size:,})", density=True)
    for e in quaia["z_q_edges"]:
        axes[2].axvline(e, color="red", ls=":", lw=0.6)
    axes[2].set_xlabel("redshift z"); axes[2].set_ylabel("density")
    axes[2].set_title("n(z), normalised; shell edges marked")
    axes[2].legend()
    fig.tight_layout()
    figs["catalog"] = fig_to_b64(fig)

    # ---- Jackknife regions panel --------------------------------------
    # Visualises the n_regions=25 jackknife regions used to derive ALL
    # error bars in the σ², σ²_LS, ξ_LS, P(N=k), VPF panels. Each
    # region is one-or-more NSIDE=4 healpix super-pixels (~215 deg²
    # per pixel) greedy-assigned to balance per-region galaxy count.
    # Region labels here are recomputed from the same
    # jackknife_region_labels(...) call the pipelines used (deterministic
    # given the catalog positions and nside_jack=4).
    print("rendering jackknife regions panel ...")
    import healpy as hp
    NSIDE_JACK = 4
    N_JACK = quaia["n_regions"]
    pix_area_deg2 = float(np.degrees(np.degrees(hp.nside2pixarea(NSIDE_JACK))))
    labels_q, counts_q = jackknife_region_labels(
        q_ra, q_dec, n_regions=N_JACK, nside_jack=NSIDE_JACK)
    labels_d, counts_d = jackknife_region_labels(
        d_cat.ra_data, d_cat.dec_data,
        n_regions=N_JACK, nside_jack=NSIDE_JACK)
    # Per-region area: count NSIDE=4 pixels assigned to each region
    # (each pixel ≈ 215 deg²).
    pix_q = hp.ang2pix(NSIDE_JACK, np.deg2rad(90.0 - q_dec),
                        np.deg2rad(q_ra))
    pix_d = hp.ang2pix(NSIDE_JACK, np.deg2rad(90.0 - d_cat.dec_data),
                        np.deg2rad(d_cat.ra_data))
    n_pix_q_per_region = np.zeros(N_JACK, dtype=np.int64)
    n_pix_d_per_region = np.zeros(N_JACK, dtype=np.int64)
    pix_to_label_q = np.full(12 * NSIDE_JACK ** 2, -1, dtype=np.int64)
    pix_to_label_d = np.full(12 * NSIDE_JACK ** 2, -1, dtype=np.int64)
    for p, l in zip(pix_q, labels_q): pix_to_label_q[p] = l
    for p, l in zip(pix_d, labels_d): pix_to_label_d[p] = l
    for r in range(N_JACK):
        n_pix_q_per_region[r] = int((pix_to_label_q == r).sum())
        n_pix_d_per_region[r] = int((pix_to_label_d == r).sum())
    area_q_per_region = n_pix_q_per_region.astype(float) * pix_area_deg2
    area_d_per_region = n_pix_d_per_region.astype(float) * pix_area_deg2

    # 2x2 layout: row 0 = sky maps, row 1 = per-region count + area.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap_jk = plt.get_cmap("tab20")

    def _sky_scatter(ax, ra, dec, lab, title):
        # Scatter at low alpha, color-coded by region.
        ax.scatter(ra, dec, c=lab, s=0.2, cmap=cmap_jk,
                    vmin=0, vmax=N_JACK - 1, alpha=0.4)
        ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    _sky_scatter(axes[0, 0], q_ra, q_dec, labels_q,
                  f"Quaia G<20 — {N_JACK} jackknife regions")
    _sky_scatter(axes[0, 1], d_cat.ra_data, d_cat.dec_data, labels_d,
                  f"DESI Y1 QSO — {N_JACK} jackknife regions")

    # Per-region count bar plot.
    ax = axes[1, 0]
    width = 0.4
    idx = np.arange(N_JACK)
    ax.bar(idx - width/2, counts_q, width, color="#1f77b4",
            label=f"Quaia (mean={counts_q.mean():.0f}, "
                  f"σ/μ={counts_q.std()/counts_q.mean():.2f})")
    ax.bar(idx + width/2, counts_d, width, color="#ff7f0e",
            label=f"DESI  (mean={counts_d.mean():.0f}, "
                  f"σ/μ={counts_d.std()/counts_d.mean():.2f})")
    ax.set_xlabel("region id (greedy-balanced by count)")
    ax.set_ylabel("# galaxies per region")
    ax.set_title("Per-region galaxy counts")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # Per-region area in deg^2.
    ax = axes[1, 1]
    ax.bar(idx - width/2, area_q_per_region, width, color="#1f77b4",
            label=f"Quaia (total={area_q_per_region.sum():.0f} deg²)")
    ax.bar(idx + width/2, area_d_per_region, width, color="#ff7f0e",
            label=f"DESI  (total={area_d_per_region.sum():.0f} deg²)")
    ax.set_xlabel("region id")
    ax.set_ylabel("region area [deg²]")
    ax.set_title(rf"Per-region sky area  "
                 rf"(NSIDE={NSIDE_JACK} pixel = {pix_area_deg2:.0f} deg²)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Jackknife regions used for ALL error bars in this presentation "
        f"({N_JACK} regions, greedy-balanced NSIDE={NSIDE_JACK} super-pixels)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figs["jackknife"] = fig_to_b64(fig)
    # Stash a few summary numbers for the panel text.
    jk_metrics = dict(
        n_jack=N_JACK,
        nside_jack=NSIDE_JACK,
        pix_area_deg2=pix_area_deg2,
        q_count_mean=float(counts_q.mean()),
        q_count_std=float(counts_q.std()),
        q_count_min=int(counts_q.min()),
        q_count_max=int(counts_q.max()),
        q_area_total=float(area_q_per_region.sum()),
        d_count_mean=float(counts_d.mean()),
        d_count_std=float(counts_d.std()),
        d_count_min=int(counts_d.min()),
        d_count_max=int(counts_d.max()),
        d_area_total=float(area_d_per_region.sum()),
    )

    # ---- 2D ⟨N⟩(θ, z) heatmaps: data vs volume-filling queries -------
    # For each catalog and each query type, compute the per-cap mean
    # neighbour count ⟨N⟩(θ; z_q = z_n) on the diagonal — a continuous
    # function of (θ, z) that underlies σ², σ²_LS, ξ_LS and the full
    # P(N=k) ladder. Two query types:
    #   - DD: cap centres are data → biased high by clustering
    #   - RD: cap centres are random (uniform on the survey window)
    #         → unbiased volume-filling sample
    # The ratio panel ⟨N⟩_DD / ⟨N⟩_RD is the data-cap selection
    # bias — finite at small θ where caps overlap individual data
    # points / clusters, → 1 at large θ where the window dominates.
    print("rendering ⟨N⟩(θ, z) 2D heatmap panel ...")
    mean_q_dd_diag = np.array([
        _diag(mean_count(quaia["res_dd"]), iq) for iq in range(n_z_q)
    ])  # (n_z_q, n_theta)
    mean_q_rd_diag = np.array([
        _diag(mean_count(quaia["res_rd"]), iq) for iq in range(n_z_q)
    ])
    mean_d_dd_diag = np.array([
        _diag(mean_count(desi["res_dd"]), iq) for iq in range(n_z_q)
    ])
    mean_d_rd_diag = np.array([
        _diag(mean_count(desi["res_rd"]), iq) for iq in range(n_z_q)
    ])
    # Pixel edges in θ (geomspace) and z (shell edges).
    # Use pcolormesh with the actual edges so the y-axis is continuous z.
    theta_deg_arr = np.degrees(quaia["theta"])
    # Theta edges: midpoints in log space, with edge extrapolation.
    log_th = np.log10(theta_deg_arr)
    log_th_edges = np.concatenate([
        [log_th[0] - 0.5 * (log_th[1] - log_th[0])],
        0.5 * (log_th[:-1] + log_th[1:]),
        [log_th[-1] + 0.5 * (log_th[-1] - log_th[-2])],
    ])
    th_edges = 10.0 ** log_th_edges
    z_edges_arr = quaia["z_q_edges"]

    # Shared log color scale — clipped to the 5th-95th percentile of
    # joint positive values. The full ⟨N⟩ range spans ~4 decades
    # (from sub-empty caps at θ=0.05° to >1000 neighbours at θ=12°),
    # but most of that is trivial geometric / density scaling and
    # washes out the visible structure. Clipping shows the
    # interesting middle decades crisply.
    all_means = np.concatenate([
        mean_q_dd_diag.ravel(), mean_q_rd_diag.ravel(),
        mean_d_dd_diag.ravel(), mean_d_rd_diag.ravel(),
    ])
    pos = all_means[all_means > 0]
    vmin_log = float(np.log10(np.percentile(pos, 5)))
    vmax_log = float(np.log10(np.percentile(pos, 95)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5),
                              gridspec_kw={"width_ratios": [1, 1, 1.05]})
    cmap_n = plt.get_cmap("viridis")
    cmap_ratio = plt.get_cmap("RdBu_r")

    def _plot_mean(ax, mean_diag, title, vmin, vmax):
        with np.errstate(divide="ignore"):
            log_mean = np.log10(np.maximum(mean_diag, 10 ** vmin))
        # mean_diag shape (n_z_q, n_theta) → pcolormesh wants
        # X (n_theta+1,), Y (n_z_q+1,), C (n_z_q, n_theta).
        pc = ax.pcolormesh(th_edges, z_edges_arr, log_mean,
                            cmap=cmap_n, vmin=vmin, vmax=vmax,
                            shading="flat")
        # Contour ⟨N⟩=1 (log = 0) — the boundary between empty-cap
        # (VPF-dominated) and dense (variance-dominated) regimes.
        # Use cell-centre coords for the contour (not edges).
        z_mid_local = 0.5 * (z_edges_arr[:-1] + z_edges_arr[1:])
        cs = ax.contour(theta_deg_arr, z_mid_local, log_mean,
                         levels=[0.0], colors="white", linewidths=1.4,
                         linestyles="--")
        ax.clabel(cs, fmt={0.0: r"$\langle N\rangle=1$"}, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]"); ax.set_ylabel("z (z_q = z_n shell)")
        ax.set_title(title, fontsize=11)
        return pc

    def _plot_ratio(ax, num, den, title):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / np.where(den > 0, den, np.nan)
            log_r = np.log10(np.where(ratio > 0, ratio, np.nan))
        # Tight color limits — clustering bias on data caps is a few
        # percent at typical θ; ±0.05 (factor 1.12 either way) gives
        # ~2× the visible contrast vs ±0.1, and the actual signal
        # never exceeds ~0.07 anyway. Saturating cells at very small
        # θ are pinned to the colorbar extremes (visible as deepest
        # red).
        pc = ax.pcolormesh(th_edges, z_edges_arr, log_r,
                            cmap=cmap_ratio, vmin=-0.05, vmax=0.05,
                            shading="flat")
        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]"); ax.set_ylabel("z")
        ax.set_title(title, fontsize=11)
        return pc

    pc0 = _plot_mean(axes[0, 0], mean_q_dd_diag,
                      r"Quaia DD: $\langle N\rangle$ (data-centred caps)",
                      vmin_log, vmax_log)
    _plot_mean(axes[0, 1], mean_q_rd_diag,
                r"Quaia RD: $\langle N\rangle$ (volume-filling caps)",
                vmin_log, vmax_log)
    pc_r0 = _plot_ratio(axes[0, 2],
                         mean_q_dd_diag, mean_q_rd_diag,
                         r"Quaia: $\log_{10}(\langle N\rangle_{\rm DD} / "
                         r"\langle N\rangle_{\rm RD})$")
    _plot_mean(axes[1, 0], mean_d_dd_diag,
                r"DESI DD: $\langle N\rangle$ (data-centred caps)",
                vmin_log, vmax_log)
    _plot_mean(axes[1, 1], mean_d_rd_diag,
                r"DESI RD: $\langle N\rangle$ (volume-filling caps)",
                vmin_log, vmax_log)
    pc_r1 = _plot_ratio(axes[1, 2],
                         mean_d_dd_diag, mean_d_rd_diag,
                         r"DESI: $\log_{10}(\langle N\rangle_{\rm DD} / "
                         r"\langle N\rangle_{\rm RD})$")

    # Shared colorbars: one for the mean-count panels (cols 0+1),
    # one for the ratio (col 2).
    cb0 = fig.colorbar(pc0, ax=axes[:, :2].ravel().tolist(),
                        shrink=0.85, pad=0.02)
    cb0.set_label(r"$\log_{10} \langle N \rangle$", rotation=270, labelpad=14)
    cb1 = fig.colorbar(pc_r1, ax=axes[:, 2].tolist(),
                        shrink=0.85, pad=0.04)
    cb1.set_label(r"$\log_{10}(\langle N\rangle_{\rm DD} / "
                  r"\langle N\rangle_{\rm RD})$",
                  rotation=270, labelpad=14)

    fig.suptitle(
        "Per-cap mean neighbour count ⟨N⟩(θ, z) — data-centred (DD) "
        "vs volume-filling (RD) queries",
        fontsize=12,
    )
    figs["pthetaz"] = fig_to_b64(fig)

    # ---- N_DD / N_RR pair-count heatmaps -----------------------------
    # The most basic clustering observable: ratio of cumulative data-
    # data pair counts (sum_n on the DD pass) to random-random
    # (sum_n on the RR pass), each per-pair-normalised. Equivalent to
    # the Peebles 1+ξ_NN at the cap scale — a sanity-check "raw
    # observable" view of the clustering signal that bypasses the LS
    # combination.
    print("rendering N_DD / N_RR heatmap panel ...")
    z_edges_arr = quaia["z_q_edges"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0),
                              squeeze=False,
                              gridspec_kw={"width_ratios": [1, 1, 1.05]})

    def _peebles_ratio(res_dd, res_rr, n_d_per_zn):
        """Per-pair-normalised DD/RR ratio at each (θ, z) on the
        diagonal. Uses sum_n / N_q for the cube means; then
        ratio = (mean_DD / N_data_per_shell) / (mean_RR / N_R_per_shell).
        For DESI weighted, N_data_per_shell is the sum-of-weights array."""
        m_dd = mean_count(res_dd)
        m_rr = mean_count(res_rr)
        n_d = np.asarray(n_d_per_zn, dtype=np.float64)
        n_r = res_rr.N_q.astype(np.float64)
        # Diagonal cubes: mean shape (n_theta, n_z); broadcast n_d
        # against the single z axis.
        if getattr(res_dd, "is_diagonal", False):
            num = m_dd / np.where(n_d > 0, n_d, np.inf)[None, :]
            den = m_rr / np.where(n_r > 0, n_r, np.inf)[None, :]
        else:
            num = m_dd / np.where(n_d > 0, n_d, np.inf)[None, None, :]
            den = m_rr / np.where(n_r > 0, n_r, np.inf)[None, None, :]
        ratio = num / np.where(den > 0, den, np.nan)
        return m_dd, m_rr, ratio

    # Quaia (unweighted): use res_dd.N_q as the per-shell count.
    n_d_q = quaia["res_dd"].N_q.astype(np.float64)
    m_dd_q, m_rr_q, ratio_q = _peebles_ratio(quaia["res_dd"],
                                              quaia["res_rr"], n_d_q)
    # DESI (weighted DD): use the per-shell weight sum.
    n_d_d = np.asarray(desi_w_sum_per_zn, dtype=np.float64)
    m_dd_d, m_rr_d, ratio_d = _peebles_ratio(desi["res_dd"],
                                              desi["res_rr"], n_d_d)

    # Joint log-scale for the pair-count panels.
    all_pc = np.concatenate([
        m_dd_q.ravel(), m_rr_q.ravel(),
        m_dd_d.ravel(), m_rr_d.ravel(),
    ])
    pos_pc = all_pc[all_pc > 0]
    vmin_pc = float(np.log10(np.percentile(pos_pc, 5)))
    vmax_pc = float(np.log10(np.percentile(pos_pc, 95)))

    # Quaia row (DD pair count, RR pair count, ratio).
    pc0 = _heatmap_panel_diag(
        axes[0, 0], m_dd_q, theta_deg, z_edges_arr,
        cmap="viridis", vmin=vmin_pc, vmax=vmax_pc, logscale=True,
        label="z", title=r"Quaia DD: $\langle N\rangle$")
    _heatmap_panel_diag(
        axes[0, 1], m_rr_q, theta_deg, z_edges_arr,
        cmap="viridis", vmin=vmin_pc, vmax=vmax_pc, logscale=True,
        title=r"Quaia RR: $\langle N\rangle$ (random-random)")
    # Ratio: Peebles 1+ξ. Tight log color around 0 (= ratio 1).
    pc_r0 = _heatmap_panel_diag(
        axes[0, 2], ratio_q, theta_deg, z_edges_arr,
        cmap="RdBu_r", vmin=-0.1, vmax=0.1, logscale=True,
        title=r"Quaia: $\log_{10}(N_{\rm DD}/N_{\rm RR})$ (Peebles)")
    # DESI row.
    _heatmap_panel_diag(
        axes[1, 0], m_dd_d, theta_deg, z_edges_arr,
        cmap="viridis", vmin=vmin_pc, vmax=vmax_pc, logscale=True,
        label="z", title=r"DESI DD: $\langle N\rangle$ (weighted)")
    _heatmap_panel_diag(
        axes[1, 1], m_rr_d, theta_deg, z_edges_arr,
        cmap="viridis", vmin=vmin_pc, vmax=vmax_pc, logscale=True,
        title=r"DESI RR: $\langle N\rangle$ (random-random)")
    pc_r1 = _heatmap_panel_diag(
        axes[1, 2], ratio_d, theta_deg, z_edges_arr,
        cmap="RdBu_r", vmin=-0.1, vmax=0.1, logscale=True,
        title=r"DESI: $\log_{10}(N_{\rm DD}/N_{\rm RR})$ (Peebles)")
    cb0 = fig.colorbar(pc0, ax=axes[:, :2].ravel().tolist(),
                        shrink=0.85, pad=0.02)
    cb0.set_label(r"$\log_{10}\langle N\rangle$", rotation=270, labelpad=14)
    cb1 = fig.colorbar(pc_r1, ax=axes[:, 2].tolist(),
                        shrink=0.85, pad=0.04)
    cb1.set_label(r"$\log_{10}(N_{\rm DD}/N_{\rm RR})$",
                   rotation=270, labelpad=14)
    fig.suptitle(
        "Pair-count heatmaps: data-data, random-random, and Peebles ratio"
        " (per-pair-normalised)", fontsize=12,
    )
    figs["n_dd_n_rr"] = fig_to_b64(fig)

    # ---- sigma^2: overlay both catalogs per shell, DD/RD/Poisson-W ---
    # σ²_clust(θ) = Var(N)/⟨N⟩² − 1/⟨N⟩ uses ONE cube (DD or RD); no
    # random catalog enters this estimator. Both DD-centred and
    # RD-centred versions are shown. A thin solid line shows what σ²
    # would be for an unbiased Poisson source population sampling the
    # SAME angular window W(Ω) and same cap-centre selection — derived
    # purely from the completeness map (Var_q(A_w)/⟨A_w⟩², where
    # A_w(q,θ) = ∫_cap W(Ω)dΩ). No randoms.
    s2_q_rd = sigma2_clust(quaia["res_rd"])
    s2_d_rd = sigma2_clust(desi["res_rd"])
    se_s2_q_rd = _se(_per_region_sigma2_samples(quaia["res_rd"]))
    se_s2_d_rd = _se(_per_region_sigma2_samples(desi["res_rd"]))

    print("computing Poisson-with-window σ² baselines (no randoms) ...")
    import healpy as hp
    quaia_sel_map_full, quaia_nside_mask = load_selection_function(
        "/Users/tabel/Research/data/quaia/selection_function_NSIDE64_G20.0.fits")
    desi_sel_N = hp.read_map(
        os.path.join(REPO_ROOT,
                     "data/desi/desi_qso_y1_completeness_N_NSIDE64.fits"),
        verbose=False)
    desi_sel_S = hp.read_map(
        os.path.join(REPO_ROOT,
                     "data/desi/desi_qso_y1_completeness_S_NSIDE64.fits"),
        verbose=False)
    desi_nside_mask = hp.npix2nside(desi_sel_N.size)
    # All Quaia caps fall in the single global region.
    s2_q_pois = sigma2_poisson_window(
        sel_maps={"X": quaia_sel_map_full},
        region_of_cap=np.full(q_ra.size, "X", dtype="U1"),
        ra_caps=q_ra, dec_caps=q_dec,
        theta_radii_rad=quaia["theta"],
        nside_mask=quaia_nside_mask,
        n_subsample=5000, seed=0,
    )
    # DESI per-region.
    s2_d_pois = sigma2_poisson_window(
        sel_maps={"N": desi_sel_N, "S": desi_sel_S},
        region_of_cap=d_cat.photsys_data,
        ra_caps=d_cat.ra_data, dec_caps=d_cat.dec_data,
        theta_radii_rad=desi["theta"],
        nside_mask=desi_nside_mask,
        n_subsample=5000, seed=0,
    )

    fig, axes = plt.subplots(1, n_z_plot, figsize=(4.2 * n_z_plot, 4.7),
                              squeeze=False, sharey=True)
    for ip, iq in enumerate(Z_PLOT_IDX):
        ax = axes[0, ip]
        ax.errorbar(theta_deg, _diag(s2_q, iq), yerr=_diag(se_s2_q, iq),
                    fmt="o-", color="#1f77b4", capsize=3, lw=1.5,
                    label=f"{quaia['label']}  (DD)")
        ax.errorbar(theta_deg, _diag(s2_q_rd, iq),
                    yerr=_diag(se_s2_q_rd, iq),
                    fmt="o--", color="#1f77b4", capsize=3, lw=1.0,
                    alpha=0.55, mfc="none",
                    label=f"{quaia['label']}  (RD)")
        ax.errorbar(theta_deg, _diag(s2_d, iq), yerr=_diag(se_s2_d, iq),
                    fmt="s-", color="#ff7f0e", capsize=3, lw=1.5,
                    label=f"{desi['label']}  (DD)")
        ax.errorbar(theta_deg, _diag(s2_d_rd, iq),
                    yerr=_diag(se_s2_d_rd, iq),
                    fmt="s--", color="#ff7f0e", capsize=3, lw=1.0,
                    alpha=0.55, mfc="none",
                    label=f"{desi['label']}  (RD)")
        ax.plot(theta_deg, s2_q_pois,
                "-", color="#1f77b4", lw=0.9, alpha=0.85,
                label="Quaia Poisson(W)")
        ax.plot(theta_deg, s2_d_pois,
                "-", color="#ff7f0e", lw=0.9, alpha=0.85,
                label="DESI Poisson(W)")
        if xi_q["xi_ls"] is not None:
            ax.errorbar(theta_deg, _diag(xi_q["xi_ls"], iq),
                        yerr=_diag(xi_q["se_xi_ls"], iq),
                        fmt="d-.", color="#1f77b4", capsize=2,
                        lw=1.0, ms=3, alpha=0.9,
                        label=f"{quaia['label']}  (LS pair-count)")
        if xi_d["xi_ls"] is not None:
            ax.errorbar(theta_deg, _diag(xi_d["xi_ls"], iq),
                        yerr=_diag(xi_d["se_xi_ls"], iq),
                        fmt="d-.", color="#ff7f0e", capsize=2,
                        lw=1.0, ms=3, alpha=0.9,
                        label=f"{desi['label']}  (LS pair-count)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("theta [deg]")
        if ip == 0:
            ax.set_ylabel(r"$\sigma^2_{\rm clust}(\theta)$")
        ax.set_title(rf"z={z_q_mid[iq]:.2f}  "
                     rf"($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
        ax.set_ylim(1e-3, 1.0)
        ax.legend(fontsize=6); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    figs["sigma2"] = fig_to_b64(fig)

    # Optional Veusz override for the σ² panel (publication-quality
    # workflow). When PAPER_USE_VEUSZ=1, also write a Veusz panel-group
    # at vsz/sigma2.vsz and an SVG at docs/sigma2.svg, then expose the
    # SVG as figs["sigma2_vsz"]. The matplotlib version stays in
    # figs["sigma2"] so the existing tab keeps working; the Veusz
    # version drives the click-through edit-and-propagate loop. See
    # tools/VEUSZ_README.md.
    figs["sigma2_vsz"] = None
    if os.environ.get("PAPER_USE_VEUSZ", "0") == "1":
        try:
            import sys as _sys
            _sys.path.insert(0, REPO_ROOT)
            from tools.build_vsz import (
                sigma2_group as _vsz_sigma2_group,
                export_svg as _vsz_export_svg,
                snapshot_vsz_dir as _vsz_snapshot,
            )
            from tools.propagate_vsz_edits import propagate as _vsz_propagate
            vsz_dir = os.path.join(REPO_ROOT, "vsz")
            os.makedirs(vsz_dir, exist_ok=True)
            # Diff hand-edited .vsz files against the latest snapshot
            # and append entries to STYLE_LOG.md, BEFORE we create a
            # new snapshot.
            _vsz_propagate(vsz_dir, dry_run=False)
            _vsz_snapshot(vsz_dir, keep=20)
            z_mids_plot = [float(z_q_mid[i]) for i in Z_PLOT_IDX]
            vsz_path = _vsz_sigma2_group(
                s2_q, se_s2_q, s2_d, se_s2_d,
                theta_deg=theta_deg, z_indices=Z_PLOT_IDX,
                z_mids=z_mids_plot, vsz_dir=vsz_dir,
            )
            svg_out = os.path.join(OUTPUT_DIR, "figures", "sigma2.svg")
            _vsz_export_svg(vsz_path, svg_out)
            # Also drop a copy under docs/ so Pages serves it next to
            # the HTML.
            docs_svg = os.path.join(REPO_ROOT, "docs", "sigma2.svg")
            os.makedirs(os.path.dirname(docs_svg), exist_ok=True)
            import shutil as _shutil
            _shutil.copy2(svg_out, docs_svg)
            figs["sigma2_vsz"] = "docs/sigma2.svg"
            print(f"  [veusz] wrote {vsz_path} and {svg_out}")
        except Exception as _e:
            print(f"  [veusz] sigma2 panel failed: {_e}")
            figs["sigma2_vsz"] = None

    # σ²_clust heatmap (full z range, log color scale): Quaia DD,
    # Quaia RD, DESI DD, DESI RD on a common color scale.
    if n_z_q > n_z_plot:
        z_edges_arr = quaia["z_q_edges"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.0),
                                  squeeze=False)
        # Joint color scale clipped to 5-95 percentile of positive values.
        all_s2 = np.concatenate([
            s2_q.ravel(), s2_q_rd.ravel(),
            s2_d.ravel(), s2_d_rd.ravel(),
        ])
        pos = all_s2[all_s2 > 0]
        vmin = float(np.log10(np.percentile(pos, 5)))
        vmax = float(np.log10(np.percentile(pos, 95)))
        for j, (cube, title) in enumerate([
            (s2_q,    r"Quaia DD: $\sigma^2_{\rm clust}$"),
            (s2_q_rd, r"Quaia RD: $\sigma^2_{\rm clust}$"),
            (s2_d,    r"DESI DD: $\sigma^2_{\rm clust}$"),
            (s2_d_rd, r"DESI RD: $\sigma^2_{\rm clust}$"),
        ]):
            ax = axes[0, j]
            pc = _heatmap_panel_diag(
                ax, cube, theta_deg, z_edges_arr,
                cmap="viridis", vmin=vmin, vmax=vmax, logscale=True,
                label="z" if j == 0 else None, title=title)
        cb = fig.colorbar(pc, ax=axes[0, :].tolist(), shrink=0.85)
        cb.set_label(r"$\log_{10}\sigma^2_{\rm clust}$",
                      rotation=270, labelpad=14)
        figs["sigma2_heat"] = fig_to_b64(fig)

    # ---- σ²_DP panel (cap-cumulative DP — relabelled from "ξ_DP") ----
    fig, axes = plt.subplots(1, n_z_plot, figsize=(4.0 * n_z_plot, 4.5),
                              squeeze=False, sharey=True)
    for ip, iq in enumerate(Z_PLOT_IDX):
        ax = axes[0, ip]
        ax.errorbar(theta_deg, _diag(xi_q["xi_dp"], iq),
                    yerr=_diag(xi_q["se_xi_dp"], iq),
                    fmt="o-", color="#1f77b4", capsize=3, lw=1.5,
                    label=quaia["label"])
        ax.errorbar(theta_deg, _diag(xi_d["xi_dp"], iq),
                    yerr=_diag(xi_d["se_xi_dp"], iq),
                    fmt="s-", color="#ff7f0e", capsize=3, lw=1.5,
                    label=desi["label"])
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("theta [deg]")
        if ip == 0:
            ax.set_ylabel(r"$\sigma^2_{\rm DP}(\theta) = \langle\xi\rangle_{\rm cap}^{\rm DP}(\theta)$")
        ax.set_title(rf"z={z_q_mid[iq]:.2f}  ($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["xi_dp"] = fig_to_b64(fig)

    # σ²_DP heatmap (full z range): Quaia + DESI side-by-side.
    if n_z_q > n_z_plot:
        z_edges_arr = quaia["z_q_edges"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                                  squeeze=False)
        all_dp = np.concatenate([
            xi_q["xi_dp"].ravel(), xi_d["xi_dp"].ravel(),
        ])
        finite = all_dp[np.isfinite(all_dp)]
        # Symmetric color limits about 0 for the divergent quantity.
        vlim = float(np.percentile(np.abs(finite), 95))
        for j, (cube, title) in enumerate([
            (xi_q["xi_dp"], r"Quaia $\sigma^2_{\rm DP}$"),
            (xi_d["xi_dp"], r"DESI $\sigma^2_{\rm DP}$"),
        ]):
            ax = axes[0, j]
            pc = _heatmap_panel_diag(
                ax, cube, theta_deg, z_edges_arr,
                cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                label="z" if j == 0 else None, title=title)
        cb = fig.colorbar(pc, ax=axes[0, :].tolist(), shrink=0.85)
        cb.set_label(r"$\sigma^2_{\rm DP}$", rotation=270, labelpad=14)
        figs["xi_dp_heat"] = fig_to_b64(fig)

    # ---- σ²_LS panel: MC-RR overlaid with window-corrected analytic-RR.
    # Analytic-RR drops MC random-catalog shot-noise from the LS combo.
    if xi_q["xi_ls"] is not None and xi_d["xi_ls"] is not None:
        from twopt_density.knn_analytic_rr import analytic_rr_cube
        print("computing analytic-RR (window-corrected) for σ²_LS overlay ...")

        def _to_diag_if_needed(res_full, like_res):
            """If like_res uses diagonal cubes, convert the full
            analytic cube to diagonal layout by extracting the
            iq=jn slice. Otherwise return as-is."""
            if not getattr(like_res, "is_diagonal", False):
                return res_full
            # Full sum_n shape: (n_theta, n_z_q, n_z_n) → diagonal
            # (n_theta, n_z) by indexing iq == jn.
            n_theta, n_z, _ = res_full.sum_n.shape
            sn_d = np.array([res_full.sum_n[:, i, i] for i in range(n_z)]).T
            sn2_d = np.array([res_full.sum_n2[:, i, i] for i in range(n_z)]).T
            # H_geq_k may be all zeros (analytic doesn't fill it).
            h_d = np.zeros((n_theta, n_z, max(res_full.H_geq_k.shape[-1], 1)),
                            dtype=np.int64)
            return KnnCdfResult(
                H_geq_k=h_d, sum_n=sn_d, sum_n2=sn2_d,
                N_q=res_full.N_q,
                theta_radii_rad=res_full.theta_radii_rad,
                z_q_edges=res_full.z_q_edges, z_n_edges=res_full.z_n_edges,
                flavor=res_full.flavor, backend_used=res_full.backend_used,
                area_per_cap=res_full.area_per_cap,
                is_diagonal=True,
            )
        # Quaia: single global mask.
        res_q_rr_an = analytic_rr_cube(
            sel_map=quaia_sel_map_full, z_data=q_z,
            theta_radii_rad=quaia["theta"],
            z_q_edges=quaia["z_q_edges"],
            z_n_edges=quaia["z_q_edges"],
            n_q_per_shell=quaia["res_rr"].N_q,
            n_random_total=int(quaia["res_rr"].N_q.sum()),
            nside=quaia_nside_mask,
            query_ra_deg=q_ra, query_dec_deg=q_dec,
            n_subsample_for_window=5000,
        )
        res_q_rr_an = _to_diag_if_needed(res_q_rr_an, quaia["res_dd"])
        n_neigh_q_dd = quaia["res_dd"].N_q.astype(np.float64)
        n_neigh_q_rr = res_q_rr_an.N_q.astype(np.float64)
        xi_ls_q_an = xi_ls(quaia["res_dd"], quaia["res_rd"], res_q_rr_an,
                           n_neigh_q_dd, n_neigh_q_dd, n_neigh_q_rr)
        # DESI: combined N+S mask. Although N and S have disjoint sky
        # footprints, large caps near the Dec=32.375° boundary CAN
        # bridge between them (a 12° cap centred at Dec=33° reaches
        # Dec=21°), and MC RR includes those cross-region pairs. The
        # combined formula sel_combined = sel_N + sel_S handles this
        # naturally because A_w_combined = A_w_N + A_w_S and the
        # per-region random densities (Ω_X^-1 × N_R_X) are nearly
        # equal (~1.06 ratio), so density × A_w factorises cleanly.
        # A per-region-then-sum reconstruction would zero the
        # cross-region area and undershoot MC by ~5% at large θ
        # (verified numerically at θ=12° z=1.90).
        desi_sel_combined = desi_sel_N + desi_sel_S
        res_d_rr_an = analytic_rr_cube(
            sel_map=desi_sel_combined, z_data=d_cat.z_data,
            theta_radii_rad=desi["theta"],
            z_q_edges=desi["z_q_edges"],
            z_n_edges=desi["z_q_edges"],
            n_q_per_shell=desi["res_rr"].N_q,
            n_random_total=int(desi["res_rr"].N_q.sum()),
            nside=desi_nside_mask,
            query_ra_deg=d_cat.ra_data, query_dec_deg=d_cat.dec_data,
            n_subsample_for_window=5000,
        )
        res_d_rr_an = _to_diag_if_needed(res_d_rr_an, desi["res_dd"])
        n_neigh_d_dd = (np.asarray(desi_w_sum_per_zn, dtype=np.float64))
        n_neigh_d_rr = res_d_rr_an.N_q.astype(np.float64)
        xi_ls_d_an = xi_ls(desi["res_dd"], desi["res_rd"], res_d_rr_an,
                           n_neigh_d_dd, n_neigh_d_dd, n_neigh_d_rr)

        # ---- Analytic μ for the Poisson reference in the P(N=k) panel.
        # μ_an_DD(θ; z_q=z_n) = N_data[jn] · ⟨A_w(q;data)⟩(θ) / Ω_mask_w
        #   (no-clustering reference for DATA-centred caps)
        # μ_an_RD(θ; z_q=z_n) = N_data[jn] · ⟨A_w(q;random)⟩(θ) / Ω_mask_w
        #   (no-clustering reference for RANDOM-centred caps)
        # For DESI (weighted DD) we use the per-shell weight sum
        # in place of N_data so the reference matches what
        # mean_count(res_dd) returns.
        from twopt_density.knn_analytic_rr import (
            _mean_windowed_cap_area, random_queries_from_selection_function,
        )
        print("computing analytic μ for P(N=k) Poisson reference ...")

        # Quaia (unweighted)
        A_w_q_dd = _mean_windowed_cap_area(
            sel_map=quaia_sel_map_full, nside=quaia_nside_mask,
            query_ra_deg=q_ra, query_dec_deg=q_dec,
            theta_radii_rad=quaia["theta"], n_subsample=5000, seed=0,
        )
        rng_q = np.random.default_rng(0)
        ra_rq, dec_rq, _ = random_queries_from_selection_function(
            sel_map=quaia_sel_map_full, n_random=5000,
            z_data=np.array([1.0]), nside=quaia_nside_mask, rng=rng_q,
        )
        A_w_q_rd = _mean_windowed_cap_area(
            sel_map=quaia_sel_map_full, nside=quaia_nside_mask,
            query_ra_deg=ra_rq, query_dec_deg=dec_rq,
            theta_radii_rad=quaia["theta"], n_subsample=5000, seed=0,
        )
        Omega_mask_q = (float(quaia_sel_map_full.sum())
                        * (4.0 * np.pi / quaia_sel_map_full.size))
        N_data_q_zn, _ = np.histogram(q_z, quaia["z_q_edges"])
        mu_q_an_dd = (N_data_q_zn[None, :].astype(np.float64)
                      * (A_w_q_dd[:, None] / Omega_mask_q))   # (n_theta, n_z_n)
        mu_q_an_rd = (N_data_q_zn[None, :].astype(np.float64)
                      * (A_w_q_rd[:, None] / Omega_mask_q))

        # DESI (weighted DD): use the per-shell weight sum as the
        # effective neighbor count.
        A_w_d_dd = _mean_windowed_cap_area(
            sel_map=desi_sel_combined, nside=desi_nside_mask,
            query_ra_deg=d_cat.ra_data, query_dec_deg=d_cat.dec_data,
            theta_radii_rad=desi["theta"], n_subsample=5000, seed=0,
        )
        rng_d = np.random.default_rng(1)
        ra_rd_d, dec_rd_d, _ = random_queries_from_selection_function(
            sel_map=desi_sel_combined, n_random=5000,
            z_data=np.array([1.0]), nside=desi_nside_mask, rng=rng_d,
        )
        A_w_d_rd = _mean_windowed_cap_area(
            sel_map=desi_sel_combined, nside=desi_nside_mask,
            query_ra_deg=ra_rd_d, query_dec_deg=dec_rd_d,
            theta_radii_rad=desi["theta"], n_subsample=5000, seed=0,
        )
        Omega_mask_d = (float(desi_sel_combined.sum())
                        * (4.0 * np.pi / desi_sel_combined.size))
        # DESI: use weighted neighbor count (matches mean_count(res_dd))
        N_neigh_d_zn = np.asarray(desi_w_sum_per_zn, dtype=np.float64)
        mu_d_an_dd = (N_neigh_d_zn[None, :]
                      * (A_w_d_dd[:, None] / Omega_mask_d))
        mu_d_an_rd = (N_neigh_d_zn[None, :]
                      * (A_w_d_rd[:, None] / Omega_mask_d))

        fig, axes = plt.subplots(1, n_z_plot, figsize=(4.0 * n_z_plot, 4.5),
                                  squeeze=False, sharey=True)
        for ip, iq in enumerate(Z_PLOT_IDX):
            ax = axes[0, ip]
            ax.errorbar(theta_deg, _diag(xi_q["xi_ls"], iq),
                        yerr=_diag(xi_q["se_xi_ls"], iq),
                        fmt="o-", color="#1f77b4", capsize=3, lw=1.5,
                        label=f"{quaia['label']} (MC RR)")
            ax.errorbar(theta_deg, _diag(xi_d["xi_ls"], iq),
                        yerr=_diag(xi_d["se_xi_ls"], iq),
                        fmt="s-", color="#ff7f0e", capsize=3, lw=1.5,
                        label=f"{desi['label']} (MC RR)")
            ax.plot(theta_deg, _diag(xi_ls_q_an, iq),
                    "-", color="#1f77b4", lw=0.9, alpha=0.85,
                    label="Quaia (analytic RR)")
            ax.plot(theta_deg, _diag(xi_ls_d_an, iq),
                    ":", color="#ff7f0e", lw=0.9, alpha=0.85,
                    label="DESI (analytic RR)")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("theta [deg]")
            if ip == 0:
                ax.set_ylabel(r"$\sigma^2_{\rm LS}(\theta) = \langle\xi\rangle_{\rm cap}(\theta)$")
            ax.set_title(rf"z={z_q_mid[iq]:.2f}  ($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout()
        figs["xi_ls"] = fig_to_b64(fig)

        # σ²_LS heatmap (full z range): Quaia + DESI side-by-side.
        if n_z_q > n_z_plot:
            z_edges_arr = quaia["z_q_edges"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                                      squeeze=False)
            all_ls = np.concatenate([
                xi_q["xi_ls"].ravel(), xi_d["xi_ls"].ravel(),
            ])
            finite = all_ls[np.isfinite(all_ls)]
            vlim = float(np.percentile(np.abs(finite), 95))
            for j, (cube, title) in enumerate([
                (xi_q["xi_ls"], r"Quaia $\sigma^2_{\rm LS}$"),
                (xi_d["xi_ls"], r"DESI $\sigma^2_{\rm LS}$"),
            ]):
                ax = axes[0, j]
                pc = _heatmap_panel_diag(
                    ax, cube, theta_deg, z_edges_arr,
                    cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                    label="z" if j == 0 else None, title=title)
            cb = fig.colorbar(pc, ax=axes[0, :].tolist(), shrink=0.85)
            cb.set_label(r"$\sigma^2_{\rm LS}$", rotation=270, labelpad=14)
            figs["xi_ls_heat"] = fig_to_b64(fig)

    # ---- True differential ξ_LS(θ) on a COARSER θ grid --------------
    # The 90-bin cumulative grid is too fine for per-annulus pair counts
    # (each annular bin gets too few pairs → enormous bin-to-bin noise).
    # Standard differential-ξ practice is ~10-15 log-spaced bins. We
    # pick 12 edges across the full θ range and rebuild the LS pair
    # count from the cumulative cubes at those edges.
    if (xi_q["xi_ls"] is not None and xi_d["xi_ls"] is not None):
        n_coarse_edges = 12
        theta_rad_full = quaia["theta"]   # both catalogs share grid
        idx_coarse = np.round(
            np.linspace(0, theta_rad_full.size - 1, n_coarse_edges)
        ).astype(int)
        idx_coarse = np.unique(idx_coarse)
        n_coarse_edges = idx_coarse.size
        theta_edges_rad = theta_rad_full[idx_coarse]
        # Annular bin midpoints (geometric mean of adjacent edges); the
        # first bin output of xi_ls_annular is the cap [0, θ_edges[0]],
        # which we drop and plot only the K-1 true annular bins.
        theta_mid_deg = np.degrees(
            np.sqrt(theta_edges_rad[:-1] * theta_edges_rad[1:])
        )

        def _coarsen(res):
            return KnnCdfResult(
                H_geq_k=res.H_geq_k,
                sum_n=res.sum_n[idx_coarse],
                sum_n2=res.sum_n2[idx_coarse],
                N_q=res.N_q,
                theta_radii_rad=res.theta_radii_rad[idx_coarse],
                z_q_edges=res.z_q_edges, z_n_edges=res.z_n_edges,
                flavor=res.flavor, backend_used=res.backend_used,
                area_per_cap=res.area_per_cap[idx_coarse],
                H_geq_k_per_region=res.H_geq_k_per_region,
                sum_n_per_region=(res.sum_n_per_region[idx_coarse]
                                  if res.sum_n_per_region is not None else None),
                sum_n2_per_region=(res.sum_n2_per_region[idx_coarse]
                                   if res.sum_n2_per_region is not None else None),
                N_q_per_region=res.N_q_per_region,
                is_diagonal=getattr(res, "is_diagonal", False),
            )

        # Re-derive ann ξ on coarse cubes for both catalogs.
        def _xi_ann_coarse(cat, n_neigh_dd_p):
            res_dd_c = _coarsen(cat["res_dd"])
            res_rd_c = _coarsen(cat["res_rd"])
            res_rr_c = _coarsen(cat["res_rr"])
            n_dd = (np.asarray(n_neigh_dd_p, dtype=np.float64)
                    if n_neigh_dd_p is not None
                    else res_dd_c.N_q.astype(np.float64))
            n_dr = n_dd
            n_rr = res_rr_c.N_q.astype(np.float64)
            cube = xi_ls_annular(res_dd_c, res_rd_c, res_rr_c,
                                  n_dd, n_dr, n_rr)
            samples = _per_region_xi_ls_annular_samples(
                res_dd_c, res_rd_c, res_rr_c, n_dd, n_dr, n_rr,
            )
            return cube, _se(samples)

        xi_q_ann_c, se_q_ann_c = _xi_ann_coarse(quaia, None)
        xi_d_ann_c, se_d_ann_c = _xi_ann_coarse(desi, desi_w_sum_per_zn)

        fig, axes = plt.subplots(1, n_z_plot, figsize=(4.0 * n_z_plot, 4.5),
                                  squeeze=False, sharey=True)
        for ip, iq in enumerate(Z_PLOT_IDX):
            ax = axes[0, ip]
            # Drop bin 0 (the cap [0, theta_edges[0]] — not an annulus
            # in the usual sense). Plot bins 1..K-1 at the K-1 midpoints.
            ax.errorbar(theta_mid_deg,
                        _diag(xi_q_ann_c[1:], iq),
                        yerr=_diag(se_q_ann_c[1:], iq),
                        fmt="o-", color="#1f77b4", capsize=3, lw=1.5,
                        label=quaia["label"])
            ax.errorbar(theta_mid_deg,
                        _diag(xi_d_ann_c[1:], iq),
                        yerr=_diag(se_d_ann_c[1:], iq),
                        fmt="s-", color="#ff7f0e", capsize=3, lw=1.5,
                        label=desi["label"])
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("theta [deg] (annular midpoint, log)")
            if ip == 0:
                ax.set_ylabel(r"$\xi_{\rm LS}(\theta)$  "
                              r"(true differential, "
                              f"{n_coarse_edges-1}-bin)")
            ax.set_title(rf"z={z_q_mid[iq]:.2f}  "
                         rf"($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
            ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        figs["xi_ls_true"] = fig_to_b64(fig)

        # True differential ξ_LS heatmap: average of Quaia and DESI
        # across the full z range. Useful to see the ξ→0 transition
        # scale and any z-evolution of the clustering amplitude.
        if n_z_q > n_z_plot:
            # xi_q_ann_c has shape (n_coarse, n_z, ...) — drop bin 0.
            theta_mid_for_heatmap = theta_mid_deg
            z_edges_arr = quaia["z_q_edges"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                                      squeeze=False)
            # Symmetric color limits about 0.
            all_ann = np.concatenate([
                xi_q_ann_c[1:].ravel(), xi_d_ann_c[1:].ravel(),
            ])
            finite = all_ann[np.isfinite(all_ann)]
            vlim = float(np.percentile(np.abs(finite), 95))
            for j, (cube, title) in enumerate([
                (xi_q_ann_c[1:], r"Quaia $\xi_{\rm LS}$ (differential)"),
                (xi_d_ann_c[1:], r"DESI $\xi_{\rm LS}$ (differential)"),
            ]):
                ax = axes[0, j]
                pc = _heatmap_panel_diag(
                    ax, cube, theta_mid_for_heatmap, z_edges_arr,
                    cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                    label="z" if j == 0 else None, title=title)
            cb = fig.colorbar(pc, ax=axes[0, :].tolist(), shrink=0.85)
            cb.set_label(r"$\xi_{\rm LS}(\theta)$",
                          rotation=270, labelpad=14)
            figs["xi_ls_true_heat"] = fig_to_b64(fig)

    # ---- CIC PMF: 2 columns (one per catalog), one row per shell -----
    # Thin solid lines = Poisson(μ=⟨N⟩(θ)) at the same per-cap mean,
    # in matching colour — the offset between the markers and the
    # solid line is the clustering signature.
    mean_q = mean_count(quaia["res_dd"])  # full: (n_theta, n_z_q, n_z_n); diag: (n_theta, n_z)
    mean_d = mean_count(desi["res_dd"])
    n_pmf_q = pmf_q.shape[-1] - 1
    n_pmf_d = pmf_d.shape[-1] - 1
    t_choices = [0, len(theta_deg) // 2]
    cic_colors = ["#1f77b4", "#d62728"]
    fig, axes = plt.subplots(2, n_z_plot, figsize=(4.0 * n_z_plot, 8),
                              squeeze=False)
    # Cube-shape-aware indexing helpers.
    def _pmf_at(pmf, t_idx, iq, n_pmf):
        # full pmf: (n_theta, n_z_q, n_z_n, k_max+1)
        # diag pmf: (n_theta, n_z, k_max+1)
        if pmf.ndim == 4:
            return pmf[t_idx, iq, iq, :n_pmf]
        return pmf[t_idx, iq, :n_pmf]
    def _mu_at(mean_arr, t_idx, iq):
        if mean_arr.ndim == 3:
            return float(mean_arr[t_idx, iq, iq])
        return float(mean_arr[t_idx, iq])

    for ip, iq in enumerate(Z_PLOT_IDX):
        # find xlim where PMF still > 1e-5 in either catalog
        max_k = 1
        for t_idx in t_choices:
            for arr in [_pmf_at(pmf_q, t_idx, iq, n_pmf_q),
                        _pmf_at(pmf_d, t_idx, iq, n_pmf_d)]:
                above = np.flatnonzero(arr > 1e-5)
                if above.size > 0:
                    max_k = max(max_k, int(above[-1]))
        xmax = int(max_k * 1.2) + 2

        # Quaia row
        ax = axes[0, ip]
        for j, t_idx in enumerate(t_choices):
            col = cic_colors[j]
            ax.plot(np.arange(n_pmf_q), _pmf_at(pmf_q, t_idx, iq, n_pmf_q),
                    "o-", color=col, lw=1.4, ms=4,
                    label=f"theta={theta_deg[t_idx]:.2f} deg")
            mu = _mu_at(mean_q, t_idx, iq)
            k_p = np.arange(0, xmax)
            p_pois = poisson.pmf(k_p, mu)
            ax.plot(k_p, p_pois,
                    "-", color=col, lw=0.9, alpha=0.85,
                    label=f"Poisson μ={mu:.1f}")
        ax.set_yscale("log"); ax.set_xlim(0, xmax)
        ax.set_ylim(1e-4, 1.0)         # cap meaningless tail noise
        ax.set_xlabel("k (cap count)")
        if ip == 0:
            ax.set_ylabel(r"$P_{N=k}$ Quaia")
        ax.set_title(rf"Quaia, z={z_q_mid[iq]:.2f}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

        # DESI row
        ax = axes[1, ip]
        for j, t_idx in enumerate(t_choices):
            col = cic_colors[j]
            ax.plot(np.arange(n_pmf_d), _pmf_at(pmf_d, t_idx, iq, n_pmf_d),
                    "s-", color=col, lw=1.4, ms=4,
                    label=f"theta={theta_deg[t_idx]:.2f} deg")
            mu = _mu_at(mean_d, t_idx, iq)
            k_p = np.arange(0, xmax)
            ax.plot(k_p, poisson.pmf(k_p, mu),
                    "-", color=col, lw=0.9, alpha=0.85,
                    label=f"Poisson μ={mu:.1f}")
        ax.set_yscale("log"); ax.set_xlim(0, xmax)
        ax.set_ylim(1e-4, 1.0)
        ax.set_xlabel("k (cap count)")
        if ip == 0:
            ax.set_ylabel(r"$P_{N=k}$ DESI")
        ax.set_title(rf"DESI, z={z_q_mid[iq]:.2f}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    figs["cic"] = fig_to_b64(fig)

    # ---- P(N=k) at fixed k vs theta and r, Quaia and DESI overlay ---
    # cic_pmf puts the cumulative tail P_{>=k_max} in the LAST array
    # slot (index k_max), not P_{N=k_max}. Plot only k strictly below
    # k_max. Thin solid lines = Poisson(μ=⟨N⟩(θ)) per (k, catalog) at
    # the same per-cap mean as the data; deviation from the data
    # markers is the clustering signature.
    print("deriving P(N=k) curves vs theta/r ...")
    pmf_q_dd = pmf_q                  # (n_theta, n_z_q, n_z_n, k_max+1)
    pmf_d_dd = pmf_d
    pmf_q_rd = cic_pmf(quaia["res_rd"]) if quaia["res_rd"].k_max > 0 else None
    pmf_d_rd = cic_pmf(desi["res_rd"]) if desi["res_rd"].k_max > 0 else None
    # Poisson reference μ: prefer the analytic prediction (window-
    # corrected, shot-noise-free) when available; fall back to the
    # empirical DD mean otherwise. μ_DD: ⟨A_w(q;data)⟩ — reference
    # for DATA-centred caps. μ_RD: ⟨A_w(q;random)⟩ — reference for
    # RANDOM-centred caps. Both broadcast (n_theta, n_z_q, n_z_n) by
    # repeating along the z_q axis (μ depends on θ, z_n only).
    n_th = quaia["theta"].size
    if "mu_q_an_dd" in locals():
        mean_q_th = np.broadcast_to(
            mu_q_an_dd[:, None, :], (n_th, n_z_q, n_z_q)).copy()
        mean_d_th = np.broadcast_to(
            mu_d_an_dd[:, None, :], (n_th, n_z_q, n_z_q)).copy()
        mean_q_th_rd = np.broadcast_to(
            mu_q_an_rd[:, None, :], (n_th, n_z_q, n_z_q)).copy()
        mean_d_th_rd = np.broadcast_to(
            mu_d_an_rd[:, None, :], (n_th, n_z_q, n_z_q)).copy()
        ref_label_suffix = " (analytic)"
    else:
        mean_q_th = mean_count(quaia["res_dd"])
        mean_d_th = mean_count(desi["res_dd"])
        mean_q_th_rd = (mean_count(quaia["res_rd"])
                        if quaia["res_rd"] is not None else mean_q_th)
        mean_d_th_rd = (mean_count(desi["res_rd"])
                        if desi["res_rd"] is not None else mean_d_th)
        ref_label_suffix = " (empirical)"

    k_max_q_dd = pmf_q_dd.shape[-1] - 1
    k_max_d_dd = pmf_d_dd.shape[-1] - 1
    k_max_q_rd = pmf_q_rd.shape[-1] - 1 if pmf_q_rd is not None else 0
    k_max_d_rd = pmf_d_rd.shape[-1] - 1 if pmf_d_rd is not None else 0
    # k_targets includes k=0 (the VPF = void probability function =
    # 1NN-CDF complement). Poisson(k=0|μ) = exp(-μ).
    k_targets_full = [0, 1, 3, 10, 30, 100, 300]
    k_min_max = min(k_max_q_dd, k_max_d_dd,
                    k_max_q_rd if pmf_q_rd is not None else 1_000_000,
                    k_max_d_rd if pmf_d_rd is not None else 1_000_000)
    k_targets = [k for k in k_targets_full if k < k_min_max]
    print(f"  k_max Q_DD={k_max_q_dd}, D_DD={k_max_d_dd}, "
          f"Q_RD={k_max_q_rd}, D_RD={k_max_d_rd}; "
          f"plotting k_targets={k_targets}  (k=0 is the VPF / 1NN-CDF)")

    theta_rad = quaia["theta"]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(k_targets) - 1, 1))
              for i in range(len(k_targets))]

    # Fine theta grid for smooth Poisson lines. We log-log interpolate
    # ⟨N⟩(θ) from the 90 measurement bins onto a 5× denser grid; the
    # data markers stay at the original 90 bins.
    n_fine = 5 * theta_rad.size
    theta_rad_fine = np.geomspace(theta_rad.min(), theta_rad.max(), n_fine)
    theta_deg_fine = np.degrees(theta_rad_fine)
    log_theta = np.log(theta_rad)
    log_theta_fine = np.log(theta_rad_fine)

    def _interp_mean(mu_arr):
        """Log-log interpolate ⟨N⟩(θ) onto the fine grid."""
        # Use log y where positive (numerical safety; ⟨N⟩ should be > 0
        # at all measured θ in this analysis).
        with np.errstate(divide="ignore"):
            log_mu = np.log(np.maximum(mu_arr, 1e-30))
        return np.exp(np.interp(log_theta_fine, log_theta, log_mu))

    # 4 rows: (row 0) Quaia vs θ, (row 1) DESI vs θ,
    #         (row 2) Quaia vs r, (row 3) DESI vs r.
    fig, axes = plt.subplots(4, n_z_plot, figsize=(4.5 * n_z_plot, 14.5),
                              squeeze=False)

    def _plot_one_panel(ax, x_data, x_pois, pmf_dd, pmf_rd,
                         k_max_rd_avail,
                         mu_dd_fine, mu_rd_fine,
                         marker_dd, show_legend, cat_label):
        """Plot one panel for one catalog at one z shell.

        ``x_data`` is the measurement-bin axis (length ~90); ``x_pois``
        is the fine-grid axis matching ``mu_*_fine``.

        Plots two Poisson references per k: solid thin (DD reference,
        from analytic ⟨A_w(q;data)⟩) for the DD markers, and dotted
        thin (RD reference, from analytic ⟨A_w(q;random)⟩) for the
        RD markers.
        """
        for jk, k in enumerate(k_targets):
            col = colors[jk]
            label = (f"k={k}" if k != 0 else "k=0 (VPF)") if show_legend else None
            ax.plot(x_data, pmf_dd[:, k], marker_dd + "-",
                    color=col, lw=1.4, ms=2, alpha=0.9,
                    label=f"DD {label}" if show_legend else None)
            if pmf_rd is not None and k < k_max_rd_avail:
                ax.plot(x_data, pmf_rd[:, k], marker_dd + "--",
                        color=col, lw=0.9, ms=1.5, alpha=0.6,
                        label=f"RD {label}" if show_legend else None)
            ax.plot(x_pois, poisson.pmf(k, mu_dd_fine),
                    "-", color=col, lw=0.8, alpha=0.85,
                    label=(f"Poisson(μ_DD{ref_label_suffix})"
                           if show_legend and jk == 0 else None))
            if mu_rd_fine is not None:
                ax.plot(x_pois, poisson.pmf(k, mu_rd_fine),
                        ":", color=col, lw=0.8, alpha=0.85,
                        label=(f"Poisson(μ_RD{ref_label_suffix})"
                               if show_legend and jk == 0 else None))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.grid(alpha=0.3, which="both")
        if show_legend:
            ax.legend(fontsize=5, ncol=2, loc="lower left")

    for ip, iq in enumerate(Z_PLOT_IDX):
        r_axis = theta_rad * chi_at_z[iq]
        r_axis_fine = theta_rad_fine * chi_at_z[iq]
        mu_q_fine_dd = _interp_mean(_diag(mean_q_th, iq))
        mu_d_fine_dd = _interp_mean(_diag(mean_d_th, iq))
        mu_q_fine_rd = _interp_mean(_diag(mean_q_th_rd, iq))
        mu_d_fine_rd = _interp_mean(_diag(mean_d_th_rd, iq))

        # Row 0: Quaia vs theta
        ax = axes[0, ip]
        _plot_one_panel(
            ax, theta_deg, theta_deg_fine,
            _diag_pmf(pmf_q_dd, iq, slice(None)),
            _diag_pmf(pmf_q_rd, iq, slice(None)) if pmf_q_rd is not None else None,
            k_max_q_rd, mu_q_fine_dd, mu_q_fine_rd, "o",
            show_legend=(iq == 0), cat_label="Q",
        )
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel(r"Quaia $P_{N=k}$" if ip == 0 else "")
        ax.set_title(rf"z={z_q_mid[iq]:.2f}  "
                     rf"($\chi$={chi_at_z[iq]:.0f} Mpc/h)")

        # Row 1: DESI vs theta
        ax = axes[1, ip]
        _plot_one_panel(
            ax, theta_deg, theta_deg_fine,
            _diag_pmf(pmf_d_dd, iq, slice(None)),
            _diag_pmf(pmf_d_rd, iq, slice(None)) if pmf_d_rd is not None else None,
            k_max_d_rd, mu_d_fine_dd, mu_d_fine_rd, "s",
            show_legend=(iq == 0), cat_label="D",
        )
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel(r"DESI $P_{N=k}$" if ip == 0 else "")

        # Row 2: Quaia vs r
        ax = axes[2, ip]
        _plot_one_panel(
            ax, r_axis, r_axis_fine,
            _diag_pmf(pmf_q_dd, iq, slice(None)),
            _diag_pmf(pmf_q_rd, iq, slice(None)) if pmf_q_rd is not None else None,
            k_max_q_rd, mu_q_fine_dd, mu_q_fine_rd, "o",
            show_legend=False, cat_label="Q",
        )
        ax.set_xlabel("r [Mpc/h] (Ωm=0.31, h=0.68)")
        ax.set_ylabel(r"Quaia $P_{N=k}$" if ip == 0 else "")

        # Row 3: DESI vs r
        ax = axes[3, ip]
        _plot_one_panel(
            ax, r_axis, r_axis_fine,
            _diag_pmf(pmf_d_dd, iq, slice(None)),
            _diag_pmf(pmf_d_rd, iq, slice(None)) if pmf_d_rd is not None else None,
            k_max_d_rd, mu_d_fine_dd, mu_d_fine_rd, "s",
            show_legend=False, cat_label="D",
        )
        ax.set_xlabel("r [Mpc/h] (Ωm=0.31, h=0.68)")
        ax.set_ylabel(r"DESI $P_{N=k}$" if ip == 0 else "")

    fig.suptitle(
        rf"P(N = k cap counts) on the diagonal $z_q = z_n$ — "
        rf"k=0 is the VPF (= 1−P[N≥1] = 1NN-CDF complement); "
        rf"DD solid, RD dashed; thin solid = Poisson(μ_DD{ref_label_suffix}); "
        rf"thin dotted = Poisson(μ_RD{ref_label_suffix}).",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figs["p_n_k"] = fig_to_b64(fig)

    # ---- P(N=k) / Poisson(μ_an) ratio panel --------------------------
    # log10 ratio of empirical PMF over the analytic-Poisson reference
    # Poisson(μ_an, k) = exp(-μ_an) μ_an^k / k!. Generalises the VPF
    # log-ratio panel (which is just k=0) to all k targets. Above 1 ⇒
    # excess of caps with k neighbors; below 1 ⇒ deficit. For
    # clustered fields the typical pattern is excess at k=0 (more
    # voids) and excess at large k (more dense caps), with a deficit
    # at the mean.
    print("rendering P(N=k)/Poisson(μ_an) ratio panel ...")
    PMF_FLOOR = 1e-3
    # Thin to 4 well-separated k values for visual clarity. Each k's
    # arc peaks around θ where μ_an ≈ k, so 4 values gives 4 arcs
    # spanning the θ axis without too much overlap.
    k_targets_ratio = [k for k in [0, 3, 10, 30, 100] if k in k_targets]
    if not k_targets_ratio:
        k_targets_ratio = k_targets[:5]
    colors_ratio = [cmap(i / max(len(k_targets_ratio) - 1, 1))
                     for i in range(len(k_targets_ratio))]
    fig, axes = plt.subplots(2, n_z_plot, figsize=(4.5 * n_z_plot, 7.0),
                              squeeze=False, sharey=True)

    def _safe_log_ratio(obs, ref):
        """log10(obs/ref) where BOTH > PMF_FLOOR; NaN elsewhere. The
        joint floor is essential because Poisson(k, μ) is sharply
        peaked at μ ≈ k — outside that window the reference is
        numerically near zero and the ratio diverges meaninglessly.
        """
        good = (obs > PMF_FLOOR) & (ref > PMF_FLOOR)
        out = np.full_like(obs, np.nan, dtype=np.float64)
        out[good] = np.log10(obs[good] / ref[good])
        return out

    for ip, iq in enumerate(Z_PLOT_IDX):
        mu_q_dd_iq = _diag(mean_q_th, iq)
        mu_d_dd_iq = _diag(mean_d_th, iq)
        mu_q_rd_iq = _diag(mean_q_th_rd, iq)
        mu_d_rd_iq = _diag(mean_d_th_rd, iq)
        # Top row: Quaia
        ax = axes[0, ip]
        for jk, k in enumerate(k_targets_ratio):
            col = colors_ratio[jk]
            r = _safe_log_ratio(_diag_pmf(pmf_q_dd, iq, k),
                                 poisson.pmf(k, mu_q_dd_iq))
            ax.plot(theta_deg, r, "o-", color=col, lw=1.2, ms=2.5,
                    alpha=0.9,
                    label=f"DD k={k}" if ip == 0 else None)
            if pmf_q_rd is not None and k < k_max_q_rd:
                r_rd = _safe_log_ratio(_diag_pmf(pmf_q_rd, iq, k),
                                        poisson.pmf(k, mu_q_rd_iq))
                ax.plot(theta_deg, r_rd, "o--", color=col, lw=0.9,
                        ms=1.5, alpha=0.55)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]")
        if ip == 0:
            ax.set_ylabel(r"$\log_{10}(P_{\rm data}/P_{\rm Poisson}(\mu_{\rm an}))$  Quaia")
            ax.legend(fontsize=6, ncol=2, loc="upper right")
        ax.set_title(rf"z={z_q_mid[iq]:.2f}  ($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
        ax.set_ylim(-0.6, 0.8)
        ax.grid(alpha=0.3, which="both")
        # Bottom row: DESI
        ax = axes[1, ip]
        for jk, k in enumerate(k_targets_ratio):
            col = colors_ratio[jk]
            r = _safe_log_ratio(_diag_pmf(pmf_d_dd, iq, k),
                                 poisson.pmf(k, mu_d_dd_iq))
            ax.plot(theta_deg, r, "s-", color=col, lw=1.2, ms=2.5,
                    alpha=0.9,
                    label=f"DD k={k}" if ip == 0 else None)
            if pmf_d_rd is not None and k < k_max_d_rd:
                r_rd = _safe_log_ratio(_diag_pmf(pmf_d_rd, iq, k),
                                        poisson.pmf(k, mu_d_rd_iq))
                ax.plot(theta_deg, r_rd, "s--", color=col, lw=0.9,
                        ms=1.5, alpha=0.55)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]")
        if ip == 0:
            ax.set_ylabel(r"$\log_{10}(P_{\rm data}/P_{\rm Poisson}(\mu_{\rm an}))$  DESI")
        ax.set_ylim(-0.6, 0.8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(
        rf"P(N=k) / Poisson(μ_an) — clustering excess vs no-clustering "
        rf"reference at the analytic mean. DD solid, RD dashed.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figs["p_n_k_ratio"] = fig_to_b64(fig)

    # ---- VPF tab: dedicated diagnostic of P(N=0|θ;z) -----------------
    # The void probability function is the most window-robust k=0 slice
    # of the kNN-CDF. For Poisson VPF=exp(-⟨N⟩(θ)); the data/Poisson
    # ratio is the suppression of empty caps by clustering and probes
    # higher-order moments via -log(VPF) = sum of cumulants.
    print("rendering VPF panel ...")
    # P(N=0|θ;z) = 1 - P(N>=1|θ;z) = 1 - H_geq_k[...,0] / N_q
    def _vpf(res):
        Nq = res.N_q.astype(np.float64)
        safe = np.where(Nq > 0, Nq, np.inf)
        H1 = res.H_geq_k[..., 0]
        # H1 is (n_theta, n_z_q, n_z_n) full or (n_theta, n_z) diag.
        if getattr(res, "is_diagonal", False):
            return 1.0 - H1 / safe[None, :]
        return 1.0 - H1 / safe[None, :, None]

    def _vpf_jackknife_se(res):
        n_regions = res.H_geq_k_per_region.shape[-1]
        is_diag = getattr(res, "is_diagonal", False)
        # H1 leading dims: (n_theta, n_z_q, n_z_n) full or (n_theta, n_z) diag.
        h_lead = res.H_geq_k.shape[:-1]
        samples = np.zeros((n_regions,) + h_lead, dtype=np.float64)
        for k in range(n_regions):
            keep = np.ones(n_regions, dtype=bool); keep[k] = False
            H1_k = res.H_geq_k_per_region[..., 0, :][..., keep].sum(axis=-1)
            Nq_k = res.N_q_per_region[..., keep].sum(axis=-1).astype(np.float64)
            safe = np.where(Nq_k > 0, Nq_k, np.inf)
            if is_diag:
                samples[k] = 1.0 - H1_k / safe[None, :]
            else:
                samples[k] = 1.0 - H1_k / safe[None, :, None]
        return _se(samples)

    vpf_q = _vpf(quaia["res_dd"])
    vpf_d = _vpf(desi["res_dd"])
    se_vpf_q = _vpf_jackknife_se(quaia["res_dd"])
    se_vpf_d = _vpf_jackknife_se(desi["res_dd"])

    # Theta grid for smooth Poisson lines (5x densified, log-log
    # interpolation of <N>(theta) — same recipe as the P(N=k) panel).
    n_fine_v = 5 * theta_rad.size
    theta_rad_fine_v = np.geomspace(theta_rad.min(), theta_rad.max(), n_fine_v)
    theta_deg_fine_v = np.degrees(theta_rad_fine_v)
    log_theta_v = np.log(theta_rad)
    log_theta_fine_v = np.log(theta_rad_fine_v)
    def _interp_log(arr):
        with np.errstate(divide="ignore"):
            return np.exp(np.interp(log_theta_fine_v, log_theta_v,
                                     np.log(np.maximum(arr, 1e-30))))

    # Layout: top row = VPF and Poisson reference vs theta;
    #         bottom row = log10(VPF/VPF_Poisson) ratio (clustering
    #         suppression of empty caps; >0 ⇒ data has more empty caps
    #         than Poisson, ⇒ clustering).
    fig, axes = plt.subplots(2, n_z_plot, figsize=(4.0 * n_z_plot, 8.5),
                              squeeze=False)
    for ip, iq in enumerate(Z_PLOT_IDX):
        mu_q_diag = _diag(mean_q_th, iq)
        mu_d_diag = _diag(mean_d_th, iq)
        mu_q_fine = _interp_log(mu_q_diag)
        mu_d_fine = _interp_log(mu_d_diag)
        vpf_q_pois_fine = np.exp(-mu_q_fine)
        vpf_d_pois_fine = np.exp(-mu_d_fine)
        vpf_q_pois = np.exp(-mu_q_diag)
        vpf_d_pois = np.exp(-mu_d_diag)

        # Top row: VPF + Poisson
        ax = axes[0, ip]
        ax.errorbar(theta_deg, _diag(vpf_q, iq),
                    yerr=_diag(se_vpf_q, iq),
                    fmt="o-", color="#1f77b4", ms=3, lw=1.4,
                    capsize=2, label=f"{quaia['label']} VPF")
        ax.errorbar(theta_deg, _diag(vpf_d, iq),
                    yerr=_diag(se_vpf_d, iq),
                    fmt="s-", color="#ff7f0e", ms=3, lw=1.4,
                    capsize=2, label=f"{desi['label']} VPF")
        ax.plot(theta_deg_fine_v, vpf_q_pois_fine,
                "-", color="#1f77b4", lw=0.9, alpha=0.85,
                label="Quaia Poisson e^{-⟨N⟩}")
        ax.plot(theta_deg_fine_v, vpf_d_pois_fine,
                ":", color="#ff7f0e", lw=0.9, alpha=0.85,
                label="DESI Poisson e^{-⟨N⟩}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel(r"VPF $=P(N=0|\theta;z)$" if ip == 0 else "")
        ax.set_ylim(1e-4, 1.5)
        ax.set_title(rf"z={z_q_mid[iq]:.2f}  "
                     rf"($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
        ax.grid(alpha=0.3, which="both")
        if ip == 0:
            ax.legend(fontsize=7, loc="lower left")

        # Bottom row: log10( VPF / VPF_Poisson ) — clustering excess
        # of empty caps. >0 ⇒ data has MORE empty caps than Poisson,
        # the signature of clustered tracers leaving large voids.
        # Mask points where measured VPF is too small to be reliable
        # (≲1e-3): a single-cap fluctuation can dominate, and the log
        # ratio diverges; show only the well-measured regime.
        VPF_FLOOR = 1e-3
        def _safe_log10_ratio(vpf, se, vpf_pois):
            ok = (vpf > VPF_FLOOR) & (vpf_pois > 1e-30)
            ratio = np.full_like(vpf, np.nan)
            ratio[ok] = np.log10(vpf[ok] / vpf_pois[ok])
            se_log = np.full_like(vpf, np.nan)
            se_log[ok] = se[ok] / (vpf[ok] * np.log(10))
            return ratio, se_log
        log10_ratio_q, se_log_q = _safe_log10_ratio(
            _diag(vpf_q, iq), _diag(se_vpf_q, iq), vpf_q_pois)
        log10_ratio_d, se_log_d = _safe_log10_ratio(
            _diag(vpf_d, iq), _diag(se_vpf_d, iq), vpf_d_pois)

        ax = axes[1, ip]
        ax.errorbar(theta_deg, log10_ratio_q, yerr=se_log_q,
                    fmt="o-", color="#1f77b4", ms=3, lw=1.4, capsize=2,
                    label=quaia["label"])
        ax.errorbar(theta_deg, log10_ratio_d, yerr=se_log_d,
                    fmt="s-", color="#ff7f0e", ms=3, lw=1.4, capsize=2,
                    label=desi["label"])
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.4, 0.6)
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel(r"$\log_{10}\left(\mathrm{VPF}/\mathrm{VPF}^{\rm Poisson}\right)$"
                      if ip == 0 else "")
        ax.grid(alpha=0.3, which="both")
        if ip == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Void Probability Function on the diagonal z_q=z_n.  "
        "Top: VPF + Poisson reference exp(−⟨N⟩(θ)).  "
        "Bottom: log10(VPF/VPF_Poisson) — clustering excess of empty caps.",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figs["vpf"] = fig_to_b64(fig)

    # ---- Cross-correlation tab (if artifacts available) -------------
    cross_metrics = dict(cross_status="cross artifacts not found")
    # Cross artifacts use a fixed (pipeline-time) z grid which may
    # differ from the auto pipelines' grid (e.g. when autos are run
    # at PAPER_N_Z_SHELLS=64 diagonal-only and cross stays at the
    # default 4-shell full cube). Skip the cross tab cleanly when
    # the z grids don't match — re-run quaia_x_desi_pipeline.py
    # with matching env vars to enable.
    if cross is not None:
        cross_n_z = cross["dd"].sum_n.shape[1]
        auto_n_z = quaia["res_dd"].N_q.size
        if cross_n_z != auto_n_z:
            print(f"  cross n_z={cross_n_z} != auto n_z={auto_n_z}; "
                  "skipping cross-correlation tab. Re-run "
                  "demos/quaia_x_desi_pipeline.py with matching "
                  "PAPER_N_Z_SHELLS / PAPER_DIAGONAL_ONLY to "
                  "enable.")
            cross_metrics = dict(
                cross_status=(f"z-grid mismatch (cross n_z={cross_n_z}, "
                              f"auto n_z={auto_n_z})"))
            cross = None
    if cross is not None:
        # Per-shell counts of the *Y* (Quaia) data and random catalogs.
        # The Quaia auto pipeline's DD pass has queries = neighbors = Quaia
        # data, so its N_q gives Quaia data per z_q shell. Likewise the
        # auto RR pass gives Quaia random per shell.
        n_dy_per_zn = quaia["res_dd"].N_q.astype(np.int64)
        n_ry_per_zn = quaia["res_rr"].N_q.astype(np.int64)

        # Davis-Peebles cross: 1 + xi_xy = nbar_DD_xy / nbar_DR_xy
        # (independent of N_R/N_D ratio because the ratio cancels the
        # neighbor density). Use Quaia data and random per-shell counts
        # to renormalise so DR is on the data-density scale.
        nbar_dd_xy = mean_count(cross["dd"])
        nbar_dr_xy = mean_count(cross["dr"])
        # Density-renormalise nbar_DR to data-equivalent density:
        #   nbar_DR_normalized = nbar_DR * (N_dy / N_ry)
        scale = (n_dy_per_zn / np.maximum(n_ry_per_zn, 1)
                 ).astype(np.float64)
        xi_dp_xy = (nbar_dd_xy
                    / np.where(nbar_dr_xy * scale[None, None, :] > 0,
                               nbar_dr_xy * scale[None, None, :], np.inf)
                    - 1.0)

        # Landy-Szalay cross via the asymmetric 4-term estimator
        xi_ls_xy = xi_ls_cross(
            cross["dd"], cross["dr"], cross["rd"], cross["rr"],
            n_dy_per_zn, n_ry_per_zn,
        )

        # Auto LS for context (already computed above as xi_q, xi_d).
        fig, axes = plt.subplots(1, n_z_plot, figsize=(4.0 * n_z_plot, 4.5),
                                  squeeze=False, sharey=False)
        for ip, iq in enumerate(Z_PLOT_IDX):
            ax = axes[0, ip]
            ax.plot(theta_deg, _diag(xi_ls_xy, iq),
                    "D-", color="#2ca02c", lw=1.7, ms=5,
                    label="DESI x Quaia (xi_LS)")
            ax.plot(theta_deg, _diag(xi_dp_xy, iq),
                    "x--", color="#9467bd", lw=1.0, ms=5,
                    label="DESI x Quaia (xi_DP)")
            if xi_q["xi_ls"] is not None:
                ax.errorbar(theta_deg, _diag(xi_q["xi_ls"], iq),
                            yerr=_diag(xi_q["se_xi_ls"], iq),
                            fmt="o-", color="#1f77b4", capsize=3,
                            lw=1.0, alpha=0.6, label="Quaia auto")
            if xi_d["xi_ls"] is not None:
                ax.errorbar(theta_deg, _diag(xi_d["xi_ls"], iq),
                            yerr=_diag(xi_d["se_xi_ls"], iq),
                            fmt="s-", color="#ff7f0e", capsize=3,
                            lw=1.0, alpha=0.6, label="DESI auto")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("theta [deg]")
            if ip == 0:
                ax.set_ylabel(r"$\hat\xi(\theta)$")
            ax.set_title(rf"z={z_q_mid[iq]:.2f}  "
                         rf"($\chi$={chi_at_z[iq]:.0f} Mpc/h)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        figs["cross_diag"] = fig_to_b64(fig)

        # Off-diagonal (z_q != z_n) — the cross between catalogs at
        # *different* redshifts should be near 0 if the two catalogs
        # only share structures within the same shell.
        fig, axes = plt.subplots(n_z_q, n_z_q,
                                  figsize=(3.2 * n_z_plot, 3.0 * n_z_q),
                                  squeeze=False, sharex=True)
        for ip, iq in enumerate(Z_PLOT_IDX):
            for jn in range(n_z_q):
                ax = axes[iq, jn]
                ax.plot(theta_deg, xi_ls_xy[:, iq, jn],
                        "D-", color="#2ca02c", lw=1.4, ms=4)
                ax.axhline(0, color="k", lw=0.4, ls=":")
                ax.set_xscale("log")
                if iq == n_z_q - 1:
                    ax.set_xlabel("theta [deg]")
                if jn == 0:
                    ax.set_ylabel(rf"z_q={z_q_mid[iq]:.2f}")
                if ip == 0:
                    ax.set_title(rf"z_n={z_q_mid[jn]:.2f}", fontsize=11)
                ax.grid(alpha=0.3)
        fig.suptitle(r"Cross $\xi_{LS}^{xy}(\theta;\,z_q,z_n)$ "
                     "DESI x Quaia (rows = z_q, cols = z_n)",
                     fontsize=13)
        fig.tight_layout()
        figs["cross_grid"] = fig_to_b64(fig)

        cross_metrics = dict(
            cross_status="available",
            n_dx=int(cross["dd"].N_q.sum()),  # DESI data total = sum of cross DD N_q
            n_dy=int(n_dy_per_zn.sum()),
            n_rx=int(cross["rd"].N_q.sum()),  # DESI random
            n_ry=int(n_ry_per_zn.sum()),
        )

    # ---- Run-time accounting --------------------------------------
    def _t(art, key):
        return float(art[key]) if key in art.files else 0.0

    timings = {
        "Quaia DD (auto)": _t(quaia["dd_artifact"], "dd_bao_elapsed_s"),
        "Quaia RD (auto)": _t(quaia["rd_artifact"], "rd_elapsed_s"),
        "Quaia RR (auto)": (_t(quaia["rr_artifact"], "rr_elapsed_s")
                            if quaia["rr_artifact"] is not None else 0.0),
        "DESI DD (auto)":  _t(desi["dd_artifact"], "dd_bao_elapsed_s"),
        "DESI RD (auto)":  _t(desi["rd_artifact"], "rd_elapsed_s"),
        "DESI RR (auto)":  (_t(desi["rr_artifact"], "rr_elapsed_s")
                            if desi["rr_artifact"] is not None else 0.0),
    }
    cross_timings = {}
    if cross is not None:
        for k, label in [("dd", "Cross DD_xy"), ("dr", "Cross DR_xy"),
                         ("rd", "Cross RD_xy"), ("rr", "Cross RR_xy")]:
            path = os.path.join(OUTPUT_DIR, f"quaia_x_desi_{k}.npz")
            art = np.load(path, allow_pickle=True)
            cross_timings[label] = _t(art, "elapsed_s")
    timings.update(cross_timings)
    t_total = sum(timings.values())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = list(timings.keys())
    secs = np.array([timings[k] for k in labels])
    pct = 100.0 * secs / max(t_total, 1.0)
    colors_t = ["#1f77b4"] * 3 + ["#ff7f0e"] * 3 + ["#2ca02c"] * len(cross_timings)
    bars = ax.barh(labels, secs / 60.0, color=colors_t)
    for b, s, p in zip(bars, secs, pct):
        if s > 0:
            ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                    f"{s/60:.1f} min  ({p:.1f}%)",
                    va="center", fontsize=9)
    ax.set_xlabel("wall time [min]  (single run, 8 numba threads)")
    ax.set_title(f"Per-pass kNN-CDF wall time  —  total = "
                 f"{t_total/60:.1f} min")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    figs["timing"] = fig_to_b64(fig)

    # ===================================================================
    # Note v4_1 §3, §4, §6 additions: higher-moment & complementary
    # diagnostics. All panels here are gated on the relevant cubes
    # being present so the page degrades gracefully when older
    # artifacts are loaded.
    # ===================================================================

    from twopt_density.knn_derived import (
        cic_skewness_raw, cic_kurtosis_raw,
        xi_hamilton, sigma2_clust_ls, cic_moment_ls,
        differential_pair_count, dlnsigma2_dlogz,
    )

    have_higher = (
        getattr(quaia["res_dd"], "sum_n3", None) is not None
        and getattr(desi["res_dd"], "sum_n3", None) is not None
    )

    # ---- S₃ skewness (raw DD form) per-shell columns + heatmap ----
    if have_higher:
        print("rendering S₃ skewness panels ...")
        s3_q = cic_skewness_raw(quaia["res_dd"])
        s3_d = cic_skewness_raw(desi["res_dd"])
        fig, axes = plt.subplots(
            2, n_z_plot, figsize=(4.0 * n_z_plot, 7.5),
            squeeze=False, sharex=True,
        )
        for ip, iq in enumerate(Z_PLOT_IDX):
            for row, (label, s3, color) in enumerate((
                ("Quaia", s3_q, "#1f77b4"),
                ("DESI",  s3_d, "#ff7f0e"),
            )):
                ax = axes[row, ip]
                y = _diag(s3, iq)
                ax.plot(theta_deg, y, "o-", color=color, lw=1.4, ms=4,
                        label=f"{label} S₃")
                ax.axhline(0, color="k", lw=0.5, ls=":")
                ax.set_xscale("log")
                if ip == 0:
                    ax.set_ylabel(rf"$S_3(\theta)$  [{label}]")
                if row == 1:
                    ax.set_xlabel("θ [deg]")
                ax.set_title(f"z={z_q_mid[iq]:.2f}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, loc="best")
        fig.suptitle(
            "Skewness $S_3 = \\langle(N-\\langle N\\rangle)^3\\rangle/\\sigma^3$ "
            "(raw DD form; note v4_1 §6)",
            y=0.995, fontsize=11)
        fig.tight_layout()
        figs["s3_skew"] = fig_to_b64(fig)
        plt.close(fig)

        if n_z_q > N_Z_PLOT:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)
            for ax, cube, lbl in zip(
                axes[0], (s3_q, s3_d),
                ("Quaia G<20", "DESI Y1 QSO"),
            ):
                pc = _heatmap_panel_diag(
                    ax, cube, theta_deg, quaia["z_q_edges"],
                    cmap="RdBu_r", vmin=-2.0, vmax=2.0,
                    label="z", title=f"{lbl}  S₃(θ; z)")
                fig.colorbar(pc, ax=ax, label=r"$S_3$")
            fig.suptitle("Skewness heatmaps (full z range)", y=0.995)
            fig.tight_layout()
            figs["s3_skew_heat"] = fig_to_b64(fig)
            plt.close(fig)

        # ---- S₄ kurtosis (excess) panels ----
        print("rendering S₄ kurtosis panels ...")
        s4_q = cic_kurtosis_raw(quaia["res_dd"])
        s4_d = cic_kurtosis_raw(desi["res_dd"])
        fig, axes = plt.subplots(
            2, n_z_plot, figsize=(4.0 * n_z_plot, 7.5),
            squeeze=False, sharex=True,
        )
        for ip, iq in enumerate(Z_PLOT_IDX):
            for row, (label, s4, color) in enumerate((
                ("Quaia", s4_q, "#1f77b4"),
                ("DESI",  s4_d, "#ff7f0e"),
            )):
                ax = axes[row, ip]
                y = _diag(s4, iq)
                ax.plot(theta_deg, y, "o-", color=color, lw=1.4, ms=4,
                        label=f"{label} kurt")
                ax.axhline(0, color="k", lw=0.5, ls=":")
                ax.set_xscale("log")
                if ip == 0:
                    ax.set_ylabel(rf"$S_4-3$  [{label}]")
                if row == 1:
                    ax.set_xlabel("θ [deg]")
                ax.set_title(f"z={z_q_mid[iq]:.2f}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, loc="best")
        fig.suptitle(
            "Excess kurtosis $S_4-3$ (raw DD form; note v4_1 §6)",
            y=0.995, fontsize=11)
        fig.tight_layout()
        figs["s4_kurt"] = fig_to_b64(fig)
        plt.close(fig)

        if n_z_q > N_Z_PLOT:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)
            for ax, cube, lbl in zip(
                axes[0], (s4_q, s4_d),
                ("Quaia G<20", "DESI Y1 QSO"),
            ):
                pc = _heatmap_panel_diag(
                    ax, cube, theta_deg, quaia["z_q_edges"],
                    cmap="RdBu_r", vmin=-5.0, vmax=5.0,
                    label="z", title=f"{lbl}  kurt(θ; z)")
                fig.colorbar(pc, ax=ax, label=r"$S_4-3$")
            fig.suptitle("Kurtosis heatmaps (full z range)", y=0.995)
            fig.tight_layout()
            figs["s4_kurt_heat"] = fig_to_b64(fig)
            plt.close(fig)
    else:
        print("skipping S₃/S₄ panels: sum_n3/sum_n4 absent from artifacts")

    # ---- Hamilton ξ vs LS ξ (only when DR is loaded) ---------------
    have_dr = (quaia["res_dr"] is not None and desi["res_dr"] is not None)
    if have_dr:
        print("rendering Hamilton ξ vs LS comparison ...")
        # Per-shell weight sums for the Hamilton/LS estimators.
        # Quaia is unweighted: n_neigh per zn = the per-shell count.
        n_d_q_per_zn = quaia["res_dd"].N_q.astype(np.float64)
        n_r_q_per_zn = quaia["res_rd"].N_q.astype(np.float64)
        n_d_d_per_zn = np.asarray(desi_w_sum_per_zn, dtype=np.float64)
        n_r_d_per_zn = desi["res_rd"].N_q.astype(np.float64)

        try:
            xi_h_q = xi_hamilton(
                quaia["res_dd"], quaia["res_dr"], quaia["res_rd"],
                quaia["res_rr"],
                n_d_q_per_zn, n_r_q_per_zn,
            )
            xi_h_d = xi_hamilton(
                desi["res_dd"], desi["res_dr"], desi["res_rd"],
                desi["res_rr"],
                n_d_d_per_zn, n_r_d_per_zn,
            )
        except Exception as e:
            print(f"  xi_hamilton failed: {e}")
            xi_h_q = xi_h_d = None

        if xi_h_q is not None and xi_h_d is not None:
            fig, axes = plt.subplots(
                2, n_z_plot, figsize=(4.0 * n_z_plot, 7.5),
                squeeze=False, sharex=True,
            )
            for ip, iq in enumerate(Z_PLOT_IDX):
                for row, (label, xi_h, xi_l, color) in enumerate((
                    ("Quaia", xi_h_q, xi_q.get("xi_ls"), "#1f77b4"),
                    ("DESI",  xi_h_d, xi_d.get("xi_ls"), "#ff7f0e"),
                )):
                    ax = axes[row, ip]
                    yh = _diag(xi_h, iq)
                    ax.plot(theta_deg, yh, "o-", color=color, lw=1.4, ms=4,
                            label="Hamilton")
                    if xi_l is not None:
                        yl = _diag(xi_l, iq)
                        ax.plot(theta_deg, yl, "x--", color=color,
                                lw=1.0, ms=4, alpha=0.65,
                                label="LS (cap-avg)")
                    ax.axhline(0, color="k", lw=0.5, ls=":")
                    ax.set_xscale("log")
                    if ip == 0:
                        ax.set_ylabel(rf"$\xi(\theta)$  [{label}]")
                    if row == 1:
                        ax.set_xlabel("θ [deg]")
                    ax.set_title(f"z={z_q_mid[iq]:.2f}")
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8, loc="best")
            fig.suptitle(
                "Hamilton ξ (note Eq. 15) vs Landy–Szalay ξ (Eq. 12); "
                "Hamilton is more robust to small RR.",
                y=0.995, fontsize=11)
            fig.tight_layout()
            figs["xi_hamilton"] = fig_to_b64(fig)
            plt.close(fig)
    else:
        print("skipping Hamilton panel: DR artifacts not loaded "
              "(rerun pipelines with PAPER_DR_PASS=1)")

    # ---- Differential pair count dn_pairs/dθ -----------------------
    print("rendering differential pair count panel ...")
    dn_q = differential_pair_count(quaia["res_dd"])
    dn_d = differential_pair_count(desi["res_dd"])
    fig, axes = plt.subplots(
        2, n_z_plot, figsize=(4.0 * n_z_plot, 7.5),
        squeeze=False, sharex=True,
    )
    for ip, iq in enumerate(Z_PLOT_IDX):
        for row, (label, dn, color) in enumerate((
            ("Quaia", dn_q, "#1f77b4"),
            ("DESI",  dn_d, "#ff7f0e"),
        )):
            ax = axes[row, ip]
            y = _diag(dn, iq)
            ax.plot(theta_deg, y, "o-", color=color, lw=1.4, ms=4,
                    label=f"{label} d⟨N⟩/dθ · θ")
            ax.set_xscale("log")
            if ip == 0:
                ax.set_ylabel(rf"$\theta\, \partial\langle N\rangle/\partial\theta$  [{label}]")
            if row == 1:
                ax.set_xlabel("θ [deg]")
            ax.set_title(f"z={z_q_mid[iq]:.2f}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Differential pair-count density θ·∂⟨N⟩/∂θ "
        "(note v4_1 Eq. 9) — alternate visual to ξ_LS",
        y=0.995, fontsize=11)
    fig.tight_layout()
    figs["dn_dtheta"] = fig_to_b64(fig)
    plt.close(fig)

    # ---- Redshift derivative ∂ln σ² / ∂ln(1+z) heatmap -------------
    if n_z_q >= 3:
        print("rendering ∂lnσ²/∂ln(1+z) panel ...")
        # s2 is already (n_theta, n_z) on a diagonal cube; pass as-is.
        deriv_q = dlnsigma2_dlogz(s2_q, z_q_mid)
        deriv_d = dlnsigma2_dlogz(s2_d, z_q_mid)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), squeeze=False)
        for ax, cube, lbl in zip(
            axes[0], (deriv_q, deriv_d),
            ("Quaia G<20", "DESI Y1 QSO"),
        ):
            pc = _heatmap_panel_diag(
                ax, cube, theta_deg, quaia["z_q_edges"],
                cmap="RdBu_r", vmin=-4.0, vmax=4.0,
                label="z", title=f"{lbl}  ∂lnσ²/∂ln(1+z)")
            fig.colorbar(pc, ax=ax,
                         label=r"$\partial\ln\sigma^2/\partial\ln(1+z)$")
        fig.suptitle(
            "Logarithmic redshift derivative ∂lnσ²/∂ln(1+z) "
            "(note v4_1 Eq. 20: 2p_bias + growth + geometry decomposition)",
            y=0.995, fontsize=11)
        fig.tight_layout()
        figs["dlogsigma2_dlogz"] = fig_to_b64(fig)
        plt.close(fig)

    # ---- LS-corrected σ² (only when DR available) ------------------
    if have_dr:
        print("rendering σ²_clust^LS panel ...")
        try:
            s2_ls_q = sigma2_clust_ls(
                quaia["res_dd"], quaia["res_dr"], quaia["res_rd"],
                quaia["res_rr"],
                quaia["res_dd"].N_q.astype(np.float64),
                quaia["res_rd"].N_q.astype(np.float64),
            )
            s2_ls_d = sigma2_clust_ls(
                desi["res_dd"], desi["res_dr"], desi["res_rd"],
                desi["res_rr"],
                np.asarray(desi_w_sum_per_zn, dtype=np.float64),
                desi["res_rd"].N_q.astype(np.float64),
            )
        except Exception as e:
            print(f"  sigma2_clust_ls failed: {e}")
            s2_ls_q = s2_ls_d = None

        if s2_ls_q is not None and s2_ls_d is not None:
            fig, axes = plt.subplots(
                2, n_z_plot, figsize=(4.0 * n_z_plot, 7.5),
                squeeze=False, sharex=True,
            )
            for ip, iq in enumerate(Z_PLOT_IDX):
                for row, (label, s2, s2_lsx, color) in enumerate((
                    ("Quaia", s2_q, s2_ls_q, "#1f77b4"),
                    ("DESI",  s2_d, s2_ls_d, "#ff7f0e"),
                )):
                    ax = axes[row, ip]
                    ax.plot(theta_deg, _diag(s2, iq), "o-",
                            color=color, lw=1.4, ms=4, label="DD raw")
                    ax.plot(theta_deg, _diag(s2_lsx, iq), "x--",
                            color=color, lw=1.0, ms=4, alpha=0.7,
                            label="LS-corrected")
                    ax.set_xscale("log"); ax.set_yscale("symlog",
                                                        linthresh=1e-3)
                    ax.axhline(0, color="k", lw=0.4, ls=":")
                    if ip == 0:
                        ax.set_ylabel(rf"$\sigma^2_{{clust}}$  [{label}]")
                    if row == 1:
                        ax.set_xlabel("θ [deg]")
                    ax.set_title(f"z={z_q_mid[iq]:.2f}")
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8, loc="best")
            fig.suptitle(
                "σ²_clust: raw DD vs Landy–Szalay-corrected "
                "(note v4_1 Eq. 14)", y=0.995, fontsize=11)
            fig.tight_layout()
            figs["sigma2_compare"] = fig_to_b64(fig)
            plt.close(fig)

    # ---- ⟨N^p⟩^LS bias-corrected moment overlays (p=1,2,3,4) -------
    if have_dr and have_higher:
        print("rendering LS moment p=1,2,3,4 panel ...")
        try:
            mom_p_lookup = {}
            for p in (1, 2, 3, 4):
                mom_p_lookup[("quaia", p)] = cic_moment_ls(
                    quaia["res_dd"], quaia["res_dr"], quaia["res_rd"],
                    quaia["res_rr"], p,
                    quaia["res_dd"].N_q.astype(np.float64),
                    quaia["res_rd"].N_q.astype(np.float64),
                )
                mom_p_lookup[("desi", p)] = cic_moment_ls(
                    desi["res_dd"], desi["res_dr"], desi["res_rd"],
                    desi["res_rr"], p,
                    np.asarray(desi_w_sum_per_zn, dtype=np.float64),
                    desi["res_rd"].N_q.astype(np.float64),
                )
            fig, axes = plt.subplots(
                2, n_z_plot, figsize=(4.0 * n_z_plot, 7.5),
                squeeze=False, sharex=True,
            )
            for ip, iq in enumerate(Z_PLOT_IDX):
                for row, (catname, color, name) in enumerate((
                    ("quaia", "#1f77b4", "Quaia"),
                    ("desi",  "#ff7f0e", "DESI"),
                )):
                    ax = axes[row, ip]
                    for p, marker, ls in ((1, "D", "-"),
                                           (2, "o", "-"),
                                           (3, "s", "--"),
                                           (4, "^", ":")):
                        mp = mom_p_lookup[(catname, p)]
                        if mp is None:
                            continue
                        y = _diag(mp, iq)
                        ax.plot(theta_deg, y, marker + ls, color=color,
                                lw=1.2, ms=4, alpha=0.85,
                                label=f"⟨N^{p}⟩^LS")
                    ax.set_xscale("log")
                    ax.set_yscale("symlog", linthresh=1e-2)
                    ax.axhline(0, color="k", lw=0.4, ls=":")
                    if ip == 0:
                        ax.set_ylabel(rf"$\langle N^p\rangle^{{LS}}$  [{name}]")
                    if row == 1:
                        ax.set_xlabel("θ [deg]")
                    ax.set_title(f"z={z_q_mid[iq]:.2f}")
                    ax.grid(alpha=0.3)
                    ax.legend(fontsize=8, loc="best")
            fig.suptitle(
                "Landy–Szalay-corrected raw moments ⟨N^p⟩^LS for p=1,2,3,4 "
                "(note v4_1 Eq. 13). p=1 is the connected pair density "
                "(≡ 1+ξ_LS up to normalisation).",
                y=0.995, fontsize=11)
            fig.tight_layout()
            figs["moment_ls"] = fig_to_b64(fig)
            plt.close(fig)
        except Exception as e:
            print(f"  moment_ls failed: {e}")

    # ===================================================================
    # End of v4_1 additions.
    # ===================================================================

    # ---- Angular ξ(θ) via cascade tree (chord-length metric) -------
    # The same morton_cascade tree, but applied to (RA, Dec) → S²
    # unit vectors with chord d = 2 sin(θ/2). Each z-shell becomes
    # one cascade pass, giving ξ_LS(θ;z) at every dyadic chord scale
    # in fractions of a second per shell.
    morton_ang_path = os.path.join(OUTPUT_DIR, "morton_angular_xi.npz")
    morton_ang_status = "not-run"
    if os.path.exists(morton_ang_path):
        morton_ang_status = "available"
        print("rendering angular cascade ξ(θ;z) panel ...")
        ma_ang = np.load(morton_ang_path, allow_pickle=True)
        z_edges_ang = ma_ang["z_edges"]
        n_z_ang = z_edges_ang.size - 1
        fig, axes = plt.subplots(1, n_z_ang, figsize=(4.0 * n_z_ang, 4.5),
                                  squeeze=False, sharey=True)
        for iq in range(n_z_ang):
            ax = axes[0, ip]
            for cat, color, marker, label in (
                ("quaia", "#1f77b4", "o", "Quaia G<20"),
                ("desi", "#ff7f0e", "s", "DESI Y1 QSO"),
            ):
                key_theta = f"{cat}_iq{iq}_theta_deg"
                if key_theta not in ma_ang.files:
                    continue
                theta = ma_ang[key_theta]
                xi = ma_ang[f"{cat}_iq{iq}_xi"]
                RR = ma_ang[f"{cat}_iq{iq}_RR"]
                good = ((RR > 0) & np.isfinite(xi)
                         & (theta > 0.05) & (theta < 12.0))
                if not good.any():
                    continue
                ax.plot(theta[good], xi[good], marker + "-",
                        color=color, lw=1.4, ms=5,
                        label=f"{label} (cascade)")
            # Overlay kNN-cube σ²_LS = ⟨ξ⟩_cap on the diagonal
            # z_q=z_n for the same shell — different smoothing
            # window (cap-cumulative vs differential dyadic shell)
            # but should agree in scale.
            if (xi_q["xi_ls"] is not None and xi_d["xi_ls"] is not None
                    and iq < n_z_q):
                ax.plot(theta_deg, _diag(xi_q["xi_ls"], iq),
                        "-", color="#1f77b4", lw=0.7, alpha=0.55,
                        label="Quaia ⟨ξ⟩_cap (kNN)")
                ax.plot(theta_deg, _diag(xi_d["xi_ls"], iq),
                        "-", color="#ff7f0e", lw=0.7, alpha=0.55,
                        label="DESI ⟨ξ⟩_cap (kNN)")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("θ [deg]   (= 2 arcsin(d/2))")
            if ip == 0:
                ax.set_ylabel(r"$\xi_{\rm LS}(\theta)$")
            ax.set_title(f"z=[{z_edges_ang[iq]:.2f}, {z_edges_ang[iq+1]:.2f}]")
            ax.set_ylim(-0.05, 0.5)
            ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        figs["morton_angular"] = fig_to_b64(fig)
        morton_ang_t_q = float(ma_ang["quaia_t_total_s"])
        morton_ang_t_d = float(ma_ang["desi_t_total_s"])
    else:
        print(f"morton angular artifact not found at {morton_ang_path}; "
              "skipping panel. Run demos/morton_angular_demo.py first.")
        morton_ang_t_q = morton_ang_t_d = 0.0

    # ---- 3D Cascade ξ(r) panel (morton_cascade Rust backend) -------
    # Read the artifact written by demos/morton_xi_demo.py if present.
    # The cascade produces ξ(r) at every dyadic shell from sub-Mpc up
    # to the box-size in a single O(N log N) pass. Complementary to
    # the angular-cap σ²_LS — this is native 3D LSS clustering.
    morton_path = os.path.join(OUTPUT_DIR, "morton_xi_quaia_desi.npz")
    morton_status = "not-run"
    if os.path.exists(morton_path):
        morton_status = "available"
        print("rendering morton_cascade ξ(r) panel ...")
        ma = np.load(morton_path, allow_pickle=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        for cat, color in (("quaia", "#1f77b4"), ("desi", "#ff7f0e")):
            r = ma[f"{cat}_r"]; xi = ma[f"{cat}_xi"]
            RR = ma[f"{cat}_RR"]
            label = str(ma[f"{cat}_label"])
            elapsed = float(ma[f"{cat}_elapsed_s"])
            good = (RR > 0) & np.isfinite(xi) & (r > 1.0)
            ax.plot(r[good], xi[good], "o-", color=color, lw=1.4, ms=4,
                    label=f"{label} ({elapsed:.1f}s)")
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("r [Mpc/h]")
        ax.set_ylabel(r"$\xi_{\rm LS}^{\rm 3D}(r)$")
        ax.set_title("Cascade Landy-Szalay ξ(r), dyadic shells")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
        ax.set_ylim(-0.05, 0.5)
        ax = axes[1]
        for cat, color in (("quaia", "#1f77b4"), ("desi", "#ff7f0e")):
            r = ma[f"{cat}_r"]; xi = ma[f"{cat}_xi"]
            RR = ma[f"{cat}_RR"]
            label = str(ma[f"{cat}_label"])
            good = (RR > 0) & np.isfinite(xi) & (r > 1.0)
            ax.plot(r[good], r[good] ** 2 * xi[good], "o-",
                    color=color, lw=1.4, ms=4, label=label)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("r [Mpc/h]")
        ax.set_ylabel(r"$r^2\,\xi_{\rm LS}(r)$  [Mpc²/h²]")
        ax.set_title("r²ξ — emphasises BAO / clustering scale")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        figs["morton_xi"] = fig_to_b64(fig)
        morton_t_q = float(ma["quaia_elapsed_s"])
        morton_t_d = float(ma["desi_elapsed_s"])
    else:
        print(f"morton_xi artifact not found at {morton_path}; "
              "skipping cascade panel. Run demos/morton_xi_demo.py first.")
        morton_t_q = morton_t_d = 0.0

    # ---- Metrics --------------------------------------------------
    metrics = dict(
        morton_status=morton_status,
        morton_t_q=morton_t_q, morton_t_d=morton_t_d,
        morton_ang_status=morton_ang_status,
        morton_ang_t_q=morton_ang_t_q,
        morton_ang_t_d=morton_ang_t_d,
        n_z_plot=n_z_plot,
        n_z_q=n_z_q,
        z_plot_summary=", ".join(f"z={z:.2f}" for z in z_plot_mid),
        **jk_metrics,
        n_quaia=quaia["n_d"], n_desi=desi["n_d"],
        n_quaia_raw=q_ra.size, n_desi_raw=d_cat.ra_data.size,
        z_q_edges=", ".join(f"{z:.2f}" for z in quaia["z_q_edges"]),
        z_centres=", ".join(f"{z:.2f}" for z in quaia["z_q_mid"]),
        chi_z=", ".join(f"{c:.0f}" for c in chi_at_z),
        theta_min=np.degrees(quaia["theta"]).min(),
        theta_max=np.degrees(quaia["theta"]).max(),
        n_theta=quaia["theta"].size,
        n_regions=quaia["n_regions"],
        k_max=quaia["k_max"],
        ls_status=("available" if xi_q["xi_ls"] is not None and xi_d["xi_ls"] is not None
                   else "DESI or Quaia RR cube missing"),
        t_total_min=t_total / 60.0,
        timing_breakdown="<br/>".join(
            f"&nbsp;&nbsp;<code>{k}</code>: {v/60:.2f} min "
            f"({100*v/max(t_total,1.0):.1f}%)"
            for k, v in timings.items() if v > 0),
        **cross_metrics,
    )

    html = render_html(figs, metrics)
    with open(HTML, "w") as f:
        f.write(html)
    print(f"\nwrote {HTML} ({len(html)/1024:.0f} KB)")

    # Also stage the HTML for GitHub Pages publishing. The docs/
    # folder on `main` is served at
    # https://yipihey.github.io/graphGP-cosmology/ (Pages picks up
    # docs/index.html as the landing page). Skip with
    # PAPER_PAGES_PUBLISH=0 to keep the docs/ copy frozen.
    if os.environ.get("PAPER_PAGES_PUBLISH", "1") != "0":
        docs_dir = os.path.join(REPO_ROOT, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        docs_html = os.path.join(docs_dir, "index.html")
        with open(docs_html, "w") as f:
            f.write(html)
        # .nojekyll suppresses Jekyll processing (we serve raw HTML).
        nojekyll = os.path.join(docs_dir, ".nojekyll")
        if not os.path.exists(nojekyll):
            open(nojekyll, "w").close()
        print(f"wrote {docs_html} (commit + push docs/ to publish)")

    # Optional: also push the rendered HTML to a Google Drive results
    # folder via tools/drive_upload.sh. Triggered by
    # PAPER_DRIVE_UPLOAD=1 (legacy, superseded by GitHub Pages).
    if os.environ.get("PAPER_DRIVE_UPLOAD") == "1":
        import subprocess
        helper = os.path.join(REPO_ROOT, "tools", "drive_upload.sh")
        try:
            subprocess.run([helper, HTML], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  drive upload failed (exit {e.returncode}); see "
                  f"tools/UPLOAD_README.md for one-time setup.")
        except FileNotFoundError:
            print(f"  drive upload helper missing at {helper}")


CSS = """
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       max-width: 1400px; margin: 24px auto; padding: 0 16px;
       color: #222; line-height: 1.55; }
h1 { font-size: 28px; margin-bottom: 4px; }
h2 { font-size: 22px; margin-top: 26px; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; }
.subtitle { color: #777; margin-bottom: 18px; }
.tabs { display: flex; gap: 4px; border-bottom: 2px solid #ccc;
        margin-bottom: 20px; flex-wrap: wrap; }
.tab { padding: 10px 14px; background: #f4f4f4; border: 1px solid #ccc;
       border-bottom: none; border-radius: 6px 6px 0 0; cursor: pointer;
       font-size: 14px; user-select: none; }
.tab.active { background: white; border-bottom: 2px solid white;
              margin-bottom: -2px; font-weight: 600; }
.panel { display: none; }
.panel.active { display: block; }
img { max-width: 100%; border-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr);
               gap: 8px 24px; background: #f9f9f9; padding: 14px;
               border-radius: 6px; margin: 14px 0; }
.metric-grid div { font-size: 14px; }
.metric-grid b { color: #c0392b; }
code { background: #eef; padding: 1px 6px; border-radius: 3px;
       font-size: 13px; }
.callout { background: #fff8dc; border-left: 4px solid #e6b800;
           padding: 10px 14px; margin: 14px 0; border-radius: 4px; }
"""

JS = """
function showTab(id) {
  document.querySelectorAll('.tab').forEach(t =>
      t.classList.toggle('active', t.dataset.target === id));
  document.querySelectorAll('.panel').forEach(p =>
      p.classList.toggle('active', p.id === id));
}
"""


def render_html(figs, m):
    img = lambda k: f'<img src="data:image/png;base64,{figs[k]}" />' if k in figs else ''
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>DESI Y1 QSO vs Quaia G&lt;20: kNN-CDF side-by-side</title>
<style>{CSS}</style></head>
<body>
<h1>DESI Y1 QSO vs Quaia G&lt;20</h1>
<div class="subtitle">Direct comparison of the joint angular kNN-CDF
on two QSO samples, run on a matched z-shell grid (z=0.8-2.1) and
matched theta grid. Quaia is photo-z-based (~1.2M sources before z
cut, large footprint, unweighted); DESI Y1 QSO is spectroscopic
(~1.2M sources before cut, ~12k deg^2 footprint, per-object
WEIGHT*WEIGHT_FKP). Equations and definitions used throughout follow
the unified four-flavor framework in
<a href="lightcone_native_v4_1.pdf"
   target="_blank">Abel (2026), <i>Survey-Native Clustering
Statistics on the Lightcone</i> (PDF)</a>.</div>

<div class="tabs">
  <div class="tab active" data-target="overview" onclick="showTab('overview')">Overview</div>
  <div class="tab" data-target="note" onclick="showTab('note')">Note (PDF)</div>
  <div class="tab" data-target="catalog" onclick="showTab('catalog')">Catalogs</div>
  <div class="tab" data-target="jackknife" onclick="showTab('jackknife')">Jackknife regions</div>
  <div class="tab" data-target="pthetaz" onclick="showTab('pthetaz')">⟨N⟩(θ,z) heatmaps</div>
  <div class="tab" data-target="n_dd_n_rr" onclick="showTab('n_dd_n_rr')">N_DD / N_RR heatmaps</div>
  <div class="tab" data-target="sigma2" onclick="showTab('sigma2')">σ²(θ;z)</div>
  <div class="tab" data-target="xi_dp" onclick="showTab('xi_dp')">σ²_DP (cap-avg)</div>
  <div class="tab" data-target="xi_ls" onclick="showTab('xi_ls')">σ²_LS (cap-avg)</div>
  <div class="tab" data-target="xi_ls_true" onclick="showTab('xi_ls_true')">ξ_LS (true differential)</div>
  <div class="tab" data-target="cic" onclick="showTab('cic')">CIC PMF</div>
  <div class="tab" data-target="p_n_k" onclick="showTab('p_n_k')">P(N=k) vs θ/r</div>
  <div class="tab" data-target="p_n_k_ratio" onclick="showTab('p_n_k_ratio')">P(N=k) / Poisson(μ_an)</div>
  <div class="tab" data-target="vpf" onclick="showTab('vpf')">VPF (= P(N=0))</div>
  <div class="tab" data-target="cross" onclick="showTab('cross')">Cross ξ (DESI×Quaia)</div>
  <div class="tab" data-target="s3_skew" onclick="showTab('s3_skew')">S₃ skewness</div>
  <div class="tab" data-target="s4_kurt" onclick="showTab('s4_kurt')">S₄ kurtosis</div>
  <div class="tab" data-target="moment_ls" onclick="showTab('moment_ls')">⟨N^p⟩ LS</div>
  <div class="tab" data-target="xi_hamilton" onclick="showTab('xi_hamilton')">ξ Hamilton vs LS</div>
  <div class="tab" data-target="sigma2_compare" onclick="showTab('sigma2_compare')">σ² LS vs raw</div>
  <div class="tab" data-target="dn_dtheta" onclick="showTab('dn_dtheta')">dn/dθ pair count</div>
  <div class="tab" data-target="dlogsigma2_dlogz" onclick="showTab('dlogsigma2_dlogz')">∂lnσ²/∂ln(1+z)</div>
  <div class="tab" data-target="morton_angular" onclick="showTab('morton_angular')">Cascade ξ(θ;z) [angular]</div>
  <div class="tab" data-target="morton_xi" onclick="showTab('morton_xi')">Cascade ξ(r) [3D]</div>
  <div class="tab" data-target="timing" onclick="showTab('timing')">Run times</div>
</div>

<div id="overview" class="panel active">
<h2>Comparison setup</h2>
<p>📄 <b>Companion note (PDF):</b>
<a href="lightcone_native_v4_1.pdf" target="_blank"><i>Survey-Native
Clustering Statistics on the Lightcone: A Unified Four-Flavor
Framework in (θ, z₁, z₂)</i></a> (T. Abel, May 2026) — gives every
equation and definition referenced in the panels below. The same PDF
is embedded in the "Note (PDF)" tab.</p>
<div class="metric-grid">
<div>N Quaia (raw, in z range): <b>{m['n_quaia_raw']:,}</b></div>
<div>N Quaia (DD pass): <b>{m['n_quaia']:,}</b></div>
<div>N DESI (raw, in z range): <b>{m['n_desi_raw']:,}</b></div>
<div>N DESI (DD pass): <b>{m['n_desi']:,}</b></div>
<div>θ range: <b>{m['theta_min']:.2f}-{m['theta_max']:.2f} deg ({m['n_theta']} bins)</b></div>
<div>z shells: <b>{m['z_q_edges']}</b></div>
<div>z centres: <b>{m['z_centres']}</b></div>
<div>χ(z) [Mpc/h]: <b>{m['chi_z']}</b></div>
<div>jackknife regions: <b>{m['n_regions']}</b></div>
<div>k_max: <b>{m['k_max']}</b></div>
<div>LS available: <b>{m['ls_status']}</b></div>
<div>headline N_R/N_D: <b>0.20 (both catalogs)</b></div>
<div>total wall time: <b>{m['t_total_min']:.1f} min</b></div>
</div>

<div class="callout">
<b>Reading the panels.</b> Blue circles + lines = Quaia (G&lt;20,
unweighted, full sky minus galactic plane). Orange squares + lines =
DESI Y1 QSO (NGC + SGC, per-object WEIGHT*WEIGHT_FKP applied). Both
are run with N_R = 0.2 N_D and the sub-pixel-jittered random-catalog
recipe. Per-shell density normalisation makes the curves directly
comparable in amplitude (no N_R-induced offset). The CIC PMF and
P(N=k) panels include thin solid lines = Poisson(μ=⟨N⟩) at the same
per-cap mean count — the offset between the data markers and the
thin line is the clustering signature.
</div>
</div>

<div id="note" class="panel">
<h2>Companion note (PDF)</h2>
<p>The PDF below is shipped alongside the HTML in the repo
(<code>docs/lightcone_native_v4_1.pdf</code>). Every "note v4_1 §X" /
"note v4_1 Eq. X" reference in the other panels links back here.</p>
<p><a href="lightcone_native_v4_1.pdf" target="_blank">Open in a
new tab</a> · <a href="lightcone_native_v4_1.pdf"
download>Download PDF</a></p>
<embed src="lightcone_native_v4_1.pdf" type="application/pdf"
       width="100%" height="900px"
       style="border: 1px solid #ccc; border-radius: 4px;" />
</div>

<div id="catalog" class="panel">
<h2>Sky distributions and n(z)</h2>
<p>Quaia covers most of the sky outside the galactic plane; DESI Y1 QSO
is split between the Northern and Southern Galactic Caps (NGC + SGC),
total ~12,000 deg². n(z) overplotted as densities show DESI peaks
~z=1.5 vs Quaia's broader bimodal distribution (Quaia photo-z is
smoother).</p>
{img('catalog')}
</div>

<div id="jackknife" class="panel">
<h2>Jackknife regions and error bars</h2>
<p><b>All error bars in this presentation</b> — on σ², σ²_LS, ξ_LS,
ξ_DP, the P(N=k) curves, and the VPF — are derived from a
<b>delete-one block jackknife</b> over <b>{m['n_jack']}</b> spatial
regions of the survey footprint. Region assignment uses
<code>twopt_density.jackknife.jackknife_region_labels</code>: each
galaxy gets binned to a HEALPix super-pixel at NSIDE={m['nside_jack']}
(~{m['pix_area_deg2']:.0f} deg² per pixel), and the populated
super-pixels are greedy-assigned to {m['n_jack']} regions in
descending-count order so per-region galaxy counts are roughly
balanced.</p>

<h3>Covariance recipe</h3>
<p>For any per-cap observable <i>O(θ)</i> we form
<i>{m['n_jack']}</i> jackknife samples by leaving one region out at a
time and re-deriving <i>O</i> from the remaining
<i>(N−1)/N</i> data. The standard error is</p>
<pre style="background:#f4f4f4;padding:8px;border-radius:4px;font-size:13px;">
  σ²(O) = ((N − 1) / N) · Σ_k (O_k − ⟨O⟩)²
        = ((N − 1)) · Var_k(O_k)
</pre>
<p>This is the textbook block-jackknife scaling — captures
<b>cosmic variance</b> from the survey's spatial structure (the
dominant off-diagonal contribution at our scales), which the
diagonal Poisson estimate misses. Per-region cubes
(<code>sum_n_per_region</code>, <code>H_geq_k_per_region</code>) are
populated by <code>joint_knn_cdf</code> in a single pass alongside
the global cube — no extra DD/RD/RR work is needed for the
jackknife.</p>

<h3>What you should know</h3>
<ul>
<li><b>Quaia</b> ({m['n_quaia']:,} objects): regions span
    {m['q_count_min']:,}–{m['q_count_max']:,} galaxies (mean
    {m['q_count_mean']:.0f}, fractional spread σ/μ
    ≈ {m['q_count_std']/m['q_count_mean']:.2f}). Total area
    ≈ {m['q_area_total']:.0f} deg² across the populated
    NSIDE={m['nside_jack']} super-pixels.</li>
<li><b>DESI Y1 QSO</b> ({m['n_desi']:,} objects):
    {m['d_count_min']:,}–{m['d_count_max']:,} galaxies per region
    (mean {m['d_count_mean']:.0f}, σ/μ
    ≈ {m['d_count_std']/m['d_count_mean']:.2f}). Total area
    ≈ {m['d_area_total']:.0f} deg² (NGC + SGC).</li>
<li><b>Region areas vary</b> because the populated super-pixels are
    not uniform across the footprint — DESI's NGC + SGC produces a
    bimodal area distribution; Quaia's near-full-sky coverage
    produces tighter balance. The greedy-by-count assignment
    rebalances <i>by galaxy count</i>, not by area, which is the
    right thing for jackknife of a clustering observable
    (variance scales with N_objects, not with area).</li>
<li><b>Per-region density</b> is roughly uniform within each catalog
    (greedy balance does most of the work) so each leave-one-out
    sample drops a comparable fraction of the data and contributes
    similarly to the variance estimate. Highly imbalanced regions
    would inflate the leave-one-out spread and bias σ(O) high.</li>
<li><b>Caveat (low-mode bias)</b>: a {m['n_jack']}-region jackknife
    underestimates uncertainty on modes whose wavelength approaches
    the survey size, because no single dropped region captures
    those modes' full variance. Adequate at the θ ≲ 1°
    scales we focus on; for θ ≳ 5° expect the true error to be
    larger than the jackknife estimate by ~10-20%.</li>
<li>Jackknife covariance has been verified to obey the consistency
    invariant <i>per-region sums = global cube</i> for both
    H_geq_k_per_region and sum_n_per_region, on both numba and
    cascade backends. See
    <code>tests/test_knn_cdf_cascade_equivalence.py::test_jackknife</code>.</li>
</ul>

{img('jackknife')}
</div>

<div id="pthetaz" class="panel">
<h2>⟨N⟩(θ, z) on the diagonal — data vs volume-filling caps</h2>
<p>Two-dimensional heatmaps of the per-cap mean neighbour count
<i>⟨N⟩(θ, z<sub>q</sub>=z<sub>n</sub>)</i> for each catalog under
both query distributions:</p>
<ul>
<li><b>DD (data-centred caps)</b> — caps placed on the data
    points themselves. Sees the local clustering boost: high-density
    regions get more queries, dense neighbourhoods.</li>
<li><b>RD (volume-filling caps)</b> — caps placed on random points
    drawn from the survey selection function (the Quaia sel-map and
    DESI's per-PHOTSYS-region completeness). This is the
    <b>unbiased volume-average</b> reference: each unit of survey
    volume is sampled in proportion to its observability.</li>
</ul>

<h3>Construction</h3>
<p>For each query point at redshift z<sub>q</sub>, count neighbours
in the same z-shell within angular cap radius θ; average over all
queries in that shell. This is the elementary moment
<code>mean_count(res) = sum_n / N_q</code> on the diagonal slice
z<sub>q</sub>=z<sub>n</sub>. The full off-diagonal cube
<code>⟨N⟩[t, iq, jn]</code> for jn≠iq is small (cross-shell
correlations are short-range in z) and not shown here.</p>

<h3>Reading the panels</h3>
<p>Color = log<sub>10</sub>⟨N⟩ on a shared scale across the four
DD/RD panels (left two columns). The third column shows
<i>log<sub>10</sub>(⟨N⟩<sub>DD</sub> / ⟨N⟩<sub>RD</sub>)</i> on a
diverging RdBu scale: <b>red</b> = excess on data caps relative to
random caps (clustering boost), <b>blue</b> = deficit, <b>white</b>
= no excess. The systematic positive shift at small θ in the ratio
panels is exactly the data-cap selection bias that
<code>σ²_LS = (DD−2DR+RR)/RR</code> subtracts off — and the same
clustering signal that
<code>VPF − exp(−⟨N⟩)</code> captures on the count-distribution
side. At large θ both ⟨N⟩s converge (red→white) because the cap
becomes large enough that the DD vs RD difference washes out.</p>

<h3>Why this is the master observable</h3>
<p>Every angular kNN-CDF observable in this presentation is a
function of either ⟨N⟩(θ, z<sub>q</sub>, z<sub>n</sub>) or its
moments:</p>
<ul>
<li>σ²_clust ← Var(N)/⟨N⟩² − 1/⟨N⟩</li>
<li>σ²_LS, ξ_LS ← (μ_DD − 2μ_DR + μ_RR) / μ_RR</li>
<li>VPF, P(N=k) ← Banerjee-Abel ladder, anchored on the H_geq_k
    cube whose first moment is ⟨N⟩.</li>
</ul>
<p>Reading the heatmap directly tells you the per-cap occupancy
regime — whether you're in the empty-cap regime (⟨N⟩ ≪ 1) where
VPF dominates the information, the unit-occupancy regime
(⟨N⟩ ~ 1) where the full PMF carries non-Gaussian information, or
the dense regime (⟨N⟩ ≫ 1) where σ²_LS converges to the
cap-averaged 2-point function.</p>
{img('pthetaz')}
</div>

<div id="n_dd_n_rr" class="panel">
<h2>Pair-count heatmaps: N_DD, N_RR, and Peebles ratio</h2>
<p>The most direct view of clustering: per-cap mean neighbour count
on the DD pass (data queries with data neighbours) and the RR pass
(random queries with random neighbours), each normalised by the
catalog's per-shell count to give a per-pair quantity. The third
column shows the Peebles 1+ξ-like ratio</p>
<pre style="background:#f4f4f4;padding:8px;border-radius:4px;font-size:13px;">
  R(θ; z) = [⟨N⟩_DD(θ; z) / N_data(z)] / [⟨N⟩_RR(θ; z) / N_random(z)]
          ≈ 1 + ξ_cap(θ; z)
</pre>
<p>For DESI we use the per-shell <b>weight sum</b> (Σ
WEIGHT·WEIGHT_FKP) for N_data, matching the weighted DD pass.
The ratio is shown on a log10 scale clipped to ±0.1; brighter red
⇒ stronger clustering excess. Compare with the σ²_LS heatmap (which
shows the same physical signal after the LS combination removes the
DR cross-bias).</p>
<p><b>What to look for.</b> The DD/RR panels span 4+ decades from
near-empty (purple) to dense (yellow); the diagonal stripes track
n(z) — bins with more data have higher ⟨N⟩ at any given θ. The
ratio panel is near 1 (white) at large θ, fades to red at small θ
where the clustering boost dominates. DESI's red region extends to
larger θ than Quaia's because DESI is a denser, more strongly
clustered tracer.</p>
{img('n_dd_n_rr')}
</div>

<div id="sigma2" class="panel">
<h2>Per-shell clustering variance σ²_clust(θ; z)</h2>
<p>σ²_clust(θ) ≡ Var(N)/⟨N⟩² − 1/⟨N⟩ from a single cube — <b>no random
catalog enters this estimator at all</b>. Solid markers = DD-centred
(cap centres on data); open + dashed = RD-centred (cap centres on
randoms). The DD-RD difference is the data-vs-random selection bias
on the <i>cap centres</i>, not a normalisation problem.</p>
<p>The thin solid lines are <b>σ² for an unbiased Poisson source
population sampling the same window function W(Ω) and same set of
cap centres — derived purely from the completeness map, no random
catalog instantiated</b>. Closed form:
σ²<sub>Poisson(W)</sub>(θ) = Var<sub>q</sub>(A<sub>w</sub>(q,θ)) /
⟨A<sub>w</sub>(q,θ)⟩², where A<sub>w</sub>(q,θ) = ∫<sub>cap(q,θ)</sub>
W(Ω) dΩ is the windowed cap area (sum of HEALPix completeness over
pixels in the disc). Both the Poisson noise (1/⟨N⟩) and the source
density (λ) cancel exactly; what's left is the relative variance of
the local windowed area at the cap centres. The DD/RD measurements
sitting <i>above</i> the Poisson(W) line is the genuine clustering
excess.</p>
<p>The diamond-marker dash-dot curves are the <b>Landy–Szalay
pair-counting estimator of σ²</b>:
σ²<sub>LS</sub>(θ) = (DD̂ − 2 DR̂ + RR̂) / RR̂, with hat-quantities the
cap-integrated pair counts on the standard per-pair normalisation
(DD̂ = DD/[N<sub>d</sub>(N<sub>d</sub>−1)/2] etc). The cap-integrated
LS estimator is mathematically identical to the cap-averaged 2-point
function ⟨ξ⟩<sub>cap</sub>(θ); in our cube formalism it is the
diagonal slice z<sub>q</sub>=z<sub>n</sub> of <code>xi_ls</code>.
Unlike σ²<sub>DD</sub> and σ²<sub>RD</sub> (moment-based, biased by
data-vs-random selection of the cap centres), σ²<sub>LS</sub> →
0 at large θ where ξ → 0 by construction. The gap between
σ²<sub>DD</sub> and σ²<sub>LS</sub> at large θ is the data-centred
cap-selection bias of the moment estimator — visible as ~0.05 in
Quaia and ~0.10 in DESI on this geometry.</p>
<p>The plateau in σ²<sub>DD</sub> ≈ 0.05 (Quaia) and ≈ 0.10–0.13
(DESI) at large θ is the data-centred cap-selection inflation —
σ²<sub>LS</sub> shows that the genuine cap-averaged 2-point signal
decays cleanly with θ. The thin Poisson(W) line shows how much of the
small-θ noise comes purely from the window function.</p>
{img('sigma2')}
{('<h3>Veusz publication-quality (clickable to edit)</h3>'
  '<p>Click the panel below to open <code>vsz/sigma2.vsz</code> '
  'in Veusz; save your edits, then POST <code>/rebuild</code> to '
  'the local helper (see <code>tools/VEUSZ_README.md</code>). '
  'Style edits propagate via <code>tools/propagate_vsz_edits.py</code> '
  'and append to <code>vsz/STYLE_LOG.md</code>.</p>'
  '<a href="http://localhost:8765/open?vsz=vsz/sigma2.vsz">'
  '<object data="sigma2.svg" type="image/svg+xml" width="100%">'
  '(Veusz SVG not yet generated; rebuild with PAPER_USE_VEUSZ=1)'
  '</object></a>'
  ) if figs.get('sigma2_vsz') else ''}
<h3>Heatmap (full z range, log color)</h3>
<p>Shows σ²_clust(θ; z) across all {m['n_z_q']} z-shells for both
catalogs and both query types. Brighter ⇒ stronger clustering signal.
The x-axis is log-θ; y-axis is z (log(1+z) bins). The color scale
clips to the 5–95th percentile of joint positive values for
contrast.</p>
{img('sigma2_heat') if 'sigma2_heat' in figs else '<i>(not available)</i>'}
</div>

<div id="xi_dp" class="panel">
<h2>σ²_DP(θ;z) = nbar_DD / nbar_RD − 1  (cap-averaged Davis-Peebles)</h2>
<p>The Davis–Peebles estimator applied to <b>cap-cumulative</b> per-query
neighbor counts (<code>nbar_DD = sum_n_DD/N_q</code>). Because both
<code>nbar_DD</code> and <code>nbar_RD</code> are integrated over the
cap, the ratio gives the cap-averaged 2-point function
σ²<sub>DP</sub>(θ) ≡ ⟨ξ⟩<sup>DP</sup><sub>cap</sub>(θ), <b>not</b>
the differential ξ(θ). For the true differential ξ(θ) see the
"ξ_LS (true differential)" tab.</p>
{img('xi_dp')}
<h3>Heatmap (full z range, divergent color centred on 0)</h3>
{img('xi_dp_heat') if 'xi_dp_heat' in figs else '<i>(not available)</i>'}
</div>

<div id="xi_ls" class="panel">
<h2>σ²_LS(θ;z) = (DD − 2DR + RR) / RR  (cap-averaged Landy-Szalay)</h2>
<p>Landy–Szalay applied to <b>cap-cumulative</b> per-query neighbor
counts. Mathematically equivalent to the cap-averaged 2-point function
σ²<sub>LS</sub>(θ) ≡ ⟨ξ⟩<sup>LS</sup><sub>cap</sub>(θ) = (1/V<sub>cap</sub>²)
∫∫ ξ(α) dΩ dΩ' over the disc of radius θ. Density-normalised LS:
each per-query nbar is divided by its neighbor catalog's per-shell
count (or sum-of-weights, for FKP-weighted DESI), putting DD/DR/RR
on a common V<sub>cap</sub> scale and removing the N_R/N_D bias of
naive LS. For the true differential ξ(θ) per angular separation,
see the next tab.</p>
<p><b>Markers + errorbars:</b> MC-RR (random catalog instantiated
from <code>random_queries_from_selection_function</code>).
<b>Thin solid (Quaia) and thin dotted (DESI) lines:</b> the
<i>same</i> LS estimator computed with a <b>window-corrected
analytic RR</b>: the per-cap expected random count is integrated
exactly over the HEALPix completeness map at a 5000-cap subsample
of cap centres (no MC random catalog instantiated). This drops the
random-catalog shot-noise contribution from the LS combination —
visible as a smoother curve coinciding with the MC-RR markers
within their errorbars. Implementation:
<code>twopt_density.knn_analytic_rr.analytic_rr_cube(...,
query_ra_deg=..., query_dec_deg=...)</code>.</p>
{img('xi_ls') if 'xi_ls' in figs else '<i>RR cubes not yet available; run desi_full_pipeline.py and the Quaia rerun.</i>'}
<h3>Heatmap (full z range, divergent color centred on 0)</h3>
{img('xi_ls_heat') if 'xi_ls_heat' in figs else '<i>(not available)</i>'}
</div>

<div id="xi_ls_true" class="panel">
<h2>True differential ξ_LS(θ;z) from per-annulus pair counts</h2>
<p>The angular two-point correlation function ξ(θ) at <b>fixed
angular separation</b> θ — the LS estimator applied to <b>per-annulus</b>
pair counts (pairs in (θ<sub>t-1</sub>, θ<sub>t</sub>], obtained by
differencing the cumulative cap-count cube along the θ axis).
Compare to the σ²_LS tab: σ²_LS is the cap-volume average of ξ over
[0, θ], whereas ξ_LS here is the differential value at separation θ.
Relationship: σ²(θ) = (1/V<sub>cap</sub>²) ∫<sub>0</sub><sup>θ</sup>
ξ(α) · 2π sin α · K(α; θ) dα with K an O(1) geometric kernel.</p>
<p><b>Coarsened θ grid</b> (~11 log-spaced annular bins from the
underlying 90-bin cumulative cube). Per-annulus pair counts are
sparse on the original fine grid, so we group ~9 fine bins per
coarse annulus to get usable signal-to-noise per bin. The
cap-cumulative σ²_LS uses all 90 bins because cap-integrated counts
are smooth; the differential estimator needs the broader binning
because <code>np.diff</code> amplifies bin-to-bin noise. Symlog
y-scale (linthresh=10⁻³) shows the small/zero-crossing regime at
large θ alongside the rising clustering signal at small θ.</p>
{img('xi_ls_true') if 'xi_ls_true' in figs else '<i>not available without RR cubes.</i>'}
</div>

<div id="cic" class="panel">
<h2>CIC PMF P_{{N=k}} per shell</h2>
<p>One row per catalog (Quaia top, DESI bottom), one column per
representative redshift shell ({m['n_z_plot']} of {m['n_z_q']} shells
shown — quartile midpoints). Two θ values overlaid per panel.</p>
{img('cic')}
</div>

<div id="p_n_k" class="panel">
<h2>P(N = k cap counts) on the diagonal z<sub>q</sub> = z<sub>n</sub></h2>
<p>Fixed-k slices of the CIC PMF on the (z<sub>q</sub>=z<sub>n</sub>)
diagonal, in <b>4 rows</b>: row 1 Quaia vs θ, row 2 DESI vs θ, row 3
Quaia vs r, row 4 DESI vs r (r = χ(z)·θ at fiducial Ω<sub>m</sub>=0.31,
h=0.68). Circles = Quaia, squares = DESI; solid markers = DD (data
centres on data), dashed markers = RD (random centres on data).
Thin solid line = Poisson(μ=⟨N⟩(θ)) for that catalog.</p>
<p><b>k=0 is the void probability function</b> (VPF) =
P[N=0|θ;z] = 1 − 1NN-CDF — the probability that a cap of radius θ
contains no data points. For Poisson it is exp(−⟨N⟩(θ)) and
appears as the deep-purple curve at the top of each panel: at small
θ the cap is small enough to be empty most of the time (P→1); at
large θ the cap always contains data (P→0). Departure of the data
markers from the thin Poisson line at k=0 measures the clustering
suppression of empty caps (clustered fields make small caps emptier,
large caps fuller, vs Poisson). For k>0 the offset between markers
and Poisson is the Banerjee-Abel CIC clustering signature. Only k
strictly below <code>k_max</code> (the cumulative-tail slot) are
plotted.</p>
{img('p_n_k')}
</div>

<div id="p_n_k_ratio" class="panel">
<h2>P(N=k) / Poisson(μ<sub>an</sub>) — clustering excess in the count distribution</h2>
<p>Direct generalisation of the VPF log-ratio panel (k=0) to all k.
For each (θ, z) we compare the empirical PMF against the
analytic-random Poisson reference
<i>P<sub>ref</sub>(N=k) = e<sup>−μ<sub>an</sub></sup>·μ<sub>an</sub><sup>k</sup>/k!</i>
where <i>μ<sub>an</sub>(θ;z) = N<sub>data</sub>·⟨A<sub>w</sub>(θ)⟩/Ω<sub>mask</sub></i>
is the window-corrected, shot-noise-free analytic mean cap count.
Two flavours per panel:</p>
<ul>
  <li><b>DD (solid markers, μ<sub>DD</sub>):</b>
      ⟨A<sub>w</sub>⟩ averaged over <i>data</i> positions.
      The Poisson reference is what a Poisson realisation of the
      same density on the same window would give for caps centred
      on the data points.</li>
  <li><b>RD (dashed markers, μ<sub>RD</sub>):</b>
      ⟨A<sub>w</sub>⟩ averaged over <i>random</i> positions
      drawn from the selection function. The Poisson reference is
      the no-clustering baseline for caps placed uniformly on the
      window — the natural reference for the survey-averaged
      count PMF.</li>
</ul>
<p>Curves above zero ⇒ data has more caps with that count than
Poisson; below zero ⇒ deficit. Clustered tracers typically show
positive offsets at <b>both ends</b> of k (more empty caps + more
overdense caps) and a deficit near the mean — the two-sided
excess is the same higher-cumulant signature −log(VPF) =
Σ<sub>p≥1</sub>⟨N<sup>p</sup>⟩<sub>c</sub>/p! probes for k=0.</p>
<p>The PMF floor 10<sup>−3</sup> masks numerically unreliable
points where the empirical PMF is dominated by Poisson sampling
noise on the kNN-CDF cube.</p>
{img('p_n_k_ratio')}
</div>

<div id="vpf" class="panel">
<h2>Void Probability Function — P(N=0|θ;z)</h2>
<p>The VPF is the probability that a cap of radius θ contains zero
data points: VPF(θ) = 1 − P(N≥1|θ) = 1 − 1NN-CDF complement.
Among the kNN-CDF derived statistics it is the <b>most window-robust</b>
because it depends only on the local density distribution at the
cap centre (not on long-range pair correlations across the
footprint). For a Poisson source population sampling the same
window VPF<sub>Poisson</sub>(θ) = exp(−⟨N⟩(θ)), where
⟨N⟩(θ) is the mean cap count.</p>
<p><b>Top row</b>: VPF(θ) for both catalogs (markers) with their
Poisson references (Quaia thin solid, DESI thin dotted), per
redshift shell on the diagonal z<sub>q</sub>=z<sub>n</sub>.</p>
<p><b>Bottom row</b>: log<sub>10</sub>(VPF / VPF<sub>Poisson</sub>) —
the clustering excess of empty caps. Positive values mean the data
field has <b>more</b> empty caps than Poisson at the same mean
density (the signature of clustered tracers, which leave large
voids while concentrating in overdense regions). The full sum
−log(VPF) = Σ<sub>p≥1</sub> ⟨N<sup>p</sup>⟩<sub>c</sub>/p! is the
cumulant-generating-function expansion: deviations from
Poisson directly probe higher-order moments
(skewness, kurtosis…) of the density field beyond the variance
captured by σ²_LS. Complementary to ξ_LS for non-Gaussian regimes.</p>
{img('vpf')}
</div>

<div id="cross" class="panel">
<h2>Cross-correlation: DESI Y1 QSO × Quaia G&lt;20</h2>
<p>Joint analysis using the same kNN-CDF primitive. Four cross passes
(<code>DD_xy, DR_xy, RD_xy, RR_xy</code>) with DESI as <i>x</i> and
Quaia as <i>y</i>. The asymmetric Landy-Szalay cross estimator is
ξ<sub>LS</sub><sup>xy</sup> = (μ<sub>DD</sub> − μ<sub>DR</sub> − μ<sub>RD</sub> + μ<sub>RR</sub>) / μ<sub>RR</sub>,
with each μ normalised by its respective neighbor catalog's per-shell
count to keep the four terms on a common per-cap-volume scale.
The Davis-Peebles cross 1+ξ = nbar<sub>DD</sub> / (nbar<sub>DR</sub>·N<sub>D<sub>y</sub></sub>/N<sub>R<sub>y</sub></sub>)
is also shown. Cross runs without jackknife (single pass each), so
the green/purple cross curves have no error bars; the auto LS curves
retain their jackknife errors for comparison.</p>
<div class="metric-grid">
<div>cross status: <b>{m['cross_status']}</b></div>
{f"<div>N_DESI_data: <b>{m['n_dx']:,}</b></div>" if m['cross_status']=='available' else ""}
{f"<div>N_Quaia_data (z-cut): <b>{m['n_dy']:,}</b></div>" if m['cross_status']=='available' else ""}
{f"<div>N_DESI_random: <b>{m['n_rx']:,}</b></div>" if m['cross_status']=='available' else ""}
{f"<div>N_Quaia_random: <b>{m['n_ry']:,}</b></div>" if m['cross_status']=='available' else ""}
</div>
<h3>Diagonal (z<sub>q</sub> = z<sub>n</sub>): cross vs auto</h3>
{img('cross_diag') if 'cross_diag' in figs else '<i>cross diag not yet rendered</i>'}
<h3>Full (z<sub>q</sub>, z<sub>n</sub>) grid: only the diagonal should be non-zero</h3>
<p>Off-diagonal panels test that DESI quasars at z<sub>q</sub> are
uncorrelated with Quaia quasars at a substantially different
z<sub>n</sub>. A clean null on the off-diagonal (with the diagonal
showing a clear positive ξ) is the signature of a working cross
estimator.</p>
{img('cross_grid') if 'cross_grid' in figs else '<i>cross grid not yet rendered</i>'}
</div>

<div id="morton_angular" class="panel">
<h2>Angular ξ<sub>LS</sub>(θ;z) via the morton_cascade tree</h2>
<p>Same Rust cascade tree as the 3D ξ(r) panel, but applied to the
celestial-sphere kNN search. <b>Mapping:</b> each catalog point
<code>(RA, Dec)</code> becomes a unit vector on S² (3D Cartesian on the
sphere, no radial direction). The cascade then organises the search
by <b>chord length</b> <i>d = 2 sin(θ/2)</i>; in the small-angle
limit chord ≈ θ (error &lt;0.07% at 12°). Each z-shell is one
cascade pass; the Morton-ordered dyadic hierarchy gives DD/RR/DR
pair counts and ξ<sub>LS</sub> at <i>every</i> dyadic chord scale
in a single O(N log N) traversal — replacing the per-query
<code>healpy.query_disc</code> + linear-pair-scan in
<code>joint_knn_cdf</code> with a single tree walk per shell.</p>
<p><b>Status:</b> {m['morton_ang_status']}.
{f"Quaia ({m['n_quaia']:,} data) all 4 z-shells in <b>{m['morton_ang_t_q']:.1f} s</b>; DESI Y1 QSO ({m['n_desi']:,} data, weighted) in <b>{m['morton_ang_t_d']:.1f} s</b>. Compare with the kNN-CDF angular pipeline at ~13 min for the same observable." if m['morton_ang_status']=='available' else "Run <code>demos/morton_angular_demo.py</code> to populate this panel."}</p>
<p><b>Reading the panels.</b> Bold markers = differential
ξ<sub>LS</sub> at each dyadic chord shell from the cascade. Thin
lines = cumulative ⟨ξ⟩<sub>cap</sub> = σ²<sub>LS</sub> from the
kNN cube (existing pipeline). The two are <i>different
smoothing windows</i> on the same underlying ξ(θ): the cascade
uses a chord-shell window [d_inner, d_outer], the kNN uses a cap
[0, θ]. They should agree in sign and approximate scale, with
the kNN curve smoother because it integrates over all radii up
to θ.</p>
<p><b>What the cascade buys you.</b> The full angular ξ<sub>LS</sub>(θ;z)
in 4.4 s wall time (4 z-shells × 2 catalogs) on a single thread —
~150× faster than the angular kNN-CDF pipeline at the cost of
fixed dyadic shells (no custom θ grid) and no per-query
distribution (no H_geq_k, so no VPF / kNN-CDF ladder). Use the
cascade tree for fast moment statistics and likelihood evaluation
in iterative inference; use the kNN-CDF cube when you need the
full count distribution.</p>
{img('morton_angular') if 'morton_angular' in figs else '<i>angular cascade panel not rendered — run demos/morton_angular_demo.py first</i>'}
</div>

<div id="morton_xi" class="panel">
<h2>3D Cartesian ξ(r) via the morton_cascade tree</h2>
<p>Native 3D Landy-Szalay correlation function via the
<code>morton_cascade</code> Rust crate (bit-vector pair cascade,
<code>hier_bitvec_pair</code>). Catalog (RA, Dec, z) is converted to
comoving Cartesian (Mpc/h) on a fiducial cosmology
(Ω<sub>m</sub>=0.31, h=0.68) and embedded in a non-periodic [0, L)³
box; the cascade builds one Morton-ordered dyadic hierarchy and
emits DD/RR/DR pair counts plus ξ<sub>LS</sub>(r) at every dyadic
scale in a single O(N log N) pass. No per-r binning loop, no
mesh — empty regions cost nothing.</p>
<p><b>Status:</b> {m['morton_status']}.
{f"Quaia ({m['n_quaia']:,} data + 0.2× randoms) cascaded in <b>{m['morton_t_q']:.1f} s</b>; DESI Y1 QSO ({m['n_desi']:,} data, weighted) in <b>{m['morton_t_d']:.1f} s</b>." if m['morton_status']=='available' else "Run <code>demos/morton_xi_demo.py</code> to populate this panel."}</p>
<p><b>Left panel:</b> ξ<sub>LS</sub>(r) on a log r-axis. Both
catalogs converge to ξ→0 at r ≳ 200 Mpc/h and rise steeply at small
r — the standard clustering signal of biased quasar tracers.
<b>Right panel:</b> r²ξ — emphasises the BAO and large-scale clustering
features. Positive bumps at r ~ 1000-2000 Mpc/h are partly survey
window imprint (the dyadic shells at L/4, L/8 see the box edges);
real clustering signatures dominate at r ≲ 200 Mpc/h.</p>
<p><b>Why a different observable than σ²<sub>LS</sub>(θ)?</b>
The angular σ²<sub>LS</sub> integrates over the survey's redshift
extent at a fixed angular cap; this 3D ξ(r) projects onto comoving
distance with full radial information. The two are related by
r↔θ·χ̄ at the central redshift but probe different smoothing
windows; running both is a self-consistency check on the
clustering pipeline.</p>
{img('morton_xi') if 'morton_xi' in figs else '<i>cascade panel not rendered — run demos/morton_xi_demo.py first</i>'}
</div>

<div id="s3_skew" class="panel">
<h2>S₃ skewness — third standardised moment of N(θ; z)</h2>
<p>Per-shell skewness ``S₃ = ⟨(N − ⟨N⟩)³⟩ / σ³`` of the per-cap
counts, derived from the third raw moment ``⟨N³⟩`` accumulated by
the kNN-CDF kernel (<a href="lightcone_native_v4_1.pdf"
target="_blank">note v4_1 §6</a>). For a Gaussian or Poisson field
``S₃ → 0``; positive ``S₃`` indicates a heavier upper tail
(clusters). The DESI sample shows characteristically larger
skewness at small angular scales than Quaia (deeper, sparser
sample → stronger small-scale clustering) and both samples shift
toward zero at large θ where the cap encloses many independent
structures.</p>
{img('s3_skew')}
<h3>Heatmap (full z range)</h3>
{img('s3_skew_heat')}
</div>

<div id="s4_kurt" class="panel">
<h2>S₄ kurtosis — fourth standardised moment</h2>
<p>Excess kurtosis ``S₄ − 3 = ⟨(N − ⟨N⟩)⁴⟩ / σ⁴ − 3`` per shell.
For a Gaussian field this is zero; for a Poisson field with mean μ
it equals ``1/μ``. Positive excess kurtosis at small angular scales
is a signature of rare, high-density structures dominating the
upper tail of the count distribution. The kurtosis is the most
sensitive of the lower-order moments to non-Gaussian features and
is a direct probe of perturbative bias parameters.</p>
{img('s4_kurt')}
<h3>Heatmap (full z range)</h3>
{img('s4_kurt_heat')}
</div>

<div id="moment_ls" class="panel">
<h2>⟨N^p⟩^LS — Landy–Szalay-corrected raw moments p=1,2,3,4</h2>
<p><a href="lightcone_native_v4_1.pdf" target="_blank">Note v4_1 Eq.
13</a> generalises the Landy–Szalay estimator from the
two-point function to higher moments by applying the same
<code>(DD − DR − RD + RR)/RR</code> combination per moment order.
This isolates the connected (clustering-driven) contribution from
each moment order while removing the geometric/selection-function
bias that affects each catalog. <b>p=1</b> is the connected pair
density (numerically equivalent to <i>1+ξ_LS</i> up to per-cap
normalisation — the "two-point clustering" trace); <b>p=2</b>
measures the clustering variance; <b>p=3</b> and <b>p=4</b> are
direct probes of the connected three- and four-point structure of
the field. Requires the DR pass (run pipelines with
<code>PAPER_DR_PASS=1</code>).</p>
{img('moment_ls')}
</div>

<div id="xi_hamilton" class="panel">
<h2>Hamilton ξ vs Landy–Szalay ξ</h2>
<p>Hamilton's estimator (<a href="lightcone_native_v4_1.pdf"
target="_blank">note v4_1 Eq. 15</a>)::

  ξ^Ham = (DD · RR) / (DR · RD) − 1

is multiplicative rather than additive, so it cancels overall
normalisation factors. It is less sensitive than LS to fluctuations
when RR is small (large angular scales) — useful as a cross-check
on the LS estimator at the boundary of the well-sampled regime.
The two estimators should agree to within their joint statistical
uncertainty; persistent disagreement at large θ indicates LS
instability rather than a genuine signal. Requires the DR pass.</p>
{img('xi_hamilton')}
</div>

<div id="sigma2_compare" class="panel">
<h2>σ²_clust: raw DD vs Landy–Szalay-corrected
(<a href="lightcone_native_v4_1.pdf" target="_blank">note v4_1
Eq. 14</a>)</h2>
<p>``σ²_clust^LS = Var^LS / ⟨N¹⟩^LS² − 1/⟨N¹⟩^LS`` uses the LS
combination of the second moment
(<a href="lightcone_native_v4_1.pdf" target="_blank">Eq. 13</a> with
p=2). Compared to the
raw-DD form (which has no random subtraction), the LS form removes
selection-function bias to first order. The difference between the
two curves at large θ traces the magnitude of the systematic that
the LS combination removes; at small θ both should agree because
the DD signal dominates. Requires DR.</p>
{img('sigma2_compare')}
</div>

<div id="dn_dtheta" class="panel">
<h2>Differential pair-count density θ·∂⟨N⟩/∂θ
(<a href="lightcone_native_v4_1.pdf" target="_blank">note v4_1
Eq. 9</a>)</h2>
<p>The per-cap mean count ``⟨N⟩(θ)`` is the cumulative pair count to
separation θ. The differential pair count
``dn_pairs/dθ ∝ θ · ∂⟨N⟩/∂θ`` traces the underlying ξ(θ) more
directly than the cap-cumulative form: features in ξ become
features in the differential. Compare to the
"ξ_LS (true differential)" tab — the differential pair count is a
direct measurement; LS is its bias-corrected analog.</p>
{img('dn_dtheta')}
</div>

<div id="dlogsigma2_dlogz" class="panel">
<h2>Logarithmic redshift derivative ∂lnσ² / ∂ln(1+z)
(<a href="lightcone_native_v4_1.pdf" target="_blank">note v4_1 §4</a>)</h2>
<p>From the diagonal σ²(θ; z) cube, the redshift log-derivative at
each angular scale is computed by central finite differences along
the z axis. <a href="lightcone_native_v4_1.pdf" target="_blank">Note
Eq. 20</a> gives the linear-bias decomposition::

  ∂lnσ² / ∂ln(1+z) = 2·p_bias  +  2·(1+z)·dlnD/dz  +  n_eff(θ·D_M)·dlnD_M/dln(1+z)

The three terms are individually constrainable: the bias-evolution
term is constant in (log θ, log(1+z)); the growth term is smooth
and monotonic; the geometry term carries the BAO standard ruler as
a localised feature near θ·D_M = r_BAO that traces a curve through
the (log θ, log(1+z)) plane. The diagnostic here is purely
empirical (no model overlay yet) — a smooth monotonic trend
indicates the data are consistent with the linear-bias
decomposition; sharp features locate the BAO scale.</p>
{img('dlogsigma2_dlogz')}
</div>

<div id="timing" class="panel">
<h2>Wall time per pass</h2>
<p>Time consumed by each <code>joint_knn_cdf</code> pass in the most
recent run (single machine, 8 numba threads). The DD passes (DESI &
Quaia auto plus the cross DD<sub>xy</sub>) dominate because both
catalog axes scale with N<sub>data</sub>; the random-side passes
scale with N<sub>R</sub> = 0.2·N<sub>D</sub> per the convergence
sweep, so they're an order of magnitude cheaper. Total compute
budget for the present analysis ≈ <b>{m['t_total_min']:.1f}
min</b>.</p>
{img('timing')}
<h3>Breakdown</h3>
<p>{m['timing_breakdown']}</p>
</div>

<script>{JS}</script>
</body></html>
"""


if __name__ == "__main__":
    main()
