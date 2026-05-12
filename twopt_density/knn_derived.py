"""Derived statistics from the joint angular kNN-CDF primitive.

Every clustering observable in ``lightcone_native_v3.pdf`` is a thin
reduction over the ``KnnCdfResult`` cube produced by
``twopt_density.knn_cdf.joint_knn_cdf``. This module collects those
reductions in one place; none of the functions here touches the catalog
directly.

References to paper equations are to ``lightcone_native_v3.pdf`` (T.
Abel, May 2026).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .knn_cdf import KnnCdfResult


# ----------------------------------------------------------------------
# Single-result reductions (DD- or RD-flavor consumed alone)
# ----------------------------------------------------------------------


def mean_count(result: KnnCdfResult) -> np.ndarray:
    """``nbar(theta; z_q, z_n) = sum_n / N_q``  (Eq. 3).

    Returns shape ``(n_theta, n_z_q, n_z_n)`` for full cubes, or
    ``(n_theta, n_z)`` for diagonal-only cubes. Where ``N_q == 0``
    the corresponding query shell yields zero (no information).
    """
    Nq = result.N_q.astype(np.float64)
    safe = np.where(Nq > 0, Nq, np.inf)
    if getattr(result, "is_diagonal", False):
        # sum_n shape (n_theta, n_z); broadcast Nq across theta.
        return result.sum_n / safe[None, :]
    return result.sum_n / safe[None, :, None]


def cic_pmf(result: KnnCdfResult) -> np.ndarray:
    """Counts-in-cells PMF ``P_{N=k}`` via the Banerjee-Abel relation
    (Eq. 4): ``P_{N=k} = P_{>=k} - P_{>=k+1}``.

    Returns shape ``(n_theta, n_z_q, n_z_n, k_max+1)`` for full cubes
    or ``(n_theta, n_z, k_max+1)`` for diagonal-only cubes. The
    leading slot is ``P_{N=0} = 1 - P_{>=1}``; subsequent slots are
    the differences. The PMF integrates to 1 only out to ``k_max``;
    tail truncation is the caller's responsibility.
    """
    if result.k_max == 0:
        raise ValueError(
            "k_max=0 results have no kNN ladder; rerun joint_knn_cdf with k_max>=1."
        )
    Nq = result.N_q.astype(np.float64)
    safe = np.where(Nq > 0, Nq, np.inf)
    if getattr(result, "is_diagonal", False):
        P_geq = result.H_geq_k / safe[None, :, None]   # (theta, z, k)
    else:
        P_geq = result.H_geq_k / safe[None, :, None, None]  # (theta, z_q, z_n, k)
    k_max = P_geq.shape[-1]
    pmf = np.zeros(P_geq.shape[:-1] + (k_max + 1,), dtype=np.float64)
    pmf[..., 0] = 1.0 - P_geq[..., 0]
    pmf[..., 1:k_max] = P_geq[..., :k_max - 1] - P_geq[..., 1:k_max]
    pmf[..., k_max] = P_geq[..., k_max - 1]  # tail = P_{>=k_max}
    return pmf


def cic_moments(result: KnnCdfResult, p: int) -> np.ndarray:
    """``<N^p>(theta; z_q, z_n) = sum_k k^p P_{N=k}``  (Eq. 5).

    For ``p=1`` this returns ``mean_count(result)`` to numerical
    precision (consistency check). For ``p=2`` it returns ``<N^2>``,
    matching ``sum_n2 / N_q`` to truncation error in the kNN ladder.

    The truncation note matters: the kNN ladder only resolves
    ``k <= k_max``, so the moment is biased low when the per-cap
    distribution has significant probability past ``k_max``. For
    ``p in {1, 2}``, prefer the direct moments
    ``sum_n / N_q`` and ``sum_n2 / N_q`` which carry no truncation
    error.
    """
    pmf = cic_pmf(result)
    k_axis = np.arange(pmf.shape[-1], dtype=np.float64)
    return (pmf * (k_axis ** p)).sum(axis=-1)


def sigma2_clust(result: KnnCdfResult) -> np.ndarray:
    """Dimensionless clustering variance  (Eq. 13)::

        sigma2_clust(theta; z_q, z_n) = Var(N) / <N>^2 - 1 / <N>

    Computed from the direct moment accumulators (no kNN-ladder
    truncation): ``Var = sum_n2 / N_q - mean^2``, ``<N> = sum_n / N_q``.

    Returns shape ``(n_theta, n_z_q, n_z_n)`` for full cubes, or
    ``(n_theta, n_z)`` for diagonal-only cubes. NaN where ``N_q`` or
    ``<N>`` is 0.
    """
    Nq = result.N_q.astype(np.float64)
    safe = np.where(Nq > 0, Nq, np.inf)
    if getattr(result, "is_diagonal", False):
        mu = result.sum_n / safe[None, :]
        var = result.sum_n2 / safe[None, :] - mu * mu
    else:
        mu = result.sum_n / safe[None, :, None]
        var = result.sum_n2 / safe[None, :, None] - mu * mu
    out = np.where(mu > 0, var / mu ** 2 - 1.0 / mu, np.nan)
    return out


def rsd_first_moment(result: KnnCdfResult) -> np.ndarray:
    """``<z_n - z_q>(theta; z_q)``: the first-moment RSD signal (paper §2.2).

    For each ``(theta, z_q)`` slice, average the difference between the
    z-shell midpoint of each neighbor bin and the z-shell midpoint of
    the query bin, weighted by ``sum_n[t, iq, jn]``. A non-zero value
    at ``z_q ~ z_n`` indicates infall asymmetry.

    Returns shape ``(n_theta, n_z_q)``. NaN where the row total is 0.
    """
    z_q_centres = 0.5 * (result.z_q_edges[:-1] + result.z_q_edges[1:])
    z_n_centres = 0.5 * (result.z_n_edges[:-1] + result.z_n_edges[1:])
    # delta_z[iq, jn] = z_n[jn] - z_q[iq]
    delta_z = z_n_centres[None, :] - z_q_centres[:, None]
    weighted = result.sum_n * delta_z[None, :, :]      # (n_theta, n_z_q, n_z_n)
    num = weighted.sum(axis=-1)                        # (n_theta, n_z_q)
    den = result.sum_n.sum(axis=-1)
    return np.where(den > 0, num / den, np.nan)


# ----------------------------------------------------------------------
# Two- and three-result reductions: Davis-Peebles and Landy-Szalay xi
# ----------------------------------------------------------------------


def xi_dp(result_dd: KnnCdfResult, result_rd: KnnCdfResult) -> np.ndarray:
    """Davis-Peebles natural estimator (Eq. 6)::

        1 + xi(theta; z_q, z_n) = nbar^DD / nbar^RD

    Both inputs must share the same ``theta_radii_rad``,
    ``z_q_edges``, ``z_n_edges``. Returns ``xi`` (i.e. minus 1) shape
    ``(n_theta, n_z_q, n_z_n)``.
    """
    _check_same_grid(result_dd, result_rd, "xi_dp")
    nbar_dd = mean_count(result_dd)
    nbar_rd = mean_count(result_rd)
    safe = np.where(nbar_rd > 0, nbar_rd, np.inf)
    return nbar_dd / safe - 1.0


def xi_ls(
    result_dd: KnnCdfResult,
    result_dr: KnnCdfResult,
    result_rr: KnnCdfResult,
    n_neigh_dd_per_zn: np.ndarray,
    n_neigh_dr_per_zn: np.ndarray,
    n_neigh_rr_per_zn: np.ndarray,
) -> np.ndarray:
    """Landy-Szalay angular two-point estimator (paper Eq. 12)::

        w(theta; z_q, z_n) = (DD - 2 DR + RR) / RR

    The kNN-CDF cubes store per-query neighbor counts ``nbar^X``, which
    scale with the *neighbor* catalog density. To put DD, DR, RR on a
    common ``V_cap`` scale before the LS combination, each per-query
    nbar must be divided by the count of its neighbor catalog in the
    relevant z_n shell. Without this normalisation LS picks up an
    O(N_R/N_D) bias (xi_DP is immune because it's a ratio that cancels
    the density).

    Parameters
    ----------
    result_dd, result_dr, result_rr
        Cubes from ``joint_knn_cdf`` (must share theta + z grids).
    n_neigh_dd_per_zn
        ``(n_z_n,)`` integer array — count of the DD pass's neighbor
        catalog (the data) in each z_n shell.
    n_neigh_dr_per_zn
        Per-shell count of the DR pass's neighbor catalog (typically
        also data, so this is usually equal to ``n_neigh_dd_per_zn``).
    n_neigh_rr_per_zn
        Per-shell count of the RR pass's neighbor catalog (the random
        catalog).
    """
    _check_same_grid(result_dd, result_dr, "xi_ls")
    _check_same_grid(result_dd, result_rr, "xi_ls")
    n_z_n = result_dd.n_z_n
    for arr, name in [(n_neigh_dd_per_zn, "n_neigh_dd_per_zn"),
                      (n_neigh_dr_per_zn, "n_neigh_dr_per_zn"),
                      (n_neigh_rr_per_zn, "n_neigh_rr_per_zn")]:
        if np.asarray(arr).shape != (n_z_n,):
            raise ValueError(
                f"{name} must have shape ({n_z_n},); got "
                f"{np.asarray(arr).shape}"
            )
    # mu_X = nbar_X / N_neighbor_per_zn  ~  V_cap on a common scale.
    nbar_dd = mean_count(result_dd)
    nbar_dr = mean_count(result_dr)
    nbar_rr = mean_count(result_rr)
    safe_dd = np.where(np.asarray(n_neigh_dd_per_zn) > 0,
                       n_neigh_dd_per_zn, np.inf).astype(np.float64)
    safe_dr = np.where(np.asarray(n_neigh_dr_per_zn) > 0,
                       n_neigh_dr_per_zn, np.inf).astype(np.float64)
    safe_rr = np.where(np.asarray(n_neigh_rr_per_zn) > 0,
                       n_neigh_rr_per_zn, np.inf).astype(np.float64)
    if getattr(result_dd, "is_diagonal", False):
        # Diagonal cubes: nbar_X has shape (n_theta, n_z); broadcast
        # the per-z_n array against the single z axis.
        mu_dd = nbar_dd / safe_dd[None, :]
        mu_dr = nbar_dr / safe_dr[None, :]
        mu_rr = nbar_rr / safe_rr[None, :]
    else:
        mu_dd = nbar_dd / safe_dd[None, None, :]
        mu_dr = nbar_dr / safe_dr[None, None, :]
        mu_rr = nbar_rr / safe_rr[None, None, :]
    safe = np.where(mu_rr > 0, mu_rr, np.inf)
    return (mu_dd - 2.0 * mu_dr + mu_rr) / safe


def xi_ls_annular(
    result_dd: KnnCdfResult,
    result_dr: KnnCdfResult,
    result_rr: KnnCdfResult,
    n_neigh_dd_per_zn: np.ndarray,
    n_neigh_dr_per_zn: np.ndarray,
    n_neigh_rr_per_zn: np.ndarray,
) -> np.ndarray:
    """**True differential** Landy–Szalay angular two-point function
    ξ(θ; z_q, z_n) from per-annulus pair counts.

    The cube primitive ``joint_knn_cdf`` stores cap-cumulative counts
    (``sum_n[t]`` = pairs at separation ≤ θ_t). To recover the
    differential ξ at angular separation θ_t — i.e. pair density in
    the annulus (θ_{t-1}, θ_t] — we take np.diff along the θ axis
    before applying the LS combination. The first bin's annular count
    is the cap from 0 to θ_0 (assuming θ_0 is small).

    Compare to ``xi_ls``, which evaluates the LS combination on
    cumulative cap counts and therefore returns the
    cap-volume-averaged σ²_LS(θ) = ⟨ξ⟩_cap(θ), not ξ(θ).

    Same density-normalisation contract as ``xi_ls`` for handling
    weighted catalogs: ``n_neigh_*_per_zn`` is the catalog (or
    sum-of-weights) count of the corresponding pass's neighbor catalog
    in each z_n shell.
    """
    _check_same_grid(result_dd, result_dr, "xi_ls_annular")
    _check_same_grid(result_dd, result_rr, "xi_ls_annular")
    n_z_n = result_dd.n_z_n
    for arr, name in [(n_neigh_dd_per_zn, "n_neigh_dd_per_zn"),
                      (n_neigh_dr_per_zn, "n_neigh_dr_per_zn"),
                      (n_neigh_rr_per_zn, "n_neigh_rr_per_zn")]:
        if np.asarray(arr).shape != (n_z_n,):
            raise ValueError(
                f"{name} must have shape ({n_z_n},); got "
                f"{np.asarray(arr).shape}"
            )

    def _annular_sum_n(res: KnnCdfResult) -> np.ndarray:
        # diff along theta axis with a zero pre-pad so output[0] is the
        # cap from 0 to theta_0.
        sn = res.sum_n
        sn_padded = np.concatenate(
            [np.zeros((1,) + sn.shape[1:]), sn], axis=0)
        return np.diff(sn_padded, axis=0)

    sn_dd_a = _annular_sum_n(result_dd)
    sn_dr_a = _annular_sum_n(result_dr)
    sn_rr_a = _annular_sum_n(result_rr)

    Nq_dd = result_dd.N_q.astype(np.float64)
    Nq_dr = result_dr.N_q.astype(np.float64)
    Nq_rr = result_rr.N_q.astype(np.float64)
    safe_Nq_dd = np.where(Nq_dd > 0, Nq_dd, np.inf)
    safe_Nq_dr = np.where(Nq_dr > 0, Nq_dr, np.inf)
    safe_Nq_rr = np.where(Nq_rr > 0, Nq_rr, np.inf)
    if getattr(result_dd, "is_diagonal", False):
        nbar_dd_a = sn_dd_a / safe_Nq_dd[None, :]
        nbar_dr_a = sn_dr_a / safe_Nq_dr[None, :]
        nbar_rr_a = sn_rr_a / safe_Nq_rr[None, :]
    else:
        nbar_dd_a = sn_dd_a / safe_Nq_dd[None, :, None]
        nbar_dr_a = sn_dr_a / safe_Nq_dr[None, :, None]
        nbar_rr_a = sn_rr_a / safe_Nq_rr[None, :, None]

    safe_dd = np.where(np.asarray(n_neigh_dd_per_zn) > 0,
                       n_neigh_dd_per_zn, np.inf).astype(np.float64)
    safe_dr = np.where(np.asarray(n_neigh_dr_per_zn) > 0,
                       n_neigh_dr_per_zn, np.inf).astype(np.float64)
    safe_rr = np.where(np.asarray(n_neigh_rr_per_zn) > 0,
                       n_neigh_rr_per_zn, np.inf).astype(np.float64)
    if getattr(result_dd, "is_diagonal", False):
        mu_dd = nbar_dd_a / safe_dd[None, :]
        mu_dr = nbar_dr_a / safe_dr[None, :]
        mu_rr = nbar_rr_a / safe_rr[None, :]
    else:
        mu_dd = nbar_dd_a / safe_dd[None, None, :]
        mu_dr = nbar_dr_a / safe_dr[None, None, :]
        mu_rr = nbar_rr_a / safe_rr[None, None, :]
    safe_denom = np.where(mu_rr > 0, mu_rr, np.inf)
    return (mu_dd - 2.0 * mu_dr + mu_rr) / safe_denom


def xi_ls_cross(
    result_dd_xy: KnnCdfResult,
    result_dr_xy: KnnCdfResult,
    result_rd_xy: KnnCdfResult,
    result_rr_xy: KnnCdfResult,
    n_neigh_dy_per_zn: np.ndarray,
    n_neigh_ry_per_zn: np.ndarray,
) -> np.ndarray:
    """Asymmetric Landy-Szalay cross-correlation estimator (paper Eq. 12,
    cross variant)::

        xi_LS_xy(theta; z_q, z_n) =
            (mu_DD_xy - mu_DR_xy - mu_RD_xy + mu_RR_xy) / mu_RR_xy

    where ``mu_X = nbar_X / N_neighbor_per_zn`` puts each per-cap count
    on a common per-cap-volume scale. For two distinct catalogs ``x``
    and ``y``, ``DR_xy != RD_xy`` in general (different query catalog,
    different sample variance), so the symmetric ``-2 mu_DR`` shortcut
    used in the autocorrelation ``xi_ls`` does not apply.

    Parameters
    ----------
    result_dd_xy
        query=D_x, neighbor=D_y. mean_count divides by N_query_x.
    result_dr_xy
        query=D_x, neighbor=R_y.
    result_rd_xy
        query=R_x, neighbor=D_y.
    result_rr_xy
        query=R_x, neighbor=R_y.
    n_neigh_dy_per_zn
        ``(n_z_n,)`` per-shell count of the Y data catalog.
    n_neigh_ry_per_zn
        ``(n_z_n,)`` per-shell count of the Y random catalog.
    """
    _check_same_grid(result_dd_xy, result_dr_xy, "xi_ls_cross")
    _check_same_grid(result_dd_xy, result_rd_xy, "xi_ls_cross")
    _check_same_grid(result_dd_xy, result_rr_xy, "xi_ls_cross")
    n_z_n = result_dd_xy.n_z_n
    for arr, name in [(n_neigh_dy_per_zn, "n_neigh_dy_per_zn"),
                      (n_neigh_ry_per_zn, "n_neigh_ry_per_zn")]:
        if np.asarray(arr).shape != (n_z_n,):
            raise ValueError(
                f"{name} must have shape ({n_z_n},); got "
                f"{np.asarray(arr).shape}"
            )
    nbar_dd = mean_count(result_dd_xy)
    nbar_dr = mean_count(result_dr_xy)
    nbar_rd = mean_count(result_rd_xy)
    nbar_rr = mean_count(result_rr_xy)
    safe_dy = np.where(np.asarray(n_neigh_dy_per_zn) > 0,
                       n_neigh_dy_per_zn, np.inf).astype(np.float64)
    safe_ry = np.where(np.asarray(n_neigh_ry_per_zn) > 0,
                       n_neigh_ry_per_zn, np.inf).astype(np.float64)
    mu_dd = nbar_dd / safe_dy[None, None, :]
    mu_dr = nbar_dr / safe_ry[None, None, :]
    mu_rd = nbar_rd / safe_dy[None, None, :]
    mu_rr = nbar_rr / safe_ry[None, None, :]
    safe = np.where(mu_rr > 0, mu_rr, np.inf)
    return (mu_dd - mu_dr - mu_rd + mu_rr) / safe


def xi_diag(xi_cube: np.ndarray) -> np.ndarray:
    """Per-shell two-point function: the diagonal ``z_q == z_n`` slice
    of an angular xi cube. Returns shape ``(n_theta, n_shells)``.
    """
    n_z_q = xi_cube.shape[1]
    n_z_n = xi_cube.shape[2]
    n = min(n_z_q, n_z_n)
    return xi_cube[:, np.arange(n), np.arange(n)]


# ----------------------------------------------------------------------
# Jackknife covariance helper
# ----------------------------------------------------------------------


def jackknife_cov(
    samples_per_region: np.ndarray,
    hartlap: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard jackknife covariance from per-region samples.

    Parameters
    ----------
    samples_per_region
        ``(n_regions, ...)`` array of per-fold derived-quantity
        samples. NaN entries (folds with no kept centres) are
        excluded.
    hartlap
        If True, multiply the inverse covariance by the Hartlap
        factor ``(n_used - p - 2) / (n_used - 1)`` where ``p`` is the
        flat data dimension. The forward covariance is unaffected;
        the Hartlap correction is only meaningful when inverting.

    Returns
    -------
    mean : ``(...)`` jackknife mean.
    cov : ``(d, d)`` covariance, where ``d = prod(samples.shape[1:])``.
    """
    s = np.asarray(samples_per_region, dtype=np.float64)
    finite = np.all(np.isfinite(s.reshape(s.shape[0], -1)), axis=1)
    n_used = int(finite.sum())
    if n_used < 2:
        raise ValueError(
            f"jackknife_cov needs >= 2 finite folds; got {n_used}"
        )
    s_use = s[finite]
    mean = s_use.mean(axis=0)
    diff = (s_use - mean[None]).reshape(n_used, -1)
    cov = ((n_used - 1) / n_used) * (diff.T @ diff)
    return mean, cov


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _check_same_grid(a: KnnCdfResult, b: KnnCdfResult, label: str) -> None:
    if not np.array_equal(a.theta_radii_rad, b.theta_radii_rad):
        raise ValueError(f"{label}: theta_radii_rad mismatch")
    if not np.array_equal(a.z_q_edges, b.z_q_edges):
        raise ValueError(f"{label}: z_q_edges mismatch")
    if not np.array_equal(a.z_n_edges, b.z_n_edges):
        raise ValueError(f"{label}: z_n_edges mismatch")


# ----------------------------------------------------------------------
# Higher moments and Hamilton/LS estimators (note v4_1 §3, §6)
# ----------------------------------------------------------------------


def _moment_p(result: KnnCdfResult, p: int) -> np.ndarray:
    """Direct ⟨N^p⟩ moment from the raw moment cubes for p ∈ {1,2,3,4}.

    No truncation error (the moments are accumulated per-cap during
    the kNN-CDF pass; the kNN ladder is not consulted). Falls back to
    raising ValueError for older results that lack sum_n3 / sum_n4
    (e.g. NPZ artifacts produced before v4_1 §6 was wired in).
    """
    if p == 1:
        cube = result.sum_n
    elif p == 2:
        cube = result.sum_n2
    elif p == 3:
        cube = getattr(result, "sum_n3", None)
    elif p == 4:
        cube = getattr(result, "sum_n4", None)
    else:
        raise ValueError(f"_moment_p: only p=1..4 supported, got {p}")
    if cube is None:
        raise ValueError(
            f"sum_n{p} missing — rerun joint_knn_cdf with the v4_1 "
            "kernels to populate higher moments.")
    Nq = result.N_q.astype(np.float64)
    safe = np.where(Nq > 0, Nq, np.inf)
    if getattr(result, "is_diagonal", False):
        return cube / safe[None, :]
    return cube / safe[None, :, None]


def cic_skewness_raw(result: KnnCdfResult) -> np.ndarray:
    """Standardised skewness of the per-cap counts ``N(θ; z)``::

        S₃ = ⟨(N - ⟨N⟩)³⟩ / σ³
           = (⟨N³⟩ - 3 ⟨N⟩⟨N²⟩ + 2 ⟨N⟩³) / σ³

    Computed from the *raw* DD moments. For a Gaussian or Poisson
    field S₃ → 0; positive S₃ indicates a fatter upper tail (clusters).

    NaN where σ ≤ 0.
    """
    m1 = _moment_p(result, 1)
    m2 = _moment_p(result, 2)
    m3 = _moment_p(result, 3)
    var = m2 - m1 * m1
    sigma = np.where(var > 0, np.sqrt(np.maximum(var, 0.0)), np.nan)
    central3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    return central3 / (sigma_safe ** 3)


def cic_kurtosis_raw(result: KnnCdfResult) -> np.ndarray:
    """Excess kurtosis of the per-cap counts (S₄ − 3)::

        Kurtosis = ⟨(N - ⟨N⟩)⁴⟩ / σ⁴ - 3
                 = (⟨N⁴⟩ − 4⟨N⟩⟨N³⟩ + 6⟨N⟩²⟨N²⟩ − 3⟨N⟩⁴) / σ⁴ − 3

    For a Gaussian field returns 0; for a Poisson field with mean μ
    returns 1/μ (shot-noise kurtosis); for a clustered field returns
    a positive value when the upper tail is fatter than Gaussian.
    """
    m1 = _moment_p(result, 1)
    m2 = _moment_p(result, 2)
    m3 = _moment_p(result, 3)
    m4 = _moment_p(result, 4)
    var = m2 - m1 * m1
    central4 = (
        m4 - 4.0 * m1 * m3 + 6.0 * (m1 * m1) * m2 - 3.0 * m1 ** 4
    )
    var_safe = np.where(var > 0, var, np.nan)
    return central4 / (var_safe ** 2) - 3.0


def _mu_p_per_cap(result: KnnCdfResult, p: int,
                  n_neigh_per_zn: np.ndarray) -> np.ndarray:
    """Per-cap-volume-normalised moment ``μ_p^X = ⟨N^p⟩^X / N_neigh``.

    Used by the LS and Hamilton bias-corrected estimators below;
    mirrors the ``mu_dd``/``mu_dr``/``mu_rr`` pattern in `xi_ls`.
    """
    nbar_p = _moment_p(result, p)
    n_z_n = result.n_z_n
    safe_neigh = np.where(np.asarray(n_neigh_per_zn) > 0,
                          n_neigh_per_zn, np.inf).astype(np.float64)
    if safe_neigh.shape != (n_z_n,):
        raise ValueError(
            f"_mu_p_per_cap: n_neigh_per_zn shape ({safe_neigh.shape}) "
            f"!= (n_z_n,) = ({n_z_n},)")
    if getattr(result, "is_diagonal", False):
        return nbar_p / safe_neigh[None, :]
    return nbar_p / safe_neigh[None, None, :]


def cic_moment_ls(
    result_dd: KnnCdfResult,
    result_dr: KnnCdfResult,
    result_rd: KnnCdfResult,
    result_rr: KnnCdfResult,
    p: int,
    n_neigh_d_per_zn: np.ndarray,
    n_neigh_r_per_zn: np.ndarray,
) -> np.ndarray:
    """Landy–Szalay-corrected ``⟨N^p⟩^LS`` (note Eq. 13)::

        ⟨N^p⟩^LS = (⟨N^p⟩^DD - ⟨N^p⟩^DR
                    - ⟨N^p⟩^RD + ⟨N^p⟩^RR) / ⟨N^p⟩^RR

    Each ``⟨N^p⟩^X`` is normalised per-cap via
    ``μ_p^X = ⟨N^p⟩^X / N_neigh^X`` so the subtraction has no
    catalog-density bias.

    For ``p=1`` this returns the connected pair-density (i.e.
    ``1 + ξ_LS`` if ``μ`` were normalised differently); higher ``p``
    give bias-corrected raw moments.
    """
    _check_same_grid(result_dd, result_dr, "cic_moment_ls")
    _check_same_grid(result_dd, result_rd, "cic_moment_ls")
    _check_same_grid(result_dd, result_rr, "cic_moment_ls")
    mu_dd = _mu_p_per_cap(result_dd, p, n_neigh_d_per_zn)
    mu_dr = _mu_p_per_cap(result_dr, p, n_neigh_r_per_zn)
    mu_rd = _mu_p_per_cap(result_rd, p, n_neigh_d_per_zn)
    mu_rr = _mu_p_per_cap(result_rr, p, n_neigh_r_per_zn)
    safe = np.where(mu_rr > 0, mu_rr, np.inf)
    return (mu_dd - mu_dr - mu_rd + mu_rr) / safe


def cic_moment_hamilton(
    result_dd: KnnCdfResult,
    result_dr: KnnCdfResult,
    result_rd: KnnCdfResult,
    result_rr: KnnCdfResult,
    p: int,
    n_neigh_d_per_zn: np.ndarray,
    n_neigh_r_per_zn: np.ndarray,
) -> np.ndarray:
    """Hamilton-corrected ``⟨N^p⟩^Ham`` (note Eq. 16)::

        ⟨N^p⟩^Ham = (⟨N^p⟩^DD ⟨N^p⟩^RR) / (⟨N^p⟩^DR ⟨N^p⟩^RD) - 1

    Multiplicative form cancels overall normalisation factors. Less
    sensitive than LS to fluctuations when RR is small (note §3.3),
    which matters at large angular scales where the random-pair
    count is sparse.
    """
    _check_same_grid(result_dd, result_dr, "cic_moment_hamilton")
    _check_same_grid(result_dd, result_rd, "cic_moment_hamilton")
    _check_same_grid(result_dd, result_rr, "cic_moment_hamilton")
    mu_dd = _mu_p_per_cap(result_dd, p, n_neigh_d_per_zn)
    mu_dr = _mu_p_per_cap(result_dr, p, n_neigh_r_per_zn)
    mu_rd = _mu_p_per_cap(result_rd, p, n_neigh_d_per_zn)
    mu_rr = _mu_p_per_cap(result_rr, p, n_neigh_r_per_zn)
    den = mu_dr * mu_rd
    safe_den = np.where(den > 0, den, np.inf)
    return (mu_dd * mu_rr) / safe_den - 1.0


def xi_hamilton(
    result_dd: KnnCdfResult,
    result_dr: KnnCdfResult,
    result_rd: KnnCdfResult,
    result_rr: KnnCdfResult,
    n_neigh_d_per_zn: np.ndarray,
    n_neigh_r_per_zn: np.ndarray,
) -> np.ndarray:
    """Hamilton ξ estimator (note Eq. 15) — alias for ``cic_moment_hamilton``
    at ``p=1``. Matches the canonical ξ definition with multiplicative
    bias removal."""
    return cic_moment_hamilton(
        result_dd, result_dr, result_rd, result_rr, 1,
        n_neigh_d_per_zn, n_neigh_r_per_zn,
    )


def sigma2_clust_ls(
    result_dd: KnnCdfResult,
    result_dr: KnnCdfResult,
    result_rd: KnnCdfResult,
    result_rr: KnnCdfResult,
    n_neigh_d_per_zn: np.ndarray,
    n_neigh_r_per_zn: np.ndarray,
) -> np.ndarray:
    """Landy-Szalay clustering variance σ²_clust^LS (note Eq. 14)::

        σ²_clust = Var^LS / ⟨N¹⟩^LS² - 1 / ⟨N¹⟩^LS

    where ``Var^LS = ⟨N²⟩^LS - (⟨N¹⟩^LS)²``. NaN where the LS mean
    is non-positive (clustering artifacts at large θ when RR is
    near zero).
    """
    m1_ls = cic_moment_ls(
        result_dd, result_dr, result_rd, result_rr, 1,
        n_neigh_d_per_zn, n_neigh_r_per_zn)
    m2_ls = cic_moment_ls(
        result_dd, result_dr, result_rd, result_rr, 2,
        n_neigh_d_per_zn, n_neigh_r_per_zn)
    var_ls = m2_ls - m1_ls * m1_ls
    out = np.where(m1_ls > 0,
                   var_ls / (m1_ls * m1_ls) - 1.0 / m1_ls,
                   np.nan)
    return out


# ----------------------------------------------------------------------
# Differential pair count and redshift derivative (note §4, Eq. 9)
# ----------------------------------------------------------------------


def differential_pair_count(result: KnnCdfResult) -> np.ndarray:
    """Differential pair count ``dn_pairs/dθ`` (note Eq. 9)::

        dn_pairs/dθ = 2π θ N · ∂⟨N¹⟩/∂θ / (2π)

    The cap is converted to an annulus by differencing along the θ
    axis. Returns the same shape as ``mean_count(result)``.

    Useful as a direct visual cross-check on the kNN-derived ξ: the
    differential pair count traces the underlying clustering more
    directly than the cumulative cap count.
    """
    nbar = mean_count(result)
    theta = np.asarray(result.theta_radii_rad, dtype=np.float64)
    # Forward-difference along theta axis with a leading 0 pre-pad so
    # output[0] is the cap from 0 to theta_0 divided by theta_0.
    nbar_pad = np.concatenate(
        [np.zeros((1,) + nbar.shape[1:]), nbar], axis=0)
    dnbar_dtheta = np.diff(nbar_pad, axis=0)
    theta_pad = np.concatenate([[0.0], theta])
    dtheta = np.diff(theta_pad)
    # Broadcast theta axis
    bcast = (slice(None),) + (None,) * (nbar.ndim - 1)
    return dnbar_dtheta * (theta[bcast] / dtheta[bcast])


def dlnsigma2_dlogz(sigma2: np.ndarray, z_centres: np.ndarray) -> np.ndarray:
    """Logarithmic redshift derivative ``∂ ln σ² / ∂ ln(1+z)``
    (note Eq. 18, Eq. 20).

    Centred-difference along the z axis (the second axis of
    ``sigma2``); endpoints use forward/backward difference. NaN
    propagates through bins with σ² ≤ 0.

    Parameters
    ----------
    sigma2 : ndarray
        ``(n_theta, n_z)`` array — typically from
        ``sigma2_clust(result)`` on a diagonal cube, or from
        ``sigma2_clust_ls(...)``.
    z_centres : ndarray
        ``(n_z,)`` array of bin centres (e.g.
        ``0.5 * (z_edges[:-1] + z_edges[1:])``).

    Returns
    -------
    deriv : ndarray
        ``(n_theta, n_z)`` array of ∂ ln σ² / ∂ ln(1+z) per bin.
        For a linear-bias scaling ``b(z)² D(z)² σ²_m(θ D_M(z); z=0)``
        this decomposes into bias-evolution, growth, and geometry
        terms (note Eq. 20) — useful as a direct cosmology-sensitive
        diagnostic.
    """
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    z_centres = np.asarray(z_centres, dtype=np.float64)
    if sigma2.shape[1] != z_centres.size:
        raise ValueError(
            f"sigma2.shape[1]={sigma2.shape[1]} but z_centres.size="
            f"{z_centres.size}")
    log1pz = np.log1p(z_centres)
    log_s2 = np.where(sigma2 > 0, np.log(np.maximum(sigma2, 1e-300)),
                      np.nan)
    n_z = z_centres.size
    out = np.full_like(log_s2, np.nan)
    if n_z >= 2:
        # Centred difference for interior; forward/backward for ends.
        out[:, 1:-1] = (
            (log_s2[:, 2:] - log_s2[:, :-2])
            / (log1pz[2:] - log1pz[:-2])[None, :]
        )
        out[:, 0] = (log_s2[:, 1] - log_s2[:, 0]) / \
            (log1pz[1] - log1pz[0])
        out[:, -1] = (log_s2[:, -1] - log_s2[:, -2]) / \
            (log1pz[-1] - log1pz[-2])
    return out


# ----------------------------------------------------------------------
# Multipole projection ξ_ℓ(s) from off-diagonal cube (note §5, Eq. 22-24)
# ----------------------------------------------------------------------


def multipole_xi_ell(
    xi_offdiag_cube: np.ndarray,
    theta_rad: np.ndarray,
    z_q_centres: np.ndarray,
    z_n_centres: np.ndarray,
    cosmo,
    ell: int,
    s_edges: np.ndarray,
) -> np.ndarray:
    """Project a (θ, z₁, z₂) cube to multipole ``ξ_ℓ(s)`` via the
    Legendre projection of note Eq. (22–24).

    Parameters
    ----------
    xi_offdiag_cube
        ``(n_theta, n_z_q, n_z_n)`` array of bias-corrected ξ on the
        full (z₁, z₂) plane (typically ``xi_ls`` evaluated on a
        non-diagonal result).
    theta_rad
        ``(n_theta,)`` angular separations in radians.
    z_q_centres, z_n_centres
        ``(n_z_q,)`` / ``(n_z_n,)`` bin centres.
    cosmo
        Object with ``D_M(z) -> Mpc/h`` (transverse comoving distance)
        and ``H_z(z) -> km/s/(Mpc/h)`` (Hubble at z).
    ell
        Multipole order (0, 2, 4 typical).
    s_edges
        ``(n_s+1,)`` ascending bin edges in Mpc/h for the output ξ_ℓ(s).

    Returns
    -------
    xi_ell : ndarray
        ``(n_s,)`` Legendre-projected multipole evaluated at the bin
        midpoints.

    Notes
    -----
    Maps each (θ, z₁, z₂) cell to ``(s, μ)`` via Eq. (23–24) using
    ``z_mid = (z₁ + z₂) / 2`` for the geometry. Bins by ``s`` and
    averages weighted by ``L_ℓ(μ)``. The averaging counts each cell
    once with equal weight; finer integration would require resampling
    on a denser (z₁, z₂) grid.
    """
    from scipy.special import eval_legendre
    n_theta, n_z_q, n_z_n = xi_offdiag_cube.shape
    if theta_rad.size != n_theta or z_q_centres.size != n_z_q \
            or z_n_centres.size != n_z_n:
        raise ValueError(
            "shape mismatch: xi_offdiag_cube vs theta_rad / z_*_centres")
    # Coordinates per cell (theta, z_q, z_n).
    z_mid = 0.5 * (z_q_centres[:, None] + z_n_centres[None, :])
    dz = z_n_centres[None, :] - z_q_centres[:, None]
    # Vectorise via list comprehension to avoid asking the cosmo for
    # arrays (some implementations are scalar-only).
    DM = np.array([[cosmo.D_M(z_mid[i, j]) for j in range(n_z_n)]
                   for i in range(n_z_q)])
    Hz = np.array([[cosmo.H_z(z_mid[i, j]) for j in range(n_z_n)]
                   for i in range(n_z_q)])
    c_kms = 299792.458
    r_perp = DM[None, :, :] * theta_rad[:, None, None]
    r_para = (c_kms * dz[None, :, :]) / Hz[None, :, :]
    s = np.sqrt(r_perp * r_perp + r_para * r_para)
    mu = np.where(s > 0, r_para / s, 0.0)
    L = eval_legendre(ell, mu)
    # Bin by s.
    n_s = s_edges.size - 1
    xi_ell = np.full(n_s, np.nan)
    weighted = (xi_offdiag_cube * L) * (2 * ell + 1) / 2.0
    s_flat = s.ravel()
    w_flat = weighted.ravel()
    L_flat = (L * (2 * ell + 1) / 2.0).ravel()
    for i in range(n_s):
        in_bin = (s_flat >= s_edges[i]) & (s_flat < s_edges[i + 1])
        if not in_bin.any():
            continue
        # Average ξ × Lℓ weighted uniformly over the bin's cells.
        # Normalisation by Σ Lℓ² recovers an unbiased ℓ-projection
        # in the limit of dense (z₁, z₂) sampling.
        denom = (L_flat[in_bin] ** 2).sum() / ((2 * ell + 1) / 2.0)
        if denom <= 0:
            continue
        xi_ell[i] = w_flat[in_bin].sum() / denom
    return xi_ell
