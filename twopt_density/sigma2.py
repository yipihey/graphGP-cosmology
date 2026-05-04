"""sigma^2(R) -- count-in-cells variance as a parallel two-point statistic.

The variance of the galaxy count in a sphere of radius R is the
exact linear projection of the two-point correlation function
xi(r) against the spherical-overlap kernel:

    sigma^2_TH(R) = (1 / V_R) <(N - <N>)^2> / <N>^2 - 1/<N>
                  = integral d^3 r  xi(r)  K_TH(r; R),

with

    K_TH(r; R) = (3 / (4 pi R^3)) * [1 - 3 r / (4 R) + (r / (2 R))^3]

for ``r <= 2 R`` and zero otherwise (the geometric overlap volume of
two top-hat spheres of radius ``R`` separated by ``r``, normalised so
``int K_TH dV = 1``). Equivalently for a Gaussian window of scale ``R``,

    K_G(r; R) = (1 / (4 pi^(3/2) R^3)) * exp(-r^2 / (4 R^2))

(the Gaussian convolution of two Gaussians of width ``R``).

The projection is linear in the pair counts: each unordered pair
contributes ``K(r_pair; R)`` to the kernel-weighted sum. Replacing
``DD - 2 DR + RR`` in the Landy-Szalay numerator with the
kernel-weighted version yields ``sigma^2(R)`` directly:

    sigma^2(R) = sum_DD K(r; R) - 2 sum_DR K(r; R) + sum_RR K(r; R)
                  ----------------------------------------------------
                                 sum_RR K(r; R)

with the same analytic-RR machinery (RR is computed once on a fine
``r`` grid, kernel-projected, and re-used across all ``R``).

Why a parallel sigma^2 pipeline:
  - Robust to small-scale systematics. The kernel ``K_TH`` strongly
    weights pairs at ``r ~ R`` and suppresses ``r << R``. Fiber
    collisions, photo-z LOS smearing, and angular-resolution effects
    that bias xi(r) at small r have a much smaller leverage on
    sigma^2(R) at scales ``R > r_systematic``.
  - Bin-free. Only one R per measurement -- no rebinning or kernel
    width choice on the data side.
  - Exact equivalence with xi-based pipeline: for a noise-free LS
    measurement, ``sigma^2_K(R)`` and ``int xi(r) K(r; R) dV`` are
    identical to within finite-bin discretisation.
  - Same analytic-RR backend. The window-aware analytic RR(r) is
    kernel-projected once per R in O(N_r * N_R) work.

Module exposes::

  kernel_TH_3d(r, R), kernel_Gauss_3d(r, R)
  sigma2_from_xi(r_grid, xi, R_grid, kernel="tophat")
        -> sigma^2(R) by integrating an existing xi(r)
  sigma2_from_pair_counts(r_centres, DD, DR, RR, R_grid,
                            N_d, N_r, kernel="tophat")
        -> sigma^2_LS(R), the parallel of xi_LS(s)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def kernel_TH_3d(r: np.ndarray, R: float) -> np.ndarray:
    """Top-hat sphere overlap kernel ``K_TH(r; R)`` (units of 1/volume).

    Two top-hat spheres of radius ``R`` separated by ``r`` overlap
    in a fraction of either sphere's volume given by
    ``f(r; R) = 1 - 3r/(4R) + (r/(2R))^3`` for ``r <= 2R``, 0 else.
    The kernel is the overlap PDF normalised so
    ``int K_TH(r; R) dV = 1``::

        K_TH(r; R) = (3 / (4 pi R^3)) * f(r; R).

    Returns 0 outside ``[0, 2R]``.
    """
    r = np.asarray(r, dtype=np.float64)
    x = r / R
    f = 1.0 - 0.75 * x + (x ** 3) / 16.0
    K = np.where(x <= 2.0, (3.0 / (4.0 * np.pi * R ** 3)) * f, 0.0)
    return K


def kernel_Gauss_3d(r: np.ndarray, R: float) -> np.ndarray:
    """Gaussian smoothing kernel ``K_G(r; R)`` of scale ``R``.

    The convolution of two Gaussians of width ``R`` is itself a
    Gaussian of width ``sqrt(2) R``. Normalised so
    ``int K_G(r; R) dV = 1``::

        K_G(r; R) = (4 pi)^(-3/2) R^(-3) exp(-r^2 / (4 R^2)).
    """
    r = np.asarray(r, dtype=np.float64)
    return ((4.0 * np.pi) ** (-1.5) / R ** 3
              * np.exp(-(r ** 2) / (4.0 * R ** 2)))


def _kernel_for(kernel: str):
    if kernel == "tophat":
        return kernel_TH_3d
    if kernel in {"gauss", "gaussian"}:
        return kernel_Gauss_3d
    raise ValueError(f"unknown kernel {kernel!r}")


def sigma2_from_xi(
    r_grid: np.ndarray, xi: np.ndarray,
    R_grid: np.ndarray, kernel: str = "tophat",
) -> np.ndarray:
    """``sigma^2(R)`` by integrating an existing ``xi(r)`` against the
    spherical-volume kernel:

        sigma^2(R) = int dr 4 pi r^2 xi(r) K(r; R).

    Trapezoidal integration on the input ``r_grid``; ``r_grid`` must
    extend to at least ``2 max(R)`` for the top-hat case, and to
    ``~ 5 max(R)`` for Gaussian (convergence at 5-sigma).

    Returns ``(n_R,)`` array.
    """
    r = np.asarray(r_grid, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    R_grid = np.asarray(R_grid, dtype=np.float64).ravel()
    K_fn = _kernel_for(kernel)
    out = np.zeros_like(R_grid)
    for i, R in enumerate(R_grid):
        K = K_fn(r, float(R))
        integrand = 4.0 * np.pi * r ** 2 * xi * K
        out[i] = np.trapezoid(integrand, r)
    return out


def sigma2_from_pair_counts(
    r_centres: np.ndarray,
    DD: np.ndarray, RR: np.ndarray,
    R_grid: np.ndarray,
    N_d: int, N_r: int,
    DR: np.ndarray = None,
    kernel: str = "tophat",
    dV_bin: np.ndarray = None,
) -> np.ndarray:
    """Landy-Szalay ``sigma^2(R)`` from kernel-weighted pair-count bins.

    Each unordered pair contributes ``K(r_pair; R)`` to the
    kernel-weighted sum. Working with binned counts:

        Sum_K(R) = sum_bins counts_bin * <K(r; R)>_bin

    where ``<K>_bin`` is approximated by ``K(r_centre; R)``. The
    Landy-Szalay form is then

        sigma^2(R) = (Sum_DD_K / Nd_pairs - 2 Sum_DR_K / (Nd Nr)
                       + Sum_RR_K / Nr_pairs) / (Sum_RR_K_norm),

    where ``Sum_RR_K_norm = Sum_RR_K / Nr_pairs`` (per-pair
    density). If ``DR`` is omitted we substitute the unclustered
    expectation ``DR = (N_d N_r / Nr_pairs) * RR / 2 / scale``.

    Parameters
    ----------
    r_centres : (n_r,) bin centres in 3D separation r [Mpc/h].
    DD, RR    : (n_r,) raw pair counts.
    R_grid    : (n_R,) sphere radii at which to evaluate sigma^2.
    N_d, N_r  : data and random sample sizes.
    DR        : optional cross-pair counts; if None, analytic
                Davis-Peebles form is used.
    kernel    : "tophat" or "gaussian".
    dV_bin    : optional (n_r,) per-bin shell volume; used to
                renormalise the bin-centre kernel value to a
                volume-averaged value (small correction at log-spaced
                bins). If None, the kernel is evaluated at r_centres.

    Returns
    -------
    sigma2 : (n_R,) array of sigma^2(R) values.
    """
    r = np.asarray(r_centres, dtype=np.float64)
    DD = np.asarray(DD, dtype=np.float64)
    RR = np.asarray(RR, dtype=np.float64)
    R_grid = np.asarray(R_grid, dtype=np.float64).ravel()

    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r * (N_r - 1) / 2.0
    DD_n = DD / max(Nd_pairs, 1.0)
    RR_n = RR / max(Nr_pairs, 1.0)
    if DR is None:
        DR_n = RR_n
    else:
        DR_n = np.asarray(DR, dtype=np.float64) / (N_d * N_r)

    K_fn = _kernel_for(kernel)
    out = np.zeros_like(R_grid)
    for i, R in enumerate(R_grid):
        K_at = K_fn(r, float(R))
        if dV_bin is not None:
            K_at = K_at * (np.asarray(dV_bin) / np.maximum(
                4.0 * np.pi * r ** 2 * np.gradient(r), 1e-30))
        S_DD = float(np.sum(DD_n * K_at))
        S_DR = float(np.sum(DR_n * K_at))
        S_RR = float(np.sum(RR_n * K_at))
        if S_RR > 0:
            out[i] = (S_DD - 2.0 * S_DR + S_RR) / S_RR
        else:
            out[i] = 0.0
    return out


def sigma2_from_rp_pi_pairs(
    rp_centres: np.ndarray, pi_centres: np.ndarray,
    DD2: np.ndarray, RR2: np.ndarray,
    R_grid: np.ndarray,
    N_d: int, N_r: int,
    DR2: np.ndarray = None,
    kernel: str = "tophat",
) -> np.ndarray:
    """``sigma^2(R)`` from the (rp, pi) pair-count tables.

    Recasts the 2D (rp, pi) bins to a 3D-separation kernel by
    s_ij = sqrt(rp_ij^2 + pi_ij^2), evaluating the kernel at the bin
    centre. Useful when the wp(rp) pipeline already produced
    (DD, RR, DR) on a (rp, pi) grid.
    """
    rp = np.asarray(rp_centres, dtype=np.float64)
    pi = np.asarray(pi_centres, dtype=np.float64)
    s = np.sqrt(rp[:, None] ** 2 + pi[None, :] ** 2)

    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r * (N_r - 1) / 2.0
    DD_n = np.asarray(DD2, dtype=np.float64) / max(Nd_pairs, 1.0)
    RR_n = np.asarray(RR2, dtype=np.float64) / max(Nr_pairs, 1.0)
    if DR2 is None:
        DR_n = RR_n
    else:
        DR_n = np.asarray(DR2, dtype=np.float64) / (N_d * N_r)

    K_fn = _kernel_for(kernel)
    R_grid = np.asarray(R_grid, dtype=np.float64).ravel()
    out = np.zeros_like(R_grid)
    for i, R in enumerate(R_grid):
        K_at = K_fn(s, float(R))
        S_DD = float(np.sum(DD_n * K_at))
        S_DR = float(np.sum(DR_n * K_at))
        S_RR = float(np.sum(RR_n * K_at))
        if S_RR > 0:
            out[i] = (S_DD - 2.0 * S_DR + S_RR) / S_RR
        else:
            out[i] = 0.0
    return out


def per_particle_kernel_counts(
    positions_a: np.ndarray, positions_b: np.ndarray,
    R: float, kernel: str = "tophat",
    auto: bool = False,
) -> np.ndarray:
    """For each particle ``i`` in ``positions_a``, return
    ``b_K_i = sum_{j in B, j != i} K(|x_i - x_j|; R)``.

    Smooth analogue of ``weights_pair_counts.per_particle_pair_counts``
    -- instead of integer counts in a hard radial bin, each partner
    contributes the smooth kernel weight, so the per-particle estimate
    integrates over many partners and is much less noisy than the
    binned version. Set ``auto=True`` when ``A == B`` to exclude
    self-pairs.

    Cost is ``O(N_a log N_b)`` via cKDTree query (kernel support is
    finite at ``2R`` for top-hat, ``~ 4R`` for Gaussian).
    """
    from scipy.spatial import cKDTree

    A = np.asarray(positions_a, dtype=np.float64)
    B = np.asarray(positions_b, dtype=np.float64)
    if kernel == "tophat":
        r_max = 2.0 * R
    else:
        r_max = 6.0 * R                 # 5-sigma cutoff for Gaussian
    K_fn = _kernel_for(kernel)
    tree = cKDTree(B)
    out = np.zeros(len(A), dtype=np.float64)
    nbrs = tree.query_ball_point(A, r=r_max)
    for i, lst in enumerate(nbrs):
        if not lst:
            continue
        j = np.asarray(lst, dtype=np.int64)
        if auto:
            j = j[j != i]
            if j.size == 0:
                continue
        d = np.linalg.norm(B[j] - A[i], axis=1)
        out[i] = float(np.sum(K_fn(d, float(R))))
    return out


def density_weights_sigma2(
    positions_d: np.ndarray, positions_r: np.ndarray,
    R: float, kernel: str = "tophat",
):
    """Per-galaxy density weights tied to the ``sigma^2(R)`` estimator.

    For each data galaxy ``i`` and a window scale ``R``, the
    window-aware Davis-Peebles per-particle overdensity is the
    smooth, kernel-weighted analogue of
    ``weights_pair_counts.per_particle_overdensity_windowed``:

        delta_i(R) = (b_DD_K_i * N_R) / (b_DR_K_i * N_D) - 1,

    with

        b_DD_K_i = sum_{j in D, j != i} K(|x_i - x_j|; R),
        b_DR_K_i = sum_{j in R}         K(|x_i - x_j|; R).

    The DD-only estimator that *fully subsumes the random catalogue*
    is then the per-particle mean

        sigma^2(R) = <delta_i>_i  (to within Poisson noise),

    which equals the LS sigma^2(R) up to one-percent-level
    differences from the average-of-ratios versus
    ratio-of-averages (see Hamilton 1993). The exact global form
    (DP, ratio-of-averages) is

        sigma^2_DP(R) = (sum_i b_DD_K_i * N_R) / (sum_i b_DR_K_i * N_D) - 1,

    which is mathematically identical to the LS DP estimator on the
    same pair counts.

    Why this is much smoother than the xi-based per-particle
    weights of ``weights_pair_counts.per_particle_overdensity_windowed``:
      - the kernel ``K_TH(r; R)`` integrates over ALL partners
        within ``r <= 2R`` weighted by the smooth overlap profile,
        not by an integer indicator function on a hard radial bin;
      - one R produces one weight per particle, so no ad-hoc
        bin-aggregation choice is needed;
      - on a clustered catalogue the kernel includes more partners
        per data point (typically 10-1000 vs. ~ a few inside one
        narrow bin), giving sqrt(N_partners) Poisson noise
        reduction.

    Returns
    -------
    w : (N_D,) per-galaxy weight ``w_i = 1 + delta_i(R)``.
    delta : (N_D,) raw per-particle overdensity. The mean
            ``<delta_i>`` is the DD-only sigma^2(R) estimate.
    aux : dict with diagnostic counts ``b_DD_K``, ``b_DR_K``, mean
          partners, etc.
    """
    pos_d = np.asarray(positions_d, dtype=np.float64)
    pos_r = np.asarray(positions_r, dtype=np.float64)
    N_d = len(pos_d); N_r = len(pos_r)
    b_DD = per_particle_kernel_counts(pos_d, pos_d, R, kernel=kernel,
                                          auto=True)
    b_DR = per_particle_kernel_counts(pos_d, pos_r, R, kernel=kernel,
                                          auto=False)
    eps = 1e-30
    delta = (b_DD * N_r) / (np.maximum(b_DR, eps) * N_d) - 1.0
    # zero-out particles with no random partners (genuinely outside window)
    delta = np.where(b_DR > 0, delta, 0.0)
    w = 1.0 + delta
    aux = {
        "R": float(R),
        "kernel": kernel,
        "b_DD_K_mean": float(np.mean(b_DD)),
        "b_DR_K_mean": float(np.mean(b_DR)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
    }
    return w, delta, aux


def sigma2_from_data_only(
    positions_d: np.ndarray, weights: np.ndarray,
    R_grid: np.ndarray,
    sigma2_RR_norm: np.ndarray,
    kernel: str = "tophat",
) -> np.ndarray:
    """``sigma^2(R)`` from a single weighted DD pass -- no random
    catalogue in the hot loop.

    Given per-galaxy weights ``w_i`` (e.g. from
    ``density_weights_sigma2`` at a representative ``R*``) and a
    pre-computed kernel-projected analytic-RR
    ``sigma2_RR_norm(R) = sum_RR K(r; R) / (N_R (N_R-1)/2)``, compute

        sigma^2_K(R) = (Sum_DD_w K(r; R) / W2 - sigma2_RR_norm(R))
                          / sigma2_RR_norm(R),

    where ``W2 = (sum_i w_i)^2 - sum_i w_i^2`` is the weighted
    unordered-pair count. With weights set to 1 this reduces to the
    Davis-Peebles natural form ``DD_K / RR_K - 1``.

    Returns ``(n_R,)`` array of ``sigma^2_K(R)``.
    """
    pos = np.asarray(positions_d, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    R_grid = np.asarray(R_grid, dtype=np.float64).ravel()
    N = len(pos)
    sum_w = float(w.sum()); sum_w2 = float(np.sum(w ** 2))
    W_pairs = 0.5 * (sum_w ** 2 - sum_w2)
    if W_pairs <= 0:
        raise ValueError("weighted unordered pair count is non-positive")
    out = np.zeros_like(R_grid)
    K_fn = _kernel_for(kernel)
    for i, R in enumerate(R_grid):
        # weighted DD kernel sum
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        rmax = 2.0 * R if kernel == "tophat" else 6.0 * R
        S_DD = 0.0
        chunk = 4000
        for start in range(0, N, chunk):
            stop = min(start + chunk, N)
            block = pos[start:stop]
            nbrs = tree.query_ball_point(block, r=rmax)
            for k_local, lst in enumerate(nbrs):
                k = start + k_local
                if not lst:
                    continue
                j = np.asarray(lst, dtype=np.int64)
                j = j[j > k]
                if j.size == 0:
                    continue
                d = np.linalg.norm(pos[j] - pos[k], axis=1)
                S_DD += float(np.sum(w[k] * w[j] * K_fn(d, float(R))))
        DD_norm = S_DD / W_pairs
        rrn = float(sigma2_RR_norm[i]) if i < len(sigma2_RR_norm) else 0.0
        if rrn > 0:
            out[i] = (DD_norm - rrn) / rrn
        else:
            out[i] = 0.0
    return out
