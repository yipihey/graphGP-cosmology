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


def kernel_TH_derivative_3d(r: np.ndarray, R: float) -> np.ndarray:
    """Analytic derivative ``partial K_TH(r; R) / partial R``.

    Differentiating the top-hat sphere kernel with respect to ``R``
    yields the kernel that, applied to LS pair counts, returns
    ``d sigma^2(R) / dR`` exactly:

        d K_TH(r; R) / d R
            = -(9 / (4 pi R^4)) * [1 - r/R + (r/R)^3 / 8]

    for ``r <= 2 R`` and 0 else. The boundary term vanishes because
    ``K_TH(r = 2R; R) = 0``. Note the kernel is *bipolar*: positive
    at intermediate ``r`` (where shrinking ``R`` removes partners)
    and negative at small ``r`` (where shrinking ``R`` retains the
    same partners but renormalises the volume). The integral
    ``int dK/dR dV = 0`` by definition of the normalised
    parent kernel.
    """
    r = np.asarray(r, dtype=np.float64)
    x = r / R
    g = 1.0 - x + (x ** 3) / 8.0
    K = np.where(x <= 2.0, -9.0 / (4.0 * np.pi * R ** 4) * g, 0.0)
    return K


def _two_sphere_overlap_volume(r: np.ndarray, R_a: float,
                                  R_b: float) -> np.ndarray:
    """Volume of intersection of two solid spheres of radii ``R_a, R_b``
    at centre-to-centre separation ``r``. Three cases:

        r >= R_a + R_b               -> 0 (no intersection)
        r <= |R_a - R_b|             -> (4/3) pi min(R_a, R_b)^3
        |R_a - R_b| < r < R_a + R_b  -> standard lens formula

    The lens formula (e.g. https://mathworld.wolfram.com/Sphere-
    SphereIntersection.html) for the overlap volume is

        V = pi (R_a + R_b - r)^2 [r^2 + 2 r (R_a + R_b)
                                    - 3 (R_a - R_b)^2] / (12 r).
    """
    r = np.asarray(r, dtype=np.float64)
    sum_R = float(R_a) + float(R_b)
    diff_R = abs(float(R_a) - float(R_b))
    out = np.zeros_like(r)
    inside = r <= diff_R
    out = np.where(inside,
                     (4.0 * np.pi / 3.0) * (min(R_a, R_b) ** 3),
                     out)
    overlap = (r > diff_R) & (r < sum_R)
    rr = np.where(overlap, r, 1.0)         # avoid div0; masked below
    lens = (np.pi * (sum_R - rr) ** 2
              * (rr ** 2 + 2.0 * rr * sum_R - 3.0 * diff_R ** 2)
              / (12.0 * rr))
    out = np.where(overlap, lens, out)
    return out


def kernel_shell_3d(r: np.ndarray, R_inner: float,
                     R_outer: float) -> np.ndarray:
    """Thick-spherical-shell overlap kernel, normalised so
    ``int K_shell(r; R_in, R_out) dV = 1``.

    A thick shell of inner radius ``R_in`` and outer radius ``R_out``
    is the set difference of two top-hat spheres, with volume
    ``V_shell = (4 pi / 3) (R_out^3 - R_in^3)``. The kernel is the
    auto-correlation of the indicator function of the shell, divided
    by ``V_shell^2``::

        K_shell(r) = (V_oo(r) - 2 V_io(r) + V_ii(r)) / V_shell^2

    where ``V_ab(r)`` is the overlap volume of two solid spheres of
    radii ``R_a, R_b`` at separation ``r``. Always non-negative, and
    in the thin-shell limit (``R_out - R_in << R``) the kernel
    approaches a delta function at ``r = 0`` -- the natural
    "variance of count in a shell" probe.

    Apply this kernel to LS pair counts to recover the
    spherical-shell two-point variance ``sigma^2_shell``, the
    physical analogue of ``d sigma^2 / dR``. With normalisation
    ``1 / (R_out - R_in)`` and ``R_out - R_in -> 0`` it converges to
    a derivative-kernel-based estimator of ``d sigma^2 / dR``.
    """
    r = np.asarray(r, dtype=np.float64)
    R1 = float(R_inner); R2 = float(R_outer)
    if R2 <= R1:
        raise ValueError("R_outer must be > R_inner")
    V_shell = (4.0 * np.pi / 3.0) * (R2 ** 3 - R1 ** 3)
    V22 = _two_sphere_overlap_volume(r, R2, R2)
    V12 = _two_sphere_overlap_volume(r, R1, R2)
    V11 = _two_sphere_overlap_volume(r, R1, R1)
    return (V22 - 2.0 * V12 + V11) / (V_shell ** 2)


def _kernel_for(kernel: str):
    if kernel == "tophat":
        return kernel_TH_3d
    if kernel in {"gauss", "gaussian"}:
        return kernel_Gauss_3d
    if kernel in {"derivative", "dkdr"}:
        return kernel_TH_derivative_3d
    raise ValueError(f"unknown kernel {kernel!r}")


def dsigma2_dR_from_xi(
    r_grid: np.ndarray, xi: np.ndarray, R_grid: np.ndarray,
) -> np.ndarray:
    """``d sigma^2(R) / d R`` by analytic-derivative-kernel projection
    of an existing ``xi(r)`` curve. Uses ``kernel_TH_derivative_3d``
    so the result is the *exact* derivative of ``sigma2_from_xi`` --
    no finite-difference noise.
    """
    r = np.asarray(r_grid, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    R_grid = np.asarray(R_grid, dtype=np.float64).ravel()
    out = np.zeros_like(R_grid)
    for i, R in enumerate(R_grid):
        K = kernel_TH_derivative_3d(r, float(R))
        out[i] = np.trapezoid(4.0 * np.pi * r ** 2 * xi * K, r)
    return out


def sigma2_shell_from_xi(
    r_grid: np.ndarray, xi: np.ndarray, R_inner: float, R_outer: float,
) -> float:
    """``sigma^2_shell(R_in, R_out)`` -- variance of count in a thick
    spherical shell, by ``kernel_shell_3d`` projection of ``xi(r)``.
    Always non-negative when ``xi >= 0``."""
    r = np.asarray(r_grid, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    K = kernel_shell_3d(r, float(R_inner), float(R_outer))
    return float(np.trapezoid(4.0 * np.pi * r ** 2 * xi * K, r))


# ----------------------------------------------------------------------
#  JAX-pure forward model: sigma^2(R, z, cosmo, b, sigma_8) and
#  dsigma^2/dR from the syren-halofit P_NL(k, z) via Fourier
#  representation of the top-hat sphere window.
# ----------------------------------------------------------------------


def _W_TH_kR(kR):
    """Fourier transform of the 3D top-hat sphere window of radius R,
    evaluated at ``x = k R``::

        W_TH(x) = 3 (sin x - x cos x) / x^3.

    Returns ``W_TH(kR)``; the small-x limit is taped via Taylor:
    ``W_TH(x) -> 1 - x^2/10 + x^4/280`` for ``x -> 0``. JAX-safe.
    """
    import jax.numpy as jnp

    x = jnp.asarray(kR, dtype=jnp.float64)
    eps = 1e-3
    safe = jnp.where(jnp.abs(x) > eps, x, jnp.where(x >= 0, eps, -eps))
    big = 3.0 * (jnp.sin(safe) - safe * jnp.cos(safe)) / (safe ** 3)
    small = 1.0 - x ** 2 / 10.0 + (x ** 4) / 280.0
    return jnp.where(jnp.abs(x) > eps, big, small)


def _dW_TH_kR_dR(k, R):
    """Derivative of the top-hat window w.r.t. R, evaluated at
    ``(k, R)``. Useful for ``d sigma^2 / d R``::

        d/dR [W_TH(kR)] = 3/(kR)^3 [3 kR cos(kR)
                                    + ((kR)^2 - 3) sin(kR)] * k
    """
    import jax.numpy as jnp

    kj = jnp.asarray(k, dtype=jnp.float64)
    x = kj * R
    eps = 1e-3
    safe = jnp.where(jnp.abs(x) > eps, x, eps)
    # d/dx W_TH(x) = (9 cos x - 9 x sin x - 9 cos x + 3 x sin x) / x^4
    #              = 3 (3 sin x - 3 cos x / x) ... let me derive directly:
    # W_TH(x) = 3 (sin x - x cos x) / x^3
    # dW/dx   = [3 cos x dx + 3 x sin x dx - 3 cos x dx] / x^3
    #             - 9 (sin x - x cos x) / x^4
    #         = 3 x sin x / x^3 - 9 (sin x - x cos x) / x^4
    #         = (3 sin x) / x^2 - 9 sin x / x^4 + 9 cos x / x^3
    big = (3.0 * jnp.sin(safe) / safe ** 2
             - 9.0 * jnp.sin(safe) / safe ** 4
             + 9.0 * jnp.cos(safe) / safe ** 3)
    small = -x / 5.0 + x ** 3 / 70.0
    dW_dx = jnp.where(jnp.abs(x) > eps, big, small)
    return dW_dx * kj


def sigma2_predicted(
    R, z_eff: float, cosmo, bias: float = 1.0, sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    k_min: float = 1e-4, k_max: float = 1e2, n_k: int = 4096,
    nowiggle: bool = False,
):
    """JAX-pure sigma^2(R, z, cosmo, bias, sigma_8) from the syren-halofit
    P_NL(k, z) via the standard Fourier-space form::

        sigma^2(R) = (b^2 / (2 pi^2))  int dk  k^2  W_TH^2(kR)  P_NL(k, z).

    Uses ``twopt_density.limber.pnl_at_z`` (full-baryon halofit) by
    default; set ``nowiggle=True`` to use the EH zero-baryon
    ``pnl_at_z_nowiggle`` for BAO-template construction.

    Differentiable in ``(cosmo, bias, sigma8)``.

    Returns
    -------
    s2 : (n_R,) jax array of sigma^2 values.
    """
    import jax.numpy as jnp
    from .limber import pnl_at_z, pnl_at_z_nowiggle

    R_arr = jnp.atleast_1d(jnp.asarray(R, dtype=jnp.float64))
    k_np = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    k_grid = jnp.asarray(k_np)
    if nowiggle:
        Pk = pnl_at_z_nowiggle(k_grid, z=z_eff, sigma8=sigma8,
                                  cosmo=cosmo, Ob=Ob, ns=ns)
    else:
        Pk = pnl_at_z(k_grid, z=z_eff, sigma8=sigma8,
                        cosmo=cosmo, Ob=Ob, ns=ns)
    # vector form: integrand_{R,k} = k^2 W_TH^2(kR) P(k); trapezoid in k
    kR = k_grid[None, :] * R_arr[:, None]
    W = _W_TH_kR(kR)
    integrand = (k_grid[None, :] ** 2) * (W ** 2) * Pk[None, :]
    s2 = (bias ** 2) / (2.0 * jnp.pi ** 2) * jnp.trapezoid(
        integrand, k_grid, axis=1
    )
    return s2


def dsigma2_dR_predicted(
    R, z_eff: float, cosmo, bias: float = 1.0, sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    k_min: float = 1e-4, k_max: float = 1e2, n_k: int = 4096,
    nowiggle: bool = False,
):
    """JAX-pure ``d sigma^2 / dR`` from the same Fourier form::

        d sigma^2 / dR = (b^2 / pi^2)  int dk  k^2  W_TH(kR) (dW_TH/dR)
                                          P_NL(k, z).

    Differentiable in ``(cosmo, bias, sigma8)`` and returns the
    *exact* derivative of ``sigma2_predicted``.
    """
    import jax.numpy as jnp
    from .limber import pnl_at_z, pnl_at_z_nowiggle

    R_arr = jnp.atleast_1d(jnp.asarray(R, dtype=jnp.float64))
    k_np = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    k_grid = jnp.asarray(k_np)
    if nowiggle:
        Pk = pnl_at_z_nowiggle(k_grid, z=z_eff, sigma8=sigma8,
                                  cosmo=cosmo, Ob=Ob, ns=ns)
    else:
        Pk = pnl_at_z(k_grid, z=z_eff, sigma8=sigma8,
                        cosmo=cosmo, Ob=Ob, ns=ns)
    kR = k_grid[None, :] * R_arr[:, None]
    W = _W_TH_kR(kR)
    dW = _dW_TH_kR_dR(k_grid[None, :], R_arr[:, None])
    integrand = (k_grid[None, :] ** 2) * (2.0 * W * dW) * Pk[None, :]
    out = (bias ** 2) / (2.0 * jnp.pi ** 2) * jnp.trapezoid(
        integrand, k_grid, axis=1
    )
    return out


def sigma2_bao_template(
    R_grid, z_eff: float, cosmo, bias: float = 1.0, sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    derivative: bool = False, alpha: float = 1.0,
    k_min: float = 1e-4, k_max: float = 1e2, n_k: int = 4096,
):
    """BAO template ``T(R) = sigma^2_full(R/alpha) - sigma^2_nowiggle(R/alpha)``,
    optionally for ``d sigma^2 / dR`` rather than ``sigma^2``.

    Returns a numpy array (jax-array materialised to host) of shape
    ``(n_R,)`` ready for the matched filter.
    """
    import jax.numpy as jnp

    R_alpha = jnp.asarray(R_grid, dtype=jnp.float64) / float(alpha)
    fn = dsigma2_dR_predicted if derivative else sigma2_predicted
    s_full = fn(R_alpha, z_eff=z_eff, cosmo=cosmo, bias=bias,
                  sigma8=sigma8, Ob=Ob, ns=ns,
                  k_min=k_min, k_max=k_max, n_k=n_k, nowiggle=False)
    s_smooth = fn(R_alpha, z_eff=z_eff, cosmo=cosmo, bias=bias,
                    sigma8=sigma8, Ob=Ob, ns=ns,
                    k_min=k_min, k_max=k_max, n_k=n_k, nowiggle=True)
    return np.asarray(s_full - s_smooth, dtype=np.float64)


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


# ----------------------------------------------------------------------
#  Direct analytic-window sigma^2 RR (no (rp, pi) discretisation).
# ----------------------------------------------------------------------


def sigma2_RR_analytic_window(
    R_grid,
    mask: np.ndarray, nside: int,
    z_data: np.ndarray, cosmo,
    kernel: str = "tophat",
    n_chi_eff: int = 60,
    n_chi_bins: int = 80,
    kde_bandwidth: float = 0.05,
    n_rp: int = 60, n_pi: int = 80,
    pi_max: float = None,
    lmax: int = None,
    theta_max_rad: float = 0.5,
    taper: str = "hann",
    N_r: int = None,
):
    """Direct analytic-window ``sigma^2_RR(R)`` without going through
    binned ``(rp, pi)`` pair counts.

    Computes the per-pair-density form of the random-random kernel
    sum ``int d^3 x_1 d^3 x_2 rho_R(x_1) rho_R(x_2) K(|x_1 - x_2|; R)``
    by evaluating the same separable-window integrand as
    ``rr_analytic`` -- ``chi_eff^2 * xi_mask(rp/chi_eff)
    * n(chi_eff + pi/2) * n(chi_eff - pi/2)`` -- on a fine continuous
    ``(rp, pi)`` grid and projecting it through the 3D-separation
    kernel ``K(sqrt(rp^2 + pi^2); R)``.

    Equivalent to ``sigma2_from_rp_pi_pairs(rp_c, pi_c, RR_analytic, ...)``
    in the limit of fine binning, but skips the binning step entirely
    so the result is independent of the choice of ``(rp_edges,
    pi_edges)`` at the science-measurement stage.

    Parameters
    ----------
    R_grid : array of sphere radii [Mpc/h] at which to evaluate sigma^2.
    mask, nside : HEALPix angular completeness map.
    z_data : observed redshifts; used to build n(chi).
    cosmo : DistanceCosmo.
    kernel : "tophat", "gaussian", or "derivative".
    n_rp, n_pi : grid resolution for the continuous (rp, pi) integration.
    pi_max : truncation of the LOS integral [Mpc/h]; default ``2 max(R)``
             for top-hat (which has compact support) or
             ``6 max(R)`` for Gaussian.
    taper, lmax : passed through to ``angular_corr_from_mask``.

    Returns
    -------
    sigma2_RR : (n_R,) array, in *per-pair-density* units (matching
    the convention of ``rr_analytic``). To convert to a count, multiply
    by ``N_r * (N_r - 1) / 2`` for some chosen ``N_r``.
    """
    from .analytic_rr import angular_corr_from_mask, radial_pair_density_from_z

    R_grid = np.asarray(R_grid, dtype=np.float64).ravel()
    R_max = float(R_grid.max())
    if pi_max is None:
        pi_max = 2.0 * R_max if kernel == "tophat" else 6.0 * R_max

    # Use the same angular-mask correlation and radial pair-density as
    # rr_analytic so the discretisation-free sigma^2 matches the
    # binned (rp, pi) projection in the fine-bin limit.
    theta_grid, xi_mask = angular_corr_from_mask(
        mask, nside, lmax=lmax, theta_max_rad=theta_max_rad, taper=taper,
    )
    chi_grid, n_chi = radial_pair_density_from_z(
        z_data, cosmo, n_chi_bins=n_chi_bins, kde_bandwidth=kde_bandwidth,
    )
    chi_min = float(chi_grid[0]); chi_max = float(chi_grid[-1])
    f_sky = float(np.mean(mask))
    chi2_p = float(np.trapezoid(n_chi * chi_grid ** 2, chi_grid))

    # Continuous (rp, pi) grid for the projection. Top-hat: rp in [0, 2R_max].
    rp_grid = np.linspace(0.0, 2.0 * R_max, n_rp)
    pi_grid = np.linspace(0.0, pi_max, n_pi)
    K_fn = _kernel_for(kernel)
    # for each (rp, pi), the per-pair density kernel:
    #   2 pi rp * int dchi_eff chi_eff^2 xi_mask(rp/chi_eff)
    #                           n(chi_eff + pi/2) n(chi_eff - pi/2)
    # is the rr_analytic integrand. We evaluate this 2D table once.
    rr_table = np.zeros((n_rp, n_pi), dtype=np.float64)
    for j, pi in enumerate(pi_grid):
        chi_lo = chi_min + pi / 2.0
        chi_hi = chi_max - pi / 2.0
        if chi_hi <= chi_lo:
            continue
        chi_eff = np.linspace(chi_lo, chi_hi, n_chi_eff)
        for i, rp in enumerate(rp_grid):
            if rp <= 0:
                continue
            theta_at = rp / chi_eff
            xi_at = np.interp(theta_at, theta_grid, xi_mask,
                               left=xi_mask[0], right=0.0)
            n1 = np.interp(chi_eff + pi / 2.0, chi_grid, n_chi,
                            left=0, right=0)
            n2 = np.interp(chi_eff - pi / 2.0, chi_grid, n_chi,
                            left=0, right=0)
            rr_table[i, j] = 2.0 * np.pi * rp * np.trapezoid(
                chi_eff ** 2 * xi_at * n1 * n2, chi_eff,
            )
    rr_table *= 1.0 / (2.0 * f_sky ** 2 * chi2_p ** 2)

    # Project rr_table onto K_TH(s = sqrt(rp^2 + pi^2); R) for each R.
    s_grid = np.sqrt(rp_grid[:, None] ** 2 + pi_grid[None, :] ** 2)
    out = np.zeros_like(R_grid)
    for i, R in enumerate(R_grid):
        K_2d = K_fn(s_grid, float(R))
        # 2D trapezoid in (rp, pi); the 2 from |pi| -> pi positive
        # already absorbed into the rr_table normalisation.
        integrand = rr_table * K_2d
        out[i] = np.trapezoid(np.trapezoid(integrand, pi_grid, axis=1),
                                rp_grid)
    if N_r is not None:
        out = out * (N_r * N_r)
    return out


# ----------------------------------------------------------------------
#  sigma^2-style g-kappa cross via 2D angular top-hat smoothing.
# ----------------------------------------------------------------------


# Precomputed 2D-top-hat Fourier window via scipy.special.j1 (used as a
# lookup table inside JAX via jnp.interp). 8 k samples per unit out to
# x = 200 captures the slow oscillating tail well; W_disk -> 0 at large
# x via 1/x^(3/2) decay.
_W_DISK_X = np.linspace(0.0, 200.0, 8000)
import scipy.special as _sp_special
_W_DISK_VAL = np.where(
    _W_DISK_X > 1e-3,
    2.0 * _sp_special.j1(_W_DISK_X) / np.maximum(_W_DISK_X, 1e-12),
    1.0 - _W_DISK_X ** 2 / 8.0 + (_W_DISK_X ** 4) / 192.0,
)


def _W_TH_2d(x):
    """2D top-hat angular window in flat-sky Fourier:
    ``W_TH^{2D}(x = ell theta_R) = 2 J_1(x) / x``.

    Implemented via a precomputed ``scipy.special.j1`` lookup
    interpolated in JAX -- ``jax.scipy.special.bessel_jn`` is
    unstable for ``x > 0`` on this build.
    """
    import jax.numpy as jnp
    return jnp.interp(jnp.asarray(x, dtype=jnp.float64),
                         jnp.asarray(_W_DISK_X), jnp.asarray(_W_DISK_VAL))


def sigma2_gkappa_predicted(
    R, z_grid, dndz, b_z, cosmo,
    sigma8: float = 0.81, Ob: float = 0.049, ns: float = 0.965,
    z_star: float = 1090.0,
    chi_eff: float = None,
    ell_min: float = 1.0, ell_max: float = 5000.0, n_ell: int = 600,
    nowiggle: bool = False,
):
    """JAX-pure sigma^2_{g-kappa}(R), the variance of the cross
    product of the 2D-top-hat-smoothed galaxy and CMB-lensing
    convergence fields, smoothed with angular radius ``theta_R = R /
    chi_eff``.

    sigma^2_{g-kappa}(R) = int d ell ell / (2 pi)
                              C_ell^{g-kappa} W_TH^{2D}(ell theta_R)^2

    The Limber-projected cross spectrum ``C_ell^{g-kappa}`` is the same
    one used in ``twopt_density.lensing.cl_gkappa_limber``. The
    chi_eff for the angle conversion is taken as the Limber
    effective comoving distance: dndz-weighted mean chi.
    """
    import jax
    import jax.numpy as jnp
    from .lensing import cl_gkappa_limber, cl_gkappa_limber_nowiggle

    R_arr = jnp.atleast_1d(jnp.asarray(R, dtype=jnp.float64))
    ell = jnp.asarray(np.linspace(float(ell_min), float(ell_max), n_ell),
                         dtype=jnp.float64)
    if nowiggle:
        cl = cl_gkappa_limber_nowiggle(
            ell, z_grid, dndz, b_z, cosmo,
            sigma8=sigma8, Ob=Ob, ns=ns, z_star=z_star,
        )
    else:
        cl = cl_gkappa_limber(
            ell, z_grid, dndz, b_z, cosmo,
            sigma8=sigma8, Ob=Ob, ns=ns, z_star=z_star,
        )

    if chi_eff is None:
        from .distance import comoving_distance
        chi_z = comoving_distance(jnp.asarray(z_grid, dtype=jnp.float64),
                                     cosmo)
        nz = jnp.asarray(dndz, dtype=jnp.float64)
        nz = nz / jnp.trapezoid(nz, jnp.asarray(z_grid, dtype=jnp.float64))
        chi_eff = float(jnp.trapezoid(
            chi_z * nz, jnp.asarray(z_grid, dtype=jnp.float64)))

    theta_R = R_arr / float(chi_eff)
    # outer product ell x R
    x = ell[None, :] * theta_R[:, None]
    W2 = _W_TH_2d(x) ** 2
    integrand = ell[None, :] * cl[None, :] * W2 / (2.0 * jnp.pi)
    return jnp.trapezoid(integrand, ell, axis=1)


# ----------------------------------------------------------------------
#  Cylindrical sigma^2: photo-z LOS-smeared variance for Quaia-type data.
# ----------------------------------------------------------------------


def sigma2_cyl_predicted(
    R, pi_max,
    z_eff: float, cosmo,
    bias: float = 1.0, sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    sigma_chi: float = 0.0,
    k_perp_min: float = 1e-4, k_perp_max: float = 1e2, n_k_perp: int = 256,
    k_par_min: float = 1e-4, k_par_max: float = 1e2, n_k_par: int = 256,
    nowiggle: bool = False,
):
    """JAX-pure sigma^2 of the count in a cylinder of transverse
    radius ``R`` and half-length ``pi_max``, evaluated at ``z_eff``
    under the syren-halofit P_NL.

        sigma^2_cyl(R, pi_max) =
           b^2 / (2 pi^2)  int d k_perp k_perp  W_disk^2(k_perp R)
                         x int d k_par / (2 pi)  W_los^2(k_par pi_max)
                                                exp(-k_par^2 sigma_chi^2)
                                                P(k = sqrt(k_perp^2 + k_par^2), z)

    The disk window is ``W_disk(x) = 2 J_1(x)/x`` and the LOS top-hat
    is ``W_los(y) = sin(y)/y``. The optional ``sigma_chi`` Gaussian
    factor models per-galaxy photo-z LOS smear: with ``sigma_chi >
    0`` the cylinder count is convolved by a Gaussian along LOS
    before differentiation, multiplying ``W_los`` by
    ``exp(-k_par^2 sigma_chi^2 / 2)``.

    The cylinder-variance is the natural sigma^2 analogue for
    photometric data (Quaia): the LOS smear destroys 3D-sphere
    estimates but cylinders at scale R > sigma_chi still have well-
    defined clustering variance.
    """
    import jax
    import jax.numpy as jnp
    from .limber import pnl_at_z, pnl_at_z_nowiggle

    R_arr = jnp.atleast_1d(jnp.asarray(R, dtype=jnp.float64))
    k_perp_np = np.logspace(np.log10(k_perp_min), np.log10(k_perp_max),
                                n_k_perp)
    k_par_np = np.logspace(np.log10(k_par_min), np.log10(k_par_max),
                               n_k_par)
    kp = jnp.asarray(k_perp_np); kz = jnp.asarray(k_par_np)
    kp_grid = kp[None, :, None]                      # (1, n_kp, 1)
    kz_grid = kz[None, None, :]                      # (1, 1, n_kz)
    R_grid = R_arr[:, None, None]                     # (n_R, 1, 1)

    # disk window via scipy.j1 lookup (jax.scipy.special.bessel_jn unstable)
    x_disk = kp_grid * R_grid
    eps = 1e-3
    W_disk = _W_TH_2d(x_disk)

    # LOS top-hat with optional photo-z Gaussian smearing
    y_los = kz_grid * float(pi_max)
    safe_los = jnp.where(jnp.abs(y_los) > eps, y_los, eps)
    W_los_big = jnp.sin(safe_los) / safe_los
    W_los_small = 1.0 - y_los ** 2 / 6.0 + (y_los ** 4) / 120.0
    W_los = jnp.where(jnp.abs(y_los) > eps, W_los_big, W_los_small)
    if sigma_chi > 0.0:
        W_los = W_los * jnp.exp(-(kz_grid ** 2)
                                  * (float(sigma_chi) ** 2) / 2.0)

    # 2D power on the (k_perp, k_par) grid via P(|k|)
    k_mag = jnp.sqrt(kp_grid ** 2 + kz_grid ** 2)
    # We need P at every grid point; vectorised via evaluating P on a
    # 1D fine grid and interpolating.
    k_eval = jnp.asarray(np.logspace(-4.5, 2.5, 2048))
    if nowiggle:
        Pk1d = pnl_at_z_nowiggle(k_eval, z=z_eff, sigma8=sigma8,
                                    cosmo=cosmo, Ob=Ob, ns=ns)
    else:
        Pk1d = pnl_at_z(k_eval, z=z_eff, sigma8=sigma8,
                          cosmo=cosmo, Ob=Ob, ns=ns)
    # interpolate -- jnp.interp is 1D so we flatten + reshape
    Pk = jnp.interp(k_mag.ravel(), k_eval, Pk1d).reshape(k_mag.shape)

    # integrand: (b^2 / (2 pi^2)) * k_perp * W_disk^2 * W_los^2 * P
    integrand = (kp_grid * (W_disk ** 2) * (W_los ** 2) * Pk)

    # 2D trapezoid: integrate over k_par first, then k_perp
    int_kz = jnp.trapezoid(integrand, kz, axis=2)         # (n_R, n_kp)
    out = (bias ** 2) / (2.0 * jnp.pi ** 2) * jnp.trapezoid(
        int_kz, kp, axis=1
    )
    return out
