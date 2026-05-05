"""Lightcone-native angular cone-shell variance ``sigma^2(theta; z, dz)``.

Forward model for the angular variance of the galaxy count in a
spherical cap of half-angle ``theta`` restricted to a redshift shell
``[z_min, z_max]``. Built directly on the existing Limber projection
``cl_gg_limber`` and the 2D top-hat angular Fourier window
``_W_TH_2d`` (both reused verbatim from this package).

The estimator paired with this forward model -- ``cone_shell_counts``
in ``sigma2_cone_shell_estimator.py`` -- counts galaxies in caps on
``(ra, dec, z)`` directly via ``healpy.query_disc``, never converting
to comoving Cartesian. The per-shell C_ell's depend only on the dN/dz
inside the shell and the survey-window-aware galaxy bias; the angular
top-hat smoothing converts C_ell to ``sigma^2(theta)`` exactly:

    sigma^2(theta_R; z_min, z_max)
        = int dell ell / (2 pi) * C_ell^{gg, shell} * |W_TH^{2D}(ell theta_R)|^2

Functions:

  sigma2_cone_shell_predicted(theta, z_min, z_max, z_grid, dndz, cosmo,
                                 bias=1.0, sigma8=0.81, ...)
      -> sigma^2(theta) for a single shell.

  sigma2_cone_shell_predicted_stack(theta, z_edges, z_grid, dndz, cosmo,
                                      ...)
      -> (n_shells, n_theta) stack across adjacent shells.

  dsigma2_dz_cone_shell_predicted(theta, z_edges, z_grid, dndz, cosmo,
                                     ...)
      -> central differences in log(1+z) across adjacent shells.

  sigma2_cone_shell_bao_template(theta, z_min, z_max, ..., alpha=1.0)
      -> sigma^2_full(theta/alpha) - sigma^2_nowiggle(theta/alpha).

  sigma2_cone_shell_decomposition(theta, z_centres, z_grid, dndz, cosmo,
                                     bias_z, sigma8, ...)
      -> (sigma2, dlns2_dlnz, contrib_bias, contrib_growth, contrib_geom)
         the explicit bias / growth / geometry split of the log-derivative
         (paper Eq. (10)).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .distance import DistanceCosmo


def _restrict_dndz_to_shell(z_grid, dndz, z_min: float, z_max: float):
    """Mask dN/dz to a redshift shell and renormalise. JAX-pure."""
    import jax.numpy as jnp
    z = jnp.asarray(z_grid, dtype=jnp.float64)
    nz = jnp.asarray(dndz, dtype=jnp.float64)
    inside = (z >= float(z_min)) & (z <= float(z_max))
    nz_shell = jnp.where(inside, nz, 0.0)
    norm = jnp.trapezoid(nz_shell, z)
    nz_shell = nz_shell / jnp.where(norm > 0.0, norm, 1.0)
    return nz_shell


def sigma2_cone_shell_predicted(
    theta,
    z_min: float, z_max: float,
    z_grid, dndz,
    cosmo: DistanceCosmo,
    bias=1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    ell_min: float = 1.0, ell_max: float = 5e4, n_ell: int = 800,
    nowiggle: bool = False,
):
    """JAX-pure ``sigma^2(theta; z_min, z_max)`` -- the angular variance
    of the galaxy count in a spherical cap of half-angle ``theta``
    restricted to the redshift shell ``[z_min, z_max]``.

    sigma^2(theta_R) = int dell ell / (2 pi) C_ell^{gg, shell}
                                              |W_TH^{2D}(ell theta_R)|^2

    The shell-restricted Limber kernel is built from the supplied
    full-survey ``dndz`` masked to ``[z_min, z_max]`` and renormalised
    inside the shell. ``bias`` is a constant linear bias over the shell.

    Parameters
    ----------
    theta : float or 1D array, cap half-angle in radians.
    z_min, z_max : shell edges.
    z_grid, dndz : the *full survey* dN/dz, sampled on ``z_grid``.
    cosmo : DistanceCosmo.
    bias, sigma8, Ob, ns : forward-model parameters.
    ell_min, ell_max, n_ell : ell quadrature.
    nowiggle : if True use the EH zero-baryon P_NL (BAO templates).

    Returns
    -------
    sigma2 : (n_theta,) jax array.
    """
    import jax.numpy as jnp
    from .limber import cl_gg_limber, cl_gg_limber_nowiggle
    from .sigma2 import _W_TH_2d

    theta_arr = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))
    nz_shell = _restrict_dndz_to_shell(z_grid, dndz, z_min, z_max)
    ell = jnp.asarray(
        np.linspace(float(ell_min), float(ell_max), int(n_ell)),
        dtype=jnp.float64,
    )
    fn = cl_gg_limber_nowiggle if nowiggle else cl_gg_limber
    cl = fn(ell, z_grid, nz_shell, cosmo,
              bias=bias, sigma8=sigma8, Ob=Ob, ns=ns)

    x = ell[None, :] * theta_arr[:, None]
    W2 = _W_TH_2d(x) ** 2
    integrand = ell[None, :] * cl[None, :] * W2 / (2.0 * jnp.pi)
    return jnp.trapezoid(integrand, ell, axis=1)


def sigma2_cone_shell_predicted_stack(
    theta,
    z_edges,
    z_grid, dndz,
    cosmo: DistanceCosmo,
    bias=1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    ell_min: float = 1.0, ell_max: float = 5e4, n_ell: int = 800,
    nowiggle: bool = False,
):
    """Stack of ``sigma^2(theta)`` across adjacent shells defined by
    ``z_edges`` (length ``n_shells + 1``). Returns shape
    ``(n_shells, n_theta)``. ``bias`` may be a scalar (constant across
    all shells) or an array of length ``n_shells``.

    Used as the building block for ``dsigma2_dz_cone_shell_predicted``
    and the joint shell-stacked likelihood.
    """
    import jax.numpy as jnp
    z_edges = np.asarray(z_edges, dtype=np.float64)
    n_shells = len(z_edges) - 1
    bias_arr = np.broadcast_to(np.asarray(bias, dtype=np.float64),
                                  (n_shells,))
    rows = []
    for i in range(n_shells):
        rows.append(sigma2_cone_shell_predicted(
            theta, float(z_edges[i]), float(z_edges[i + 1]),
            z_grid, dndz, cosmo,
            bias=float(bias_arr[i]), sigma8=sigma8, Ob=Ob, ns=ns,
            ell_min=ell_min, ell_max=ell_max, n_ell=n_ell,
            nowiggle=nowiggle,
        ))
    return jnp.stack(rows, axis=0)


def dsigma2_dz_cone_shell_predicted(
    theta,
    z_edges,
    z_grid, dndz,
    cosmo: DistanceCosmo,
    bias=1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    ell_min: float = 1.0, ell_max: float = 5e4, n_ell: int = 800,
    nowiggle: bool = False,
    log1pz: bool = True,
):
    """Adjacent-shell finite-difference derivative
    ``d sigma^2 / d ln(1+z)`` (default) or ``d sigma^2 / dz``
    (``log1pz=False``).

    Central differences across consecutive shells. The returned redshift
    pivots are the geometric centres of pairs of adjacent shells -- the
    natural bin for finite-difference operators on the lightcone.

    Returns
    -------
    z_centres : (n_pivots,) numpy array, redshift pivots.
    dsigma2  : (n_pivots, n_theta) jax array, the FD derivative.
    """
    import jax.numpy as jnp
    z_edges = np.asarray(z_edges, dtype=np.float64)
    s2 = sigma2_cone_shell_predicted_stack(
        theta, z_edges, z_grid, dndz, cosmo,
        bias=bias, sigma8=sigma8, Ob=Ob, ns=ns,
        ell_min=ell_min, ell_max=ell_max, n_ell=n_ell,
        nowiggle=nowiggle,
    )
    # shell centres (mean of edges)
    z_shell = 0.5 * (z_edges[:-1] + z_edges[1:])
    if log1pz:
        x = np.log(1.0 + z_shell)
    else:
        x = z_shell
    # central differences across adjacent shells
    dx = jnp.asarray(x[2:] - x[:-2], dtype=jnp.float64)
    ds = (s2[2:] - s2[:-2]) / dx[:, None]
    z_centres = z_shell[1:-1]
    return z_centres, ds


def sigma2_cone_shell_bao_template(
    theta,
    z_min: float, z_max: float,
    z_grid, dndz,
    cosmo: DistanceCosmo,
    bias=1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    alpha: float = 1.0,
    ell_min: float = 1.0, ell_max: float = 5e4, n_ell: int = 800,
):
    """BAO template
    ``T(theta) = sigma^2_full(theta/alpha) - sigma^2_nowiggle(theta/alpha)``
    for a single shell. ``alpha`` is the angular-BAO dilation parameter:
    a feature at angular scale ``theta_BAO`` in the smoothed signal
    appears at ``alpha * theta_BAO`` in the full signal.

    Returns a numpy array of shape ``(n_theta,)``.
    """
    import jax.numpy as jnp
    theta_a = jnp.asarray(theta, dtype=jnp.float64) / float(alpha)
    s_full = sigma2_cone_shell_predicted(
        theta_a, z_min, z_max, z_grid, dndz, cosmo,
        bias=bias, sigma8=sigma8, Ob=Ob, ns=ns,
        ell_min=ell_min, ell_max=ell_max, n_ell=n_ell, nowiggle=False,
    )
    s_smooth = sigma2_cone_shell_predicted(
        theta_a, z_min, z_max, z_grid, dndz, cosmo,
        bias=bias, sigma8=sigma8, Ob=Ob, ns=ns,
        ell_min=ell_min, ell_max=ell_max, n_ell=n_ell, nowiggle=True,
    )
    return np.asarray(s_full - s_smooth, dtype=np.float64)


def sigma2_cone_shell_decomposition(
    theta,
    z_edges,
    z_grid, dndz,
    cosmo: DistanceCosmo,
    bias_z=1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    ell_min: float = 1.0, ell_max: float = 5e4, n_ell: int = 800,
):
    """Bias / growth / geometry split of ``d ln sigma^2 / d ln(1+z)``.

    Paper Eq. (10) -- valid in the thin-shell limit::

        d ln sigma^2 / d ln(1+z) ~ 2 d ln b / d ln(1+z)
                                       + 2 d ln D / d ln(1+z)
                                       + n_eff(theta D_M) d ln D_M / d ln(1+z)

    where ``D(z)`` is the linear-growth factor,
    ``D_M(z) = chi(z)`` is the comoving angular-diameter distance
    (flat geometry), and ``n_eff = d ln sigma^2 / d ln theta`` is the
    local logarithmic slope of ``sigma^2(theta)`` (geometric/dilation
    coupling at fixed theta).

    For finite shell widths the equality is only approximate -- the
    Limber projection picks up additional ``d chi / d z`` weighting
    that this leading-order decomposition does not capture. The split
    is most useful as a *diagnostic trace*: which physical effect is
    driving the LHS log-derivative at a given (theta, z).

    Inputs
    ------
    theta : float or (n_theta,) array, the angular pivot(s).
    z_edges : 1D array of redshift edges; pivots are mid-shell.
    bias_z : either scalar (constant b) or callable ``b(z) -> b``.

    Returns
    -------
    z_centres : (n_pivots,) numpy array.
    dlns2_dlnz : (n_pivots, n_theta) numpy array (the LHS).
    contrib_bias : (n_pivots,) numpy array (term 1).
    contrib_growth : (n_pivots,) numpy array (term 2).
    contrib_geom : (n_pivots, n_theta) numpy array (term 3).
    """
    import jax.numpy as jnp
    from .distance import comoving_distance
    from .limber import linear_growth

    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    z_edges = np.asarray(z_edges, dtype=np.float64)
    z_shell = 0.5 * (z_edges[:-1] + z_edges[1:])
    z_centres = z_shell[1:-1]

    # bias as scalar or callable
    if callable(bias_z):
        b_shell = np.array([float(bias_z(z)) for z in z_shell],
                              dtype=np.float64)
    else:
        b_shell = np.full_like(z_shell, float(bias_z))

    # full sigma^2 stack
    s2 = np.asarray(sigma2_cone_shell_predicted_stack(
        theta_arr, z_edges, z_grid, dndz, cosmo,
        bias=b_shell, sigma8=sigma8, Ob=Ob, ns=ns,
        ell_min=ell_min, ell_max=ell_max, n_ell=n_ell,
    ), dtype=np.float64)
    # log-derivative LHS via central differences in ln(1+z)
    x = np.log(1.0 + z_shell)
    dx = (x[2:] - x[:-2])
    dlns2_dlnz = (np.log(s2[2:]) - np.log(s2[:-2])) / dx[:, None]

    # bias term: 2 d ln b / d ln(1+z)
    lnb = np.log(np.maximum(b_shell, 1e-12))
    dlnb_dlnz = (lnb[2:] - lnb[:-2]) / dx
    contrib_bias = 2.0 * dlnb_dlnz

    # growth term: 2 d ln D / d ln(1+z)
    D_shell = np.asarray(linear_growth(
        jnp.asarray(z_shell, dtype=jnp.float64), cosmo
    ), dtype=np.float64)
    lnD = np.log(np.maximum(D_shell, 1e-12))
    dlnD_dlnz = (lnD[2:] - lnD[:-2]) / dx
    contrib_growth = 2.0 * dlnD_dlnz

    # geometry term: n_eff(theta D_M(z)) * d ln D_M / d ln(1+z)
    # n_eff is the local log-slope of sigma^2(theta) at the central
    # shell (the one we are taking the derivative around).
    # Compute n_eff = d ln sigma^2 / d ln theta via central diff in
    # log theta, evaluated at z = z_centres for each theta pivot.
    log_theta = np.log(theta_arr)
    # need log-slope of sigma^2 in theta for the central shells.
    # s2[1:-1] are the centred shell values shape (n_centres, n_theta).
    s2_centre = s2[1:-1]
    # central log-theta differences (n_theta-2,)
    log_s2 = np.log(np.maximum(s2_centre, 1e-300))
    # for endpoints in theta use one-sided differences
    n_eff = np.zeros_like(s2_centre)
    n_eff[:, 1:-1] = (
        (log_s2[:, 2:] - log_s2[:, :-2])
        / (log_theta[2:] - log_theta[:-2])[None, :]
    )
    if theta_arr.size >= 2:
        n_eff[:, 0] = (log_s2[:, 1] - log_s2[:, 0]) / (
            log_theta[1] - log_theta[0]
        )
        n_eff[:, -1] = (log_s2[:, -1] - log_s2[:, -2]) / (
            log_theta[-1] - log_theta[-2]
        )

    # d ln D_M / d ln(1+z) at centred shell redshifts
    chi_shell = np.asarray(comoving_distance(
        jnp.asarray(z_shell, dtype=jnp.float64), cosmo
    ), dtype=np.float64)
    ln_chi = np.log(np.maximum(chi_shell, 1e-12))
    dlnchi_dlnz = (ln_chi[2:] - ln_chi[:-2]) / dx
    contrib_geom = n_eff * dlnchi_dlnz[:, None]

    return z_centres, dlns2_dlnz, contrib_bias, contrib_growth, contrib_geom


def sigma2_cone_shell_gaussian_covariance(
    theta_radii_rad,
    z_edges,
    z_grid, dndz,
    cosmo: DistanceCosmo,
    bias=1.0,
    sigma8: float = 0.81,
    Ob: float = 0.049, ns: float = 0.965,
    f_sky: float = 1.0,
    n_bar_per_steradian=None,
    ell_min: float = 1.0, ell_max: float = 5e4, n_ell: int = 800,
):
    """Analytic cosmic-variance + shot-noise Gaussian covariance of
    ``sigma^2(theta; z, dz)`` across the (theta, z) grid.

    Under the Gaussian-field approximation, the per-shell estimator
    ``sigma^2(theta) = int dell ell/(2 pi) C_ell W_TH^{2D}(ell theta)^2``
    has covariance::

        Cov[sigma^2_a, sigma^2_b]
            = (2 / (4 pi f_sky)) integral dell (2 ell + 1)
                                          [C_ell^obs]^2 W_a^2(ell) W_b^2(ell)

    where ``C_ell^obs = C_ell + 1/n_bar`` includes the shot-noise
    contribution and ``W_a(ell) = W_TH^{2D}(ell theta_a)``. Different
    redshift shells are uncorrelated in the Gaussian limit (they are
    independent angular projections), so the matrix is block-diagonal
    in the shell index.

    Parameters
    ----------
    theta_radii_rad : (n_theta,) ascending cap half-angles [rad].
    z_edges : (n_zshell + 1,) shell edges.
    z_grid, dndz : full-survey ``dN/dz`` to be restricted per shell.
    cosmo, bias, sigma8, Ob, ns : forward-model parameters; ``bias``
        may be a scalar or a length-``n_zshell`` array.
    f_sky : effective fractional sky coverage (in [0, 1]). Default 1.
    n_bar_per_steradian : optional shot-noise. Either ``None`` (no
        shot noise -- cosmic-variance-only diagonal), a scalar (same
        for all shells), or a length-``n_zshell`` array.
    ell_min, ell_max, n_ell : ell quadrature.

    Returns
    -------
    cov : ``(n_theta * n_zshell, n_theta * n_zshell)`` covariance.
        Layout: ``cov[i_theta * n_zshell + k_z, j_theta * n_zshell +
        l_z]`` matches the ``s2.flatten()`` order used by
        ``sigma2_estimate_cone_shell`` returns. Block-diagonal in
        ``(k_z, l_z)``.
    """
    import jax.numpy as jnp
    from .limber import cl_gg_limber
    from .sigma2 import _W_TH_2d

    theta_radii_rad = np.asarray(theta_radii_rad, dtype=np.float64)
    z_edges = np.asarray(z_edges, dtype=np.float64)
    n_theta = theta_radii_rad.size
    n_zshell = z_edges.size - 1
    n_total = n_theta * n_zshell

    bias_arr = np.broadcast_to(np.asarray(bias, dtype=np.float64),
                                  (n_zshell,))
    if n_bar_per_steradian is None:
        nbar_arr = np.full(n_zshell, np.inf)         # no shot noise
    else:
        nbar_arr = np.broadcast_to(
            np.asarray(n_bar_per_steradian, dtype=np.float64),
            (n_zshell,),
        )

    ell = jnp.asarray(
        np.linspace(float(ell_min), float(ell_max), int(n_ell)),
        dtype=jnp.float64,
    )
    # 2D top-hat window squared for each theta: (n_theta, n_ell)
    W2 = np.asarray(
        _W_TH_2d(ell[None, :] * jnp.asarray(theta_radii_rad)[:, None]) ** 2,
        dtype=np.float64,
    )

    cov = np.zeros((n_total, n_total), dtype=np.float64)
    ell_np = np.asarray(ell, dtype=np.float64)
    for k in range(n_zshell):
        nz_shell = _restrict_dndz_to_shell(
            z_grid, dndz, float(z_edges[k]), float(z_edges[k + 1]),
        )
        cl = np.asarray(
            cl_gg_limber(ell, z_grid, nz_shell, cosmo,
                          bias=float(bias_arr[k]),
                          sigma8=sigma8, Ob=Ob, ns=ns),
            dtype=np.float64,
        )
        cl_obs = cl + 1.0 / nbar_arr[k]                 # add shot noise
        # mode-density factor (2 ell + 1) / (4 pi f_sky)
        prefactor = (2.0 * ell_np + 1.0) / (4.0 * np.pi * float(f_sky))
        # Cov[a, b] within this shell:
        #   integrate over ell of prefactor * cl_obs^2 * W_a^2 * W_b^2
        cl_obs_sq = cl_obs ** 2
        # block: (n_theta, n_theta)
        block = np.zeros((n_theta, n_theta), dtype=np.float64)
        for i in range(n_theta):
            for j in range(i, n_theta):
                integrand = prefactor * cl_obs_sq * W2[i] * W2[j]
                val = 2.0 * float(np.trapezoid(integrand, ell_np))
                block[i, j] = val
                block[j, i] = val
        # write block at (k, k) of the (n_theta * n_zshell)^2 matrix
        # using the same flatten convention as sigma^2 grids:
        # flat_index = i_theta * n_zshell + k_z (matches ndarray.ravel
        # of (n_theta, n_zshell)).
        for i in range(n_theta):
            for j in range(n_theta):
                cov[i * n_zshell + k, j * n_zshell + k] = block[i, j]
    return cov
