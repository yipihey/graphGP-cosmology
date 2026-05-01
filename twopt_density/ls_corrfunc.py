"""Landy-Szalay two-point estimator backed by Corrfunc.

Replaces the O(N^2) ``graphGP_cosmo.compute_two_point_function`` for catalogs
where Corrfunc is available. Falls back to the existing scipy-based
implementation otherwise.

The function returns ``RR(r)`` alongside ``xi(r)`` because the binned
weight-assignment system (Eq. 2 of ``twopt_density.pdf``) is written in terms
of ``RR_j``, not just ``xi_j``.
"""

from __future__ import annotations

import numpy as np

try:
    from Corrfunc.theory.DD import DD as _corrfunc_DD
    from Corrfunc.utils import convert_3d_counts_to_cf as _to_cf
    _HAS_CORRFUNC = True
except ImportError:
    _HAS_CORRFUNC = False


def _shell_volumes(r_edges: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)


def xi_landy_szalay(
    positions: np.ndarray,
    randoms: np.ndarray | None = None,
    r_edges: np.ndarray | None = None,
    box_size: float | None = None,
    nthreads: int = 4,
    weights: np.ndarray | None = None,
):
    """Compute the LS two-point correlation function.

    Parameters
    ----------
    positions
        ``(N_D, 3)`` data positions.
    randoms
        ``(N_R, 3)`` random catalog. If ``None`` and ``box_size`` is given,
        ``RR`` and ``DR`` are computed analytically (uniform random in a
        periodic box).
    r_edges
        Bin edges in Mpc/h. Default: 25 log-spaced bins per decade from
        0.1 to 200 Mpc/h.
    box_size
        Periodic box length in Mpc/h. ``None`` for a survey geometry.
    nthreads
        OpenMP threads passed to Corrfunc.
    weights
        Optional ``(N_D,)`` per-point weights. Pair counts use the
        ``pair_product`` weighting -- this is what the validation
        ``weighted_DD`` check needs.

    Returns
    -------
    r_centers, xi_j, RR_j, DD_j, DR_j
        ``DR_j`` is ``RR_j`` when ``randoms is None``.
    """
    if r_edges is None:
        r_edges = np.logspace(np.log10(0.1), np.log10(200.0), 76)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    if not _HAS_CORRFUNC:
        # Fallback path -- documented in the plan, intentionally minimal.
        from graphGP_cosmo import compute_two_point_function
        n_bins = len(r_edges) - 1
        rc, xi, _ = compute_two_point_function(
            positions, n_bins=n_bins, r_max=float(r_edges[-1]),
            box_size=box_size,
        )
        N = len(positions)
        V = box_size ** 3 if box_size else 1.0
        RR = 0.5 * N * (N - 1) * _shell_volumes(r_edges) / V
        DD = (1.0 + xi) * RR
        return rc, xi, RR, DD, RR

    x, y, z = positions.T.astype(np.float64)
    N_D = len(positions)

    DD_kw = dict(
        autocorr=1, nthreads=nthreads, binfile=r_edges,
        X1=x, Y1=y, Z1=z,
        periodic=bool(box_size), boxsize=box_size or 0.0,
    )
    if weights is not None:
        DD_kw.update(weights1=weights.astype(np.float64),
                     weight_type="pair_product")
    dd_res = _corrfunc_DD(**DD_kw)
    DD_j = dd_res["npairs"].astype(np.float64)
    if weights is not None:
        DD_j = dd_res["weightavg"] * DD_j  # sum of w_i*w_k per bin

    if randoms is None:
        if box_size is None:
            raise ValueError("Either ``randoms`` or ``box_size`` must be given.")
        V = box_size ** 3
        RR_j = N_D * (N_D - 1) * _shell_volumes(r_edges) / V
        DR_j = RR_j  # analytically equal in the periodic-box limit
        # Eq. (1): xi = DD/RR - 1 when DR == RR.
        xi_j = np.where(RR_j > 0, DD_j / RR_j - 1.0, 0.0)
        return r_centers, xi_j, RR_j, DD_j, DR_j

    xR, yR, zR = randoms.T.astype(np.float64)
    N_R = len(randoms)
    DR_j = _corrfunc_DD(
        autocorr=0, nthreads=nthreads, binfile=r_edges,
        X1=x, Y1=y, Z1=z, X2=xR, Y2=yR, Z2=zR,
        periodic=bool(box_size), boxsize=box_size or 0.0,
    )["npairs"].astype(np.float64)
    RR_j = _corrfunc_DD(
        autocorr=1, nthreads=nthreads, binfile=r_edges,
        X1=xR, Y1=yR, Z1=zR,
        periodic=bool(box_size), boxsize=box_size or 0.0,
    )["npairs"].astype(np.float64)

    xi_j = _to_cf(N_D, N_D, N_R, N_R, DD_j, DR_j, DR_j, RR_j)
    return r_centers, xi_j, RR_j, DD_j, DR_j


def local_mean_density(
    positions: np.ndarray,
    randoms: np.ndarray | None,
    box_size: float | None = None,
    h: float | None = None,
) -> np.ndarray:
    """Estimate ``nbar(x_i)`` at each data point from the random catalog.

    Periodic-box survey: returns a constant ``N_D / L^3`` array.
    Survey geometry: top-hat KDE on ``randoms`` of bandwidth ``h``
    (default: 5x mean random separation).
    """
    N_D = len(positions)
    if randoms is None:
        if box_size is None:
            raise ValueError("Need either ``randoms`` or ``box_size``.")
        return np.full(N_D, N_D / box_size ** 3, dtype=np.float64)

    from scipy.spatial import cKDTree

    if h is None:
        V = float(np.prod(randoms.max(0) - randoms.min(0)))
        h = 5.0 * (V / len(randoms)) ** (1.0 / 3.0)

    tree = cKDTree(randoms, boxsize=box_size if box_size else None)
    counts = tree.query_ball_point(positions, r=h, return_length=True)
    shell_V = (4.0 / 3.0) * np.pi * h ** 3
    # Convert random density to expected data density via N_D / N_R rescaling.
    n_R_local = np.asarray(counts) / shell_V
    return n_R_local * (N_D / len(randoms))
