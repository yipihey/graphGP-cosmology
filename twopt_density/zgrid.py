"""Reference-redshift grid constructors for the kNN-CDF pipeline.

Note v4_1 Appendix A.2 advocates two complementary grid choices:

- **R-centered** (random queries on randoms) prefers a *coarse* grid
  because each reference defines an independent measurement and there
  is little gain from oversampling: ``decile_grid`` places references
  at the deciles of the data N(z), giving each measurement comparable
  shot-noise structure.

- **D-centered** (data queries on data) prefers a *fine* grid because
  each data galaxy carries its own redshift and we want to preserve
  the intrinsic resolution: ``quantile_edges`` carves the inner 80%
  of the data N(z) into 90 bins each containing ~1% of galaxies.

Both functions take the data N(z) sample and return either a single
1-D array of references (decile_grid) or bin-edge array (quantile_edges,
log1pz_grid), suitable for direct use as ``z_q_edges`` / ``z_n_edges``
in ``twopt_density.knn_cdf.joint_knn_cdf``.

The pipeline scripts ``demos/quaia_full_pipeline.py`` and
``demos/desi_full_pipeline.py`` select between grids via the
``PAPER_Z_GRID`` env var:

    log1pz       — current behaviour, uniform in log(1+z) (default)
    rdeciles     — N(z)-decile reference points for R-centered passes
    dquantiles   — 1%-quantile edges for D-centered passes
    both         — D-centered DD/DR get dquantiles, R-centered RD/RR get rdeciles
"""

from __future__ import annotations

import numpy as np


def log1pz_grid(z_min: float, z_max: float, n_shells: int) -> np.ndarray:
    """Uniform log(1+z) bin edges from ``z_min`` to ``z_max``.

    Returns ``(n_shells + 1,)`` array. Reproduces the legacy default
    used in the v3-era pipelines.
    """
    if z_max <= z_min:
        raise ValueError(f"need z_max > z_min, got [{z_min}, {z_max}]")
    return np.expm1(np.linspace(
        np.log1p(z_min), np.log1p(z_max), n_shells + 1))


def decile_grid(
    z_data: np.ndarray,
    n_deciles: int = 9,
    q_lo: float = 0.1,
    q_hi: float = 0.9,
) -> np.ndarray:
    """Reference redshifts at empirical N(z) deciles (note A.2,
    R-centered).

    Returns ``(n_deciles,)`` array of reference points placed at
    quantiles ``[q_lo, ..., q_hi]`` of the data N(z). The default
    {0.1, 0.2, ..., 0.9} matches note A.2's recipe.

    The pipeline uses each reference redshift as the centre of a
    *single* z-shell wide enough to capture local clustering. Edges
    halfway to the neighbours give each reference an independent,
    non-overlapping bin (caller's responsibility — this function
    returns reference points only).
    """
    z_data = np.asarray(z_data, dtype=np.float64)
    if z_data.size == 0:
        raise ValueError("decile_grid: z_data is empty")
    qs = np.linspace(q_lo, q_hi, n_deciles)
    return np.quantile(z_data, qs)


def decile_edges(
    z_data: np.ndarray,
    n_deciles: int = 9,
    q_lo: float = 0.1,
    q_hi: float = 0.9,
) -> np.ndarray:
    """Bin EDGES bracketing the decile reference points.

    The pipeline needs ``(n_deciles + 1,)`` edges to define z-shells
    around each decile centre. This helper places edges at the
    midpoints between adjacent decile centres, with the first/last
    edge extending to the ``q_lo/2`` / ``(1+q_hi)/2`` quantiles so
    the outer shells have similar widths to the interior ones.
    """
    centres = decile_grid(z_data, n_deciles=n_deciles,
                          q_lo=q_lo, q_hi=q_hi)
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    z_data = np.asarray(z_data, dtype=np.float64)
    lo = float(np.quantile(z_data, q_lo / 2))
    hi = float(np.quantile(z_data, (1.0 + q_hi) / 2))
    return np.concatenate([[lo], midpoints, [hi]])


def quantile_edges(
    z_data: np.ndarray,
    q_lo: float = 0.095,
    q_hi: float = 0.905,
    n_bins: int = 90,
) -> np.ndarray:
    """``(n_bins + 1,)`` edges at uniformly-spaced N(z) quantiles
    (note A.2, D-centered).

    The default ``q_lo=0.095, q_hi=0.905, n_bins=90`` reproduces the
    note's recipe: 91 edges at quantiles {0.095, 0.105, ..., 0.905}
    so each of the 90 inner bins contains ~1% of the data sample.
    The outer 5% on each end of N(z) is trimmed to suppress edge
    effects from the selection function.
    """
    z_data = np.asarray(z_data, dtype=np.float64)
    if z_data.size == 0:
        raise ValueError("quantile_edges: z_data is empty")
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError(
            f"need 0 <= q_lo < q_hi <= 1, got [{q_lo}, {q_hi}]")
    qs = np.linspace(q_lo, q_hi, n_bins + 1)
    return np.quantile(z_data, qs)


def construct_z_grid(
    spec: str,
    z_data: np.ndarray,
    z_min: float,
    z_max: float,
    n_shells: int,
) -> np.ndarray:
    """Dispatch to one of the grid constructors by name.

    Returns ``(n_shells + 1,)`` edge array suitable for use as
    ``z_q_edges`` / ``z_n_edges``. Used by the pipeline scripts to
    centralise PAPER_Z_GRID interpretation.

    Parameters
    ----------
    spec
        ``"log1pz"`` (default), ``"rdeciles"`` (decile-edge variant
        of A.2 R-centered, ``n_shells`` controls n_deciles),
        ``"dquantiles"`` (note A.2 D-centered, ``n_shells`` controls
        n_bins).
    z_data
        ``(N_d,)`` data redshift sample (used by the quantile-based
        constructors). Ignored by ``log1pz``.
    z_min, z_max
        Range used by ``log1pz``; ignored by quantile constructors
        (which derive their range from the data quantiles).
    n_shells
        Number of bins/shells.
    """
    spec = spec.lower()
    if spec == "log1pz":
        return log1pz_grid(z_min, z_max, n_shells)
    if spec in ("rdeciles", "decile", "deciles"):
        return decile_edges(z_data, n_deciles=n_shells)
    if spec in ("dquantiles", "quantile", "quantiles"):
        return quantile_edges(z_data, n_bins=n_shells)
    raise ValueError(
        f"unknown z-grid spec {spec!r}; "
        "expected log1pz | rdeciles | dquantiles")
