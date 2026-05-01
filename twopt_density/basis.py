"""Basis families for the SFH continuous-function estimator.

Each ``Basis`` exposes ``evaluate(r) -> (n_basis, len(r))`` and a ``support``
property used by ``basis_projection`` to clip Corrfunc bin contributions.

Three families are sketched here. The cubic-spline basis is the default.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline


@dataclass
class Basis:
    n_basis: int
    r_min: float
    r_max: float

    def evaluate(self, r: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def support(self) -> tuple[float, float]:
        return (self.r_min, self.r_max)


class CubicSplineBasis(Basis):
    """Log-spaced cubic B-spline basis on ``[r_min, r_max]``."""

    def __init__(self, n_basis: int = 12, r_min: float = 0.1, r_max: float = 200.0):
        super().__init__(n_basis=n_basis, r_min=r_min, r_max=r_max)
        # ``n_basis`` interior knots + 4 boundary clamps on each side.
        interior = np.logspace(np.log10(r_min), np.log10(r_max), n_basis)
        self._knots = np.r_[[r_min] * 3, interior, [r_max] * 3]

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros((self.n_basis, r.size))
        for a in range(self.n_basis):
            c = np.zeros(self.n_basis + 2)
            c[a] = 1.0
            spline = BSpline(self._knots, c, k=3, extrapolate=False)
            out[a] = np.nan_to_num(spline(r), nan=0.0)
        return out


class BesselBasis(Basis):
    """Spherical Bessel modes ``f_alpha(r) = j_0(k_alpha r)``.

    Coefficients ``theta_alpha`` directly estimate ``P(k_alpha)`` via
    the Hankel transform; this gives the doc's real+Fourier unification.
    """

    def __init__(self, k_grid: np.ndarray, r_min: float = 0.1, r_max: float = 200.0):
        super().__init__(n_basis=len(k_grid), r_min=r_min, r_max=r_max)
        self.k_grid = np.asarray(k_grid, dtype=np.float64)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        kr = np.outer(self.k_grid, r)
        with np.errstate(invalid="ignore", divide="ignore"):
            j0 = np.where(kr == 0.0, 1.0, np.sin(kr) / kr)
        return j0


class CompensatedBandpassBasis(Basis):
    """Polynomial-compensated wavelet-style kernels with int f r^2 dr = 0.

    Stub: dyadic levels indexed by ``ell``; the doc connects these to the
    variance cascade ``sigma^2(R_ell) = (1/4) sigma^2(R_{ell-1}) +
    (3/4) xi_bar_{ell-1}``. Full polynomial design left for the
    compensated-bandpass implementation step in the plan.
    """

    def __init__(self, n_levels: int = 6, r_min: float = 0.1, r_max: float = 200.0):
        super().__init__(n_basis=n_levels, r_min=r_min, r_max=r_max)
        self._scales = np.logspace(np.log10(r_min), np.log10(r_max), n_levels)

    def evaluate(self, r: np.ndarray) -> np.ndarray:  # pragma: no cover - stub
        raise NotImplementedError(
            "CompensatedBandpassBasis.evaluate is a stretch-goal stub; "
            "see IMPLEMENTATION_PLAN.md step 7."
        )
