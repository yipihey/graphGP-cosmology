"""Joint cosmology + HOD MAP fit using scipy + jax.

The forward model is::

    xi_pred(s | theta) = b^2(theta_HOD, halocat) * xi_syren_NL(s | theta_cosmo)

where ``b^2(theta_HOD, halocat)`` is the *effective tracer bias* set by
the per-halo HOD weights through pair counts:

    b^2_eff = sum_pairs w_i w_k 1[r_ik in [s_min, s_max]]
            / sum_pairs 1[r_ik in [s_min, s_max]]

evaluated on a chosen calibration window (where the toy halocat / mock
clustering is well represented by linear bias). Computed once per HOD
draw via the existing ``DD`` and ``RR_per_bin`` from the frozen graph.

The loss is a chi^2 over a fixed s-grid (FFTLog interpolation):

    -2 log L = sum_s (xi_data(s) - xi_pred(s))^2 / sigma^2(s)

``jax.grad`` provides analytic gradients; ``scipy.optimize.minimize`` is
used as the optimizer (no extra deps). For real surveys a covariance
matrix from mocks would replace the diagonal sigma; this is a clean
diagonal-noise demo of the pipeline.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


jax.config.update("jax_enable_x64", True)


class FitResult(NamedTuple):
    theta: np.ndarray         # best-fit parameters
    loss: float               # final loss
    success: bool             # scipy convergence flag
    nfev: int                 # forward evaluations
    message: str              # scipy message


def map_fit(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: Sequence[float],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    method: str = "L-BFGS-B",
    tol: float = 1e-8,
    options: Optional[dict] = None,
) -> FitResult:
    """Minimise ``loss_fn(theta)`` via scipy + jax-jitted gradient.

    Parameters
    ----------
    loss_fn : callable
        Pure JAX function returning a scalar.
    theta0 : sequence of float
        Initial parameter vector.
    bounds : list of (lo, hi) pairs, optional
        Box constraints for ``L-BFGS-B``.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)

    @jax.jit
    def loss_and_grad(theta):
        v, g = jax.value_and_grad(loss_fn)(theta)
        return v, g

    def obj(theta_np):
        v, g = loss_and_grad(jnp.asarray(theta_np))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(
        obj, theta0, jac=True, method=method,
        bounds=bounds, tol=tol, options=options or {},
    )
    return FitResult(
        theta=np.asarray(res.x, dtype=np.float64),
        loss=float(res.fun),
        success=bool(res.success),
        nfev=int(res.nfev),
        message=str(res.message),
    )
