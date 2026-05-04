"""JAX-native HOD (Halo Occupation Distribution) for use as per-halo weights.

Treat a halo's expected galaxy count <N_gal>(M_h | theta_HOD) as a
differentiable per-halo weight that feeds straight into the LISA / SF&H
machinery. No Monte-Carlo sampling needed for inference: the expected
occupation IS the weight, gradients flow cleanly through theta_HOD.

Currently implemented::

    Zheng07 (zheng07, leauthaud-style threshold sample, arXiv:0703457)
        <N_cen>(M) = 0.5 [1 + erf((log M - log Mmin) / sigma_logM)]
        <N_sat>(M) = ((M - M0)/M1)^alpha          for M > M0
                    0                              otherwise
        Optionally modulate <N_sat> by <N_cen>.

Mass column convention follows halotools (``halo_mvir`` etc). ``M`` here
is a linear mass in Msun; pass log10 mass through ``10 ** logm``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


jax.config.update("jax_enable_x64", True)


class Zheng07Params(NamedTuple):
    """Five Zheng+ (2007) HOD parameters. Defaults are halotools' threshold=-21."""
    logMmin: float = 12.79
    sigma_logM: float = 0.39
    logM0: float = 11.92
    logM1: float = 13.94
    alpha: float = 1.15


def mean_ncen_zheng07(M: jnp.ndarray, p: Zheng07Params) -> jnp.ndarray:
    """Expected number of central galaxies per halo, eq. 2 of Zheng+ 2007."""
    logM = jnp.log10(M)
    return 0.5 * (1.0 + erf((logM - p.logMmin) / p.sigma_logM))


def mean_nsat_zheng07(
    M: jnp.ndarray,
    p: Zheng07Params,
    modulate_with_ncen: bool = True,
) -> jnp.ndarray:
    """Expected number of satellite galaxies per halo, eq. 5 of Zheng+ 2007.

    ``modulate_with_ncen=True`` matches halotools' default (multiply by
    <N_cen>) -- ensures satellites only attach to halos that already host
    a central, which the inference literature usually wants.
    """
    M0 = 10.0 ** p.logM0
    M1 = 10.0 ** p.logM1
    arg = jnp.maximum(M - M0, 0.0) / M1
    # alpha is a positive real; jnp.where avoids NaN gradient at arg=0
    nsat = jnp.where(arg > 0, arg ** p.alpha, 0.0)
    if modulate_with_ncen:
        nsat = nsat * mean_ncen_zheng07(M, p)
    return nsat


def mean_ngal_zheng07(
    M: jnp.ndarray,
    p: Zheng07Params,
    modulate_with_ncen: bool = True,
) -> jnp.ndarray:
    """Total expected galaxy count: <N_cen> + <N_sat>."""
    return mean_ncen_zheng07(M, p) + mean_nsat_zheng07(
        M, p, modulate_with_ncen=modulate_with_ncen,
    )
