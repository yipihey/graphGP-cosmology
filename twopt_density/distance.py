"""JAX cosmological distance: (RA, Dec, z) -> comoving (x, y, z).

For a flat (w0, wa) cosmology with massless neutrinos, the dimensionless
Hubble parameter is

    E^2(z) = Om (1+z)^3 + (1-Om) * (1+z)^{3(1+w0+wa)} * exp(-3 wa z/(1+z))

(Linder 2003 parameterisation). The comoving distance is

    D_C(z) = c/H0 integral_0^z dz' / E(z'),  c/H0 = 2997.92458 / h Mpc

evaluated via JAX cumulative-trapezoid on a dense, fixed redshift grid;
the per-galaxy distance is then a 1D linear interpolation at each
sample's z. The (RA, Dec) angular-position part is the standard
spherical-to-Cartesian transform. Differentiable end-to-end -- ``jax.grad``
flows from each galaxy's comoving (x, y, z) back to (Om, w0, wa, h).

Output convention: ``c/H0`` is divided by ``h`` so distances are in
Mpc/h, the same units our pair-counting and basis machinery use.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


C_OVER_H100_MPCH = 2997.92458   # c/(100 km/s/Mpc) [Mpc/h];   D_C[Mpc/h] = (c/H100) * integ


class DistanceCosmo(NamedTuple):
    """Minimal cosmological-distance parameters.

    Notes
    -----
    Massless neutrinos and pure radiation contribution are ignored
    (negligible at z << 100). For survey use cases the dominant
    parameters are ``Om``, ``w0``, ``wa``. ``h`` is carried for
    completeness but distances are returned in Mpc/h, which are
    independent of ``h`` -- gradients with respect to ``h`` will
    therefore be zero through the comoving-distance step.
    """
    Om: float = 0.31
    h: float = 0.68
    w0: float = -1.0
    wa: float = 0.0


def E_of_z(z: jnp.ndarray, cosmo: DistanceCosmo) -> jnp.ndarray:
    """Dimensionless Hubble parameter ``E(z) = H(z)/H0``."""
    Om, w0, wa = cosmo.Om, cosmo.w0, cosmo.wa
    matter = Om * (1.0 + z) ** 3
    de_exp = 3.0 * (1.0 + w0 + wa)
    de = (1.0 - Om) * (1.0 + z) ** de_exp * jnp.exp(-3.0 * wa * z / (1.0 + z))
    return jnp.sqrt(matter + de)


def comoving_distance(
    z: jnp.ndarray,
    cosmo: DistanceCosmo,
    z_max: float = 4.0,
    n_grid: int = 4000,
) -> jnp.ndarray:
    """Comoving distance D_C(z) [Mpc/h], differentiable in ``cosmo``.

    Builds a cumulative-trapezoid table of ``int_0^z dz'/E(z')`` on a
    fixed ``[0, z_max]`` grid with ``n_grid`` points, then linearly
    interpolates at the sample redshifts. ``c/H0/h`` sets the Mpc/h
    amplitude.
    """
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    integrand = 1.0 / E_of_z(z_grid, cosmo)
    # Cumulative trapezoid: D[k] = sum_{j<=k} 0.5 (f[j] + f[j+1]) dz
    dz = z_grid[1] - z_grid[0]
    cum = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz),
    ])
    return C_OVER_H100_MPCH * jnp.interp(z, z_grid, cum)


def radec_z_to_cartesian(
    ra_deg: jnp.ndarray,
    dec_deg: jnp.ndarray,
    z: jnp.ndarray,
    cosmo: DistanceCosmo,
    z_max: float = 4.0,
    n_grid: int = 4000,
) -> jnp.ndarray:
    """Convert (RA, Dec, z) to comoving Cartesian (x, y, z) in Mpc/h.

    Returns shape ``(N, 3)``. ``ra_deg``, ``dec_deg`` are in degrees.
    Right-handed (x, y, z): x is RA=0 / Dec=0, z is the celestial pole.
    """
    ra = jnp.deg2rad(ra_deg)
    dec = jnp.deg2rad(dec_deg)
    D = comoving_distance(z, cosmo, z_max=z_max, n_grid=n_grid)
    cos_dec = jnp.cos(dec)
    x = D * cos_dec * jnp.cos(ra)
    y = D * cos_dec * jnp.sin(ra)
    z_cart = D * jnp.sin(dec)
    return jnp.stack([x, y, z_cart], axis=-1)


def cartesian_to_radec_z(
    xyz: jnp.ndarray,
    cosmo: DistanceCosmo,
    z_max: float = 4.0,
    n_grid: int = 4000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Inverse mapping: comoving (x, y, z) -> (RA[deg], Dec[deg], z).

    Numerically inverts ``D_C(z)`` via a 1D interpolation table.
    """
    D = jnp.linalg.norm(xyz, axis=-1)
    # invert D -> z by tabulating D_C on a dense z-grid
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    D_grid = comoving_distance(z_grid, cosmo, z_max=z_max, n_grid=n_grid)
    z = jnp.interp(D, D_grid, z_grid)
    dec = jnp.arcsin(xyz[..., 2] / (D + 1e-30))
    ra = jnp.arctan2(xyz[..., 1], xyz[..., 0])
    ra = jnp.where(ra < 0, ra + 2 * jnp.pi, ra)
    return jnp.rad2deg(ra), jnp.rad2deg(dec), z
