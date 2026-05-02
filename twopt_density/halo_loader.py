"""Adapter from halotools (or any halo catalog) to the LISA + SF&H pipeline.

A halo catalog enters the pipeline as ``(positions, w_data)`` where
``w_data = <N_gal(M_h | theta_HOD)>`` is the expected galaxy occupation
under the HOD. ``build_state(positions, ...)`` then produces the frozen
pair graph, and ``xi_LS_basis_AP(state, ..., w_data, ...)`` returns the
weighted xi(s).

Two entry points::

    halos_to_positions_and_mass(halocat_or_table) -> (positions, mvir)
    apply_hod(positions, mvir, hod_params, ...)   -> (positions, w_data)

``halocat_or_table`` may be a halotools ``CachedHaloCatalog`` /
``FakeSim`` (which exposes ``.halo_table`` and ``.Lbox``) or an astropy
``Table`` / dict-like with ``halo_x, halo_y, halo_z, halo_mvir`` columns.
A bare numpy ``(N, 3)`` positions array + 1D mass array is also accepted
for non-halotools workflows.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .hod import Zheng07Params, mean_ngal_zheng07


def halos_to_positions_and_mass(
    halocat_or_table,
    mass_key: str = "halo_mvir",
    pos_keys: Tuple[str, str, str] = ("halo_x", "halo_y", "halo_z"),
    host_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract ``(positions, M_h, Lbox)`` from a halotools-style halo catalog.

    Parameters
    ----------
    halocat_or_table : object
        Either a halotools simulation object (``FakeSim``, ``CachedHaloCatalog``;
        accessed via ``.halo_table`` and ``.Lbox``) OR a table / dict-like with
        the four columns ``mass_key`` and ``pos_keys``.
    host_only : bool
        If True and the catalog has ``halo_upid``, drop subhalos
        (``halo_upid != -1``). HOD models live on host halos.

    Returns
    -------
    positions : (N, 3) float64
    M_h : (N,) float64
    Lbox : float (0.0 if not deducible)
    """
    if hasattr(halocat_or_table, "halo_table"):
        ht = halocat_or_table.halo_table
        Lbox = halocat_or_table.Lbox
        Lbox = float(Lbox[0]) if hasattr(Lbox, "__len__") else float(Lbox)
    else:
        ht = halocat_or_table
        Lbox = 0.0

    if host_only and "halo_upid" in ht.colnames if hasattr(ht, "colnames") else \
       host_only and "halo_upid" in ht:
        mask = np.asarray(ht["halo_upid"]) == -1
    else:
        mask = slice(None)

    x = np.asarray(ht[pos_keys[0]], dtype=np.float64)[mask]
    y = np.asarray(ht[pos_keys[1]], dtype=np.float64)[mask]
    z = np.asarray(ht[pos_keys[2]], dtype=np.float64)[mask]
    M = np.asarray(ht[mass_key], dtype=np.float64)[mask]
    return np.column_stack([x, y, z]), M, Lbox


def apply_hod_zheng07(
    M_h: jnp.ndarray,
    params: Optional[Zheng07Params] = None,
    modulate_with_ncen: bool = True,
) -> jnp.ndarray:
    """Per-halo expected galaxy occupation under Zheng07 HOD.

    Returns ``w_halo[i] = <N_cen(M_i)> + <N_sat(M_i)>``, ready to pass to
    ``xi_LS_basis_AP(..., w_data=w_halo, ...)``. Differentiable in
    ``params`` via ``jax.grad``.
    """
    if params is None:
        params = Zheng07Params()
    return mean_ngal_zheng07(M_h, params, modulate_with_ncen=modulate_with_ncen)


def halocat_to_state_inputs(
    halocat_or_table,
    hod_params: Optional[Zheng07Params] = None,
    mass_key: str = "halo_mvir",
    host_only: bool = True,
):
    """Convenience: halos -> ``(positions, M_h, Lbox, w_halo)``.

    Use as::

        positions, M_h, Lbox, w_halo = halocat_to_state_inputs(fakesim)
        state = build_state(positions, r_edges, Lbox, randoms=randoms,
                            cache_rr=True)
        xi = xi_LS_basis_AP(state, jb, w_halo, w_rand, 1.0, 1.0, query_r)
    """
    positions, M_h, Lbox = halos_to_positions_and_mass(
        halocat_or_table, mass_key=mass_key, host_only=host_only,
    )
    M_h = jnp.asarray(M_h)
    w_halo = apply_hod_zheng07(M_h, hod_params)
    return positions, M_h, Lbox, w_halo
