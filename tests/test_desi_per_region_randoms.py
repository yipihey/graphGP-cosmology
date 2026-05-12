"""Tests for the per-PHOTSYS-region random generator for DESI Y1 QSO.

Three properties to pin:
1. Per-region random angular distribution traces the per-region sel_map
   (within disjoint hemispherical mock footprints).
2. Per-region random redshift histogram matches the per-region data
   z-pool, AND no cross-contamination occurs between regions.
3. The global random n(z) recovers the global data n(z) — the
   factorisation invariant.

Plus the trivial split helper.
"""

from __future__ import annotations

import numpy as np
import pytest


def _disjoint_hemisphere_maps(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint sel_maps: 'N' = northern hemisphere, 'S' = southern.
    Each pixel is 1.0 in its map and 0.0 in the other."""
    import healpy as hp
    npix = 12 * nside ** 2
    theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
    dec_pix = 90.0 - np.degrees(theta_pix)
    sel_N = (dec_pix > 0.0).astype(np.float64)
    sel_S = (dec_pix <= 0.0).astype(np.float64)
    return sel_N, sel_S


def test_split_n_random_by_data_fraction_balances():
    from twopt_density.desi import split_n_random_by_data_fraction

    # 30% N, 70% S
    photsys = np.concatenate([np.full(300, "N"), np.full(700, "S")])
    n_per = split_n_random_by_data_fraction(1000, photsys)
    assert sum(n_per.values()) == 1000
    assert n_per["N"] == 300
    assert n_per["S"] == 700


def test_split_n_random_by_data_fraction_handles_remainder():
    from twopt_density.desi import split_n_random_by_data_fraction

    # 1/3, 2/3 with a non-divisible total; remainder goes to S (largest).
    photsys = np.concatenate([np.full(100, "N"), np.full(200, "S")])
    n_per = split_n_random_by_data_fraction(1001, photsys)
    assert sum(n_per.values()) == 1001
    # floor splits: 1001*100/300=333.6→333, 1001*200/300=667.3→667; total 1000;
    # remainder 1 goes to the bigger region (S).
    assert n_per["N"] == 333
    assert n_per["S"] == 668


def test_per_region_angular_distribution_matches_data():
    """Drawing from the disjoint-hemisphere maps must put N randoms
    only above Dec=0 and S randoms only below Dec=0 (within sub-pixel
    jitter at the boundary)."""
    pytest.importorskip("healpy")
    from twopt_density.desi import random_queries_desi_per_region
    nside = 16
    sel_N, sel_S = _disjoint_hemisphere_maps(nside)
    # Synthetic z pools (don't matter for the angular test).
    z_pool = np.linspace(1.0, 1.5, 1000)
    rng = np.random.default_rng(3)
    ra, dec, z, lbl = random_queries_desi_per_region(
        region_sel_maps={"N": sel_N, "S": sel_S},
        region_z_pools={"N": z_pool, "S": z_pool},
        n_random_per_region={"N": 5000, "S": 5000},
        nside=nside, rng=rng,
    )
    # Sub-pixel jitter is bounded by the pixel size at NSIDE=16 (~3.7 deg);
    # allow up to one pixel of leakage across Dec=0.
    pix_deg = np.degrees(np.sqrt(4 * np.pi / (12 * nside ** 2)))
    assert (dec[lbl == "N"] > -pix_deg).all()
    assert (dec[lbl == "S"] <  pix_deg).all()
    # And the bulk (>95%) must be cleanly on the right side.
    n_clean = (dec[lbl == "N"] > 0).sum() + (dec[lbl == "S"] <= 0).sum()
    assert n_clean / lbl.size > 0.95


def test_per_region_z_histogram_matches_region_data():
    """Each region's random z-histogram must look like that region's
    data z-pool, with no cross-contamination."""
    pytest.importorskip("healpy")
    from scipy import stats
    from twopt_density.desi import random_queries_desi_per_region
    nside = 16
    sel_N, sel_S = _disjoint_hemisphere_maps(nside)
    rng = np.random.default_rng(11)
    # Two clearly-separated z pools
    z_pool_N = rng.normal(1.0, 0.05, 5000)
    z_pool_S = rng.normal(1.8, 0.05, 5000)
    rng_draw = np.random.default_rng(7)
    ra, dec, z, lbl = random_queries_desi_per_region(
        region_sel_maps={"N": sel_N, "S": sel_S},
        region_z_pools={"N": z_pool_N, "S": z_pool_S},
        n_random_per_region={"N": 5000, "S": 5000},
        nside=nside, rng=rng_draw,
    )
    # Each region's randoms should match its pool by KS test
    _, p_N = stats.ks_2samp(z[lbl == "N"], z_pool_N)
    _, p_S = stats.ks_2samp(z[lbl == "S"], z_pool_S)
    assert p_N > 0.001, f"N randoms KS p={p_N:.3g} suggests bad shuffling"
    assert p_S > 0.001, f"S randoms KS p={p_S:.3g} suggests bad shuffling"
    # Cross-contamination check: N randoms should NOT match S pool, etc.
    _, p_cross_NS = stats.ks_2samp(z[lbl == "N"], z_pool_S)
    _, p_cross_SN = stats.ks_2samp(z[lbl == "S"], z_pool_N)
    assert p_cross_NS < 1e-6
    assert p_cross_SN < 1e-6


def test_global_n_z_matches_global_data():
    """The factorisation invariant: union of per-region randoms recovers
    the union of per-region data z-distributions."""
    pytest.importorskip("healpy")
    from scipy import stats
    from twopt_density.desi import random_queries_desi_per_region
    nside = 16
    sel_N, sel_S = _disjoint_hemisphere_maps(nside)
    rng = np.random.default_rng(5)
    z_pool_N = rng.normal(1.0, 0.05, 4000)
    z_pool_S = rng.normal(1.8, 0.05, 6000)
    z_data_global = np.concatenate([z_pool_N, z_pool_S])
    # Mirror the data fractions in the random split
    photsys_data = np.concatenate(
        [np.full(z_pool_N.size, "N"), np.full(z_pool_S.size, "S")])
    from twopt_density.desi import split_n_random_by_data_fraction
    n_per = split_n_random_by_data_fraction(20000, photsys_data)
    rng_draw = np.random.default_rng(7)
    ra, dec, z, lbl = random_queries_desi_per_region(
        region_sel_maps={"N": sel_N, "S": sel_S},
        region_z_pools={"N": z_pool_N, "S": z_pool_S},
        n_random_per_region=n_per,
        nside=nside, rng=rng_draw,
    )
    _, p_global = stats.ks_2samp(z, z_data_global)
    assert p_global > 0.001, (
        f"global random n(z) deviates from global data n(z): KS p={p_global:.3g}"
    )
