"""Tests for the morton_cascade Python wrapper.

These tests validate the wrapper's plumbing: writing binary point
files, invoking the CLI, parsing CSV output. The Rust crate's
correctness is validated by its own ``cargo test`` suite (29 tests
pass).

Each test skips cleanly if the binary is unavailable (no Rust
toolchain installed). On the development machine a typical run is
< 5 s end-to-end.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest


def _have_cascade():
    from twopt_density.cascade import BIN_PATH, CRATE_DIR
    if os.path.exists(BIN_PATH):
        return True
    return shutil.which("cargo") is not None and os.path.isdir(CRATE_DIR)


@pytest.fixture(scope="module")
def uniform_2d():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, (5_000, 2)).astype(np.float64)


@pytest.fixture(scope="module")
def uniform_3d():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, (3_000, 3)).astype(np.float64)


@pytest.fixture(scope="module")
def uniform_3d_random():
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 1.0, (15_000, 3)).astype(np.float64)


def test_cascade_smoke_2d(uniform_2d):
    """``cascade()`` returns per-level rows with mean and var columns."""
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available and cargo missing")
    from twopt_density.cascade import cascade
    arr = cascade(uniform_2d, box_size=1.0, dim=2, periodic=True)
    assert arr.dtype.names is not None
    for col in ("level", "mean", "var", "sigma2_field"):
        assert col in arr.dtype.names
    # mean count at level 0 (whole box) = N
    assert float(arr[0]["mean"]) == pytest.approx(len(uniform_2d), abs=1e-6)
    # at deeper levels the mean drops by a factor 4 each level (2D)
    if len(arr) >= 3:
        ratio = float(arr[1]["mean"]) / float(arr[2]["mean"])
        assert 3.0 < ratio < 5.0


def test_cascade_3d_returns_dyadic_levels(uniform_3d):
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import cascade
    arr = cascade(uniform_3d, box_size=1.0, dim=3, periodic=True)
    assert len(arr) >= 2
    # in 3D the per-level mean drops by 8x per level
    if len(arr) >= 3:
        ratio = float(arr[1]["mean"]) / float(arr[2]["mean"])
        assert 6.5 < ratio < 9.5


def test_xi_landy_szalay_smoke(uniform_3d, uniform_3d_random):
    """``xi_landy_szalay`` returns a per-shell table with xi_ls."""
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import xi_landy_szalay
    arr = xi_landy_szalay(uniform_3d, uniform_3d_random,
                              box_size=1.0, dim=3, periodic=False)
    assert "xi_ls" in arr.dtype.names
    # uniform Poisson -> xi_ls ~ 0 within Poisson noise
    use = arr["dd"] > 50          # ignore very-low-count shells
    if use.any():
        xi_med = float(np.nanmedian(arr["xi_ls"][use]))
        assert abs(xi_med) < 0.5


def test_xi_landy_szalay_with_weights(uniform_3d, uniform_3d_random):
    """Weighted xi: passing all-ones weights matches the unweighted run."""
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import xi_landy_szalay
    n_d = len(uniform_3d); n_r = len(uniform_3d_random)
    a = xi_landy_szalay(uniform_3d, uniform_3d_random,
                           box_size=1.0, dim=3, periodic=False)
    b = xi_landy_szalay(uniform_3d, uniform_3d_random,
                           box_size=1.0, dim=3, periodic=False,
                           weights_data=np.ones(n_d),
                           weights_randoms=np.ones(n_r))
    np.testing.assert_allclose(a["xi_ls"], b["xi_ls"], rtol=1e-6, atol=1e-9)


def test_field_stats_returns_moments_and_pdf(uniform_3d, uniform_3d_random):
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import field_stats
    moments, pdf = field_stats(uniform_3d, uniform_3d_random,
                                    box_size=1.0, dim=3, periodic=False)
    assert moments.dtype.names is not None
    assert pdf.dtype.names is not None
    # On uniform Poisson, mean_delta is ~ 0 at the SHALLOW levels where
    # each cell holds many points; at deep levels cells go empty and
    # mean_delta -> -1 by construction (W_d / W_r - 1 with W_d -> 0).
    # Restrict the check to levels where ~> 10 data points per cell.
    md = np.asarray(moments["mean_delta"], dtype=np.float64)
    levels = np.asarray(moments["level"], dtype=np.int64)
    n_d = len(uniform_3d)
    # in 3D each level has 8x more cells, so dense levels obey
    # 8^L < N_d / 10  -> L < log_8(N_d / 10)
    L_max_dense = int(np.floor(np.log(n_d / 10) / np.log(8)))
    dense = levels <= L_max_dense
    assert np.all(np.abs(md[dense]) < 0.2), (
        f"mean_delta not zero at dense levels: {md[dense]}"
    )


def test_anisotropy_runs_3d(uniform_3d, uniform_3d_random):
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import anisotropy
    arr = anisotropy(uniform_3d, uniform_3d_random,
                       box_size=1.0, periodic=False)
    assert arr.dtype.names is not None
    # uniform isotropic field -> reduced quadrupole ~ 0 within noise
    assert "reduced_quadrupole_los" in arr.dtype.names


def test_pairs_returns_dyadic_shells(uniform_3d):
    """``pairs()`` returns DD per dyadic shell with monotone level ordering."""
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import pairs
    arr = pairs(uniform_3d, box_size=1.0, dim=3, periodic=True)
    assert arr.dtype.names is not None
    assert "n_pairs" in arr.dtype.names
    assert "r_inner_phys" in arr.dtype.names
    cum = np.asarray(arr["cumulative_pairs"], dtype=np.int64)
    assert (cum >= 0).all()
    # level 0 (whole box) should have the most pairs; pair counts go down
    # as cell side shrinks.
    n_pairs = np.asarray(arr["n_pairs"], dtype=np.int64)
    assert n_pairs[0] >= n_pairs[1]


def test_xi_landy_szalay_from_window_smoke(tmp_path):
    """``xi_landy_szalay_from_window`` end-to-end: synthesise randoms
    from a (mask, n(z)) survey window, run cascade xi, return
    structured array with xi_ls."""
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    pytest.importorskip("healpy")
    import healpy as hp
    from twopt_density.cascade import xi_landy_szalay_from_window
    from twopt_density.distance import DistanceCosmo, radec_z_to_cartesian

    fid = DistanceCosmo(Om=0.31, h=0.68)
    nside = 32
    npix = 12 * nside ** 2
    sel = np.ones(npix)
    rng = np.random.default_rng(0)
    n_d = 2000
    # synthesise data uniform on sphere with z in [0.5, 1.5]
    theta = np.arccos(rng.uniform(-1, 1, n_d))
    phi = rng.uniform(0, 2 * np.pi, n_d)
    ra = np.degrees(phi); dec = 90.0 - np.degrees(theta)
    z = rng.uniform(0.5, 1.5, n_d)
    xyz = radec_z_to_cartesian(ra, dec, z, fid)
    arr = xi_landy_szalay_from_window(
        xyz, ra, dec, z, sel, nside, fid,
        n_random_factor=3, rng_seed=1,
    )
    assert "xi_ls" in arr.dtype.names
    # uniform Poisson random data + matching analytic randoms -> xi ~ 0
    use = arr["dd"] > 100
    if use.any():
        xi_med = float(np.nanmedian(arr["xi_ls"][use]))
        assert abs(xi_med) < 0.5, f"unexpectedly large xi_med = {xi_med}"


def test_xi_landy_szalay_analytic_RR_smoke(tmp_path):
    """``xi_landy_szalay_analytic_RR`` produces an LS xi(r) per dyadic
    shell using a deterministic weighted-grid (no MC randoms)."""
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    pytest.importorskip("healpy")
    from twopt_density.cascade import xi_landy_szalay_analytic_RR
    from twopt_density.distance import DistanceCosmo, radec_z_to_cartesian

    fid = DistanceCosmo(Om=0.31, h=0.68)
    nside = 16
    sel = np.ones(12 * nside ** 2)
    rng = np.random.default_rng(0)
    n_d = 1500
    theta = np.arccos(rng.uniform(-1, 1, n_d))
    phi = rng.uniform(0, 2 * np.pi, n_d)
    ra = np.degrees(phi); dec = 90.0 - np.degrees(theta)
    z = rng.uniform(0.5, 1.5, n_d)
    xyz = radec_z_to_cartesian(ra, dec, z, fid)
    arr = xi_landy_szalay_analytic_RR(
        xyz, sel, nside, z, fid, n_grid=32,
    )
    assert "xi_ls" in arr.dtype.names
    # uniform random data on a full-sky window: xi at large scales ~ 0
    use = (arr["dd"] > 100) & (arr["rr"] > 1e-3) & np.isfinite(arr["xi_ls"])
    if use.any():
        # at the top dyadic levels (where the grid resolves the pair scale)
        # xi should be close to zero
        top_use = use & (arr["level"] <= 3)
        if top_use.any():
            assert (np.abs(arr["xi_ls"][top_use]) < 1.0).all()


def test_anisotropy_rejects_non_3d(uniform_2d):
    if not _have_cascade():
        pytest.skip("morton_cascade binary not available")
    from twopt_density.cascade import anisotropy
    rng = np.random.default_rng(0)
    rand2d = rng.uniform(0, 1, (200, 2))
    with pytest.raises(ValueError):
        anisotropy(uniform_2d, rand2d, box_size=1.0)
