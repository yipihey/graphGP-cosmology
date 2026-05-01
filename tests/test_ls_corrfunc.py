"""Tests for the Corrfunc-backed Landy-Szalay estimator."""

import numpy as np
import pytest

from twopt_density.ls_corrfunc import (
    xi_landy_szalay,
    local_mean_density,
    _HAS_CORRFUNC,
    _shell_volumes,
)


pytestmark = pytest.mark.skipif(
    not _HAS_CORRFUNC, reason="Corrfunc not installed"
)


def test_uniform_xi_consistent_with_zero():
    """Uniform Poisson catalog: xi(r) statistically consistent with 0."""
    box = 200.0
    N = 4000
    r_edges = np.logspace(np.log10(2.0), np.log10(50.0), 11)
    xis = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, box, size=(N, 3))
        _, xi, _, _, _ = xi_landy_szalay(
            pts, r_edges=r_edges, box_size=box, nthreads=2
        )
        xis.append(xi)
    xis = np.array(xis)
    sem = xis.std(axis=0) / np.sqrt(len(xis))
    assert np.all(np.abs(xis.mean(axis=0)) < 3.5 * sem), (
        f"xi(r) not consistent with 0: mean={xis.mean(axis=0)}, sem={sem}"
    )


def test_dd_normalization_matches_analytic_rr(uniform_box, small_box_log_bins):
    """Sum of DD over all bins equals the analytic expectation."""
    pts, box = uniform_box
    N = len(pts)
    r_edges = small_box_log_bins

    _, _, RR, DD, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2
    )

    expected_RR = N * (N - 1) * _shell_volumes(r_edges) / box**3
    np.testing.assert_allclose(RR, expected_RR)
    # DD should match RR in mean (uniform field), within ~5sigma per bin
    sigma = np.sqrt(DD)
    deviation = np.abs(DD - RR) / sigma
    assert np.all(deviation < 5.0), f"DD-RR deviation = {deviation}"


def test_xi_increases_for_clustered_catalog(clustered_box, small_box_log_bins):
    """Clustered catalog must give xi(r) > 0 at small r."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    _, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2
    )
    assert xi[0] > xi[-1], "xi should decrease with r for a clustered catalog"
    assert xi[0] > 1.0, f"expected strong clustering at small r, got xi={xi[0]}"


def test_pair_product_weights():
    """Constant unit weights reproduce unweighted DD count."""
    box = 200.0
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, box, size=(2000, 3))
    r_edges = np.logspace(np.log10(2.0), np.log10(40.0), 9)

    _, _, _, DD_unw, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2
    )
    weights = np.ones(len(pts))
    _, _, _, DD_w, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2, weights=weights
    )
    np.testing.assert_allclose(DD_w, DD_unw, rtol=1e-10)


def test_local_mean_density_periodic(uniform_box):
    pts, box = uniform_box
    nbar = local_mean_density(pts, randoms=None, box_size=box)
    assert nbar.shape == (len(pts),)
    np.testing.assert_allclose(nbar, len(pts) / box**3)
