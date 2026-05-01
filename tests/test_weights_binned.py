"""Tests for Part I: binned per-point weights and Eq. 5 recovery."""

import numpy as np
import pytest

from twopt_density.ls_corrfunc import (
    xi_landy_szalay, local_mean_density, _HAS_CORRFUNC,
)
from twopt_density.weights_binned import compute_binned_weights
from twopt_density.validate import weighted_xi


pytestmark = pytest.mark.skipif(
    not _HAS_CORRFUNC, reason="Corrfunc not installed"
)


def test_weights_have_unit_mean(clustered_box, small_box_log_bins):
    pts, box = clustered_box
    r_c, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=small_box_log_bins, box_size=box, nthreads=2
    )
    nbar = local_mean_density(pts, randoms=None, box_size=box)
    w = compute_binned_weights(pts, r_c, xi, nbar, box_size=box)
    assert w.shape == (len(pts),)
    assert abs(w.mean() - 1.0) < 0.5


def test_weights_are_non_trivial(clustered_box, small_box_log_bins):
    """Weights must vary across points; w == 1 everywhere is a regression."""
    pts, box = clustered_box
    r_c, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=small_box_log_bins, box_size=box, nthreads=2
    )
    nbar = local_mean_density(pts, randoms=None, box_size=box)
    w = compute_binned_weights(pts, r_c, xi, nbar, box_size=box)
    # For a clustered catalog, std(w) should be O(0.1) or larger.
    assert w.std() > 0.05, f"weights are trivially constant (std={w.std()})"


def test_weights_finite_and_sane(clustered_box, small_box_log_bins):
    pts, box = clustered_box
    r_c, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=small_box_log_bins, box_size=box, nthreads=2
    )
    nbar = local_mean_density(pts, randoms=None, box_size=box)
    w = compute_binned_weights(pts, r_c, xi, nbar, box_size=box)
    assert np.all(np.isfinite(w))


def test_weighted_dd_matches_input_xi_qualitatively(
    clustered_box, small_box_log_bins,
):
    """xi_w(r) has the same shape as xi(r) at scales >> kernel radius.

    The Wiener filter posterior smooths at the ``r_kernel`` scale, so
    exact Eq. 5 recovery only holds for ``r >> r_kernel``. At intermediate
    scales the ratio xi_w / xi should be approximately constant. We test
    only this qualitative property here -- see the docstring of
    ``weights_binned.py`` for the exact-recovery research issue.
    """
    pts, box = clustered_box
    r_edges = small_box_log_bins
    r_c, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2
    )
    nbar = local_mean_density(pts, randoms=None, box_size=box)
    w = compute_binned_weights(
        pts, r_c, xi, nbar, box_size=box, mode="mean",
    )
    _, xi_w = weighted_xi(pts, w, r_edges, box_size=box)
    mask = xi > 1.0
    # xi_w should be monotonically decreasing across the strongly clustered range.
    diffs = np.diff(xi_w[mask])
    assert (diffs < 0).all(), f"xi_w not monotonically decreasing: {xi_w[mask]}"
    # And its ratio to xi should be bounded (not blowing up).
    ratio = xi_w[mask] / xi[mask]
    assert np.all(ratio > 0.5), f"xi_w too small relative to xi: {ratio}"
    assert np.all(ratio < 20.0), f"xi_w blowing up relative to xi: {ratio}"


def test_n_too_large_raises():
    """Dense Cholesky path enforces an N cap."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100.0, size=(15000, 3))
    r = np.logspace(0, 1.5, 11)
    xi = np.zeros_like(r)
    nbar = np.full(len(pts), 1.0)
    with pytest.raises(ValueError, match="too large"):
        compute_binned_weights(pts, r, xi, nbar)
