"""Tests for the weighted-DD validation utility."""

import numpy as np
import pytest

from twopt_density.ls_corrfunc import xi_landy_szalay, _HAS_CORRFUNC
from twopt_density.validate import weighted_xi, assert_recovery


pytestmark = pytest.mark.skipif(
    not _HAS_CORRFUNC, reason="Corrfunc not installed"
)


def test_unit_weights_recover_xi_ls(clustered_box, small_box_log_bins):
    """w=1 reduces the weighted-DD check to the LS estimator itself."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    _, xi_ls, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2
    )
    weights = np.ones(len(pts))
    _, xi_w = weighted_xi(pts, weights, r_edges, box_size=box)
    np.testing.assert_allclose(xi_w, xi_ls, rtol=1e-10, atol=1e-12)


def test_assert_recovery_passes_when_close():
    xi_target = np.array([1.0, 0.5, 0.1, 0.01])
    xi_w = xi_target * 1.05
    assert_recovery(xi_target, xi_w, rtol=0.1)


def test_assert_recovery_fails_when_off():
    xi_target = np.array([1.0, 0.5, 0.1, 0.01])
    xi_w = xi_target * 1.5
    with pytest.raises(AssertionError, match="recovery failed"):
        assert_recovery(xi_target, xi_w, rtol=0.1)
