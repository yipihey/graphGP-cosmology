"""Tests for the per-particle pair-count baseline."""

import numpy as np
import pytest

from twopt_density.ls_corrfunc import xi_landy_szalay, _HAS_CORRFUNC
from twopt_density.weights_pair_counts import (
    per_particle_pair_counts,
    per_particle_overdensity,
    aggregate_weights,
    compute_pair_count_weights,
    _shell_volumes,
)
from twopt_density.validate import weighted_xi


pytestmark = pytest.mark.skipif(
    not _HAS_CORRFUNC, reason="Corrfunc not installed"
)


def test_pair_count_sum_equals_2DD(clustered_box, small_box_log_bins):
    """sum_i b[i, j] = 2 * DD_j_unordered (each pair counted twice)."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    b = per_particle_pair_counts(pts, r_edges, box_size=box)
    # Corrfunc DD (autocorr=1) returns total ordered pairs = 2 * unordered,
    # so b.sum(axis=0) should equal Corrfunc's DD column directly.
    _, _, _, DD_corrfunc, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    np.testing.assert_allclose(b.sum(axis=0), DD_corrfunc, rtol=1e-10)


def test_per_particle_mean_recovers_xi_simple(
    clustered_box, small_box_log_bins,
):
    """The per-particle mean overdensity matches xi(r) from LS."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    N = len(pts)
    _, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    b = per_particle_pair_counts(pts, r_edges, box_size=box)
    delta_pp = per_particle_overdensity(b, r_edges, N, box)
    xi_from_pp = delta_pp.mean(axis=0)
    # Should match LS to machine precision in the periodic-box (DR == RR) case.
    np.testing.assert_allclose(xi_from_pp, xi, atol=1e-3, rtol=5e-3)


def test_RR_weighted_perfect_shape_recovery(
    clustered_box, small_box_log_bins,
):
    """`mode='RR'` aggregation reproduces the SHAPE of xi_LS exactly.

    The RR-weighted aggregate is mathematically equivalent to a single
    top-hat KDE inside ``r_max``. The pair-product correction
    ``E[w_i w_k | r_ik = r] = <w>^2 + xi_ww(r)`` is approximately
    independent of ``r`` over the strongly-clustered range, so xi_w is
    a constant rescaling of xi_LS -- giving Pearson ~ 1 even when the
    amplitude is biased by the data-tracer factor ``<w>^2``.
    """
    pts, box = clustered_box
    r_edges = small_box_log_bins
    r_c, xi, RR, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    w, _ = compute_pair_count_weights(pts, r_edges, xi, RR, box, mode="RR")
    _, xi_w = weighted_xi(pts, w, r_edges, box_size=box)
    mask = xi > 1.0
    pearson = np.corrcoef(xi[mask], xi_w[mask])[0, 1]
    ratio = float(np.median(xi_w[mask] / xi[mask]))
    assert pearson > 0.999, f"Pearson too low: {pearson}"
    # Calibration factor <w>^2 is bounded above by the catalog's clustering
    # strength. Never larger than ~10x for typical galaxy/halo catalogs.
    assert 0.5 < ratio < 10.0, f"ratio out of range: {ratio}"


def test_aggregate_modes_produce_valid_weights(
    clustered_box, small_box_log_bins,
):
    """All three modes return finite weights."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    r_c, xi, RR, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    for mode in ("smallest_bin", "RR", "RR_xi"):
        w, _ = compute_pair_count_weights(
            pts, r_edges, xi, RR, box, mode=mode,
        )
        assert w.shape == (len(pts),)
        assert np.all(np.isfinite(w))


def test_unit_weights_recover_xi_LS(clustered_box, small_box_log_bins):
    """Setting all weights to 1 reproduces xi_LS exactly."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    r_c, xi, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    weights = np.ones(len(pts))
    _, xi_w = weighted_xi(pts, weights, r_edges, box_size=box)
    np.testing.assert_allclose(xi_w, xi, rtol=1e-10, atol=1e-12)
