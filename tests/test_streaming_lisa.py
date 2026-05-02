"""Tests for the streaming LISA implementation."""

import numpy as np
import pytest

from twopt_density.ls_corrfunc import xi_landy_szalay, _HAS_CORRFUNC
from twopt_density.weights_pair_counts import compute_pair_count_weights
from twopt_density.streaming_lisa import StreamingLISA


pytestmark = pytest.mark.skipif(
    not _HAS_CORRFUNC, reason="Corrfunc not installed"
)


def test_streaming_xi_simple_matches_corrfunc(clustered_box, small_box_log_bins):
    """xi_simple from streaming = DD/RR_periodic - 1 = LS xi (DR == RR)."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    _, xi_ref, _, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box, multipoles=(0,),
    ).fit()
    np.testing.assert_allclose(sl.xi_simple(), xi_ref, rtol=1e-12, atol=1e-14)


def test_streaming_weights_match_matrix(clustered_box, small_box_log_bins):
    """Periodic per-particle weights agree with the matrix-based path."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    _, xi_ref, RR_ref, _, _ = xi_landy_szalay(
        pts, r_edges=r_edges, box_size=box, nthreads=2,
    )
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box, multipoles=(0,),
    ).fit()
    w_stream = sl.per_particle_weights(estimator="simple", aggregation="RR")
    w_matrix, _ = compute_pair_count_weights(
        pts, r_edges, xi_ref, RR_ref, box_size=box, mode="RR",
    )
    np.testing.assert_allclose(w_stream, w_matrix, rtol=1e-10, atol=1e-12)


def test_streaming_xi_LS_with_randoms(clustered_box, small_box_log_bins):
    """xi_LS via streaming matches Corrfunc with explicit randoms."""
    pts, box = clustered_box
    r_edges = small_box_log_bins
    rng = np.random.default_rng(1)
    randoms = rng.uniform(0, box, size=(3 * len(pts), 3)).astype(np.float64)
    _, xi_ref, _, _, _ = xi_landy_szalay(
        pts, randoms=randoms, r_edges=r_edges,
        box_size=box, nthreads=2,
    )
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box,
        randoms=randoms, multipoles=(0,),
    ).fit()
    np.testing.assert_allclose(sl.xi_LS(), xi_ref, rtol=1e-10, atol=1e-12)


def test_streaming_estimators_agree_in_periodic_limit(
    uniform_box, small_box_log_bins,
):
    """Simple, DP, LS, Hamilton all agree to within shot noise on uniform."""
    pts, box = uniform_box
    r_edges = small_box_log_bins
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box, multipoles=(0,),
    ).fit()
    for est in ("simple", "DP", "LS", "Hamilton"):
        xi = sl.xi(estimator=est)
        # All consistent with 0 within ~3 sigma of Poisson noise
        assert np.all(np.abs(xi) < 0.6), f"{est}: xi out of range, max={np.abs(xi).max()}"


def test_streaming_unknown_estimator_raises():
    pts = np.random.default_rng(0).uniform(0, 100, size=(100, 3))
    sl = StreamingLISA(positions=pts, r_edges=np.array([1.0, 5.0, 20.0]),
                        box_size=100.0, multipoles=(0,))
    with pytest.raises(KeyError):
        sl.fit().xi(estimator="bogus")


def test_streaming_multipole_quadrupole_isotropic_zero(
    uniform_box, small_box_log_bins,
):
    """For uniform isotropic, per-particle L=2 overdensity averages to ~0."""
    pts, box = uniform_box
    r_edges = small_box_log_bins
    sl = StreamingLISA(
        positions=pts, r_edges=r_edges, box_size=box, multipoles=(0, 2),
    ).fit()
    delta_2 = sl.per_particle_overdensity(
        L=2, aggregation="RR", estimator="simple",
    )
    # mean should be << std on a uniform isotropic catalog
    assert abs(delta_2.mean()) < 0.05, f"mean {delta_2.mean()}"
