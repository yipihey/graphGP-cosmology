"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def uniform_box():
    """Uniform Poisson catalog in a periodic box."""
    box = 200.0
    N = 4000
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, box, size=(N, 3))
    return pts, box


@pytest.fixture
def clustered_box():
    """Clustered catalog: sum of Gaussian blobs in a periodic box."""
    box = 200.0
    rng = np.random.default_rng(7)
    n_centers = 25
    n_per = 200
    centers = rng.uniform(0, box, size=(n_centers, 3))
    pts = np.vstack([rng.normal(c, 5.0, size=(n_per, 3)) for c in centers])
    pts = np.mod(pts, box).astype(np.float64)
    return pts, box


@pytest.fixture
def small_box_log_bins():
    """Standard log-spaced bins suited to a 200 Mpc box test."""
    return np.logspace(np.log10(2.0), np.log10(50.0), 11)
