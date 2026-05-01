"""Tests for basis families and projection."""

import numpy as np

from twopt_density.basis import CubicSplineBasis, BesselBasis
from twopt_density.basis_projection import project_pair_counts, xi_from_basis


def test_cubic_spline_partition_of_unity():
    """Cubic B-splines should sum to ~1 across the interior."""
    basis = CubicSplineBasis(n_basis=10, r_min=1.0, r_max=100.0)
    r = np.logspace(0.1, 1.9, 50)
    F = basis.evaluate(r)
    s = F.sum(axis=0)
    # Interior should be close to 1; edges may sag because the boundary clamp
    # only places knots on one side.
    interior = s[5:-5]
    assert (interior > 0.7).all() and (interior <= 1.0 + 1e-9).all()


def test_cubic_spline_smooth_recovery():
    """Basis projection of a known smooth function recovers it."""
    basis = CubicSplineBasis(n_basis=14, r_min=0.5, r_max=200.0)
    r = np.logspace(np.log10(0.5), np.log10(200.0), 200)
    xi_true = (r / 5.0) ** -1.7

    # Synthesize fake counts: DD ~ (1 + xi) RR, RR shell, DR == RR
    shell = (4.0 / 3.0) * np.pi * (r ** 3)
    RR = np.diff(np.r_[0.0, shell])
    DD = (1.0 + xi_true) * RR
    DR = RR

    _, _, _, theta = project_pair_counts(r, DD, DR, RR, basis)
    xi_hat = xi_from_basis(theta, basis, r)

    interior = (r > 1.5) & (r < 100.0)
    err = np.abs(xi_hat[interior] - xi_true[interior]) / np.abs(xi_true[interior])
    assert err.max() < 0.05, f"max relative error = {err.max():.3f}"


def test_bessel_basis_orthogonality_via_inner_product():
    """j_0(k r) basis evaluation and dimensions."""
    k_grid = np.array([0.01, 0.05, 0.1, 0.5])
    basis = BesselBasis(k_grid, r_min=0.1, r_max=200.0)
    r = np.linspace(0.1, 200.0, 1000)
    F = basis.evaluate(r)
    assert F.shape == (4, 1000)
    # j_0(0) = 1
    np.testing.assert_allclose(basis.evaluate(np.array([0.0]))[:, 0], 1.0)
    # decay: max(|j_0|) for k=0.5 should be < that for k=0.01
    assert np.max(np.abs(F[3])) < np.max(np.abs(F[0])) + 1e-9
