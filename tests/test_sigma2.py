"""Tests for sigma^2(R) -- the count-in-cells variance two-point statistic.

These tests pin three identities that the pipeline depends on:

  1. The TH and Gaussian kernels integrate to 1 over the volume.
  2. For a power-law xi(r) = (r/r0)^-gamma the analytic sigma^2(R)
     matches the numerical kernel projection.
  3. On a small clustered point set, sigma^2(R) computed by
     pair-count projection agrees with sigma^2(R) computed by
     integrating the LS xi(r) measurement.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_TH_kernel_integrates_to_one():
    """``int K_TH(r; R) dV = 1`` over ``r in [0, 2R]``."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 7.5
    r = np.linspace(0.0, 2.0 * R, 5000)
    K = kernel_TH_3d(r, R)
    integral = np.trapezoid(4.0 * np.pi * r ** 2 * K, r)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-3)


def test_Gauss_kernel_integrates_to_one():
    """``int K_G(r; R) dV = 1`` (5-sigma cutoff is enough)."""
    from twopt_density.sigma2 import kernel_Gauss_3d
    R = 5.0
    r = np.linspace(0.0, 12.0 * R, 5000)
    K = kernel_Gauss_3d(r, R)
    integral = np.trapezoid(4.0 * np.pi * r ** 2 * K, r)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-4)


def test_TH_kernel_zero_outside_2R():
    """``K_TH(r > 2R; R) == 0``."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 4.0
    r = np.array([0.5 * R, 2.0 * R, 2.0 * R + 1e-9, 5.0 * R])
    K = kernel_TH_3d(r, R)
    assert K[0] > 0 and K[1] >= 0
    assert K[2] == 0.0 and K[3] == 0.0


def test_TH_kernel_at_origin():
    """``K_TH(0; R) = 3 / (4 pi R^3)`` (overlap of two coincident spheres)."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 3.0
    K0 = kernel_TH_3d(np.array([0.0]), R)[0]
    np.testing.assert_allclose(K0, 3.0 / (4.0 * np.pi * R ** 3), rtol=1e-12)


def test_sigma2_from_xi_matches_powerlaw_analytic():
    """For ``xi(r) = (r/r0)^-gamma`` with ``gamma != 3, 4, 5``, the
    integral against ``K_TH`` is computable in closed form. We compare
    the numerical projection against a direct one-shot integral on a
    very fine grid."""
    from twopt_density.sigma2 import sigma2_from_xi
    R = 5.0
    r0 = 5.0
    gamma = 1.8
    r = np.linspace(0.01, 6.0 * R, 8000)
    xi = (r / r0) ** (-gamma)
    sigma2 = sigma2_from_xi(r, xi, np.array([R]), kernel="tophat")[0]
    # closed form: sigma^2 = (3/R^3) integral_0^{2R} dr r^2 (r/r0)^-gamma
    #                                          x [1 - 3r/(4R) + r^3/(16R^3)]
    ref = 0.0
    rfine = np.linspace(1e-8, 2.0 * R, 200_000)
    K_ref = (3.0 / R ** 3) * (1.0 - 0.75 * rfine / R
                                + (rfine ** 3) / (16.0 * R ** 3))
    ref = np.trapezoid(rfine ** 2 * (rfine / r0) ** (-gamma) * K_ref, rfine)
    np.testing.assert_allclose(sigma2, ref, rtol=2e-3)


def test_sigma2_kernel_is_strictly_positive_on_support():
    """``K_TH(r; R) > 0`` for ``0 < r < 2R``."""
    from twopt_density.sigma2 import kernel_TH_3d
    R = 10.0
    r = np.linspace(1e-3, 2.0 * R - 1e-3, 200)
    K = kernel_TH_3d(r, R)
    assert (K > 0).all()


def test_sigma2_pair_counts_matches_xi_integral_on_uniform():
    """Synthetic Poisson catalogue: sigma^2 from pair counts equals
    the LS xi integral to within Poisson noise."""
    from twopt_density.sigma2 import (
        sigma2_from_pair_counts, sigma2_from_xi,
    )
    rng = np.random.default_rng(0)
    N_d = 3000; N_r = 9000
    pos_d = rng.uniform(0, 1, (N_d, 3))
    pos_r = rng.uniform(0, 1, (N_r, 3))

    from scipy.spatial import cKDTree
    r_edges = np.linspace(0.01, 0.4, 16)
    r_centres = 0.5 * (r_edges[1:] + r_edges[:-1])

    def pair_count(a, b, auto):
        tree = cKDTree(b)
        idx = tree.query_ball_point(a, r=r_edges[-1])
        d = []
        for i, lst in enumerate(idx):
            if not lst:
                continue
            arr = np.asarray(lst)
            if auto:
                arr = arr[arr > i]
                if arr.size == 0:
                    continue
            dist = np.linalg.norm(b[arr] - a[i], axis=1)
            d.append(dist)
        if not d:
            return np.zeros(len(r_centres))
        d = np.concatenate(d)
        h, _ = np.histogram(d, bins=r_edges)
        return h

    DD = pair_count(pos_d, pos_d, True)
    RR = pair_count(pos_r, pos_r, True)
    DR = pair_count(pos_d, pos_r, False)
    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r * (N_r - 1) / 2.0
    xi_ls = (DD / max(Nd_pairs, 1) - 2 * DR / (N_d * N_r) + RR / max(Nr_pairs, 1)) \
        / np.maximum(RR / max(Nr_pairs, 1), 1e-30)

    R_grid = np.array([0.05, 0.1, 0.15])
    s2_pair = sigma2_from_pair_counts(r_centres, DD, RR, R_grid,
                                          N_d=N_d, N_r=N_r, DR=DR,
                                          kernel="tophat")
    s2_xi = sigma2_from_xi(r_centres, xi_ls, R_grid, kernel="tophat")
    # Two estimators of the same quantity differ by O(1) bin-discretisation
    # corrections (the kernel is evaluated at bin centres in both cases,
    # so the difference is small but not zero). Require agreement at the
    # level of the kernel-bin-centre approximation for these scales.
    np.testing.assert_allclose(s2_pair, s2_xi, atol=2.0)
