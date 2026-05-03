"""Projected correlation function wp(rp) via 2D Landy-Szalay.

For each pair we decompose the comoving separation into:

    pi = |(p_i - p_j) . l_hat|       LOS-parallel
    rp = sqrt(|p_i - p_j|^2 - pi^2)   LOS-perpendicular

with l_hat = (p_i + p_j) / |p_i + p_j| the bisector LOS (the standard
choice for a wide-area survey -- equivalent to the midpoint LOS in
flat-sky and reduces to the cosmic LOS in the radial limit).

Pair counts DD(rp, pi), DR(rp, pi), RR(rp, pi) feed Landy-Szalay::

    xi(rp, pi) = (DD - 2*DR + RR) / RR     (all normalised by N_d/r ratios)
    wp(rp)     = 2 * int_0^pi_max xi(rp, pi) d(pi)

The pi integral suppresses redshift-space anisotropy along the LOS,
which for Quaia is dominated by spectro-photometric z error
(sigma_z/(1+z) ~ 0.03 -> ~100 Mpc/h smearing). Picking pi_max well
above that scale recovers the real-space clustering at rp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _count_pairs_rp_pi_z(
    pos1: np.ndarray,
    pos2: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    rp_edges: np.ndarray,
    pi_edges: np.ndarray,
    z_pair_edges: np.ndarray,
    auto: bool = False,
    chunk: int = 4000,
) -> np.ndarray:
    """Bin all pairs (i in pos1, j in pos2) into a 3D histogram of
    (rp, pi, z_pair=(z_i+z_j)/2).

    The third axis carries the per-pair redshift information. Combining
    this with a Gaussian kernel ``w(z_pair; z*)`` over the z_pair axis
    gives a continuous-in-z* estimator wp(rp; z*) without binning
    galaxies by z (Newman+ 2008-style "clustering-redshift" weighting).
    """
    from scipy.spatial import cKDTree

    rp_max = float(rp_edges[-1])
    pi_max = float(pi_edges[-1])
    s_max = np.sqrt(rp_max ** 2 + pi_max ** 2)

    tree2 = cKDTree(pos2)
    counts = np.zeros(
        (len(rp_edges) - 1, len(pi_edges) - 1, len(z_pair_edges) - 1),
        dtype=np.float64,
    )

    n1 = len(pos1)
    for start in range(0, n1, chunk):
        end = min(start + chunk, n1)
        block = pos1[start:end]
        idx_lists = tree2.query_ball_point(block, r=s_max)
        rows, cols = [], []
        for ki, neigh in enumerate(idx_lists):
            if not neigh:
                continue
            if auto:
                neigh = [j for j in neigh if j > start + ki]
                if not neigh:
                    continue
            rows.extend([start + ki] * len(neigh))
            cols.extend(neigh)
        if not rows:
            continue
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        rp, pi = _bisector_rp_pi(pos1[rows], pos2[cols])
        z_pair = 0.5 * (z1[rows] + z2[cols])
        m = (rp < rp_max) & (pi < pi_max) & (z_pair >= z_pair_edges[0]) \
            & (z_pair < z_pair_edges[-1])
        if not m.any():
            continue
        h, _ = np.histogramdd(
            np.column_stack([rp[m], pi[m], z_pair[m]]),
            bins=[rp_edges, pi_edges, z_pair_edges],
        )
        counts += h
    return counts


@dataclass
class WpZpairCounts:
    """3D pair-count histograms (rp, pi, z_pair) for the LS estimator.

    Combine with ``wp_kernel_z(z_eff, sigma_z, ...)`` to evaluate a
    continuous-in-z wp(rp; z*) estimator: each pair contributes through
    a Gaussian kernel on its z_pair=(z_i+z_j)/2.
    """
    rp_edges: np.ndarray
    pi_edges: np.ndarray
    z_pair_edges: np.ndarray
    DD: np.ndarray              # (n_rp, n_pi, n_zpair)
    DR: np.ndarray
    RR: np.ndarray
    N_d: int
    N_r: int

    @property
    def rp_centres(self) -> np.ndarray:
        return 0.5 * (self.rp_edges[:-1] + self.rp_edges[1:])

    @property
    def z_pair_centres(self) -> np.ndarray:
        return 0.5 * (self.z_pair_edges[:-1] + self.z_pair_edges[1:])


def wp_landy_szalay_zpaired(
    pos_data: np.ndarray, pos_random: np.ndarray,
    z_data: np.ndarray, z_random: np.ndarray,
    rp_edges: np.ndarray,
    pi_max: float = 200.0, n_pi: int = 40,
    z_pair_edges=None, n_z_pair: int = 60,
    chunk: int = 4000,
) -> WpZpairCounts:
    """Pair-count 3D histogram (rp, pi, z_pair) for kernel-weighted
    wp(rp; z*) estimation.

    ``z_pair_edges`` defaults to a fine linear binning over the joint
    range of (z_data, z_random). The fine z_pair axis is what makes the
    kernel-weighted wp estimator continuous in z*.
    """
    pi_edges = np.linspace(0.0, pi_max, n_pi + 1)
    if z_pair_edges is None:
        z_lo = float(min(z_data.min(), z_random.min()))
        z_hi = float(max(z_data.max(), z_random.max()))
        z_pair_edges = np.linspace(z_lo, z_hi, n_z_pair + 1)

    DD = _count_pairs_rp_pi_z(
        pos_data, pos_data, z_data, z_data,
        rp_edges, pi_edges, z_pair_edges, auto=True, chunk=chunk,
    )
    DR = _count_pairs_rp_pi_z(
        pos_data, pos_random, z_data, z_random,
        rp_edges, pi_edges, z_pair_edges, auto=False, chunk=chunk,
    )
    RR = _count_pairs_rp_pi_z(
        pos_random, pos_random, z_random, z_random,
        rp_edges, pi_edges, z_pair_edges, auto=True, chunk=chunk,
    )
    return WpZpairCounts(
        rp_edges=rp_edges, pi_edges=pi_edges, z_pair_edges=z_pair_edges,
        DD=DD, DR=DR, RR=RR, N_d=len(pos_data), N_r=len(pos_random),
    )


def wp_kernel_z(
    z_eff,
    sigma_z: float,
    counts: WpZpairCounts,
):
    """Kernel-weighted wp(rp; z_eff) Landy-Szalay estimator.

    Each pair contributes through a Gaussian kernel
    G(z_pair; z_eff, sigma_z), so wp(rp; z_eff) is a continuous,
    JAX-differentiable function of z_eff -- no binning of galaxies by
    redshift.

        DD_w(rp, pi; z*) = sum_{z_pair_bin} G(z_pair_bin; z*, sigma_z)
                            * DD_bin(rp, pi, z_pair_bin)
        wp(rp; z*) = 2 sum_pi xi_LS(rp, pi; z*) d(pi)

    The normalisation N_d, N_r is local: when integrating over a
    kernel of width sigma_z, the effective sample size is
    N_eff = N * <kernel weight> at z_eff. We propagate the kernel
    weight through the LS normalisation so the estimator is unbiased.

    Returns a jax array of shape (n_rp,).
    """
    import jax.numpy as jnp

    z_centres = jnp.asarray(counts.z_pair_centres, dtype=jnp.float64)
    weights = jnp.exp(-0.5 * ((z_centres - z_eff) / sigma_z) ** 2)
    weights = weights / jnp.sum(weights)        # normalise the z kernel

    DD = jnp.asarray(counts.DD)
    DR = jnp.asarray(counts.DR)
    RR = jnp.asarray(counts.RR)

    DD_w = jnp.sum(DD * weights[None, None, :], axis=2)        # (n_rp, n_pi)
    DR_w = jnp.sum(DR * weights[None, None, :], axis=2)
    RR_w = jnp.sum(RR * weights[None, None, :], axis=2)

    # LS estimator on the kernel-weighted counts. The kernel is normalised
    # in z so the *expected* DD/DR/RR are reduced by the same factor for
    # both signal and noise -- the LS ratio is unchanged.
    nd_pairs = counts.N_d * (counts.N_d - 1) / 2.0
    nr_pairs = counts.N_r * (counts.N_r - 1) / 2.0
    DD_n = DD_w / nd_pairs
    DR_n = DR_w / (counts.N_d * counts.N_r)
    RR_n = RR_w / nr_pairs
    xi = jnp.where(RR_n > 0, (DD_n - 2 * DR_n + RR_n) / RR_n, 0.0)

    pi_edges = jnp.asarray(counts.pi_edges)
    d_pi = jnp.diff(pi_edges)
    return 2.0 * jnp.sum(xi * d_pi[None, :], axis=1)


@dataclass
class WpMeasurement:
    rp_centres: np.ndarray
    rp_edges: np.ndarray
    pi_edges: np.ndarray
    DD: np.ndarray         # (n_rp, n_pi) raw pair counts
    DR: np.ndarray
    RR: np.ndarray
    N_d: int
    N_r: int

    @property
    def xi_rp_pi(self) -> np.ndarray:
        """Landy-Szalay xi(rp, pi) on the bin grid."""
        nd_pairs = self.N_d * (self.N_d - 1) / 2.0
        nr_pairs = self.N_r * (self.N_r - 1) / 2.0
        DD_n = self.DD / nd_pairs
        DR_n = self.DR / (self.N_d * self.N_r)
        RR_n = self.RR / nr_pairs
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (DD_n - 2.0 * DR_n + RR_n) / RR_n
        xi = np.where(RR_n > 0, xi, 0.0)
        return xi

    @property
    def wp(self) -> np.ndarray:
        """wp(rp) = 2 sum xi(rp, pi) * d(pi)."""
        xi = self.xi_rp_pi
        d_pi = np.diff(self.pi_edges)
        return 2.0 * np.sum(xi * d_pi[None, :], axis=1)


def _bisector_rp_pi(p1: np.ndarray, p2: np.ndarray):
    """Bisector-LOS (rp, pi) for arrays of paired positions."""
    s = p1 - p2
    mid = p1 + p2
    mid_norm = np.linalg.norm(mid, axis=1)
    safe = mid_norm > 0
    l_hat = np.zeros_like(mid)
    l_hat[safe] = mid[safe] / mid_norm[safe, None]
    pi_pair = np.abs(np.einsum("ij,ij->i", s, l_hat))
    s2 = np.einsum("ij,ij->i", s, s)
    rp_pair = np.sqrt(np.maximum(s2 - pi_pair ** 2, 0.0))
    return rp_pair, pi_pair


def _count_pairs_rp_pi(
    pos1: np.ndarray,
    pos2: np.ndarray,
    rp_edges: np.ndarray,
    pi_edges: np.ndarray,
    auto: bool = False,
    chunk: int = 4000,
) -> np.ndarray:
    """Bin all pairs (i in pos1, j in pos2) into a (rp, pi) histogram.

    For ``auto=True`` (pos1 is pos2), each unordered pair is counted
    once: the chunked query enforces ``j > i_global``.
    """
    from scipy.spatial import cKDTree

    rp_max = float(rp_edges[-1])
    pi_max = float(pi_edges[-1])
    s_max = np.sqrt(rp_max ** 2 + pi_max ** 2)

    tree2 = cKDTree(pos2)
    counts = np.zeros((len(rp_edges) - 1, len(pi_edges) - 1), dtype=np.float64)

    n1 = len(pos1)
    for start in range(0, n1, chunk):
        end = min(start + chunk, n1)
        block = pos1[start:end]
        idx_lists = tree2.query_ball_point(block, r=s_max)
        # flatten with a per-chunk i index
        rows, cols = [], []
        for ki, neigh in enumerate(idx_lists):
            if not neigh:
                continue
            if auto:
                neigh = [j for j in neigh if j > start + ki]
                if not neigh:
                    continue
            rows.extend([start + ki] * len(neigh))
            cols.extend(neigh)
        if not rows:
            continue
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        rp, pi = _bisector_rp_pi(pos1[rows], pos2[cols])
        # mask outside bin range
        m = (rp < rp_max) & (pi < pi_max)
        if not m.any():
            continue
        h, _, _ = np.histogram2d(rp[m], pi[m], bins=[rp_edges, pi_edges])
        counts += h
    return counts


def wp_landy_szalay(
    pos_data: np.ndarray,
    pos_random: np.ndarray,
    rp_edges: np.ndarray,
    pi_max: float = 80.0,
    n_pi: int = 40,
    chunk: int = 4000,
) -> WpMeasurement:
    """Measure wp(rp) via 2D Landy-Szalay on a comoving point cloud.

    Parameters
    ----------
    pos_data, pos_random : (N, 3) comoving Mpc/h, *not* shifted.
        ``_bisector_rp_pi`` uses the absolute origin as the LOS anchor;
        do NOT pass shift-to-positive coordinates.
    rp_edges : 1D array of rp bin edges, increasing.
    pi_max, n_pi : LOS bin extent and number of linear bins.
    """
    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    pi_edges = np.linspace(0.0, pi_max, n_pi + 1)
    rp_centres = 0.5 * (rp_edges[:-1] + rp_edges[1:])

    DD = _count_pairs_rp_pi(pos_data, pos_data, rp_edges, pi_edges,
                            auto=True, chunk=chunk)
    DR = _count_pairs_rp_pi(pos_data, pos_random, rp_edges, pi_edges,
                            auto=False, chunk=chunk)
    RR = _count_pairs_rp_pi(pos_random, pos_random, rp_edges, pi_edges,
                            auto=True, chunk=chunk)

    return WpMeasurement(
        rp_centres=rp_centres, rp_edges=rp_edges, pi_edges=pi_edges,
        DD=DD, DR=DR, RR=RR,
        N_d=len(pos_data), N_r=len(pos_random),
    )
