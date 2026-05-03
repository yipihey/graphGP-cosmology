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
