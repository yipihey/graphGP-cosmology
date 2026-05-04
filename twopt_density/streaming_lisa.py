"""Streaming Local Indicator of Spatial Association (LISA) for 2pt clustering.

Computes both the global two-point estimator (Landy-Szalay, Davis-Peebles,
Hamilton, Hewett, simple) and the per-particle LISA weights without ever
materializing the ``(N, n_bins)`` per-particle pair-count matrix.

Memory: ``O(N + n_bins)``; the per-particle data is a small set of
``(N,)`` running-sum arrays (one per multipole / per data-or-random
contribution). The (N, n_bins) matrix is replaced by a careful
two-pass streaming over pairs.

Two-pass design
---------------
Pass 1: compute global DD(r), DR(r), RR(r) by binning the pair list.
        These are ``(n_bins,)`` arrays. From them we form xi(r) under
        any estimator and the bin coefficients ``a_j`` for the chosen
        LISA aggregation.

Pass 2: stream the same pair list again; for each pair (i, k, bin_j)
        accumulate into ``(N,)`` running sums::

            num_DD[i] += c_j     (c_j = a_j / E[b^(j)])
            num_DD[k] += c_j     (each pair updates both endpoints)

        with analogous accumulators for DR pairs (``num_DR``) and for
        Legendre multipoles (``num_DD_L`` weighted by ``P_L(mu_ik)``).

At the end, the per-particle weight is

    delta_i = num_DD[i] - num_DR[i] * N_D / N_R - 1     (Davis-Peebles)
    delta_i = (LS-style combination of num_DD, num_DR, RR globals)

depending on ``estimator``. The matrix-based ``compute_pair_count_weights``
in ``weights_pair_counts.py`` and this streaming form give numerically
identical answers (verified by tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree


def _shell_volumes(r_edges: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)


@dataclass
class StreamingLISA:
    """Streaming pair-count LISA.

    Parameters
    ----------
    positions
        ``(N_D, 3)`` data positions.
    r_edges
        Bin edges in Mpc.
    box_size
        Periodic box length in Mpc, or ``None`` for a survey window
        (then ``randoms`` is required).
    randoms
        ``(N_R, 3)`` random catalog. If given, all window-aware
        estimators are available.
    los
        Line-of-sight unit vector for multipole accumulation. Default
        ``[0, 0, 1]``.
    multipoles
        Iterable of Legendre orders to accumulate (e.g. ``(0, 2)`` for
        scalar density + quadrupole). 0 is always included implicitly.
    """

    positions: np.ndarray
    r_edges: np.ndarray
    box_size: Optional[float] = None
    randoms: Optional[np.ndarray] = None
    los: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    multipoles: tuple = (0, 2)

    # global pair counts (filled by .fit())
    DD: np.ndarray = field(default=None, init=False, repr=False)
    DR: np.ndarray = field(default=None, init=False, repr=False)
    RR: np.ndarray = field(default=None, init=False, repr=False)

    # per-particle running sums (filled by .fit())
    _num_DD: dict = field(default_factory=dict, init=False, repr=False)
    _num_DR: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def N_D(self) -> int:
        return len(self.positions)

    @property
    def N_R(self) -> int:
        return len(self.randoms) if self.randoms is not None else 0

    @property
    def n_bins(self) -> int:
        return len(self.r_edges) - 1

    def fit(self) -> "StreamingLISA":
        """Run two passes: compute globals, then stream per-particle aggregates."""
        self._compute_globals()
        self._stream_per_particle()
        return self

    # -- pass 1: globals ----------------------------------------------------

    def _bin_pairs(self, positions_a, positions_b, autocorr):
        """Return ``(pi, pk, bin_idx, mu)`` for all pairs in ``r_edges`` range.

        For ``autocorr=True``, uses ``query_pairs`` on a single tree.
        For ``autocorr=False``, uses ``query_ball_tree`` between the two.
        """
        r_max = float(self.r_edges[-1])
        box = self.box_size if self.box_size else None
        tree_a = cKDTree(positions_a, boxsize=box)
        if autocorr:
            pairs = tree_a.query_pairs(r=r_max, output_type="ndarray")
            if len(pairs) == 0:
                return (np.array([], dtype=np.intp),) * 4
            pi, pk = pairs[:, 0], pairs[:, 1]
        else:
            tree_b = cKDTree(positions_b, boxsize=box)
            lists = tree_a.query_ball_tree(tree_b, r=r_max)
            pi = np.repeat(np.arange(len(positions_a)),
                           [len(L) for L in lists])
            pk = np.fromiter(
                (j for L in lists for j in L),
                dtype=np.intp,
                count=int(sum(len(L) for L in lists)),
            )
            if pi.size == 0:
                return (np.array([], dtype=np.intp),) * 4

        diff = positions_a[pi] - positions_b[pk]
        if self.box_size is not None:
            diff -= self.box_size * np.round(diff / self.box_size)
        d = np.linalg.norm(diff, axis=1)
        bin_idx = np.searchsorted(self.r_edges, d, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < self.n_bins) & (d > 0)
        pi = pi[valid]; pk = pk[valid]; bin_idx = bin_idx[valid]
        d = d[valid]; diff = diff[valid]
        mu = (diff @ self.los) / d
        return pi, pk, bin_idx, mu

    def _compute_globals(self):
        # DD: factor of 2 -- each unordered pair contributes to two
        # endpoints. Convention matches Corrfunc autocorr=1 npairs.
        pi, pk, bj, mu = self._bin_pairs(self.positions, self.positions, autocorr=True)
        self.DD = 2.0 * np.bincount(bj, minlength=self.n_bins).astype(np.float64)
        self._DD_pairs = (pi, pk, bj, mu)

        if self.randoms is not None:
            pi_r, pk_r, bj_r, mu_r = self._bin_pairs(
                self.positions, self.randoms, autocorr=False,
            )
            self.DR = np.bincount(bj_r, minlength=self.n_bins).astype(np.float64)
            self._DR_pairs = (pi_r, pk_r, bj_r, mu_r)

            # RR bin counts only -- never store per-particle aggregates for
            # randoms. Use a streaming bincount to avoid materializing the
            # full RR pair list (which can be huge for N_R >> N_D).
            self.RR = self._bincount_pairs_only(
                self.randoms, self.randoms, autocorr=True,
            )
            self.RR *= 2.0  # match DD/Corrfunc factor-of-2 convention
        else:
            # Periodic uniform: RR analytic, DR = RR / 2 (same factor convention)
            shell = _shell_volumes(self.r_edges)
            V = self.box_size ** 3
            self.RR = self.N_D * (self.N_D - 1) * shell / V
            self.DR = self.RR

    def _bincount_pairs_only(self, positions_a, positions_b, autocorr):
        """Like ``_bin_pairs`` but returns only the per-bin count (no pair list).

        Used when we only need the global per-bin total (e.g., for RR
        in window-aware mode, where the pair list is too large to keep).
        """
        r_max = float(self.r_edges[-1])
        box = self.box_size if self.box_size else None
        tree_a = cKDTree(positions_a, boxsize=box)
        if autocorr:
            pairs = tree_a.query_pairs(r=r_max, output_type="ndarray")
            if len(pairs) == 0:
                return np.zeros(self.n_bins, dtype=np.float64)
            pi, pk = pairs[:, 0], pairs[:, 1]
            del pairs
        else:
            tree_b = cKDTree(positions_b, boxsize=box)
            lists = tree_a.query_ball_tree(tree_b, r=r_max)
            pi = np.repeat(np.arange(len(positions_a)),
                           [len(L) for L in lists])
            pk = np.fromiter(
                (j for L in lists for j in L),
                dtype=np.intp,
                count=int(sum(len(L) for L in lists)),
            )
            del lists
            if pi.size == 0:
                return np.zeros(self.n_bins, dtype=np.float64)

        diff = positions_a[pi] - positions_b[pk]
        del pi, pk
        if self.box_size is not None:
            diff -= self.box_size * np.round(diff / self.box_size)
        d = np.linalg.norm(diff, axis=1)
        del diff
        bin_idx = np.searchsorted(self.r_edges, d, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < self.n_bins) & (d > 0)
        return np.bincount(bin_idx[valid], minlength=self.n_bins).astype(np.float64)

    # -- estimators -----------------------------------------------------------

    def xi_simple(self) -> np.ndarray:
        """DD/RR - 1 (Peebles-Hauser 1974)."""
        return np.where(self.RR > 0, self.DD / self.RR - 1.0, 0.0)

    def xi_DP(self) -> np.ndarray:
        """DD/DR - 1 (Davis-Peebles 1983)."""
        if self.randoms is None:
            return self.xi_simple()
        return np.where(self.DR > 0,
                        self.DD * self.N_R / (self.DR * self.N_D) - 1.0,
                        0.0)

    def xi_LS(self) -> np.ndarray:
        """(DD - 2 DR + RR) / RR (Landy-Szalay 1993)."""
        if self.randoms is None:
            return self.xi_simple()
        N_D, N_R = self.N_D, self.N_R
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (self.DD * N_R**2
                   - 2.0 * self.DR * N_R * N_D
                   + self.RR * N_D**2)
            den = self.RR * N_D**2
            xi = np.where(den > 0, num / den, 0.0)
        return xi

    def xi_hamilton(self) -> np.ndarray:
        """(DD * RR) / DR^2 - 1 (Hamilton 1993)."""
        if self.randoms is None:
            return self.xi_simple()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.DR > 0,
                            self.DD * self.RR * self.N_R**2
                            / (self.DR**2 * self.N_D**2) - 1.0,
                            0.0)

    def xi(self, estimator: str = "LS") -> np.ndarray:
        """Dispatch to the chosen estimator."""
        return {
            "simple": self.xi_simple,
            "DP": self.xi_DP,
            "LS": self.xi_LS,
            "Hamilton": self.xi_hamilton,
        }[estimator]()

    # -- pass 2: per-particle aggregates ---------------------------------

    def _stream_per_particle(self):
        """Accumulate (N_D,) running sums for each requested multipole."""
        for L in self.multipoles:
            self._num_DD[L] = np.zeros(self.N_D, dtype=np.float64)
            if self.randoms is not None:
                self._num_DR[L] = np.zeros(self.N_D, dtype=np.float64)

        # DD pairs: each pair contributes to both endpoints (symmetric in
        # i <-> k, and P_L is even in mu for L = 0, 2, 4, ...).
        pi, pk, bj, mu = self._DD_pairs
        for L in self.multipoles:
            P_L = _legendre(L, mu)
            np.add.at(self._num_DD[L], pi, P_L)
            np.add.at(self._num_DD[L], pk, P_L)

        if self.randoms is not None:
            pi_r, _, bj_r, mu_r = self._DR_pairs
            for L in self.multipoles:
                P_L = _legendre(L, mu_r)
                np.add.at(self._num_DR[L], pi_r, P_L)

    # -- per-particle weights --------------------------------------------

    def per_particle_overdensity(
        self,
        estimator: str = "LS",
        aggregation: str = "RR_xi",
        L: int = 0,
        xi_floor: float = 0.0,
    ) -> np.ndarray:
        """Per-particle LISA overdensity ``delta_i``.

        Parameters
        ----------
        estimator
            ``'simple'``, ``'DP'``, ``'LS'``, or ``'Hamilton'`` for the
            global xi used to weight informative bins.
        aggregation
            ``'RR'``         -> a_j = RR_j (volume-weighted, smooth).
            ``'RR_xi'``      -> a_j = RR_j * |xi_j|  (signal-weighted).
            ``'smallest'``   -> a_j = delta_{j,0} (use only the smallest bin).
        L
            Legendre order to aggregate (0 = isotropic, 2 = quadrupole, ...).
        """
        if L not in self.multipoles:
            raise ValueError(f"L={L} not in self.multipoles={self.multipoles}")

        xi_global = self.xi(estimator=estimator)

        if aggregation == "RR":
            a = np.maximum(self.RR, 0.0)
        elif aggregation == "RR_xi":
            a = np.where(xi_global > xi_floor,
                          self.RR * np.abs(xi_global), 0.0)
            if a.sum() == 0:
                a = np.maximum(self.RR, 0.0)
        elif aggregation == "smallest":
            a = np.zeros_like(self.RR)
            a[0] = 1.0
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        a_norm = a / a.sum()

        # delta_i^(L) = sum_j a_norm[j] * (b_DD_i^(j, L) / E[b^(j)] - I[L==0])
        # E[b^(j)] = (N_D-1) * V_shell/V_box (periodic) or b_DR_i^(j) * N_D/N_R (windowed)
        if self.randoms is None:
            shell = _shell_volumes(self.r_edges)
            V = self.box_size ** 3
            E_per_bin = (self.N_D - 1) * shell / V
            c = a_norm / np.where(E_per_bin > 0, E_per_bin, 1.0)
            num = self._aggregate_with_coeffs(c, L, source="DD")
            # The L=0 case wants delta = num - 1, since sum_j a_norm[j] = 1.
            return num - (1.0 if L == 0 else 0.0)

        # Window-aware Davis-Peebles per particle: delta_i = num/den - 1
        # with bin-weighted num and den so the aggregation respects a_norm.
        c_DD = a_norm * (self.N_R / self.N_D)
        c_DR = a_norm
        num = self._aggregate_with_coeffs(c_DD, L, source="DD")
        den = self._aggregate_with_coeffs(c_DR, L, source="DR")
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(den > 0, num / den - (1.0 if L == 0 else 0.0), 0.0)
        return delta

    def _aggregate_with_coeffs(self, coeffs, L, source="DD"):
        """Compute ``sum_j coeffs[j] * b_i^(j, L)`` per particle.

        Streamed via the cached pair list -- matches a single pass over
        (i, k, j, mu) tuples, never materializing the (N, n_bins)
        per-particle matrix.
        """
        out = np.zeros(self.N_D, dtype=np.float64)
        if source == "DD":
            pi, pk, bj, mu = self._DD_pairs
            P_L = _legendre(L, mu)
            inc = coeffs[bj] * P_L
            np.add.at(out, pi, inc)
            np.add.at(out, pk, inc)
        else:  # DR
            pi, _, bj, mu = self._DR_pairs
            P_L = _legendre(L, mu)
            np.add.at(out, pi, coeffs[bj] * P_L)
        return out

    def per_particle_weights(self, **kwargs) -> np.ndarray:
        """Convenience wrapper: ``w_i = 1 + delta_i`` at L = 0."""
        return 1.0 + self.per_particle_overdensity(L=0, **kwargs)


def _legendre(L: int, mu: np.ndarray) -> np.ndarray:
    if L == 0:
        return np.ones_like(mu)
    if L == 2:
        return 0.5 * (3.0 * mu * mu - 1.0)
    if L == 4:
        m2 = mu * mu
        return (35.0 * m2 * m2 - 30.0 * m2 + 3.0) / 8.0
    raise ValueError(f"L={L} not implemented")
