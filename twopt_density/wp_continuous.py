"""SF21 continuous-function estimator for wp(rp, z) -- 2D extension.

Implements the Storey-Fisher & Hogg (2021) `Continuous-Function
Estimator' (arXiv:2011.01836) generalised to a 2D (rp, z_pair) basis,
so wp is a continuous function of *both* projected separation and
redshift, with no binning at any stage. The redshift derivative is
analytic from the basis, so ``dwp/dz`` falls out of the same forward
model. JAX-pure for the evaluator side; pair counting is numpy +
scipy.cKDTree.

Math (following SF21 eq. 11+ generalised to 2D):

  Choose a separable basis ``phi_kl(rp, z) = T_k(rp) * T_l(z)`` and
  expand the LS estimator amplitude in it. The basis-projected
  pair-count vector is

    DD_kl = sum_{pairs in DD, |pi|<pi_max} T_k(rp_ij) T_l(z_pair_ij)
    DR_kl, RR_kl       analogous for cross/random pairs

  With proper LS normalisation (Nd_pairs, NdNr, Nr_pairs) the basis
  amplitudes ``a_kl`` solve the linear system

    sum_{k'l'} Q_(kl)(k'l') * a_(k'l') = (DD - 2DR + RR)_(kl)

  where the Q matrix ("random Gram matrix") is the basis Gram on the
  random-pair density:

    Q_(kl)(k'l') = sum_{pairs in RR, |pi|<pi_max}
                    T_k(rp) T_l(z) T_k'(rp) T_l'(z) / Nr_pairs

  (i.e. an outer-product accumulation of the basis vector). Solving
  by ``np.linalg.solve(Q, b)`` and reshaping gives
  a_kl = (K_rp x K_z); then

    wp(rp, z*) = 2 * pi_max * Sum_kl a_kl T_k(rp) T_l(z*)

  (the leading factor absorbs the per-pair |pi|<pi_max indicator's
  "effective Delta_pi"; equivalent to the d_pi sum factor in the
  binned LS wp).

Dimension choices for Quaia: K_rp=8 (log-Chebyshev on rp in [5, 80]
Mpc/h) + K_z=5 (Chebyshev on z in [0.85, 2.45]) -> K=40 amplitudes;
Q is 40x40, linsolve is trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----- Chebyshev basis (numpy + jax variants share recursion) -----------

def _cheb_T(x, K):
    """Chebyshev T_0..T_{K-1} evaluated at x in [-1, 1].

    Vectorised: x can be a scalar or 1-D array; output has shape
    ``x.shape + (K,)``. Works with numpy or jax.numpy.
    """
    xp = type(x)
    # Use the underlying array library
    if hasattr(x, "_jax_array_") or "jax" in type(x).__module__.lower():
        import jax.numpy as np_
    else:
        np_ = np
    x = np_.asarray(x)
    out = [np_.ones_like(x), x]
    for k in range(2, K):
        out.append(2.0 * x * out[-1] - out[-2])
    return np_.stack(out, axis=-1)                       # (..., K)


def _cheb_T_jax(x, K):
    """JAX-pure Chebyshev T_0..T_{K-1} at x in [-1, 1] -- shape x.shape + (K,)."""
    import jax.numpy as jnp

    x = jnp.asarray(x)
    if K <= 0:
        return jnp.zeros(x.shape + (0,))
    if K == 1:
        return jnp.ones(x.shape + (1,))
    out = [jnp.ones_like(x), x]
    for k in range(2, K):
        out.append(2.0 * x * out[-1] - out[-2])
    return jnp.stack(out, axis=-1)


def _cheb_dT_dx_jax(x, K):
    """JAX dT_k/dx at x in [-1, 1]: d/dx T_k(x) = k * U_{k-1}(x)
    where U is the Chebyshev polynomial of the second kind."""
    import jax.numpy as jnp

    x = jnp.asarray(x)
    if K <= 1:
        return jnp.zeros(x.shape + (K,))
    # U_0(x)=1, U_1(x)=2x, U_{k+1}=2x U_k - U_{k-1}
    U = [jnp.ones_like(x), 2 * x]
    for k in range(2, K - 1):
        U.append(2 * x * U[-1] - U[-2])
    while len(U) < K:                  # pad to size K-1
        U.append(2 * x * U[-1] - U[-2])
    # dT_0/dx = 0, dT_k/dx = k * U_{k-1}
    derivs = [jnp.zeros_like(x)]
    for k in range(1, K):
        derivs.append(float(k) * U[k - 1])
    return jnp.stack(derivs, axis=-1)


# ----- pair-projection routine ------------------------------------------

def _bisector_rp_pi(p1: np.ndarray, p2: np.ndarray):
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


def _project_pairs_basis(
    pos1: np.ndarray, pos2: np.ndarray,
    z1: np.ndarray, z2: np.ndarray,
    rp_min: float, rp_max: float,
    z_min: float, z_max: float,
    pi_max: float,
    K_rp: int, K_z: int,
    auto: bool = False,
    chunk: int = 4000,
    accumulate_Q: bool = False,
):
    """Per-pair basis projection.

    Returns
    -------
    b : (K_rp, K_z) sum of T_k(rp) T_l(z_pair) over pairs inside the
        (rp, z, |pi|) acceptance window.
    Q : (K_rp*K_z, K_rp*K_z) outer-product accumulation if
        ``accumulate_Q`` (used for the random Gram matrix); else None.
    n_pairs : number of pairs that landed inside the window.
    """
    from scipy.spatial import cKDTree

    s_max = float(np.sqrt(rp_max ** 2 + pi_max ** 2))
    tree2 = cKDTree(pos2)
    K = K_rp * K_z
    b = np.zeros((K_rp, K_z), dtype=np.float64)
    Q = np.zeros((K, K), dtype=np.float64) if accumulate_Q else None
    n_pairs = 0

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
        m = (
            (rp >= rp_min) & (rp < rp_max)
            & (pi < pi_max)
            & (z_pair >= z_min) & (z_pair < z_max)
        )
        if not m.any():
            continue
        rp_in = rp[m]; z_in = z_pair[m]
        # log-rp -> [-1, 1]
        y_rp = 2.0 * np.log(rp_in / rp_min) / np.log(rp_max / rp_min) - 1.0
        y_z = 2.0 * (z_in - z_min) / (z_max - z_min) - 1.0
        # numpy Chebyshev recurrence
        T_rp = np.empty((len(rp_in), K_rp), dtype=np.float64)
        T_rp[:, 0] = 1.0
        if K_rp > 1:
            T_rp[:, 1] = y_rp
            for k in range(2, K_rp):
                T_rp[:, k] = 2.0 * y_rp * T_rp[:, k - 1] - T_rp[:, k - 2]
        T_z = np.empty((len(rp_in), K_z), dtype=np.float64)
        T_z[:, 0] = 1.0
        if K_z > 1:
            T_z[:, 1] = y_z
            for l in range(2, K_z):
                T_z[:, l] = 2.0 * y_z * T_z[:, l - 1] - T_z[:, l - 2]
        # outer product per pair, summed: O(K_rp K_z) per pair
        b += np.einsum("nk,nl->kl", T_rp, T_z)
        n_pairs += int(m.sum())
        if accumulate_Q:
            # T_outer[n, k, l] = T_rp[n,k] * T_z[n,l] -> flatten to T_outer_flat[n, K]
            T_outer = (T_rp[:, :, None] * T_z[:, None, :]).reshape(-1, K)
            Q += T_outer.T @ T_outer
    return b, Q, n_pairs


# ----- continuous estimator dataclass ----------------------------------

@dataclass
class WpContinuousEstimator:
    """SF21 continuous-function estimator solution for wp(rp, z*).

    Holds the basis amplitudes ``a_kl`` and the basis support so the
    JAX-pure ``wp_eval`` and ``dwp_dz`` methods can evaluate the
    continuous wp and its z-derivative at any (rp, z*) inside support.
    """
    a_kl: np.ndarray        # (K_rp, K_z) basis amplitudes
    rp_min: float; rp_max: float
    z_min: float; z_max: float
    K_rp: int; K_z: int
    pi_max: float           # for the leading 2*pi_max wp normalisation
    cov_kl: np.ndarray      # (K_rp*K_z, K_rp*K_z) basis-amplitude covariance
    n_d: int; n_r: int      # for diagnostics
    info: dict              # condition number, n_pairs, etc.

    def _y_rp(self, rp):
        import jax.numpy as jnp
        return 2.0 * jnp.log(rp / self.rp_min) / np.log(self.rp_max
                                                          / self.rp_min) - 1.0

    def _y_z(self, z):
        return 2.0 * (z - self.z_min) / (self.z_max - self.z_min) - 1.0

    def wp_eval(self, rp, z):
        """Evaluate wp(rp, z) continuously. JAX-pure; rp and z can be
        scalar or array (broadcastable shapes).
        """
        import jax.numpy as jnp
        a = jnp.asarray(self.a_kl)
        rp_j = jnp.asarray(rp)
        z_j = jnp.asarray(z)
        T_rp = _cheb_T_jax(self._y_rp(rp_j), self.K_rp)        # (..., K_rp)
        T_z = _cheb_T_jax(self._y_z(z_j), self.K_z)            # (..., K_z)
        # 2 pi_max wp normalisation absorbs the |pi|<pi_max indicator
        out = jnp.einsum("...k,kl,...l->...", T_rp, a, T_z)
        return 2.0 * self.pi_max * out

    def dwp_dz(self, rp, z):
        """Analytic ``d wp(rp, z) / dz`` via Chebyshev derivative."""
        import jax.numpy as jnp
        a = jnp.asarray(self.a_kl)
        rp_j = jnp.asarray(rp)
        z_j = jnp.asarray(z)
        T_rp = _cheb_T_jax(self._y_rp(rp_j), self.K_rp)
        dT_z_dy = _cheb_dT_dx_jax(self._y_z(z_j), self.K_z)
        dy_dz = 2.0 / (self.z_max - self.z_min)
        out = jnp.einsum("...k,kl,...l->...", T_rp, a, dT_z_dy)
        return 2.0 * self.pi_max * out * dy_dz


def wp_continuous_estimator(
    pos_data: np.ndarray, pos_random: np.ndarray,
    z_data: np.ndarray, z_random: np.ndarray,
    rp_min: float = 5.0, rp_max: float = 80.0,
    z_min: float = None, z_max: float = None,
    K_rp: int = 8, K_z: int = 5,
    pi_max: float = 200.0,
    chunk: int = 4000,
) -> WpContinuousEstimator:
    """SF21 continuous-function estimator for wp(rp, z*).

    Single end-to-end build: project DD/DR/RR pair counts onto a
    (rp, z) Chebyshev basis, accumulate the random-pair Gram matrix
    ``Q``, solve ``Q a = (DD - 2DR + RR)``, and return a
    JAX-evaluable estimator. ``WpContinuousEstimator.wp_eval(rp, z)``
    and ``.dwp_dz(rp, z)`` then deliver continuous wp(rp, z) and its
    redshift derivative.

    Notes on the Q matrix: it's the basis-Gram of the random pair
    density, K_rp*K_z square. For the basis sizes used here (~40)
    inversion is trivially conditioned.
    """
    if z_min is None: z_min = float(min(z_data.min(), z_random.min()))
    if z_max is None: z_max = float(max(z_data.max(), z_random.max()))
    N_d = len(pos_data); N_r = len(pos_random)
    Nd_pairs = N_d * (N_d - 1) / 2.0
    Nr_pairs = N_r * (N_r - 1) / 2.0

    # DD, DR, RR basis projections + Q (random Gram)
    DD, _, n_DD = _project_pairs_basis(
        pos_data, pos_data, z_data, z_data,
        rp_min, rp_max, z_min, z_max, pi_max,
        K_rp, K_z, auto=True, chunk=chunk, accumulate_Q=False,
    )
    DR, _, n_DR = _project_pairs_basis(
        pos_data, pos_random, z_data, z_random,
        rp_min, rp_max, z_min, z_max, pi_max,
        K_rp, K_z, auto=False, chunk=chunk, accumulate_Q=False,
    )
    RR, Q, n_RR = _project_pairs_basis(
        pos_random, pos_random, z_random, z_random,
        rp_min, rp_max, z_min, z_max, pi_max,
        K_rp, K_z, auto=True, chunk=chunk, accumulate_Q=True,
    )

    # LS-style normalisation
    DD_n = DD / Nd_pairs
    DR_n = DR / (N_d * N_r)
    RR_n = RR / Nr_pairs
    Q_n = Q / Nr_pairs

    # solve Q a = (DD - 2DR + RR) in flattened form
    K = K_rp * K_z
    b_flat = (DD_n - 2.0 * DR_n + RR_n).reshape(K)
    # condition number diagnostic
    cond = np.linalg.cond(Q_n)
    a_flat = np.linalg.solve(Q_n, b_flat)
    # asymptotic Gaussian covariance: cov(a) ~ Q^{-1} * (sigma^2 of b)
    # Approximate sigma^2 as Poisson on the normalised counts; for
    # a quick uncertainty: cov_a ~ Q^{-1} (DD/Nd_pairs^2 ...) Q^{-1}.
    # Here we use the simpler Q^{-1} approximation -- gives the right
    # order of magnitude for diagnostic purposes.
    Q_inv = np.linalg.inv(Q_n)
    cov_a = Q_inv * (1.0 / Nd_pairs)            # rough scale
    a_kl = a_flat.reshape(K_rp, K_z)

    return WpContinuousEstimator(
        a_kl=a_kl, rp_min=rp_min, rp_max=rp_max,
        z_min=z_min, z_max=z_max,
        K_rp=K_rp, K_z=K_z, pi_max=pi_max,
        cov_kl=cov_a, n_d=N_d, n_r=N_r,
        info={
            "n_DD": n_DD, "n_DR": n_DR, "n_RR": n_RR,
            "Q_cond": cond,
        },
    )
