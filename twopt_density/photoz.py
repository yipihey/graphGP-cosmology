"""Empirical k-NN photometric-redshift POSTERIOR (scipy only).

For completing the BOSS catalog we need, for each spectroscopically-missing
galaxy, a *redshift posterior* p(z | colours) we can sample — not just a point
estimate — sourced from its SDSS imaging (ugriz). We learn it non-parametrically
from the observed spectroscopic sample (colours + spec-z, already in the LSS
file): the posterior for a query object is the empirical, inverse-distance-
weighted redshift distribution of its k nearest neighbours in colour space.

Why k-NN (vs a parametric regressor or conditional KDE):
- it returns a genuine posterior (the neighbour z's) to sample, with no Gaussian
  assumption — essential for spanning the redshift uncertainty across catalog
  realizations;
- it is dependency-light (one ``scipy.spatial.cKDTree``; sklearn is not
  installed) and matches the colour-local physics of a narrow colour-selected
  sample like CMASS.

Features are supplied by the caller (typically the reliable g−r, r−i, i−z colours
plus the i-band magnitude; the u-band is dropped because CMASS galaxies are very
red and u flux is mostly noise). They are whitened internally.
"""

from __future__ import annotations

import numpy as np


def photoz_features(colors, mags):
    """Standard photo-z features for CMASS: g−r, r−i, i−z, i_mag.

    Drops the u-band (CMASS galaxies are very red → u flux is mostly noise).
    ``colors`` is (N,4) [u-g,g-r,r-i,i-z]; ``mags`` is (N,5) ugriz. Use this for
    BOTH training and the missing-target query so the colour space matches.
    """
    colors = np.asarray(colors); mags = np.asarray(mags)
    return np.column_stack([colors[:, 1], colors[:, 2], colors[:, 3], mags[:, 3]])


class PhotoZKNN:
    """k-NN colour→redshift posterior sampler."""

    def __init__(self, k: int = 100, eps: float = 1e-6):
        self.k = int(k)
        self.eps = float(eps)
        self._tree = None
        self._z = None
        self._mu = None
        self._sd = None

    def _whiten(self, X):
        return (np.asarray(X, np.float64) - self._mu) / self._sd

    def fit(self, features, z):
        """Build the tree on whitened training features with spec-z labels."""
        from scipy.spatial import cKDTree

        X = np.asarray(features, np.float64)
        z = np.asarray(z, np.float64)
        good = np.isfinite(X).all(axis=1) & np.isfinite(z)
        X, z = X[good], z[good]
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-12
        self._tree = cKDTree(self._whiten(X))
        self._z = z
        return self

    def posterior(self, features):
        """Return ``(z_neighbours[M,k], weights[M,k])`` for query features.

        Weights are inverse-distance ``1/(d²+eps)``, normalised per row. Rows
        with non-finite features get NaN (caller handles fallback).
        """
        Xq = self._whiten(features)
        finite = np.isfinite(Xq).all(axis=1)
        zk = np.full((len(Xq), self.k), np.nan)
        wk = np.full((len(Xq), self.k), np.nan)
        if finite.any():
            d, idx = self._tree.query(Xq[finite], k=self.k, workers=-1)
            d = np.atleast_2d(d); idx = np.atleast_2d(idx)
            w = 1.0 / (d ** 2 + self.eps)
            w /= w.sum(axis=1, keepdims=True)
            zk[finite] = self._z[idx]
            wk[finite] = w
        return zk, wk

    def sample(self, features, rng, n: int = 1, reweight=None):
        """Draw ``n`` redshift(s) per query object from its (re)weighted posterior.

        ``reweight`` optionally multiplies the neighbour weights by an extra
        per-(object, neighbour) factor (shape (M,k)) — used to fold in the
        close-pair clustering prior. Returns ``z[M]`` for n=1 else ``z[M,n]``.
        """
        zk, wk = self.posterior(features)
        if reweight is not None:
            wk = wk * np.asarray(reweight)
        out = np.full((len(zk), n), np.nan)
        for i in range(len(zk)):
            w = wk[i]
            if not np.isfinite(w).any() or np.nansum(w) <= 0:
                continue
            w = np.where(np.isfinite(w), w, 0.0)
            w = w / w.sum()
            out[i] = rng.choice(zk[i], size=n, p=w)
        return out[:, 0] if n == 1 else out

    def point(self, features, stat: str = "median"):
        """Point photo-z (weighted median/mean of the posterior) — diagnostics."""
        zk, wk = self.posterior(features)
        out = np.full(len(zk), np.nan)
        for i in range(len(zk)):
            w = wk[i]
            if not np.isfinite(w).any():
                continue
            order = np.argsort(zk[i]); zz = zk[i][order]; ww = w[order]
            if stat == "mean":
                out[i] = np.sum(zz * ww)
            else:
                c = np.cumsum(ww); out[i] = zz[np.searchsorted(c, 0.5 * c[-1])]
        return out
