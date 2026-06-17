"""Density-coupling of spectroscopic selection — measured, cosmology-free.

Risa Wechsler's v0 forward-model document (the LGCP / GraphGP field pipeline)
makes one point we had not tested: spectroscopic *selection* (redshift failures,
fiber collisions) can be **density-coupled** — its rate depends on the local
galaxy overdensity. Modelled as a redshift-success probability
``S_zsucc = σ(β·o + ε(n̂) + h·δ_g)``, the coupling ``h`` is *identifiable*, and
ignoring a non-zero ``h`` is exactly the mechanism (Thomas et al. 2011, MegaZ)
by which redshift failures imprint spurious large-scale power.

This module measures that coupling **observationally and cosmology-free**:

  * ``local_overdensity`` — a purely angular, random-normalised local overdensity
    1+δ at arbitrary sky positions. Counts of spectroscopic successes in an
    aperture are divided by the count expected from the random catalogue (which
    encodes the angular selection), so δ is a true local overdensity that needs
    no distances, no fiducial cosmology, and is automatically footprint/
    completeness-aware.
  * ``measure_failure_coupling`` — the coupling ``h`` itself: at every member of
    the *total target sample* (successes ∪ a given failure kind) we evaluate the
    success-field δ, then fit the success indicator against δ with a logistic
    (IRLS) model. ``h`` is the data-space analogue of Risa's density coupling,
    with a bootstrap error and a label-shuffle **null test** (shuffling must give
    h≈0). Reported separately for redshift failures (the S_zsucc analogue) and
    fiber collisions (expected to over-occupy dense regions by construction).
  * ``total_target_density`` — the *selection-immune* amplitude anchor: the
    angular density of the total target sample (successes + failures). Because
    every colour-selected target enters regardless of whether it yielded a
    redshift, the spectroscopic-success selection cancels in the total, so this
    map is the cleanest cosmology-free reference for the completion amplitude.

Everything is angular and random-normalised: cosmology-free, fully data-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .observed import _radec_to_nhat


def _aperture_chord(aperture_deg: float) -> float:
    """Euclidean chord length on the unit sphere for an angular aperture."""
    return 2.0 * np.sin(np.radians(aperture_deg) / 2.0)


def local_overdensity(
    query_ra, query_dec,
    field_ra, field_dec,
    rand_ra, rand_dec,
    *,
    aperture_deg: float = 0.5,
    min_rand: float = 10.0,
    self_exclude: bool = False,
):
    """Random-normalised angular local overdensity 1+δ−1 = δ at query positions.

    For each query point we count the *field* galaxies (spectroscopic successes)
    and the *randoms* within an angular aperture. The randoms encode the angular
    selection function, so the count expected at δ=0 is
    ``(N_field/N_rand)·n_rand``; the local overdensity is

        δ = n_field / [ (N_field/N_rand)·n_rand ] − 1 .

    Purely angular and random-normalised — no distances, no cosmology, and
    automatically corrected for footprint edges and angular completeness. Pixels
    with fewer than ``min_rand`` randoms in the aperture are returned as NaN
    (too close to an edge/hole to estimate). ``self_exclude`` subtracts the query
    point itself from the field count (use when the query points ARE field
    galaxies, e.g. the success sample evaluated against itself).
    """
    from scipy.spatial import cKDTree

    r = _aperture_chord(aperture_deg)
    q = _radec_to_nhat(np.asarray(query_ra, np.float64), np.asarray(query_dec, np.float64))
    field = _radec_to_nhat(np.asarray(field_ra, np.float64), np.asarray(field_dec, np.float64))
    rand = _radec_to_nhat(np.asarray(rand_ra, np.float64), np.asarray(rand_dec, np.float64))

    n_field = cKDTree(field).query_ball_point(q, r, return_length=True).astype(np.float64)
    n_rand = cKDTree(rand).query_ball_point(q, r, return_length=True).astype(np.float64)
    if self_exclude:
        n_field = np.maximum(n_field - 1.0, 0.0)

    norm = len(field) / len(rand)                  # mean data-per-random
    expected = norm * n_rand
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(n_rand >= min_rand, n_field / expected - 1.0, np.nan)
    return delta


def _logistic_irls(x, y, iters: int = 100, ridge: float = 1e-8):
    """Logistic regression of binary ``y`` on ``[1, x]`` by IRLS.

    Returns ``(intercept, slope)``; the slope is the density coupling ``h``
    (log-odds of success per unit δ). scipy-free, transparent, cosmology-free.
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-6, 1.0 - 1e-6)
        W = p * (1.0 - p)
        z = eta + (y - p) / W
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X + ridge * np.eye(2), XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


@dataclass
class CouplingResult:
    """Measured density coupling of a selection kind (cosmology-free)."""
    kind: str                  # 'zfail' | 'collided'
    h: float                   # logistic density coupling (log-odds of success per unit δ)
    h_err: float               # bootstrap 1σ
    h_null_mean: float         # label-shuffle null mean (≈0 expected)
    h_null_std: float          # label-shuffle null scatter
    z_score: float             # (h − h_null_mean) / sqrt(h_err² + h_null_std²)
    pearson_r: float           # model-free corr(success indicator, δ)
    n_success: int
    n_fail: int
    delta_bin_centres: np.ndarray   # binned S(δ) curve for plotting
    S_of_delta: np.ndarray
    S_err: np.ndarray
    aperture_deg: float

    @property
    def detected(self) -> bool:
        return abs(self.z_score) >= 3.0


def measure_failure_coupling(
    catalog, targets,
    *,
    rand_ra, rand_dec,
    kind: str = "zfail",
    aperture_deg: float = 0.5,
    n_bins: int = 8,
    n_boot: int = 200,
    min_rand: float = 10.0,
    seed: int = 0,
) -> CouplingResult:
    """Measure the density coupling ``h`` of a selection ``kind`` (cosmology-free).

    The *total target sample* is the spectroscopic successes (``catalog``) plus
    the missing targets of the requested kind (``targets`` with
    ``miss_kind==kind``) — both colour-selected, so the only thing that differs
    is whether a redshift was obtained. We evaluate the **success-field** local
    overdensity δ (``local_overdensity`` against the observed successes and the
    randoms) at every member, then fit the success indicator against δ with a
    logistic IRLS model. The slope ``h`` is the data-space analogue of Risa's
    ``h δ_g`` coupling:

      * ``kind='zfail'`` — the S_zsucc analogue. ``h≈0`` ⇒ redshift failures are
        density-blind and our local-placement completion needs no density term;
        ``h≠0`` ⇒ failures cluster (or anti-cluster) with the field, the MegaZ
        spurious-power risk, which our real-position completion must reproduce.
      * ``kind='collided'`` — fiber collisions, expected to over-occupy dense
        regions by construction (close pairs ⇒ collisions), i.e. h<0 for success.

    Errors come from an object-resampling bootstrap; the **null** comes from
    shuffling the success/fail labels (destroying any real coupling), so a
    detection is ``|h − h_null| ≫ √(h_err² + h_null_std²)``. Fully observational.
    """
    rng = np.random.default_rng(seed)
    ra_d = np.asarray(catalog.ra_data, np.float64)
    dec_d = np.asarray(catalog.dec_data, np.float64)
    mask = np.asarray(targets.miss_kind) == kind
    fra = np.asarray(targets.ra, np.float64)[mask]
    fdec = np.asarray(targets.dec, np.float64)[mask]

    # success-field δ at successes (self-excluded) and at the failures
    d_succ = local_overdensity(ra_d, dec_d, ra_d, dec_d, rand_ra, rand_dec,
                               aperture_deg=aperture_deg, min_rand=min_rand, self_exclude=True)
    d_fail = local_overdensity(fra, fdec, ra_d, dec_d, rand_ra, rand_dec,
                               aperture_deg=aperture_deg, min_rand=min_rand, self_exclude=False)

    delta = np.concatenate([d_succ, d_fail])
    label = np.concatenate([np.ones(len(d_succ)), np.zeros(len(d_fail))])
    ok = np.isfinite(delta)
    delta = delta[ok]; label = label[ok]

    _, h = _logistic_irls(delta, label)
    pear = float(np.corrcoef(label, delta)[0, 1])

    # bootstrap over objects
    n = len(delta)
    hb = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        _, hb[b] = _logistic_irls(delta[idx], label[idx])
    h_err = float(np.std(hb))

    # null: shuffle labels
    hn = np.empty(n_boot)
    for b in range(n_boot):
        _, hn[b] = _logistic_irls(delta, rng.permutation(label))
    h_null_mean = float(np.mean(hn)); h_null_std = float(np.std(hn))
    z = (h - h_null_mean) / np.sqrt(h_err ** 2 + h_null_std ** 2 + 1e-30)

    # binned S(δ) for plotting (equal-count bins)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(delta, qs))
    which = np.clip(np.digitize(delta, edges) - 1, 0, len(edges) - 2)
    cen = np.empty(len(edges) - 1); S = np.empty(len(edges) - 1); Se = np.empty(len(edges) - 1)
    for j in range(len(edges) - 1):
        m = which == j
        cen[j] = np.median(delta[m]) if m.any() else np.nan
        nj = max(int(m.sum()), 1)
        S[j] = label[m].mean() if m.any() else np.nan
        Se[j] = np.sqrt(S[j] * (1 - S[j]) / nj) if m.any() else np.nan

    return CouplingResult(
        kind=kind, h=h, h_err=h_err, h_null_mean=h_null_mean, h_null_std=h_null_std,
        z_score=float(z), pearson_r=pear,
        n_success=int(len(d_succ)), n_fail=int(np.isfinite(d_fail).sum()),
        delta_bin_centres=cen, S_of_delta=S, S_err=Se, aperture_deg=aperture_deg)


def total_target_density(catalog, targets, *, nside: int = 256):
    """Selection-immune angular density of the total target sample (cosmology-free).

    The total target sample = spectroscopic successes (``catalog``) ∪ all missing
    targets (``targets``). Every member is a colour-selected target irrespective
    of whether it yielded a redshift, so the spectroscopic-success selection
    *cancels* in the total. Histogrammed on HEALPix and normalised by its median
    populated pixel, this is the cleanest cosmology-free reference density for
    the completion amplitude (a real underdensity, not a selection loss).

    Returns ``(counts, density, nside)`` where ``density`` is counts normalised
    to the median populated pixel (≈ 1+δ of the *target* field).
    """
    import healpy as hp

    ra = np.concatenate([np.asarray(catalog.ra_data, np.float64),
                         np.asarray(targets.ra, np.float64)])
    dec = np.concatenate([np.asarray(catalog.dec_data, np.float64),
                          np.asarray(targets.dec, np.float64)])
    npix = 12 * nside ** 2
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec), np.deg2rad(ra))
    counts = np.bincount(pix, minlength=npix).astype(np.float64)
    med = np.median(counts[counts > 0])
    return counts, counts / med, nside
