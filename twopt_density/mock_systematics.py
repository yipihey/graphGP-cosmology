"""Forward model of BOSS spectroscopic systematics, for truth-recovery tests.

The completion pipeline claims to recover the galaxies that would have been
observed absent fiber collisions, redshift failures and imaging systematics. To
test that claim against a KNOWN truth (not just by-construction closure), we take
a *truth* catalogue (e.g. the real BOSS galaxies — a fair sample of the true
field, with realistic clustering, colours and n(z)), inject a controlled,
realistic systematics model, and hand the completion the degraded "observed"
catalogue + missing targets in exactly the BOSS schema. Recovering the truth
statistics from the completed ensemble is then a genuine inject-and-recover test.

``apply_survey_systematics`` injects, in order:
  1. **Imaging systematics**: thin each galaxy with keep-probability 1/w_systot
     (so survivors upweighted by w_systot recover the truth density); survivors
     carry w_systot. The w_systot field is supplied (e.g. the real BOSS pattern).
  2. **Fiber collisions**: within the collision scale, remove one member of a
     fraction of close pairs -> "collided" targets (true position+colours kept,
     redshift removed); the surviving partner is the host.
  3. **Redshift failures**: remove a fraction of the remainder (optionally
     faint-i biased, density coupling configurable) -> "zfail" targets.

Returns a ``MockObserved`` (BOSS-catalogue-like, accepted by
``complete_catalog_photoz``) and a :class:`~twopt_density.cmass_targets.CMASSTargets`.
The mask-hole veto is deliberately NOT applied here — hole inpainting is a
separate product validated separately; this isolates the completion's domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .observed import _radec_to_nhat
from .cmass_targets import CMASSTargets


def load_patchy_truth(path, real_cat, *, z_min=0.43, z_max=0.7, nside_wsys=128, seed=0):
    """Load a MultiDark-Patchy SGC COMPSAM mock as a TRUTH catalogue for recovery.

    Patchy mocks have realistic, N-body-calibrated clustering INDEPENDENT of the
    real data (a stronger truth than using the real galaxies). Columns are
    ``RA Dec z log10(Mstar) nbar bias veto w_cp``; we keep the CMASS z-range with
    ``veto==1`` (the complete parent). Patchy has no colours and no imaging-
    systematic field, so we attach realistic ones for the forward model: colours
    & mags are drawn from real CMASS galaxies matched in redshift (preserving the
    colour-z relation + scatter), and a per-galaxy w_systot is looked up from a
    HEALPix map of the real BOSS WEIGHT_SYSTOT (the true spatial pattern; 1 where
    the footprint is uncovered). Returns ``(ra, dec, z, colors, mags, w_systot)``."""
    import healpy as hp
    rng = np.random.default_rng(seed)
    arr = np.loadtxt(path, usecols=(0, 1, 2, 6))
    ra, dec, z, veto = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    keep = (z >= z_min) & (z <= z_max) & (veto == 1)
    ra, dec, z = ra[keep], dec[keep], z[keep]

    # colours/mags from real CMASS matched in z (fine z-bins, random draw per bin)
    zr = np.asarray(real_cat.z_data); cr = np.asarray(real_cat.colors_data)
    mr = np.asarray(real_cat.mags_data) if real_cat.mags_data is not None else None
    fin = np.isfinite(cr).all(axis=1)
    zr, cr = zr[fin], cr[fin]; mr = None if mr is None else mr[fin]
    edges = np.linspace(z_min, z_max, 200)
    which_r = np.clip(np.digitize(zr, edges) - 1, 0, len(edges) - 2)
    bins = {b: np.where(which_r == b)[0] for b in np.unique(which_r)}
    which_p = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
    sel = np.empty(len(z), int)
    allidx = np.arange(len(zr))
    for b in np.unique(which_p):
        pool = bins.get(b, allidx)
        m = which_p == b
        sel[m] = rng.choice(pool, int(m.sum()), replace=True)
    colors = cr[sel]; mags = None if mr is None else mr[sel]

    # w_systot from a HEALPix map of the real BOSS WEIGHT_SYSTOT
    wmap = np.ones(hp.nside2npix(nside_wsys))
    pr = hp.ang2pix(nside_wsys, np.radians(90 - np.asarray(real_cat.dec_data)),
                    np.radians(np.asarray(real_cat.ra_data) % 360))
    wsys_r = np.asarray(real_cat.w_sys_data)
    num = np.bincount(pr, weights=wsys_r, minlength=len(wmap))
    den = np.bincount(pr, minlength=len(wmap))
    wmap[den > 0] = num[den > 0] / den[den > 0]
    pp = hp.ang2pix(nside_wsys, np.radians(90 - dec), np.radians(ra % 360))
    w_systot = wmap[pp]
    return ra, dec, z, colors, mags, w_systot


def load_patchy_randoms(path, *, z_min=0.43, z_max=0.7, max_n=None, seed=0):
    """Load Patchy randoms (RA Dec z nbar bias veto w_cp); keep z-range, veto==1."""
    arr = np.loadtxt(path, usecols=(0, 1, 2, 5))
    ra, dec, z, veto = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    keep = (z >= z_min) & (z <= z_max) & (veto == 1)
    ra, dec, z = ra[keep], dec[keep], z[keep]
    if max_n is not None and len(ra) > max_n:
        j = np.random.default_rng(seed).choice(len(ra), max_n, replace=False)
        ra, dec, z = ra[j], dec[j], z[j]
    return ra, dec, z


@dataclass
class MockObserved:
    """Minimal BOSS-catalogue interface used by complete_catalog_photoz / photoz."""
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray
    w_sys_data: np.ndarray
    w_cp_data: np.ndarray
    w_noz_data: np.ndarray
    colors_data: np.ndarray
    mags_data: Optional[np.ndarray]
    imatch_data: np.ndarray
    colors_finite: np.ndarray

    @property
    def N_data(self):
        return len(self.ra_data)


def apply_survey_systematics(
    ra, dec, z, colors, mags, w_systot,
    *,
    collision_scale_deg: float = 62.0 / 3600.0,
    coll_frac: float = 0.6,
    zfail_frac: float = 0.014,
    zfail_faint_bias: float = 1.0,
    zfail_density_coupling: float = 0.0,
    tie_scale_deg: float = 15.0 / 60.0,
    seed: int = 0,
):
    """Inject imaging-systematic thinning + fiber collisions + redshift failures
    into a truth catalogue. Returns ``(observed: MockObserved, targets: CMASSTargets,
    truth_kept_mask)`` where ``truth_kept_mask`` flags which input galaxies survive
    as observed (the rest are collided/zfail targets or systot-thinned)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    ra = np.asarray(ra, float); dec = np.asarray(dec, float); z = np.asarray(z, float)
    colors = np.asarray(colors); mags = None if mags is None else np.asarray(mags)
    w_systot = np.asarray(w_systot, float)
    n = len(ra)
    idx = np.arange(n)

    # 1. imaging-systematic thinning: keep with prob 1/w_systot (survivors *w_systot = truth)
    keep_sys = rng.random(n) < 1.0 / np.clip(w_systot, 1e-3, None)

    # 2. fiber collisions among the systot survivors: remove one of a fraction of
    #    close pairs (the "collided" galaxy); the partner is its host.
    surv = idx[keep_sys]
    nhat = _radec_to_nhat(ra[surv], dec[surv])
    chord = 2.0 * np.sin(np.radians(collision_scale_deg) / 2.0)
    pairs = cKDTree(nhat).query_pairs(chord, output_type="ndarray")
    collided_local = np.zeros(len(surv), bool)
    coll_host_local = np.full(len(surv), -1)
    if len(pairs):
        order = rng.permutation(len(pairs))
        for p in pairs[order]:
            a, b = int(p[0]), int(p[1])
            if collided_local[a] or collided_local[b]:
                continue
            if rng.random() < coll_frac:
                rm, host = (a, b) if rng.random() < 0.5 else (b, a)
                if not collided_local[host]:
                    collided_local[rm] = True; coll_host_local[rm] = host
    collided = surv[collided_local]
    coll_host_surv = coll_host_local[collided_local]               # local survivor idx of host

    # 3. redshift failures among the non-collided survivors (optionally faint/density biased)
    avail_local = np.where(~collided_local)[0]
    p_fail = np.ones(len(avail_local))
    if mags is not None and zfail_faint_bias != 1.0:               # fainter i -> more failures
        imag = mags[surv[avail_local], 3]
        p_fail *= zfail_faint_bias ** ((imag - np.median(imag)) / max(imag.std(), 1e-6))
    p_fail *= len(avail_local) * zfail_frac / p_fail.sum()
    zfail_local = avail_local[rng.random(len(avail_local)) < np.clip(p_fail, 0, 1)]

    # final observed = survivors that are neither collided nor zfail
    removed_local = np.zeros(len(surv), bool)
    removed_local[collided_local] = True
    removed_local[zfail_local] = True
    obs_local = np.where(~removed_local)[0]
    obs = surv[obs_local]                                          # global truth idx of observed

    # map local-survivor index -> position in the observed array (host_index target)
    loc_to_obs = -np.ones(len(surv), int)
    loc_to_obs[obs_local] = np.arange(len(obs_local))
    obs_nhat = _radec_to_nhat(ra[obs], dec[obs])
    otree = cKDTree(obs_nhat)

    def _nearest_obs(global_idx):
        d, j = otree.query(_radec_to_nhat(ra[global_idx], dec[global_idx]))
        return j

    # collided host: prefer its true partner if that partner is observed, else nearest obs
    coll_host_obs = loc_to_obs[coll_host_surv]
    bad = coll_host_obs < 0
    if bad.any():
        coll_host_obs[bad] = _nearest_obs(collided[bad])
    zfail = surv[zfail_local]
    zfail_host_obs = _nearest_obs(zfail)

    # assemble observed catalogue (BOSS schema)
    fin = np.isfinite(colors[obs]).all(axis=1) if colors.ndim == 2 else np.ones(len(obs), bool)
    observed = MockObserved(
        ra_data=ra[obs], dec_data=dec[obs], z_data=z[obs],
        w_sys_data=w_systot[obs], w_cp_data=np.ones(len(obs)), w_noz_data=np.ones(len(obs)),
        colors_data=colors[obs], mags_data=None if mags is None else mags[obs],
        imatch_data=np.ones(len(obs), int), colors_finite=fin)

    # assemble missing targets (collided + zfail), host_index into observed
    t_idx = np.concatenate([collided, zfail])
    t_kind = np.array(["collided"] * len(collided) + ["zfail"] * len(zfail))
    t_host = np.concatenate([coll_host_obs, zfail_host_obs]).astype(int)
    targets = CMASSTargets(
        ra=ra[t_idx], dec=dec[t_idx], colors=colors[t_idx],
        mags=None if mags is None else mags[t_idx], miss_kind=t_kind, host_index=t_host)

    kept_mask = np.zeros(n, bool); kept_mask[obs] = True
    targets_true_z = z[t_idx]            # ORACLE: the true redshifts of the missing (for diagnostics)
    return observed, targets, kept_mask, targets_true_z
