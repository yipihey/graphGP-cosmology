"""The spectroscopically-missing CMASS targets (positions + ugriz colours).

BOSS targets came from SDSS DR8 imaging, so every spectroscopically-missing
galaxy (fiber collision w_cp, redshift failure w_noz) has a real photometric
detection — known angular position and colours; only its redshift is missing.
This module loads that missing set so the completion can place each missing
galaxy at its TRUE position with a redshift drawn from its photo-z.

``load_cmass_targets`` has two modes:

- **placeholder** (``path=None``, buildable now): synthesise the missing set from
  the LSS completeness weights — ``round(w_cp−1)`` collided + ``round(w_noz−1)``
  z-failures per observed host, at the host position (collided jittered within
  the collision scale) with the host's colours as a stand-in. Lets the whole
  completion pipeline be exercised before the real catalogue is fetched. (With
  the host's colours it cannot demonstrate the photo-z's discriminating power —
  it only exercises the plumbing; the real loader is needed for the science.)

- **real** (``path`` given): the fetched DR12 CMASS target catalogue cross-matched
  against the good-z LSS galaxies (see ``demos/fetch_cmass_targets.py``); the
  unmatched / ZWARNING≠0 objects are the missing set, with their real colours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CMASSTargets:
    """The spectroscopically-missing CMASS galaxies (one row per missing object)."""
    ra: np.ndarray            # (M,)
    dec: np.ndarray           # (M,)
    colors: np.ndarray        # (M,4) ugriz colours u-g,g-r,r-i,i-z (host stand-in in placeholder)
    mags: Optional[np.ndarray]    # (M,5) or None
    miss_kind: np.ndarray     # (M,) 'collided' | 'zfail'
    host_index: np.ndarray    # (M,) index into the observed LSS array (-1 if none)

    @property
    def N(self) -> int:
        return len(self.ra)


def load_cmass_targets(
    catalog,
    path: Optional[str] = None,
    *,
    seed: int = 0,
    collision_scale_deg: float = 62.0 / 3600.0,
) -> CMASSTargets:
    """Load (or synthesise) the missing CMASS targets for ``catalog``.

    ``catalog`` must have been loaded with ``with_photometry=True`` (for colours).
    Returns the missing set as a :class:`CMASSTargets`.
    """
    if path is not None:
        return _load_real_targets(catalog, path)

    # ---- placeholder: synthesise the missing set from the LSS weights ----
    rng = np.random.default_rng(seed)
    ra = np.asarray(catalog.ra_data, np.float64)
    dec = np.asarray(catalog.dec_data, np.float64)
    colors = np.asarray(catalog.colors_data)
    mags = np.asarray(catalog.mags_data) if catalog.mags_data is not None else None
    wcp = np.asarray(catalog.w_cp_data); wnoz = np.asarray(catalog.w_noz_data)

    n_cp = np.maximum(np.round(wcp - 1.0).astype(int), 0)    # collided partners per host
    n_noz = np.maximum(np.round(wnoz - 1.0).astype(int), 0)  # z-failures per host

    host = np.concatenate([np.repeat(np.arange(len(ra)), n_cp),
                           np.repeat(np.arange(len(ra)), n_noz)])
    kind = np.concatenate([np.full(int(n_cp.sum()), "collided"),
                           np.full(int(n_noz.sum()), "zfail")])
    # collided partners sit within the collision scale; z-failures at the target
    # position (the failed object's own imaging position ≈ host's neighbourhood)
    s = np.radians(collision_scale_deg) / 3.0
    jit = (kind == "collided").astype(float)
    m = len(host)
    dra = np.degrees(rng.normal(0, s, m) * jit / np.cos(np.radians(dec[host])))
    ddec = np.degrees(rng.normal(0, s, m) * jit)
    return CMASSTargets(
        ra=ra[host] + dra, dec=dec[host] + ddec,
        colors=colors[host], mags=(mags[host] if mags is not None else None),
        miss_kind=kind, host_index=host)


def _load_real_targets(catalog, path, collision_scale_deg=62.0 / 3600.0):
    """Load the fetched CMASS target catalogue and build the missing set.

    The fetch (``demos/fetch_cmass_targets.py``) gives all CMASS *colour-selected*
    targets with extinction-corrected ugriz mags + spec match (spec_z, zwarning).
    Classify:
      - **z-failure**: has a spectrum but ``zwarning≠0`` → real position+colours,
        missing redshift. host = nearest good-z LSS galaxy.
      - **collided**: no spectrum (never fibered). The colour-selected no-spec
        pool over-counts (it includes never-BOSS-targeted objects), so we tie it
        to the LSS bookkeeping: for each LSS survivor with ``w_cp>1``, claim its
        ``round(w_cp−1)`` nearest *unclaimed* no-spec targets within the collision
        scale. This matches both the weight-implied count and the collision
        geometry, using real imaging positions+colours. Unclaimed pool objects
        (never-targeted / vetoed) are discarded.
    """
    from astropy.io import fits
    from scipy.spatial import cKDTree
    from .observed import _radec_to_nhat

    t = fits.open(path)[1].data
    tra = np.asarray(t["ra"], np.float64); tdec = np.asarray(t["dec"], np.float64)
    mags = np.column_stack([t["u"], t["g"], t["r"], t["i_mod"], t["z_mod"]]).astype(np.float64)
    colors = mags[:, :-1] - mags[:, 1:]
    spec_z = np.asarray(t["spec_z"], np.float64)
    zw = np.asarray(t["zwarning"], np.float64)
    matched = np.isfinite(spec_z)

    ra_d = np.asarray(catalog.ra_data, np.float64)
    dec_d = np.asarray(catalog.dec_data, np.float64)
    wcp = np.asarray(catalog.w_cp_data); wnoz = np.asarray(catalog.w_noz_data)
    nhat_d = _radec_to_nhat(ra_d, dec_d)

    def _tie(pool_global, weight, scale_deg):
        """Greedily claim, for each survivor with weight>1, its round(weight−1)
        nearest *unclaimed* pool targets within ``scale_deg`` — matching the
        weight-implied count and proximity. Returns (claimed_global, host)."""
        if not len(pool_global):
            return np.zeros(0, int), np.zeros(0, int)
        nhat_pool = _radec_to_nhat(tra[pool_global], tdec[pool_global])
        ptree = cKDTree(nhat_pool)
        chord = 2.0 * np.sin(np.radians(scale_deg) / 2.0)
        surv = np.where(weight > 1.0)[0]
        surv = surv[np.argsort(-weight[surv])]           # largest first
        claimed = np.zeros(len(pool_global), bool)
        out_g, out_h = [], []
        for s in surv:
            need = int(round(weight[s] - 1.0))
            if need <= 0:
                continue
            cand = [n for n in ptree.query_ball_point(nhat_d[s], chord) if not claimed[n]]
            if len(cand) > need:
                d = np.linalg.norm(nhat_pool[cand] - nhat_d[s], axis=1)
                cand = [cand[j] for j in np.argsort(d)[:need]]
            for n in cand:
                claimed[n] = True; out_g.append(pool_global[n]); out_h.append(s)
        return np.asarray(out_g, int), np.asarray(out_h, int)

    # collided: no-spec pool tied to w_cp survivors within the collision scale
    coll_idx, coll_host = _tie(np.where(~matched)[0], wcp, collision_scale_deg)
    # z-failures: matched-but-zwarning≠0 pool tied to w_noz survivors (the noz
    # upweight goes to the nearest good-z neighbour, ~CMASS NN distance, so a
    # generous ~15' scale)
    zf_idx, zf_host = _tie(np.where(matched & (zw != 0))[0], wnoz, 15.0 / 60.0)

    idx = np.concatenate([coll_idx, zf_idx]).astype(int)
    kind = np.concatenate([np.full(len(coll_idx), "collided"),
                           np.full(len(zf_idx), "zfail")])
    host = np.concatenate([coll_host, zf_host]).astype(int)
    return CMASSTargets(
        ra=tra[idx], dec=tdec[idx], colors=colors[idx], mags=mags[idx],
        miss_kind=kind, host_index=host)
