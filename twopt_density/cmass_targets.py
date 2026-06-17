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


def _load_real_targets(catalog, path):
    """Load the fetched CMASS target catalogue and flag the missing set.

    Expects a FITS/npz with RA, DEC, ugriz model fluxes (or mags) + extinction,
    ZWARNING, and a spec-match flag (specObjID present / NULL). Cross-match to the
    good-z LSS galaxies (cKDTree on unit vectors, ~1″) → unmatched = collided
    (never fibered); matched-with-ZWARNING≠0 = z-failure. Assign host_index via
    the nearest good-z galaxy (and the INGROUP/MULTGROUP collision groups when
    available). NotImplemented until the fetch lands — see fetch_cmass_targets.py.
    """
    raise NotImplementedError(
        "Real CMASS target loader pending the fetch (demos/fetch_cmass_targets.py). "
        "Use load_cmass_targets(catalog, path=None, ...) for the placeholder.")
