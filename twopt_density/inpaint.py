"""Cosmology-independent inpainting of interior survey-mask holes.

Bright-star masks, bad fields and tiling gaps remove area entirely from a
spectroscopic survey: inside these interior holes there are no galaxies, no
randoms, and no usable imaging. For two-point clustering with masked randoms the
holes cancel, but a downstream theorist who wants hole-free catalogs needs them
filled. We fill each hole by **transplanting real galaxies** — with their
colours, magnitudes and local spatial configuration — from environment-matched
nearby clean regions. This preserves higher-order clustering and the
colour/luminosity–clustering joint by construction, uses no field model (it is
data resampling), and stays in observed (RA, Dec, z) coordinates: cosmology-free.

Pipeline:
  1. ``fine_completeness_map`` — a finer HEALPix completeness/count map from the
     dense random catalogue (resolves arcmin-scale holes the nside=256 map misses;
     sub-arcmin masks remain unresolved — we lack the mangle veto polygons).
  2. ``find_interior_holes`` — connected components of empty interior pixels.
  3. ``inpaint_holes`` — analog-transplant fill, multiple realizations.

This is an optional, clearly-flagged hole-free *field product*, separate from the
unbiased masked clustering catalogs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .observed import _radec_to_nhat


def fine_completeness_map(ra_random, dec_random, nside: int = 1024):
    """HEALPix random-COUNT and completeness maps at a fine nside.

    Returns ``(counts, completeness)``: per-pixel random counts and counts
    normalised by the median populated pixel (∈[0,1]). The dense random
    catalogue (~hundreds per pixel at nside=1024) makes a count-0 pixel
    surrounded by populated pixels an unambiguous hole.
    """
    import healpy as hp
    npix = 12 * nside ** 2
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - np.asarray(dec_random)),
                     np.deg2rad(np.asarray(ra_random)))
    counts = np.bincount(pix, minlength=npix).astype(np.float64)
    rho = np.median(counts[counts > 0])
    return counts, np.clip(counts / rho, 0.0, 1.0)


@dataclass
class Hole:
    pixels: np.ndarray        # HEALPix pixel ids (the empty interior pixels)
    ra: float                 # centroid RA [deg]
    dec: float                # centroid Dec [deg]
    radius_deg: float         # bounding angular radius from centroid
    area_deg2: float


def find_interior_holes(counts, nside: int, *, empty_count: float = 0.0,
                        min_neighbour_frac: float = 0.75, min_pixels: int = 1):
    """Connected interior holes: genuinely unsurveyed pixels enclosed by footprint.

    A pixel is 'empty' if its random count is ``<= empty_count`` (default 0, i.e.
    no randoms = unsurveyed; with a robust ~tens-per-pixel density a truly-filled
    pixel reads 0 with negligible probability). It is 'interior' if at least
    ``min_neighbour_frac`` of its 8 HEALPix neighbours are populated (excludes the
    survey edge). Partial-completeness pixels (count>0) are NOT holes — they still
    contain galaxies and are handled by the weighting/randoms, not inpainted.
    Contiguous empty-interior pixels are grouped (BFS over HEALPix adjacency).
    Returns a list of :class:`Hole` (centroid, bounding radius, area).
    """
    import healpy as hp
    counts = np.asarray(counts)
    populated = counts > empty_count
    empty = ~populated
    npix = len(counts)
    allpix = np.arange(npix)
    # interior empty pixels (most neighbours populated)
    nb = hp.get_all_neighbours(nside, allpix)              # (8, npix)
    valid = nb >= 0
    nfrac = np.where(valid, populated[np.clip(nb, 0, npix - 1)], 0).sum(0) / np.maximum(valid.sum(0), 1)
    interior_empty = np.where(empty & (nfrac >= min_neighbour_frac))[0]
    iset = set(int(p) for p in interior_empty)

    # group contiguous interior-empty pixels (BFS)
    holes = []
    seen = set()
    for p0 in interior_empty:
        p0 = int(p0)
        if p0 in seen:
            continue
        comp = [p0]; seen.add(p0); stack = [p0]
        while stack:
            q = stack.pop()
            for nn in hp.get_all_neighbours(nside, q):
                nn = int(nn)
                if nn >= 0 and nn in iset and nn not in seen:
                    seen.add(nn); comp.append(nn); stack.append(nn)
        comp = np.array(comp)
        if len(comp) < min_pixels:
            continue
        vec = np.array(hp.pix2vec(nside, comp))            # (3, n)
        cvec = vec.mean(1); cvec /= np.linalg.norm(cvec)
        cdec = np.degrees(np.arcsin(cvec[2])); cra = np.degrees(np.arctan2(cvec[1], cvec[0])) % 360
        # bounding radius = max angular sep from centroid
        cos_sep = np.clip(cvec @ vec, -1, 1)
        radius = float(np.degrees(np.arccos(cos_sep.min())))
        area = len(comp) * hp.nside2pixarea(nside, degrees=True)
        holes.append(Hole(pixels=comp, ra=cra, dec=cdec, radius_deg=radius, area_deg2=area))
    holes.sort(key=lambda h: -h.area_deg2)
    return holes


def inpaint_holes(holes, counts, nside, *, donor_ra, donor_dec, donor_z,
                  rand_ra, rand_dec, donor_colors=None, donor_mags=None,
                  seed=0, n_real=1, density_boost=1.0, search_deg=6.0, max_tries=80):
    """Fill holes by transplanting real galaxies from environment-matched donors.

    For each hole we pick a nearby *clean* donor centre (an observed galaxy whose
    surrounding disk of the hole's radius contains no hole pixels and is fully in
    the footprint), translate that donor disk onto the hole in a local tangent
    plane, and keep the donor galaxies whose mapped position lands in the hole's
    empty pixels — carrying their redshift, colours and magnitudes *and their
    relative configuration* (so higher-order clustering and the colour/luminosity
    structure transfer by construction). ``density_boost`` (e.g. <w_c>) scales the
    transplanted count to the completeness-corrected density. Each realization
    uses independent donor centres; returns a list of ``n_real`` dicts with arrays
    ``ra, dec, z, colors, mags, hole_id``.

    Amplitude is set *selection-immune* by construction: the per-hole target
    count uses the local data/random ratio of the collar (``g_collar/r_collar``,
    a random-normalised 1+δ), so it matches the surrounding field's true density
    rather than any raw count, exactly as the total-target density anchor of
    ``selection_coupling.total_target_density`` does globally. ``density_boost``
    is the residual completeness factor (<w_c>); for a fully selection-immune
    amplitude pass the total-target density ratio as ``density_boost`` instead.
    """
    import healpy as hp
    donor_ra = np.asarray(donor_ra); donor_dec = np.asarray(donor_dec); donor_z = np.asarray(donor_z)
    empty_pix = np.where(counts <= 0)[0]
    empty_set = set(int(p) for p in empty_pix)
    donor_nhat = _radec_to_nhat(donor_ra, donor_dec)
    from scipy.spatial import cKDTree
    dtree = cKDTree(donor_nhat)
    rtree = cKDTree(_radec_to_nhat(np.asarray(rand_ra), np.asarray(rand_dec)))
    med = float(np.median(counts[counts > 0]))     # randoms per hole pixel (window)

    def disk_clean(cvec, radius_deg):
        ipix = hp.query_disc(nside, cvec, np.radians(radius_deg), inclusive=True)
        return not any(int(p) in empty_set for p in ipix)

    out = []
    for r in range(n_real):
        rng = np.random.default_rng(seed + r)
        ra_o, dec_o, z_o, hid_o = [], [], [], []
        col_o, mag_o = [], []
        for hi, h in enumerate(holes):
            R = max(h.radius_deg, hp.nside2resol(nside, arcmin=True) / 60.0)
            chal = _radec_to_nhat(np.array([h.ra]), np.array([h.dec]))[0]
            # candidate donor galaxies within search_deg of the hole
            cand = dtree.query_ball_point(chal, np.radians(min(search_deg, 20)))
            if not cand:
                continue
            cand = np.array(cand); rng.shuffle(cand)
            donor_c = None
            for ci in cand[:max_tries]:
                cvec = donor_nhat[ci]
                if disk_clean(cvec, R * 1.05):
                    donor_c = ci; break
            if donor_c is None:
                continue
            rad, decd = donor_ra[donor_c], donor_dec[donor_c]
            # donor galaxies within R of the donor centre
            din = np.array(dtree.query_ball_point(donor_nhat[donor_c], np.radians(R)))
            if len(din) == 0:
                continue
            cd = np.cos(np.radians(decd))
            # translate donor disk -> hole centre in local tangent plane
            ra_m = h.ra + (donor_ra[din] - rad) * cd / np.cos(np.radians(h.dec))
            dec_m = h.dec + (donor_dec[din] - decd)
            pix_m = hp.ang2pix(nside, np.radians(90 - dec_m), np.radians(ra_m % 360))
            keep = np.isin(pix_m, h.pixels)
            if not keep.any():
                continue
            sel = din[keep]; ra_m = ra_m[keep]; dec_m = dec_m[keep]
            # target count so the hole's galaxy/random ratio equals the collar's
            # (= local 1+δ), robust to footprint coverage & completeness because
            # it normalises by the randoms. filled randoms in hole = med·n_pix;
            # collar = annulus [R,2R]. Random resampling to target preserves the
            # transplanted clustering. density_boost (≈<w_c>) adds completeness.
            g_collar = (len(dtree.query_ball_point(chal, np.radians(2 * R)))
                        - len(dtree.query_ball_point(chal, np.radians(R))))
            r_collar = (len(rtree.query_ball_point(chal, np.radians(2 * R)))
                        - len(rtree.query_ball_point(chal, np.radians(R))))
            local_ratio = g_collar / max(r_collar, 1)
            filled_rand = med * len(h.pixels)
            target = int(rng.poisson(filled_rand * local_ratio * density_boost))
            if target == 0:
                continue
            idx2 = rng.choice(len(sel), size=target, replace=(target > len(sel)))
            sel = sel[idx2]; ra_m = ra_m[idx2]; dec_m = dec_m[idx2]
            ra_o.append(ra_m % 360); dec_o.append(dec_m); z_o.append(donor_z[sel])
            hid_o.append(np.full(len(sel), hi))
            if donor_colors is not None:
                col_o.append(np.asarray(donor_colors)[sel])
            if donor_mags is not None:
                mag_o.append(np.asarray(donor_mags)[sel])
        d = {"ra": np.concatenate(ra_o) if ra_o else np.zeros(0),
             "dec": np.concatenate(dec_o) if dec_o else np.zeros(0),
             "z": np.concatenate(z_o) if z_o else np.zeros(0),
             "hole_id": np.concatenate(hid_o) if hid_o else np.zeros(0, int)}
        if donor_colors is not None:
            d["colors"] = np.concatenate(col_o) if col_o else np.zeros((0, np.shape(donor_colors)[1]))
        if donor_mags is not None:
            d["mags"] = np.concatenate(mag_o) if mag_o else np.zeros((0, np.shape(donor_mags)[1]))
        out.append(d)
    return out
