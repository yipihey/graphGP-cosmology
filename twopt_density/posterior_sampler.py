"""Compact, shareable posterior over completed BOSS CMASS catalogs + a fast sampler.

The completion's structure makes it cheaply shareable:

  * Every realization keeps the SAME observed galaxies (RA, Dec, spec-z). They are
    stored ONCE, not duplicated per sample.
  * Only the ~9% missing galaxies' redshifts vary between realizations, and each
    missing galaxy's redshift posterior p(z | n̂, colours) is FIXED (it is built
    from the observed field + colours, independent of the random seed). Only the
    DRAW from it changes. So we precompute each missing galaxy's posterior once,
    store it as a compact inverse-CDF (quantile function, uint16), and a "sample"
    is just seeded inverse-CDF sampling — microseconds, unlimited draws, ~zero
    storage per sample.

This module:
  * :func:`build_package` — runs the expensive GP/local-density posterior ONCE and
    returns a compact package (immutable observed base + missing positions + per-
    object inverse-CDF + systot weights + provenance).
  * :func:`write_package` / :func:`load_package` — heavily-compressed (.npz, uint16
    quantised) serialization; a few MB for the whole posterior.
  * :func:`draw` — reconstruct a full equal-weight realization from a seed. Fast
    and exactly equivalent (in distribution) to
    :func:`observed_ls.complete_catalog_photoz` with ``z_mode='field'``.

The samples are inherently RELATIVE: the shared observed base is the bulk of every
catalog and is stored once; a fixed reproducible ensemble of K realizations needs
only K seeds (or, if materialized, K small redshift-delta arrays for the missing
galaxies), never K copies of the observed galaxies.
"""
from __future__ import annotations

import numpy as np

from .observed_ls import (_radec_to_nhat, _clpair_density, _systot_restore_extras,
                           measure_close_pair_dz, PROV)

QUANT_VERSION = 1


def build_package(catalog, targets, photoz, *, dz_pool=None, nq=65, ngrid=256,
                  K=150, bw_f=0.004, bw_p=0.02, jitter=None, verbose=False):
    """Precompute the compact posterior package (the expensive step, done once).

    Mirrors the ``z_mode='field'`` posterior of
    :func:`observed_ls.complete_catalog_photoz`: for each missing target it builds
    p(z|n̂,colours) ∝ (1+δ_g(n̂,z))·n̄(z)·p_photoz(z) (× close-pair prior for
    collisions) and stores its inverse-CDF on ``nq`` quantile levels. Returns a dict
    of plain arrays (see module docstring)."""
    from .photoz import photoz_features

    ra_o = np.asarray(catalog.ra_data, np.float64)
    dec_o = np.asarray(catalog.dec_data, np.float64)
    z_o = np.asarray(catalog.z_data, np.float64)
    wsys_o = np.asarray(catalog.w_sys_data if catalog.w_sys_data is not None
                        else np.ones(len(ra_o)), np.float64)
    host = np.asarray(targets.host_index)
    z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
    miss_kind = np.asarray(targets.miss_kind)
    if dz_pool is None:
        dz_pool = measure_close_pair_dz(catalog)
    dz_pool = np.asarray(dz_pool, np.float64)
    if jitter is None:
        jitter = bw_f * 0.5

    from scipy.spatial import cKDTree
    feat = photoz_features(targets.colors, targets.mags)
    zk, wk = photoz.posterior(feat)
    M = len(host)
    Kq = min(K, len(z_o))
    _, nn = cKDTree(_radec_to_nhat(ra_o, dec_o)).query(
        _radec_to_nhat(np.asarray(targets.ra), np.asarray(targets.dec)), k=Kq, workers=-1)
    zmin, zmax = float(z_o.min()), float(z_o.max())
    zgrid = np.linspace(zmin, zmax, ngrid)
    pcl = _clpair_density(dz_pool)
    coll_i = (miss_kind == "collided") & (host >= 0)
    qlev = np.linspace(0.0, 1.0, nq)

    invcdf = np.empty((M, nq), np.float64)
    fallback = np.zeros(M, bool)
    for i in range(M):
        znb = z_o[nn[i]]
        pf = np.exp(-0.5 * ((zgrid[:, None] - znb[None, :]) / bw_f) ** 2).sum(1)
        w = wk[i]; ok = np.isfinite(w) & (w > 0)
        pp = ((w[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
              if ok.any() else np.ones_like(zgrid))
        p = pf * pp
        if coll_i[i]:
            p = p * pcl(zgrid - z_host[i])
        s = p.sum()
        if s > 0:
            cdf = np.cumsum(p) / s
            # make cdf strictly increasing for interp (dedupe flat tails)
            cdf = np.maximum.accumulate(cdf)
            u, idx = np.unique(cdf, return_index=True)
            invcdf[i] = np.interp(qlev, u, zgrid[idx])
        else:
            zfb = z_host[i] if np.isfinite(z_host[i]) else float(np.median(z_o))
            invcdf[i] = zfb
            fallback[i] = True
    if verbose:
        print(f"[build_package] {M:,} missing posteriors, nq={nq}, fallback={int(fallback.sum())}")

    miss_prov = np.where(fallback, PROV["zhost"],
                         np.where(miss_kind == "collided", PROV["collided"], PROV["zfail"]))
    base_ra = np.concatenate([ra_o, np.asarray(targets.ra, np.float64)]).astype(np.float32)
    base_dec = np.concatenate([dec_o, np.asarray(targets.dec, np.float64)]).astype(np.float32)
    base_wsys = np.concatenate([wsys_o, wsys_o[np.clip(host, 0, len(z_o) - 1)]]).astype(np.float32)
    base_prov = np.concatenate([np.full(len(ra_o), PROV["observed"], np.int8), miss_prov.astype(np.int8)])

    return {
        "n_obs": len(ra_o), "n_miss": M, "zmin": zmin, "zmax": zmax,
        "qlev": qlev.astype(np.float32), "jitter": float(jitter),
        "obs_z": z_o.astype(np.float32),                 # fixed observed redshifts
        "base_ra": base_ra, "base_dec": base_dec,        # obs + missing positions (fixed)
        "base_wsys": base_wsys, "base_prov": base_prov,
        "invcdf": invcdf.astype(np.float32),             # (M, nq) per-object quantile fn
    }


def _quant(a, lo, hi):
    return np.clip(np.round((a - lo) / (hi - lo) * 65535.0), 0, 65535).astype(np.uint16)


def _dequant(u, lo, hi):
    return lo + u.astype(np.float32) / 65535.0 * (hi - lo)


def write_package(pkg, path):
    """Heavily-compressed serialization (.npz, uint16-quantised redshifts/CDF)."""
    zmin, zmax = pkg["zmin"], pkg["zmax"]
    np.savez_compressed(
        path,
        version=QUANT_VERSION, n_obs=pkg["n_obs"], n_miss=pkg["n_miss"],
        zmin=zmin, zmax=zmax, jitter=pkg["jitter"], qlev=pkg["qlev"].astype(np.float32),
        obs_z_q=_quant(pkg["obs_z"], zmin, zmax),                       # uint16
        invcdf_q=_quant(pkg["invcdf"], zmin, zmax),                     # uint16 (M, nq)
        base_ra=pkg["base_ra"], base_dec=pkg["base_dec"],              # float32 positions
        base_wsys=pkg["base_wsys"].astype(np.float16),
        base_prov=pkg["base_prov"],
    )


def load_package(path):
    d = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
    zmin, zmax = float(d["zmin"]), float(d["zmax"])
    return {
        "n_obs": int(d["n_obs"]), "n_miss": int(d["n_miss"]), "zmin": zmin, "zmax": zmax,
        "qlev": d["qlev"].astype(np.float64), "jitter": float(d["jitter"]),
        "obs_z": _dequant(d["obs_z_q"], zmin, zmax),
        "invcdf": _dequant(d["invcdf_q"], zmin, zmax).astype(np.float64),
        "base_ra": d["base_ra"], "base_dec": d["base_dec"],
        "base_wsys": d["base_wsys"].astype(np.float32), "base_prov": d["base_prov"],
    }


def draw(pkg, seed=0, *, systot=True):
    """Draw one equal-weight completed realization from the package (fast).

    Returns ``dict(ra, dec, z, prov)``. Equivalent in distribution to
    ``complete_catalog_photoz(..., z_mode='field')``. With ``systot=False`` the
    WEIGHT_SYSTOT analog excess is skipped (just observed + missing)."""
    rng = np.random.default_rng(seed)
    n_obs, M = pkg["n_obs"], pkg["n_miss"]
    qlev, invcdf = pkg["qlev"], pkg["invcdf"]
    nq = len(qlev)

    # vectorized inverse-CDF sampling: z_miss[i] = invcdf_i(u_i), shared qlev
    u = rng.random(M)
    j = np.clip(np.searchsorted(qlev, u), 1, nq - 1)
    q0, q1 = qlev[j - 1], qlev[j]
    v0 = invcdf[np.arange(M), j - 1]; v1 = invcdf[np.arange(M), j]
    z_miss = v0 + (v1 - v0) * (u - q0) / np.maximum(q1 - q0, 1e-12)
    z_miss = z_miss + rng.normal(0.0, pkg["jitter"], M)
    z_miss = np.clip(z_miss, pkg["zmin"], pkg["zmax"])

    base_ra, base_dec = pkg["base_ra"], pkg["base_dec"]
    base_z = np.concatenate([pkg["obs_z"], z_miss]).astype(np.float32)
    base_prov = pkg["base_prov"]
    if not systot:
        return {"ra": base_ra.copy(), "dec": base_dec.copy(), "z": base_z, "prov": base_prov.copy(),
                "N": len(base_ra)}

    wsys = pkg["base_wsys"]
    n_extra = np.floor(np.maximum(wsys - 1.0, 0.0) + rng.random(len(wsys))).astype(int)
    src = np.repeat(np.arange(len(base_ra)), n_extra)
    ex_ra, ex_dec, ex_z = _systot_restore_extras(
        base_ra.astype(np.float64), base_dec.astype(np.float64), base_z.astype(np.float64), src, rng)
    return {
        "ra": np.concatenate([base_ra, ex_ra]).astype(np.float32),
        "dec": np.concatenate([base_dec, ex_dec]).astype(np.float32),
        "z": np.concatenate([base_z, ex_z]).astype(np.float32),
        "prov": np.concatenate([base_prov, np.full(len(ex_ra), PROV["systot"], np.int8)]),
        "N": len(base_ra) + len(ex_ra),
    }
