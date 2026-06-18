#!/usr/bin/env python3
"""Draw equal-weight completed BOSS CMASS-South catalogs from the posterior package.

Self-contained (numpy only; astropy optional for FITS output). The 2 MB package
`cmass_south_posterior.npz` holds the immutable observed galaxies once plus each
missing galaxy's redshift posterior as a compact inverse-CDF, so a "sample" is just
a seed: this draws as many statistically-independent completed catalogs as you want,
each ~120k galaxies, in milliseconds, with no per-sample storage.

    python draw_samples.py --seed 0 --out catalog_0.fits
    python draw_samples.py --seed 0 --n 100 --out-prefix cat_   # 100 realizations
    # in code:
    from draw_samples import load_package, draw
    pkg = load_package("cmass_south_posterior.npz")
    cat = draw(pkg, seed=0)         # dict(ra, dec, z, prov, N)

Columns: RA, DEC [deg], Z (redshift), PROV (0 observed-specz, 1 fiber-collided,
2 redshift-failure, 3 systot-analog, 4 zhost-fallback). The catalog is equal-weight
and cosmology-free; pair it with the bundled `cmass_south_randoms.npz` (uniform over
the footprint). See README.md and DATA_MODEL.md.
"""
import argparse
import numpy as np

PROV_NAME = {0: "observed", 1: "collided", 2: "zfail", 3: "systot", 4: "zhost", 5: "inpaint"}


def _dequant(u, lo, hi):
    return lo + u.astype(np.float32) / 65535.0 * (hi - lo)


def load_package(path):
    """Load and de-quantize the posterior package (.npz)."""
    d = np.load(path)
    zmin, zmax = float(d["zmin"]), float(d["zmax"])
    return {
        "n_obs": int(d["n_obs"]), "n_miss": int(d["n_miss"]), "zmin": zmin, "zmax": zmax,
        "qlev": d["qlev"].astype(np.float64), "jitter": float(d["jitter"]),
        "obs_z": _dequant(d["obs_z_q"], zmin, zmax),
        "invcdf": _dequant(d["invcdf_q"], zmin, zmax).astype(np.float64),
        "base_ra": d["base_ra"], "base_dec": d["base_dec"],
        "base_wsys": d["base_wsys"].astype(np.float32), "base_prov": d["base_prov"],
    }


def draw(pkg, seed=0, systot=True):
    """Draw one equal-weight completed realization. Returns dict(ra, dec, z, prov, N)."""
    rng = np.random.default_rng(seed)
    M = pkg["n_miss"]; qlev, invcdf = pkg["qlev"], pkg["invcdf"]; nq = len(qlev)
    # vectorized inverse-CDF sampling of each missing galaxy's redshift
    u = rng.random(M)
    j = np.clip(np.searchsorted(qlev, u), 1, nq - 1)
    q0, q1 = qlev[j - 1], qlev[j]
    v0 = invcdf[np.arange(M), j - 1]; v1 = invcdf[np.arange(M), j]
    z_miss = v0 + (v1 - v0) * (u - q0) / np.maximum(q1 - q0, 1e-12)
    z_miss = np.clip(z_miss + rng.normal(0.0, pkg["jitter"], M), pkg["zmin"], pkg["zmax"])

    base_ra, base_dec = pkg["base_ra"], pkg["base_dec"]
    base_z = np.concatenate([pkg["obs_z"], z_miss]).astype(np.float32)
    base_prov = pkg["base_prov"]
    if not systot:
        return {"ra": base_ra.copy(), "dec": base_dec.copy(), "z": base_z,
                "prov": base_prov.copy(), "N": len(base_ra)}
    # WEIGHT_SYSTOT analog excess: restore floor(max(w-1,0)+U) galaxies at the
    # survivor position + ~1" jitter (smooth imaging-systematic boost; no Dtheta=0).
    wsys = pkg["base_wsys"]
    n_extra = np.floor(np.maximum(wsys - 1.0, 0.0) + rng.random(len(wsys))).astype(int)
    src = np.repeat(np.arange(len(base_ra)), n_extra)
    sig = 1.0 / 3600.0
    cd = np.cos(np.radians(base_dec[src].astype(np.float64)))
    ex_ra = (base_ra[src] + rng.normal(0, 1, len(src)) * sig / np.maximum(cd, 1e-3)) % 360.0
    ex_dec = base_dec[src] + rng.normal(0, 1, len(src)) * sig
    return {
        "ra": np.concatenate([base_ra, ex_ra]).astype(np.float32),
        "dec": np.concatenate([base_dec, ex_dec]).astype(np.float32),
        "z": np.concatenate([base_z, base_z[src]]).astype(np.float32),
        "prov": np.concatenate([base_prov, np.full(len(src), 3, np.int8)]),
        "N": len(base_ra) + len(src),
    }


def _write(cat, path):
    if path.endswith(".fits"):
        from astropy.table import Table
        Table({"RA": cat["ra"], "DEC": cat["dec"], "Z": cat["z"], "PROV": cat["prov"]}).write(
            path, overwrite=True)
    else:
        np.savez_compressed(path, ra=cat["ra"], dec=cat["dec"], z=cat["z"], prov=cat["prov"])


def main():
    p = argparse.ArgumentParser(description="Draw completed CMASS-South catalogs from the posterior package.")
    p.add_argument("--package", default="cmass_south_posterior.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="number of realizations (seeds seed..seed+n-1)")
    p.add_argument("--out", default="catalog.fits", help="output (single realization)")
    p.add_argument("--out-prefix", default=None, help="prefix for --n>1 (writes <prefix><seed>.fits)")
    p.add_argument("--no-systot", action="store_true")
    args = p.parse_args()
    pkg = load_package(args.package)
    if args.n == 1:
        cat = draw(pkg, seed=args.seed, systot=not args.no_systot)
        _write(cat, args.out)
        print(f"seed {args.seed}: {cat['N']:,} galaxies -> {args.out}")
    else:
        pre = args.out_prefix or "catalog_"
        for s in range(args.seed, args.seed + args.n):
            cat = draw(pkg, seed=s, systot=not args.no_systot)
            _write(cat, f"{pre}{s}.fits")
        print(f"wrote {args.n} realizations {pre}{args.seed}..{pre}{args.seed+args.n-1}.fits")


if __name__ == "__main__":
    main()
