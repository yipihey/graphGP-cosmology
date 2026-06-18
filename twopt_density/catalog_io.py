"""Write distributable, documented FITS catalogues from completion realizations.

Two products (see DATA_MODEL.md):

* **ensemble** — N realization catalogues (``realization_###.fits``), each a
  complete equal-weight catalogue (RA, DEC, Z, PROV) that samples the posterior
  over the galaxies that would have been observed absent the systematics. Draw
  from these for any analysis; the spread across realizations is the completion
  uncertainty.
* **summary** — one ``summary.fits`` for quick use: every observed galaxy plus
  every missing target (fixed positions), with the per-object redshift best
  estimate (ensemble mean) and uncertainty (ensemble std), provenance flag, and
  the imaging-systematic weight. The stochastic WEIGHT_SYSTOT *additions* are not
  in the summary (they live in the ensemble); the summary carries WEIGHT_SYSTOT
  as a column so a user can instead weight.

Per-object PROV codes match :data:`twopt_density.observed_ls.PROV`:
  0 observed-specz · 1 collided · 2 zfail · 3 systot-analog · 4 zhost-fallback · 5 inpaint
"""

from __future__ import annotations

import os
import numpy as np

from .observed_ls import PROV, PROV_NAME

_VERSION = "v1"


def _table_hdu(arrays, names, units=None, name=None):
    from astropy.io import fits
    units = units or {}
    cols = []
    for nm in names:
        a = np.asarray(arrays[nm])
        fmt = {"f4": "E", "f8": "D", "i1": "B", "i2": "I", "i4": "J", "i8": "K"}[a.dtype.str[1:].replace(">", "").replace("<", "")] \
            if a.dtype.kind in "fi" else "E"
        cols.append(fits.Column(name=nm.upper(), array=a, format=fmt, unit=units.get(nm)))
    hdu = fits.BinTableHDU.from_columns(cols, name=name)
    return hdu


def write_release(realizations, outdir, *, w_systot=None, randoms=None, meta=None, version=_VERSION):
    """Write the ensemble + summary FITS release to ``outdir``.

    ``realizations``: list of dicts ``{ra,dec,z,prov,N}`` from
    ``complete_catalog_photoz`` (analog mode; the non-systot prefix is in a fixed
    per-object order across realizations). ``w_systot``: per-object WEIGHT_SYSTOT
    for the non-systot prefix (observed+missing), optional. ``randoms``:
    ``(ra,dec,z)`` to write as ``randoms.fits``. ``meta``: dict of header cards."""
    from astropy.io import fits
    os.makedirs(outdir, exist_ok=True)
    hdr = dict(VERSION=version, NREAL=len(realizations), SAMPLE="CMASS-South",
               COORDS="observed RA/Dec/z (cosmology-free)")
    hdr.update(meta or {})

    # ---- ensemble ----
    ensdir = os.path.join(outdir, "ensemble"); os.makedirs(ensdir, exist_ok=True)
    for i, c in enumerate(realizations):
        hdu = _table_hdu({"ra": np.asarray(c["ra"], "f4"), "dec": np.asarray(c["dec"], "f4"),
                          "z": np.asarray(c["z"], "f4"), "prov": np.asarray(c["prov"], "i1")},
                         ["ra", "dec", "z", "prov"],
                         units={"ra": "deg", "dec": "deg"}, name="CATALOG")
        for k, v in hdr.items():
            hdu.header[k] = v
        hdu.header["REALIZ"] = i
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            os.path.join(ensdir, f"realization_{i:03d}.fits"), overwrite=True)

    # ---- summary (common non-systot prefix: observed + missing) ----
    c0 = realizations[0]; prov0 = np.asarray(c0["prov"])
    nbase = int((prov0 != PROV["systot"]).sum())            # prefix length (constant)
    zall = np.array([np.asarray(c["z"])[:nbase] for c in realizations])   # (Nreal, nbase)
    summ = {"ra": np.asarray(c0["ra"][:nbase], "f4"), "dec": np.asarray(c0["dec"][:nbase], "f4"),
            "z": zall.mean(0).astype("f4"), "z_err": zall.std(0).astype("f4"),
            "prov": prov0[:nbase].astype("i1")}
    names = ["ra", "dec", "z", "z_err", "prov"]
    if w_systot is not None:
        summ["weight_systot"] = np.asarray(w_systot[:nbase], "f4"); names.append("weight_systot")
    shdu = _table_hdu(summ, names, units={"ra": "deg", "dec": "deg"}, name="SUMMARY")
    for k, v in hdr.items():
        shdu.header[k] = v
    shdu.header["NBASE"] = nbase
    fits.HDUList([fits.PrimaryHDU(), shdu]).writeto(os.path.join(outdir, "summary.fits"), overwrite=True)

    if randoms is not None:
        rr, rd, rz = randoms
        rhdu = _table_hdu({"ra": np.asarray(rr, "f4"), "dec": np.asarray(rd, "f4"), "z": np.asarray(rz, "f4")},
                          ["ra", "dec", "z"], units={"ra": "deg", "dec": "deg"}, name="RANDOMS")
        fits.HDUList([fits.PrimaryHDU(), rhdu]).writeto(os.path.join(outdir, "randoms.fits"), overwrite=True)

    # provenance summary text
    counts = {PROV_NAME[k]: int((prov0 == k).sum()) for k in PROV.values() if (prov0 == k).any()}
    with open(os.path.join(outdir, "PROVENANCE.txt"), "w") as f:
        f.write(f"# completion release {version}; {len(realizations)} realizations\n")
        f.write(f"# realization 0 provenance counts: {counts}\n")
    return outdir
