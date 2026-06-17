"""Fetch the SDSS DR12 CMASS photometric-target catalogue (CMASS-South / SGC).

Recovers the spectroscopically-MISSING CMASS galaxies (fiber collisions never
fibered; redshift failures with ZWARNING≠0) — their positions + ugriz colours —
which are absent from the LSS spectroscopic file but present in the DR8 imaging.

We select CMASS *targets* directly from the imaging (PhotoPrimary) by applying
the Reid et al. (2016) CMASS colour/magnitude cuts (so never-fibered collided
objects are included), and LEFT JOIN SpecObjAll for ZWARNING/z so each target is
flagged good-z / z-failure / never-fibered. Anonymous SkyServer SqlSearch is
capped (~500k rows), so we tile over RA. Output: a trimmed FITS under data/boss/.

    ~/.venv/k3d/bin/python3 demos/fetch_cmass_targets.py --test     # small patch
    ~/.venv/k3d/bin/python3 demos/fetch_cmass_targets.py            # full SGC

CMASS selection (extinction-corrected; i is cmodel, g/r/i/z model):
  d_perp = (r-i) - (g-r)/8 > 0.55 ;  17.5 < i < 19.9 ;  r-i < 2
  i < 19.86 + 1.6*(d_perp - 0.8) ;  i_fib2 < 21.5
  star-galaxy: i_psf-i_cmod > 0.2+0.2*(20-i_cmod) ;  z_psf-z_cmod > 9.125-0.46*z_cmod
SGC footprint (matches twopt_density/boss.py): (RA<28 or RA>335) and Dec>-6.
"""
import argparse, io, os, sys, time, urllib.parse, urllib.request

URL = "https://skyserver.sdss.org/dr12/SkyServerWS/SearchTools/SqlSearch"

CMASS_SQL = """
SELECT p.objID, p.ra, p.dec,
  p.modelMag_u-p.extinction_u AS u, p.modelMag_g-p.extinction_g AS g,
  p.modelMag_r-p.extinction_r AS r, p.modelMag_i-p.extinction_i AS i_mod,
  p.modelMag_z-p.extinction_z AS z_mod,
  p.cModelMag_i-p.extinction_i AS i_cmod, p.cModelMag_z-p.extinction_z AS z_cmod,
  p.psfMag_i-p.extinction_i AS i_psf, p.psfMag_z-p.extinction_z AS z_psf,
  p.fiber2Mag_i-p.extinction_i AS i_fib2,
  s.specObjID, s.z AS spec_z, s.zWarning AS zwarning, s.class AS spec_class
FROM PhotoPrimary AS p
LEFT JOIN SpecObjAll AS s ON s.bestObjID = p.objID
WHERE p.type = 3
  AND ({radec})
  AND (p.modelMag_r-p.extinction_r) - (p.modelMag_i-p.extinction_i) < 2.0
  AND (p.cModelMag_i-p.extinction_i) BETWEEN 17.5 AND 19.9
  AND ((p.modelMag_r-p.extinction_r)-(p.modelMag_i-p.extinction_i))
      - ((p.modelMag_g-p.extinction_g)-(p.modelMag_r-p.extinction_r))/8.0 > 0.55
  AND (p.cModelMag_i-p.extinction_i) < 19.86 + 1.6*(
      ((p.modelMag_r-p.extinction_r)-(p.modelMag_i-p.extinction_i))
      - ((p.modelMag_g-p.extinction_g)-(p.modelMag_r-p.extinction_r))/8.0 - 0.8)
  AND (p.fiber2Mag_i-p.extinction_i) < 21.5
  AND (p.psfMag_i-p.cModelMag_i) > 0.2 + 0.2*(20.0 - (p.cModelMag_i-p.extinction_i))
  AND (p.psfMag_z-p.cModelMag_z) > 9.125 - 0.46*(p.cModelMag_z-p.extinction_z)
"""


def run_sql(sql, retries=3):
    q = urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(URL + "?" + q, timeout=300) as resp:
                txt = resp.read().decode()
            lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
            return lines
        except Exception as e:
            print(f"  retry {a+1}: {e}", file=sys.stderr); time.sleep(5)
    raise RuntimeError("SkyServer query failed")


def fetch(radec_clause):
    lines = run_sql(CMASS_SQL.format(radec=radec_clause))
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]
    return header, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="small 1deg patch only")
    p.add_argument("--out", default="data/boss/cmass_targets_South.fits")
    p.add_argument("--ra-step", type=float, default=2.0)
    args = p.parse_args()

    if args.test:
        header, rows = fetch("p.ra BETWEEN 20 AND 21 AND p.dec BETWEEN -2 AND 2")
        print(f"TEST patch: {len(rows)} CMASS targets, columns={header}")
        import numpy as np
        if rows:
            arr = {h: np.array([r[j] for r in rows]) for j, h in enumerate(header)}
            zw = arr["zwarning"]
            spec = arr["specObjID"]
            nomatch = np.sum((spec == "") | (spec == "0"))
            print(f"  no spec match (collided/never-fibered): {nomatch}")
            print(f"  zwarning!=0 (z-failures, among matched): "
                  f"{np.sum([z not in ('', '0') for z in zw])}")
            print(f"  example row: {rows[0]}")
        return

    # full SGC: tile over RA (RA<28 or RA>335), Dec>-6, in ra-step chunks
    import numpy as np
    from astropy.io import fits
    edges_lo = np.arange(0.0, 28.0 + 1e-9, args.ra_step)
    edges_hi = np.arange(335.0, 360.0 + 1e-9, args.ra_step)
    tiles = ([(a, b) for a, b in zip(edges_lo[:-1], edges_lo[1:])]
             + [(a, b) for a, b in zip(edges_hi[:-1], edges_hi[1:])])
    all_rows = []; header = None
    for a, b in tiles:
        h, rows = fetch(f"p.ra BETWEEN {a} AND {b} AND p.dec > -6.0")
        header = header or h
        all_rows += rows
        print(f"  RA[{a:.0f},{b:.0f}]: {len(rows)} (total {len(all_rows)})")
    arr = {h: np.array([r[j] for r in all_rows]) for j, h in enumerate(header)}
    cols = [fits.Column(name=h, format="D" if h not in ("objID", "specObjID", "zwarning", "spec_class")
                        else "K" if h != "spec_class" else "20A",
                        array=np.array([_num(x) for x in arr[h]]) if h not in ("spec_class",) else arr[h])
            for h in header]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fits.BinTableHDU.from_columns(cols).writeto(args.out, overwrite=True)
    print(f"Saved {len(all_rows)} CMASS targets -> {args.out}")


def _num(x):
    try:
        return float(x) if x not in ("", "null") else float("nan")
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
