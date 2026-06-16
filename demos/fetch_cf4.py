"""Download Cosmicflows-4 catalog from VizieR.

Usage::

    python demos/fetch_cf4.py [--dest data/cf4] [--table galaxies|groups]

Downloads from VizieR catalog J/ApJ/944/94 (Tully et al. 2023).
Two tables are available:
    galaxies  — 55,877 individual distances  (default)
    groups    — 38,065 group-averaged catalog (preferred for density field)
"""

import argparse
import os
import sys


def fetch_cf4(dest: str = "data/cf4", table: str = "galaxies") -> str:
    os.makedirs(dest, exist_ok=True)
    fname = {"galaxies": "kallcf4.fits", "groups": "kallcf4_groups.fits"}[table]
    out_path = os.path.join(dest, fname)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Already downloaded: {out_path}  ({size_mb:.1f} MB)")
        return out_path

    print(f"Fetching CF4 {table} table from VizieR J/ApJ/944/94 ...")

    try:
        from astroquery.vizier import Vizier

        v = Vizier(columns=["*"], row_limit=-1)
        catalog_list = v.get_catalogs("J/ApJ/944/94")
        if not catalog_list:
            raise RuntimeError("VizieR returned no tables for J/ApJ/944/94")

        # Find table with RA and distance modulus
        target = None
        for t in catalog_list:
            cols = [c.upper() for c in t.colnames]
            has_radec = "RAJ2000" in cols or "RA" in cols
            has_dm = "DM" in cols or "MU" in cols
            has_z = "ZCMB" in cols or "VCMB" in cols
            if has_radec and has_dm and has_z:
                if table == "groups" and ("NG" in cols or "NMEM" in cols):
                    target = t; break
                elif table == "galaxies":
                    target = t; break

        if target is None:
            target = max(catalog_list, key=len)

        print(f"Downloaded {len(target)} rows  columns: "
              f"{', '.join(target.colnames[:8])} ...")
        target.write(out_path, format="fits", overwrite=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved: {out_path}  ({size_mb:.1f} MB)")
        return out_path

    except Exception as e:
        print(f"astroquery fetch failed: {e}")
        print("\nManual download instructions:")
        print("  VizieR: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/944/94")
        print("  EDD:    https://edd.ifa.hawaii.edu/  (table: CF4 All Group Distances)")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CF4 from VizieR")
    parser.add_argument("--dest", default="data/cf4")
    parser.add_argument("--table", default="galaxies", choices=["galaxies", "groups"])
    args = parser.parse_args()
    fetch_cf4(dest=args.dest, table=args.table)
