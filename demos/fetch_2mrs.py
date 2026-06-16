"""Download 2MRS (2MASS Redshift Survey) catalog from VizieR.

Usage::

    python demos/fetch_2mrs.py [--dest data/2mrs]

Downloads ``2mrs_1175_done.fits`` from VizieR catalog J/ApJS/199/26
(Huchra et al. 2012) into the specified directory.
"""

import argparse
import os
import sys


def fetch_2mrs(dest: str = "data/2mrs") -> str:
    """Download the 2MRS catalog. Returns path to the downloaded file."""
    os.makedirs(dest, exist_ok=True)
    out_path = os.path.join(dest, "2mrs_1175_done.fits")

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Already downloaded: {out_path}  ({size_mb:.1f} MB)")
        return out_path

    print("Fetching 2MRS from VizieR J/ApJS/199/26 ...")
    print("This may take a minute (~15 MB).")

    try:
        from astroquery.vizier import Vizier
        import astropy.units as u

        # The catalog is large; remove row limit
        v = Vizier(columns=["*"], row_limit=-1)
        catalog_list = v.get_catalogs("J/ApJS/199/26")
        if not catalog_list:
            raise RuntimeError("VizieR returned no tables for J/ApJS/199/26")

        # The main galaxy table is the first/largest one
        # Try to find the table with RA and CZ columns
        table = None
        for t in catalog_list:
            cols = [c.upper() for c in t.colnames]
            if "RAJ2000" in cols and "VH" in cols:
                table = t
                break
        if table is None:
            # Fall back to the largest table
            table = max(catalog_list, key=len)

        print(f"Downloaded {len(table)} rows with columns: "
              f"{', '.join(table.colnames[:8])} ...")
        table.write(out_path, format="fits", overwrite=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved: {out_path}  ({size_mb:.1f} MB)")
        return out_path

    except Exception as e:
        print(f"astroquery fetch failed: {e}")
        print("\nManual download instructions:")
        print("  1. Visit https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/199/26")
        print("  2. Click 'FTP' → download table1.dat and ReadMe")
        print("  OR use the direct CDS FTP:")
        print("     wget -P data/2mrs/ "
              "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/199/26/2mrs_1175_done.fits.gz")
        print("     gunzip data/2mrs/2mrs_1175_done.fits.gz")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 2MRS from VizieR")
    parser.add_argument("--dest", default="data/2mrs",
                        help="Destination directory (default: data/2mrs)")
    args = parser.parse_args()
    fetch_2mrs(dest=args.dest)
