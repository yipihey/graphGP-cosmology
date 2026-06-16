"""Download BOSS DR12 LSS catalogs (simBIG subsamples) from SDSS.

Usage::

    python demos/fetch_boss.py [--dest data/boss] [--sample CMASS|LOWZ|both]
                               [--n-randoms 1]

Downloads galaxy and random FITS files for the South Galactic Cap
from the SDSS DR12 public release at::

    https://data.sdss.org/sas/dr12/boss/lss/

File sizes:
    galaxy_DR12v5_CMASS_South.fits.gz   ~160 MB
    galaxy_DR12v5_LOWZ_South.fits.gz    ~ 45 MB
    random0_DR12v5_CMASS_South.fits.gz  ~650 MB  (one random realization)
    random0_DR12v5_LOWZ_South.fits.gz   ~180 MB
"""

import argparse
import os
import urllib.request

_BASE_URL = "https://data.sdss.org/sas/dr12/boss/lss/"

_FILES = {
    "CMASS": {
        "data": ["galaxy_DR12v5_CMASS_South.fits.gz"],
        "randoms": [f"random{i}_DR12v5_CMASS_South.fits.gz" for i in range(18)],
    },
    "LOWZ": {
        "data": ["galaxy_DR12v5_LOWZ_South.fits.gz"],
        "randoms": [f"random{i}_DR12v5_LOWZ_South.fits.gz" for i in range(18)],
    },
}


def _download(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  Already present: {os.path.basename(dest_path)}  ({size_mb:.0f} MB)")
        return
    print(f"  Downloading {os.path.basename(dest_path)} ... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"{size_mb:.0f} MB")
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


def fetch_boss(
    dest: str = "data/boss",
    samples: list = None,
    n_randoms: int = 1,
) -> dict:
    """Download BOSS DR12 files.

    Parameters
    ----------
    dest        : destination directory
    samples     : list of samples to download, e.g. ['CMASS'] or ['CMASS', 'LOWZ']
    n_randoms   : number of random realisations (0-18; default 1)

    Returns
    -------
    dict mapping sample → {'data': [...paths...], 'randoms': [...paths...]}
    """
    if samples is None:
        samples = ["CMASS"]
    os.makedirs(dest, exist_ok=True)
    result = {}

    for sample in samples:
        spec = _FILES[sample]
        print(f"\n=== {sample} ===")
        data_paths = []
        for fname in spec["data"]:
            url = _BASE_URL + fname
            path = os.path.join(dest, fname)
            _download(url, path)
            data_paths.append(path)

        rand_paths = []
        for fname in spec["randoms"][:n_randoms]:
            url = _BASE_URL + fname
            path = os.path.join(dest, fname)
            _download(url, path)
            rand_paths.append(path)

        result[sample] = {"data": data_paths, "randoms": rand_paths}

    print("\nDone.")
    print("\nTo load in Python:")
    for sample in samples:
        print(f"  from twopt_density.boss import load_boss")
        print(f"  cat = load_boss(")
        print(f"      {result[sample]['data']},")
        print(f"      {result[sample]['randoms']},")
        print(f"      sample='{sample}')")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BOSS DR12 from SDSS")
    parser.add_argument("--dest", default="data/boss")
    parser.add_argument("--sample", default="CMASS",
                        help="CMASS, LOWZ, or both (comma-separated)")
    parser.add_argument("--n-randoms", type=int, default=1,
                        help="Number of random realisations to download (default 1)")
    args = parser.parse_args()
    samples = [s.strip().upper() for s in args.sample.split(",")]
    fetch_boss(dest=args.dest, samples=samples, n_randoms=args.n_randoms)
