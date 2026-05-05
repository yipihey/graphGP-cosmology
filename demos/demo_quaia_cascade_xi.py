"""Quaia xi(r) at every dyadic shell via the cascade with the random
catalogue auto-generated from the (mask, n(z)) survey window.

Wires together:
  - twopt_density.cascade.xi_landy_szalay_from_window
       which under the hood calls quaia.make_random_from_selection_function
       to synthesise an N_r-point random from sel_map x n(z), then
       runs the morton_cascade ``xi`` subcommand. The user never has
       to instantiate randoms manually.

Result is the full Landy-Szalay xi(r) at every dyadic shell from
~box-size down to a few Mpc/h, in O(N log N).

Output: demos/figures/quaia_cascade_xi.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.cascade import xi_landy_szalay_from_window
from twopt_density.distance import DistanceCosmo
from twopt_density.quaia import load_quaia, load_selection_function


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    n_data = int(os.environ.get("PAPER_N_DATA", 80_000))
    n_random_factor = int(os.environ.get("PAPER_N_RANDOM_FACTOR", 5))

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    sel_map, nside = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    md = (cat.z_data >= 0.8) & (cat.z_data <= 2.5)
    rng = np.random.default_rng(0)
    iD = rng.choice(int(md.sum()),
                       min(n_data, int(md.sum())),
                       replace=False)
    where = np.where(md)[0][iD]
    pos_d = cat.xyz_data[where]
    ra_d = cat.ra_data[where]; dec_d = cat.dec_data[where]
    z_d = cat.z_data[where]
    print(f"  N_data = {len(pos_d):,}, "
          f"N_random_factor = {n_random_factor}")

    print("\ncascade xi(r) with window-synthesised randoms ...")
    t0 = time.perf_counter()
    arr = xi_landy_szalay_from_window(
        pos_d, ra_d, dec_d, z_d, sel_map, nside, fid,
        n_random_factor=n_random_factor, rng_seed=42,
    )
    t = time.perf_counter() - t0
    print(f"  {t:.1f}s end-to-end ({len(pos_d):,} data + "
          f"{n_random_factor * len(pos_d):,} synth randoms)")

    # report the headline shells
    print(f"\nLandy-Szalay xi(r) at significant shells (DD > 1000):")
    print(f"  level   r_in    r_out       DD          RR        DR        xi_LS")
    for r in arr:
        if r["dd"] >= 1000 and r["r_outer_phys"] > r["r_inner_phys"]:
            print(f"   L{r['level']:2d}  "
                  f"{r['r_inner_phys']:7.1f} {r['r_outer_phys']:7.1f}"
                  f"  {r['dd']:10.0f}  {r['rr']:10.0f}"
                  f"  {r['dr']:9.0f}  {r['xi_ls']:+.4f}")

    # figure: xi(r) per shell
    fig, ax = plt.subplots(figsize=(8, 5))
    use = (arr["dd"] >= 100) & (arr["r_outer_phys"] > arr["r_inner_phys"])
    r_mid = 0.5 * (arr["r_inner_phys"][use] + arr["r_outer_phys"][use])
    sigma = np.maximum(np.abs(arr["xi_ls"][use])
                          / np.sqrt(np.maximum(arr["dd"][use], 1.0)), 1e-3)
    ax.errorbar(r_mid, arr["xi_ls"][use], yerr=sigma, fmt="ko-", ms=5,
                  capsize=3, label=fr"cascade LS xi(r) (auto randoms)")
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.5)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"$r$ [Mpc/h] (shell midpoint)")
    ax.set_ylabel(r"$\xi_{\rm LS}(r)$")
    ax.set_title(r"Quaia G$<$20 xi(r) -- cascade DD + cascade RR (auto-"
                  fr"randoms), N$_d$={len(pos_d):,}")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_cascade_xi.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
