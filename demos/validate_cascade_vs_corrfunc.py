"""Cross-validate the morton_cascade Landy-Szalay ξ(r) against Corrfunc.

Both estimators are run on the *identical* BOSS CMASS-SGC data + random
subsample. The cascade bins at dyadic shells; we feed Corrfunc the same
dyadic r-edges so the two ξ(r) are directly comparable shell-by-shell.

Corrfunc path uses the 3-D real-space pair counter (theory.DD) and the
Landy-Szalay combination computed by hand:

    ξ_LS(r) = (DD - 2 DR + RR) / RR

with the standard count normalisations (RR, DR rescaled by the data/random
ratio).  No GSL needed (theory module only).

Usage::

    python demos/validate_cascade_vs_corrfunc.py [--n-data 60000]
                                                 [--n-rand 200000]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twopt_density.boss import load_boss
from twopt_density.cascade import xi_landy_szalay


def corrfunc_xi_ls(xyz_d, xyz_r, r_edges, nthreads=16):
    """Landy-Szalay ξ(r) with Corrfunc theory.DD on 3-D comoving positions."""
    from Corrfunc.theory.DD import DD

    Nd = len(xyz_d)
    Nr = len(xyz_r)

    xd, yd, zd = (np.ascontiguousarray(xyz_d[:, i], dtype=np.float64)
                  for i in range(3))
    xr, yr, zr = (np.ascontiguousarray(xyz_r[:, i], dtype=np.float64)
                  for i in range(3))

    # autocorr=1 for DD and RR; autocorr=0 for DR
    dd = DD(1, nthreads, r_edges, xd, yd, zd, periodic=False)
    rr = DD(1, nthreads, r_edges, xr, yr, zr, periodic=False)
    dr = DD(0, nthreads, r_edges, xd, yd, zd,
            X2=xr, Y2=yr, Z2=zr, periodic=False)

    DD_c = dd["npairs"].astype(np.float64)
    RR_c = rr["npairs"].astype(np.float64)
    DR_c = dr["npairs"].astype(np.float64)

    # Normalisation: Corrfunc returns raw (directed, incl. self at r=0 only if
    # bins start at 0). With r_edges[0] > 0 there are no self-pairs.
    # DD, RR here count *ordered* pairs (i != j) → total ordered = N(N-1).
    norm_dd = Nd * (Nd - 1.0)
    norm_rr = Nr * (Nr - 1.0)
    norm_dr = Nd * float(Nr)

    dd_n = DD_c / norm_dd
    rr_n = RR_c / norm_rr
    dr_n = DR_c / norm_dr

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(rr_n > 0, (dd_n - 2.0 * dr_n + rr_n) / rr_n, np.nan)
    return xi, DD_c, RR_c, DR_c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms", default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-data",  type=int, default=60_000)
    p.add_argument("--n-rand",  type=int, default=200_000)
    p.add_argument("--nthreads", type=int, default=16)
    p.add_argument("--out",     default="output/cascade_vs_corrfunc.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── Load + subsample ──────────────────────────────────────────────────
    print("Loading BOSS CMASS-SGC ...")
    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    rng = np.random.default_rng(7)

    Nd_full = cat.N_data
    Nr_full = len(cat.ra_random)
    nd = min(args.n_data, Nd_full)
    nr = min(args.n_rand, Nr_full)
    id_ = rng.choice(Nd_full, nd, replace=False)
    ir_ = rng.choice(Nr_full, nr, replace=False)

    xyz_d = np.ascontiguousarray(cat.xyz_data[id_], dtype=np.float64)
    xyz_r = np.ascontiguousarray(cat.xyz_random[ir_], dtype=np.float64)

    # Shift to non-negative (cascade requirement; harmless for Corrfunc)
    shift = -np.vstack([xyz_d, xyz_r]).min(axis=0) + 100.0
    xyz_d += shift
    xyz_r += shift
    box_size = float(np.max(np.vstack([xyz_d, xyz_r]))) + 200.0
    print(f"  N_data={nd:,}  N_rand={nr:,}  box={box_size:.0f} Mpc/h")

    # ── Cascade ξ(r) ──────────────────────────────────────────────────────
    print("\ncascade ξ(r) ...")
    t0 = time.time()
    arr = xi_landy_szalay(xyz_d, xyz_r, box_size=box_size, dim=3,
                          periodic=False)
    t_casc = time.time() - t0
    print(f"  {t_casc:.2f}s")

    # Keep dyadic shells with finite width and meaningful counts
    ok = ((arr["r_outer_phys"] > arr["r_inner_phys"]) &
          (arr["dd"] > 200) & np.isfinite(arr["xi_ls"]))
    r_in  = arr["r_inner_phys"][ok]
    r_out = arr["r_outer_phys"][ok]
    r_cen = 0.5 * (r_in + r_out)
    xi_casc = arr["xi_ls"][ok]

    # ── Corrfunc ξ(r) on the SAME dyadic r-edges ─────────────────────────
    # Build matching edges from the cascade shells (ascending)
    order = np.argsort(r_in)
    r_in_s  = r_in[order]
    r_out_s = r_out[order]
    r_cen_s = r_cen[order]
    xi_casc_s = xi_casc[order]
    # Dyadic shells are contiguous: edges = [r_in_0, r_out_0=r_in_1, ...]
    r_edges = np.concatenate([r_in_s, [r_out_s[-1]]])

    print("\nCorrfunc ξ(r) on identical dyadic edges ...")
    t0 = time.time()
    xi_cf, DD_c, RR_c, DR_c = corrfunc_xi_ls(
        xyz_d, xyz_r, r_edges, nthreads=args.nthreads)
    t_cf = time.time() - t0
    print(f"  {t_cf:.2f}s")

    # ── Compare ───────────────────────────────────────────────────────────
    print(f"\n{'r[Mpc/h]':>10} {'ξ_cascade':>12} {'ξ_corrfunc':>12} "
          f"{'ratio':>8} {'Δ/ξ':>8}")
    for i in range(len(r_cen_s)):
        rc, xc, xf = r_cen_s[i], xi_casc_s[i], xi_cf[i]
        ratio = xc / xf if xf != 0 else np.nan
        frac = (xc - xf) / abs(xf) if xf != 0 else np.nan
        print(f"  {rc:8.2f} {xc:12.5f} {xf:12.5f} {ratio:8.3f} {frac:8.3f}")

    print(f"\nCascade: {t_casc:.2f}s (all dyadic shells in one pass)")
    print(f"Corrfunc: {t_cf:.2f}s ({len(r_edges)-1} bins)")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.loglog(r_cen_s, np.abs(xi_casc_s), "o-", color="#4a90d9",
               label="cascade (morton_cascade)", ms=6)
    ax1.loglog(r_cen_s, np.abs(xi_cf), "s--", color="#f5a623",
               label="Corrfunc theory.DD (LS)", ms=5, alpha=0.8)
    ax1.set_ylabel(r"$|\xi(r)|$")
    ax1.set_title("Cascade vs Corrfunc — BOSS CMASS-SGC Landy-Szalay "
                  f"$\\xi(r)$\n($N_d$={nd:,}, $N_r$={nr:,}, identical "
                  "dyadic bins)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.2)

    ratio = xi_casc_s / np.where(xi_cf != 0, xi_cf, np.nan)
    ax2.semilogx(r_cen_s, ratio, "o-", color="#333")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.fill_between(r_cen_s, 0.9, 1.1, color="green", alpha=0.1,
                     label="±10%")
    ax2.set_ylabel("cascade / Corrfunc")
    ax2.set_xlabel(r"$r$ [Mpc/$h$]")
    ax2.set_ylim(0.5, 1.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")

    # Quantitative verdict on the well-measured intermediate shells
    good = (r_cen_s > 5) & (r_cen_s < 120) & np.isfinite(ratio)
    if good.any():
        med_ratio = np.nanmedian(ratio[good])
        max_dev = np.nanmax(np.abs(ratio[good] - 1.0))
        print(f"\nVERDICT (5 < r < 120 Mpc/h):")
        print(f"  median cascade/Corrfunc = {med_ratio:.3f}")
        print(f"  max deviation from 1     = {max_dev*100:.1f}%")


if __name__ == "__main__":
    main()
