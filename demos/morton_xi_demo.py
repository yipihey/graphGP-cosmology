"""3D Cartesian Landy-Szalay ξ(r) via the morton_cascade Rust crate.

Computes ξ(r) on Quaia and DESI Y1 QSO using the bit-vector pair
cascade. All dyadic scales from sub-Mpc to ~box-half are produced in
one O(N log N) traversal — no per-r Corrfunc loop. Compares against
the angular cap-averaged σ²_LS(θ) we already have, with the
identification r ↔ θ · χ̄(z̄) at the central redshift.

Outputs ``output/morton_xi_quaia_desi.png`` and
``output/morton_xi_quaia_desi.npz``.

Build the Rust binary first: ``cd morton_cascade && cargo build --release``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "output")


def _to_cartesian_box(ra_d, dec_d, z_d, ra_r, dec_r, z_r, fid):
    """Convert two catalogs to a common Cartesian box [0,L)^3."""
    from twopt_density.morton_backend import radec_z_to_cartesian_for_cascade
    pts_d, _, _ = radec_z_to_cartesian_for_cascade(ra_d, dec_d, z_d, fid)
    pts_r, _, _ = radec_z_to_cartesian_for_cascade(ra_r, dec_r, z_r, fid)
    pts_all = np.vstack([pts_d, pts_r])
    L = float((pts_all.max(axis=0) - pts_all.min(axis=0)).max() * 1.05)
    pts_min = pts_all.min(axis=0)
    pad = 0.5 * (L - (pts_all.max(axis=0) - pts_all.min(axis=0)))
    pts_d2 = pts_d - pts_min + pad
    pts_r2 = pts_r - pts_min + pad
    return pts_d2, pts_r2, L


def main():
    from twopt_density.distance import DistanceCosmo, comoving_distance
    from twopt_density.morton_backend import run_xi
    from twopt_density.knn_analytic_rr import random_queries_from_selection_function
    import jax.numpy as jnp

    fid = DistanceCosmo(Om=0.31, h=0.68)
    z_min, z_max = 0.8, 2.1
    z_mid = 0.5 * (z_min + z_max)
    chi_mid = float(comoving_distance(jnp.asarray(z_mid), fid))
    print(f"Fiducial: z_mid={z_mid:.2f}, χ(z_mid)={chi_mid:.0f} Mpc/h")

    results = {}

    # ---- Quaia ----
    print("\n=== Quaia G<20 ===")
    from twopt_density.quaia import load_quaia
    q = load_quaia(
        catalog_path="/Users/tabel/Research/data/quaia/quaia_G20.0.fits",
        selection_path="/Users/tabel/Research/data/quaia/selection_function_NSIDE64_G20.0.fits",
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    m = (q.z_data >= z_min) & (q.z_data < z_max)
    ra_q, dec_q, z_q = q.ra_data[m], q.dec_data[m], q.z_data[m]
    sel_q = hp.read_map(
        "/Users/tabel/Research/data/quaia/selection_function_NSIDE64_G20.0.fits"
    )
    n_R = ra_q.size // 5
    ra_qr, dec_qr, z_qr = random_queries_from_selection_function(
        sel_map=sel_q, z_data=z_q, n_random=n_R, nside=64,
        rng=np.random.default_rng(0),
    )
    print(f"  N_d={ra_q.size}, N_r={n_R}")
    pts_d, pts_r, L_q = _to_cartesian_box(
        ra_q, dec_q, z_q, ra_qr, dec_qr, z_qr, fid)
    print(f"  box L = {L_q:.0f} Mpc/h")
    t0 = time.time()
    res_q = run_xi(pts_d, pts_r, box_size=L_q, quiet=True)
    print(f"  cascade xi: {time.time()-t0:.2f}s, {res_q.n_levels} levels")
    results["quaia"] = {
        "label": "Quaia G<20", "color": "#1f77b4",
        "n_d": ra_q.size, "n_r": n_R, "L": L_q,
        "r": res_q.r, "xi": res_q.xi,
        "DD": res_q.DD, "RR": res_q.RR, "DR": res_q.DR,
        "elapsed_s": res_q.elapsed_s,
    }

    # ---- DESI ----
    print("\n=== DESI Y1 QSO ===")
    from twopt_density.desi import (
        load_desi_qso, random_queries_desi_per_region,
        split_n_random_by_data_fraction,
    )
    d = load_desi_qso(
        catalog_paths=[os.path.join(REPO_ROOT, "data/desi/QSO_NGC_clustering.dat.fits"),
                       os.path.join(REPO_ROOT, "data/desi/QSO_SGC_clustering.dat.fits")],
        randoms_paths=None, fid_cosmo=fid, z_min=z_min, z_max=z_max,
        with_weight_fkp=True,
    )
    sel_N = hp.read_map(
        os.path.join(REPO_ROOT, "data/desi/desi_qso_y1_completeness_N_NSIDE64.fits"))
    sel_S = hp.read_map(
        os.path.join(REPO_ROOT, "data/desi/desi_qso_y1_completeness_S_NSIDE64.fits"))
    n_rand_per = split_n_random_by_data_fraction(
        d.ra_data.size // 5, d.photsys_data)
    region_z = {"N": d.z_data[d.photsys_data == "N"],
                "S": d.z_data[d.photsys_data == "S"]}
    ra_dr, dec_dr, z_dr, _ = random_queries_desi_per_region(
        region_sel_maps={"N": sel_N, "S": sel_S},
        region_z_pools=region_z,
        n_random_per_region=n_rand_per,
        nside=64, rng=np.random.default_rng(0),
    )
    print(f"  N_d={d.ra_data.size}, N_r={ra_dr.size}")
    pts_d, pts_r, L_d = _to_cartesian_box(
        d.ra_data, d.dec_data, d.z_data,
        ra_dr, dec_dr, z_dr, fid)
    print(f"  box L = {L_d:.0f} Mpc/h")
    t0 = time.time()
    res_d = run_xi(pts_d, pts_r, box_size=L_d,
                    weights_data=d.w_data, quiet=True)
    print(f"  cascade xi: {time.time()-t0:.2f}s, {res_d.n_levels} levels")
    results["desi"] = {
        "label": "DESI Y1 QSO", "color": "#ff7f0e",
        "n_d": d.ra_data.size, "n_r": ra_dr.size, "L": L_d,
        "r": res_d.r, "xi": res_d.xi,
        "DD": res_d.DD, "RR": res_d.RR, "DR": res_d.DR,
        "elapsed_s": res_d.elapsed_s,
    }

    # ---- Plot ξ(r) ---------------------------------------------------
    print("\nplotting ...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    for k, v in results.items():
        good = (v["RR"] > 0) & np.isfinite(v["xi"]) & (v["r"] > 1.0)
        ax.plot(v["r"][good], v["xi"][good], "o-", color=v["color"],
                lw=1.4, ms=4, label=f"{v['label']} ({v['elapsed_s']:.1f}s)")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("r [Mpc/h]")
    ax.set_ylabel(r"$\xi_{\rm LS}^{\rm 3D}(r)$")
    ax.set_title("Cascade Landy-Szalay ξ(r) — dyadic shells")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-0.05, 0.5)

    ax = axes[1]
    for k, v in results.items():
        good = (v["RR"] > 0) & np.isfinite(v["xi"]) & (v["r"] > 1.0)
        # Bias-magnitude plot: r²ξ shows the BAO scale better.
        ax.plot(v["r"][good], v["r"][good] ** 2 * v["xi"][good], "o-",
                color=v["color"], lw=1.4, ms=4, label=v["label"])
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("r [Mpc/h]")
    ax.set_ylabel(r"$r^2 \, \xi_{\rm LS}(r)$  [Mpc²/h²]")
    ax.set_title("r²ξ — emphasises BAO / clustering scale")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "morton_xi_quaia_desi.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_png}")

    np.savez_compressed(
        os.path.join(OUT_DIR, "morton_xi_quaia_desi.npz"),
        **{f"{cat}_{k}": v
           for cat, d in results.items()
           for k, v in d.items() if isinstance(v, np.ndarray)},
        quaia_label=results["quaia"]["label"],
        desi_label=results["desi"]["label"],
        quaia_n_d=results["quaia"]["n_d"], quaia_n_r=results["quaia"]["n_r"],
        desi_n_d=results["desi"]["n_d"], desi_n_r=results["desi"]["n_r"],
        quaia_L=results["quaia"]["L"], desi_L=results["desi"]["L"],
        quaia_elapsed_s=results["quaia"]["elapsed_s"],
        desi_elapsed_s=results["desi"]["elapsed_s"],
    )
    print(f"  wrote morton_xi_quaia_desi.npz")


if __name__ == "__main__":
    main()
