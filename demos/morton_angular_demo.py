"""Angular ξ_LS(θ; z_shell) via the morton_cascade tree.

Maps Quaia/DESI ``(RA, Dec)`` to **unit vectors** on S² and runs the
bit-vector pair cascade in 3D Cartesian — the cascade then organises
neighbour-search by dyadic **chord length** ``d = 2 sin(θ/2)``,
which is identical to angular separation θ to ~0.07% accuracy at
12° (better at smaller θ). One cascade pass per z-shell yields
``DD(d)``, ``RR(d)``, ``DR(d)`` at every dyadic chord-shell, plus
ξ_LS(d) via Landy-Szalay; we convert d → θ for plotting.

Compared to the angular kNN-CDF pipeline this gives the same
cap-averaged σ²_LS information **per query catalog in a fraction of
a second per z-shell**, at the cost of dyadic chord shells (no
custom θ grid) and no per-query distribution (no H_geq_k, no VPF).
Use as the speed-up cousin of joint_knn_cdf when only the moment
statistics are needed.

Run: ``python demos/morton_angular_demo.py``. Requires the cascade
binary built (``cd morton_cascade && cargo build --release``).
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


def _load_quaia(z_min, z_max, fid):
    from twopt_density.quaia import load_quaia
    q = load_quaia(
        catalog_path="/Users/tabel/Research/data/quaia/quaia_G20.0.fits",
        selection_path="/Users/tabel/Research/data/quaia/selection_function_NSIDE64_G20.0.fits",
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    sel = hp.read_map(
        "/Users/tabel/Research/data/quaia/selection_function_NSIDE64_G20.0.fits",
    )
    m = (q.z_data >= z_min) & (q.z_data < z_max)
    return q.ra_data[m], q.dec_data[m], q.z_data[m], sel


def _load_desi(z_min, z_max, fid):
    from twopt_density.desi import load_desi_qso
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
    return d, sel_N, sel_S


def _make_random_quaia(ra_d, dec_d, z_d, sel, n_random, rng):
    from twopt_density.knn_analytic_rr import random_queries_from_selection_function
    return random_queries_from_selection_function(
        sel_map=sel, z_data=z_d, n_random=n_random, nside=64, rng=rng,
    )


def _make_random_desi(d_cat, sel_N, sel_S, n_random, rng):
    from twopt_density.desi import (
        random_queries_desi_per_region,
        split_n_random_by_data_fraction,
    )
    n_per = split_n_random_by_data_fraction(n_random, d_cat.photsys_data)
    region_z = {"N": d_cat.z_data[d_cat.photsys_data == "N"],
                "S": d_cat.z_data[d_cat.photsys_data == "S"]}
    ra, dec, z, _ = random_queries_desi_per_region(
        region_sel_maps={"N": sel_N, "S": sel_S},
        region_z_pools=region_z,
        n_random_per_region=n_per,
        nside=64, rng=rng,
    )
    return ra, dec, z


def _xi_per_shell(ra_d, dec_d, z_d, ra_r, dec_r, z_r, z_edges,
                   weights_d=None, label="cat"):
    """Run the cascade on each z-shell separately. Returns a list of
    XiResult-like dicts (one per shell)."""
    from twopt_density.morton_backend import (
        run_xi, radec_to_chord_box, chord_to_theta_deg,
    )
    n_z = z_edges.size - 1
    results = []
    for iq in range(n_z):
        z_lo, z_hi = z_edges[iq], z_edges[iq + 1]
        m_d = (z_d >= z_lo) & (z_d < z_hi)
        m_r = (z_r >= z_lo) & (z_r < z_hi)
        if not m_d.any() or not m_r.any():
            results.append(None)
            continue
        pts_d, L_d = radec_to_chord_box(ra_d[m_d], dec_d[m_d])
        pts_r, L_r = radec_to_chord_box(ra_r[m_r], dec_r[m_r])
        # Both maps share the same L (= 2 * 1.005), so use either.
        L = L_d
        wd = weights_d[m_d] if weights_d is not None else None
        t0 = time.time()
        res = run_xi(
            pts_d, pts_r, box_size=L,
            weights_data=wd,
            quiet=True,
            periodic=False,
        )
        dt = time.time() - t0
        # Convert chord shells to angular θ shells.
        # res.r is geometric mean of r_inner and r_outer, both in
        # box-coordinate units (= chord length here).
        theta_deg = chord_to_theta_deg(res.r)
        results.append(dict(
            iq=iq, z_lo=z_lo, z_hi=z_hi,
            n_d=int(m_d.sum()), n_r=int(m_r.sum()),
            chord=res.r, theta_deg=theta_deg,
            xi=res.xi, DD=res.DD, RR=res.RR, DR=res.DR,
            elapsed_s=dt,
        ))
        print(f"  shell {iq} (z=[{z_lo:.2f},{z_hi:.2f}], "
              f"N_d={m_d.sum()}, N_r={m_r.sum()}): {dt:.2f}s, "
              f"{res.n_levels} levels  [{label}]")
    return results


def main():
    from twopt_density.distance import DistanceCosmo

    fid = DistanceCosmo(Om=0.31, h=0.68)
    z_edges = np.array([0.8, 1.062, 1.362, 1.706, 2.1])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    print("loading Quaia ...")
    ra_q, dec_q, z_q, sel_q = _load_quaia(z_edges[0], z_edges[-1], fid)
    rng_q = np.random.default_rng(0)
    n_R_q = ra_q.size // 5
    ra_qr, dec_qr, z_qr = _make_random_quaia(
        ra_q, dec_q, z_q, sel_q, n_R_q, rng_q)
    print(f"  N_d={ra_q.size}, N_r={n_R_q}")

    print("loading DESI ...")
    d_cat, sel_N, sel_S = _load_desi(z_edges[0], z_edges[-1], fid)
    rng_d = np.random.default_rng(0)
    n_R_d = d_cat.ra_data.size // 5
    ra_dr, dec_dr, z_dr = _make_random_desi(
        d_cat, sel_N, sel_S, n_R_d, rng_d)
    print(f"  N_d={d_cat.ra_data.size}, N_r={ra_dr.size}")

    print("\n=== Quaia per-shell cascade ξ(θ) ===")
    res_q = _xi_per_shell(
        ra_q, dec_q, z_q, ra_qr, dec_qr, z_qr, z_edges,
        weights_d=None, label="Quaia",
    )
    t_q_total = sum(r["elapsed_s"] for r in res_q if r is not None)

    print("\n=== DESI per-shell cascade ξ(θ) ===")
    res_d = _xi_per_shell(
        d_cat.ra_data, d_cat.dec_data, d_cat.z_data,
        ra_dr, dec_dr, z_dr, z_edges,
        weights_d=d_cat.w_data, label="DESI",
    )
    t_d_total = sum(r["elapsed_s"] for r in res_d if r is not None)
    print(f"\nTotals: Quaia {t_q_total:.2f}s, DESI {t_d_total:.2f}s")

    # Plot ξ(θ) per shell, side-by-side with the angular σ²_LS
    # (loaded from the existing pipeline artifact if present).
    print("\nplotting ...")
    n_z = z_edges.size - 1
    fig, axes = plt.subplots(1, n_z, figsize=(4.0 * n_z, 4.5),
                              squeeze=False, sharey=True)

    # Try to overlay the angular kNN σ²_LS = ⟨ξ⟩_cap from the
    # existing pipeline for cross-check. The kNN observable is
    # cap-cumulative (sum of per-shell ξ inside a cap); the cascade
    # observable is per-shell differential ξ at each dyadic chord
    # bin. They are not literally the same number but should agree
    # in scale and sign — and converge at the smallest θ where the
    # cap is dominated by its outermost shell.
    knn_q_dd = knn_q_rd = knn_q_rr = None
    knn_d_dd = knn_d_rd = knn_d_rr = None
    cmp_path = os.path.join(OUT_DIR, "quaia_full_dd.npz")
    if os.path.exists(cmp_path):
        from twopt_density.knn_cdf import KnnCdfResult
        from twopt_density.knn_derived import xi_ls

        def _load_cube(prefix, theta_key, art_path):
            art = np.load(art_path)
            sum_n = art[f"{prefix}_sum_n"]
            return KnnCdfResult(
                H_geq_k=np.zeros(sum_n.shape + (1,), dtype=np.int64),
                sum_n=sum_n, sum_n2=art[f"{prefix}_sum_n2"],
                N_q=art[f"{prefix}_N_q"],
                theta_radii_rad=art[theta_key],
                z_q_edges=art["z_q_edges"], z_n_edges=art["z_n_edges"],
                flavor=prefix.upper(), backend_used="numba",
                area_per_cap=2 * np.pi * (1 - np.cos(art[theta_key])),
            )
        try:
            knn_q_dd = _load_cube("dd_bao", "theta_radii_bao_rad",
                                   os.path.join(OUT_DIR, "quaia_full_dd.npz"))
            knn_q_rd = _load_cube("rd", "theta_radii_rad",
                                   os.path.join(OUT_DIR, "quaia_full_rd_1x_kmax.npz"))
            knn_q_rr = _load_cube("rr", "theta_radii_rad",
                                   os.path.join(OUT_DIR, "quaia_full_rr.npz"))
            knn_d_dd = _load_cube("dd_bao", "theta_radii_bao_rad",
                                   os.path.join(OUT_DIR, "desi_full_dd.npz"))
            knn_d_rd = _load_cube("rd", "theta_radii_rad",
                                   os.path.join(OUT_DIR, "desi_full_rd_1x_kmax.npz"))
            knn_d_rr = _load_cube("rr", "theta_radii_rad",
                                   os.path.join(OUT_DIR, "desi_full_rr.npz"))
            print("  loaded kNN-CDF cubes for comparison overlay")
        except Exception as e:
            print(f"  could not load kNN cubes for overlay: {e}")
            knn_q_dd = None

    # DESI weighted normaliser (kept consistent with the rest of the demo).
    def _per_shell_w_sum(z, w, edges):
        out = np.zeros(edges.size - 1)
        for i in range(edges.size - 1):
            m = (z >= edges[i]) & (z < edges[i+1])
            out[i] = w[m].sum()
        return out

    if knn_d_dd is not None:
        d_w = _per_shell_w_sum(d_cat.z_data, d_cat.w_data, knn_d_dd.z_q_edges)
        knn_q_xi = xi_ls(knn_q_dd, knn_q_rd, knn_q_rr,
                          knn_q_dd.N_q.astype(float),
                          knn_q_dd.N_q.astype(float),
                          knn_q_rr.N_q.astype(float))
        knn_d_xi = xi_ls(knn_d_dd, knn_d_rd, knn_d_rr,
                          d_w, d_w,
                          knn_d_rr.N_q.astype(float))
        knn_theta_deg = np.degrees(knn_q_dd.theta_radii_rad)
    else:
        knn_q_xi = knn_d_xi = knn_theta_deg = None

    for iq in range(n_z):
        ax = axes[0, iq]
        for res, label, color, marker in (
            (res_q, "Quaia G<20", "#1f77b4", "o"),
            (res_d, "DESI Y1 QSO", "#ff7f0e", "s"),
        ):
            if res[iq] is None:
                continue
            theta = res[iq]["theta_deg"]
            xi = res[iq]["xi"]
            RR = res[iq]["RR"]
            good = (RR > 0) & np.isfinite(xi) & (theta > 0.05) & (theta < 12.0)
            if not good.any():
                continue
            ax.plot(theta[good], xi[good], marker + "-",
                    color=color, lw=1.4, ms=5,
                    label=f"{label} cascade")
        # kNN σ²_LS = ⟨ξ⟩_cap overlay (thin lines).
        if knn_q_xi is not None:
            ax.plot(knn_theta_deg, knn_q_xi[:, iq, iq],
                    "-", color="#1f77b4", lw=0.8, alpha=0.6,
                    label=r"Quaia σ²_LS (kNN cube, ⟨ξ⟩_cap)")
            ax.plot(knn_theta_deg, knn_d_xi[:, iq, iq],
                    "-", color="#ff7f0e", lw=0.8, alpha=0.6,
                    label=r"DESI σ²_LS (kNN cube, ⟨ξ⟩_cap)")
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("θ [deg]   (= 2 arcsin(d/2), d = chord)")
        if iq == 0:
            ax.set_ylabel(r"$\xi_{\rm LS}(\theta)$  cascade  vs.  $\sigma^2_{\rm LS}$ kNN")
        ax.set_title(f"z=[{z_edges[iq]:.2f}, {z_edges[iq+1]:.2f}]")
        ax.set_ylim(-0.05, 0.5)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Angular ξ_LS(θ;z) via morton_cascade  —  "
        f"Quaia {t_q_total:.1f}s, DESI {t_d_total:.1f}s total "
        f"({n_z} z-shells × 2 catalogs)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(OUT_DIR, "morton_angular_xi.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"  wrote {out_png}")

    # Save artifact.
    payload = {
        "z_edges": z_edges, "z_mid": z_mid,
        "quaia_t_total_s": t_q_total, "desi_t_total_s": t_d_total,
        "quaia_n_d": ra_q.size, "quaia_n_r": n_R_q,
        "desi_n_d": d_cat.ra_data.size, "desi_n_r": ra_dr.size,
    }
    for cat, res in (("quaia", res_q), ("desi", res_d)):
        for iq, r in enumerate(res):
            if r is None:
                continue
            payload[f"{cat}_iq{iq}_chord"] = r["chord"]
            payload[f"{cat}_iq{iq}_theta_deg"] = r["theta_deg"]
            payload[f"{cat}_iq{iq}_xi"] = r["xi"]
            payload[f"{cat}_iq{iq}_DD"] = r["DD"]
            payload[f"{cat}_iq{iq}_RR"] = r["RR"]
            payload[f"{cat}_iq{iq}_DR"] = r["DR"]
            payload[f"{cat}_iq{iq}_n_d"] = r["n_d"]
            payload[f"{cat}_iq{iq}_n_r"] = r["n_r"]
            payload[f"{cat}_iq{iq}_elapsed_s"] = r["elapsed_s"]
    np.savez_compressed(
        os.path.join(OUT_DIR, "morton_angular_xi.npz"), **payload,
    )
    print("  wrote morton_angular_xi.npz")


if __name__ == "__main__":
    main()
