"""Validate the analytic-window FKP-KDE field on BOSS CMASS-SGC.

No MC random catalog is used for the density estimate: the expected random
density is the analytic survey window ρ̂_W ∝ S_ang·n(z)/χ². Candidate points
for the posterior-predictive catalog are drawn on the fly from the same
window (sel_map + n(z)) and thinned by 1+δ. We then measure ξ(r) with
Corrfunc for the observed data and for a thinned realization and compare.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from twopt_density.boss import load_boss
from twopt_density.window import build_survey_window
from twopt_density.density_field import _fkp_kde_analytic
from twopt_density.distance import radec_z_to_cartesian
from twopt_density.quaia import make_random_from_selection_function
from Corrfunc.theory.DD import DD


def xi_corrfunc(xyz_d, xyz_r, r_edges, nthreads=16):
    nd, nr = len(xyz_d), len(xyz_r)
    xd, yd, zd = (np.ascontiguousarray(xyz_d[:, i]) for i in range(3))
    xr, yr, zr = (np.ascontiguousarray(xyz_r[:, i]) for i in range(3))
    dd = DD(1, nthreads, r_edges, xd, yd, zd, periodic=False)["npairs"].astype(float)
    rr = DD(1, nthreads, r_edges, xr, yr, zr, periodic=False)["npairs"].astype(float)
    dr = DD(0, nthreads, r_edges, xd, yd, zd, X2=xr, Y2=yr, Z2=zr,
            periodic=False)["npairs"].astype(float)
    ddn = dd / (nd * (nd - 1.0)); rrn = rr / (nr * (nr - 1.0)); drn = dr / (nd * nr)
    return np.where(rrn > 0, (ddn - 2 * drn + rrn) / rrn, np.nan)


def main():
    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256)
    print(f"N_data={cat.N_data:,}")
    cosmo = cat.fid_cosmo
    w_comp = (np.asarray(cat.w_sys_data) * np.asarray(cat.w_noz_data)
              * np.asarray(cat.w_cp_data))
    print(f"completeness weight: mean={w_comp.mean():.3f}")

    # ── Build analytic window ─────────────────────────────────────────────
    win = build_survey_window(cat, kde_bandwidth=0.02)
    print(f"window: omega_eff={win.omega_eff:.4f} sr  "
          f"f_sky={win.omega_eff/(4*np.pi):.4f}  "
          f"chi=[{win.chi_min:.0f},{win.chi_max:.0f}] Mpc/h")

    xyz_d = np.ascontiguousarray(np.asarray(cat.xyz_data), dtype=np.float64)
    radecz_d = (np.asarray(cat.ra_data), np.asarray(cat.dec_data),
                np.asarray(cat.z_data))

    # ── 1+δ at data positions (sanity) ────────────────────────────────────
    t0 = time.time()
    opd_d, h_d = _fkp_kde_analytic(xyz_d, radecz_d, xyz_d, win,
                                   w_data=w_comp, k_bw=8, k_sum=40, h_min=2.0)
    print(f"\n1+δ at data: mean={opd_d.mean():.3f} median={np.median(opd_d):.3f} "
          f"std={opd_d.std():.3f}  ({time.time()-t0:.1f}s)")
    print(f"adaptive bandwidth h: median={np.median(h_d):.1f} "
          f"[{np.percentile(h_d,5):.1f},{np.percentile(h_d,95):.1f}] Mpc/h")

    # ── Draw candidate points from the window (analytic random) ───────────
    rng = np.random.default_rng(1)
    n_cand = 4 * cat.N_data
    ra_c, dec_c, z_c = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=n_cand, z_data=np.asarray(cat.z_data),
        nside=cat.nside, rng=rng)
    xyz_c = np.ascontiguousarray(
        np.asarray(radec_z_to_cartesian(ra_c, dec_c, z_c, cosmo)), dtype=np.float64)
    print(f"\nwindow candidates: {len(ra_c):,}")

    # ── 1+δ at candidates, then Poisson-thin to N_data ────────────────────
    opd_c, _ = _fkp_kde_analytic(xyz_c, (ra_c, dec_c, z_c), xyz_d, win,
                                 w_data=w_comp, k_bw=8, k_sum=40, h_min=2.0)
    w_sum = float(w_comp.sum())
    alpha = w_sum / opd_c.sum()
    p = np.clip(alpha * opd_c, 0.0, 1.0)
    accept = rng.uniform(size=len(p)) < p
    xyz_s = xyz_c[accept]
    print(f"thinned sample: N={accept.sum():,}  (target {w_sum:.0f}, "
          f"clip_frac={np.mean(alpha*opd_c>1):.4f})")

    # ── ξ(r): observed vs analytic-window realization ─────────────────────
    r_edges = np.logspace(np.log10(1.0), np.log10(60.0), 13)
    rc = np.sqrt(r_edges[:-1] * r_edges[1:])
    # subsample a fresh window set as the ξ random reference
    ra_rr, dec_rr, z_rr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=600000, z_data=np.asarray(cat.z_data),
        nside=cat.nside, rng=np.random.default_rng(7))
    xyz_rr = np.ascontiguousarray(
        np.asarray(radec_z_to_cartesian(ra_rr, dec_rr, z_rr, cosmo)), dtype=np.float64)

    xi_obs = xi_corrfunc(xyz_d, xyz_rr, r_edges)
    xi_smp = xi_corrfunc(xyz_s, xyz_rr, r_edges)

    print(f"\n{'r[Mpc/h]':>9} {'xi_obs':>9} {'xi_sample':>10} {'ratio':>7}")
    for i in range(len(rc)):
        ratio = xi_smp[i] / xi_obs[i] if xi_obs[i] else np.nan
        print(f"  {rc[i]:7.2f} {xi_obs[i]:9.4f} {xi_smp[i]:10.4f} {ratio:7.3f}")


if __name__ == "__main__":
    main()
