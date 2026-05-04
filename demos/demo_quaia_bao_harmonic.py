"""Harmonic-space BAO matched filter on Quaia C_ell^gg and Quaia x
Planck C_ell^{g-kappa}.

The BAO bump in wp(rp) is smeared by Quaia's photo-z LOS error
(sigma_chi ~ 170 Mpc/h vs the 105 Mpc/h sound horizon), capping the
matched-filter SNR at ~1-2 sigma. The BAO is also imprinted in
*angular* observables -- C_ell^gg and C_ell^{g-kappa} -- where there
is no LOS smearing, only the standard Limber kernel projection. The
ringing sits at ell ~ k_BAO * chi(z_eff) ~ 200 for z_eff ~ 1.5.

For each angular probe X in {gg, gk}, we measure the residual

    delta_X(ell) = C_ell^X_obs - b^2 (or b) * C_ell^X_smooth(no-wiggle)

and project it onto the BAO template

    T_X(ell) = b^p * [C_ell^X_full(BAO) - C_ell^X_nowiggle]

where p=2 for gg and p=1 for gk. The matched-filter amplitude is

    A_X = T_X^T C^-1 (delta_X) / T_X^T C^-1 T_X

with the same form as the wp BAO matched filter (twopt_density.bao_filter).
The Gaussian Knox covariance per multipole is

    Var(C_ell^X) = (1 / (2 ell + 1) f_sky) * sigma2_X(ell)

with sigma2_gg = (C_gg + N_gg)^2 and sigma2_gk = (C_gg + N_gg)(C_kk + N_kk)
+ C_gk^2.

Joint detection: stack the residual vectors and templates,

    delta = [delta_gg, delta_gk],   T = [T_gg, T_gk]

and run a single matched filter on the stacked vector with a block-
diagonal covariance (assuming negligible correlation between gg and
gk Gaussian variances at lowest order, which is fine when the two
share most ell modes -- the Cauchy-Schwarz cross is small).

Output: ``demos/figures/quaia_bao_harmonic.png`` with three panels:
    (left)   delta_gg vs T_gg, with best-fit A_gg
    (middle) delta_gk vs T_gk, with best-fit A_gk
    (right)  cumulative SNR(<ell) for gg, gk, joint.
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.angular import compute_cl_gg, compute_cl_gkappa
from twopt_density.distance import DistanceCosmo
from twopt_density.lensing import (
    cl_gkappa_limber, cl_gkappa_limber_nowiggle,
)
from twopt_density.limber import cl_gg_limber, cl_gg_limber_nowiggle
from twopt_density.quaia import load_quaia, load_selection_function


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
PLANCK_DIR = os.path.join(REPO_ROOT, "data", "planck")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _env_float(n, d):
    v = os.environ.get(n)
    return float(v) if v else d


def _build_planck_kappa_eq(nside: int, planck_dir: str):
    """Identical to demo_quaia_planck_lensing._build_planck_kappa_eq."""
    import healpy as hp

    lmax = 2 * nside
    fn_dat = os.path.join(planck_dir, "PR4_variations",
                            "PR42018like_klm_dat_MV.fits")
    fn_mf = os.path.join(planck_dir, "PR4_variations",
                            "PR42018like_klm_mf_MV.fits")
    fn_mask_g = os.path.join(planck_dir, "PR4_variations", "mask.fits.gz")
    alm_dat = hp.read_alm(fn_dat)
    alm_mf = hp.read_alm(fn_mf)
    alm_in_lmax = hp.Alm.getlmax(alm_dat.size)
    alm_clean = alm_dat - alm_mf

    alm_trunc = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    for l in range(lmax + 1):
        for m in range(l + 1):
            alm_trunc[hp.Alm.getidx(lmax, l, m)] = alm_clean[
                hp.Alm.getidx(alm_in_lmax, l, m)
            ]
    rot = hp.Rotator(coord=["G", "C"])
    alm_eq = rot.rotate_alm(alm_trunc.copy(), lmax=lmax)
    kappa_eq = hp.alm2map(alm_eq, nside, lmax=lmax, pol=False)

    mask_high_g = hp.read_map(fn_mask_g)
    mask_high_eq = rot.rotate_map_pixel(mask_high_g)
    mask_kappa_eq = hp.ud_grade(mask_high_eq, nside,
                                  order_in="RING", order_out="RING")
    cl_alm = hp.alm2cl(alm_trunc)
    return kappa_eq, mask_kappa_eq, cl_alm


def matched_filter_diag(delta, T, sigma):
    """Diagonal-Gaussian matched filter: A_hat, sigma_A, SNR."""
    invs2 = 1.0 / np.maximum(sigma, 1e-50) ** 2
    den = float(np.sum(T ** 2 * invs2))
    num = float(np.sum(T * delta * invs2))
    if den <= 0:
        return float("nan"), float("inf"), 0.0
    A = num / den
    sA = 1.0 / np.sqrt(den)
    return A, sA, A / sA


def matched_filter_marginalised(d, S, T, sigma):
    """Two-parameter joint fit ``d = A_S * S + A_T * T + noise`` with
    diagonal Gaussian ``sigma``. Returns ``(A_T_hat, sigma_A_T, SNR_A_T,
    A_S_hat)`` after analytically marginalising out ``A_S`` (the
    smooth-model amplitude, which absorbs any bias-squared normalisation
    error).
    """
    invs2 = 1.0 / np.maximum(sigma, 1e-50) ** 2
    SS = float(np.sum(S * S * invs2))
    TT = float(np.sum(T * T * invs2))
    ST = float(np.sum(S * T * invs2))
    Sd = float(np.sum(S * d * invs2))
    Td = float(np.sum(T * d * invs2))
    det = SS * TT - ST ** 2
    if det <= 0:
        return float("nan"), float("inf"), 0.0, float("nan")
    A_S = (TT * Sd - ST * Td) / det
    A_T = (SS * Td - ST * Sd) / det
    sigma_A_T = np.sqrt(SS / det)
    SNR = A_T / sigma_A_T
    return float(A_T), float(sigma_A_T), float(SNR), float(A_S)


def main():
    import healpy as hp

    fid = DistanceCosmo(Om=0.31, h=0.68)
    sigma8 = _env_float("QUAIA_SIGMA8", 0.81)
    z_min = _env_float("QUAIA_Z_MIN", 0.8)
    z_max = _env_float("QUAIA_Z_MAX", 2.5)
    nside = int(_env_float("QUAIA_BAO_NSIDE", 256))
    b_fid = _env_float("QUAIA_BIAS", 2.6)

    print(f"working NSIDE = {nside} (lmax = {3*nside-1})")
    print("loading Quaia ...")
    cat = load_quaia(
        catalog_path=os.path.join(DATA_DIR, "quaia_G20.0.fits"),
        selection_path=os.path.join(
            DATA_DIR, "selection_function_NSIDE64_G20.0.fits"),
        fid_cosmo=fid, n_random_factor=1, rng_seed=0,
    )
    mask_g_64, _ = load_selection_function(
        os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits"))
    mask_g = hp.ud_grade(mask_g_64, nside, order_in="RING", order_out="RING")
    md = (cat.z_data >= z_min) & (cat.z_data <= z_max)
    print(f"  N_data = {md.sum():,}, f_sky_g = {mask_g.mean():.3f}")

    # Planck PR4 kappa map
    print("building Planck PR4 kappa (galactic -> equatorial) ...")
    t0 = time.perf_counter()
    kappa, mask_kappa, _ = _build_planck_kappa_eq(nside, PLANCK_DIR)
    print(f"  {time.perf_counter()-t0:.0f}s, "
          f"f_sky_kappa = {mask_kappa.mean():.3f}")
    f_sky_gg = float(mask_g.mean())
    f_sky_gk = float((mask_g * mask_kappa).mean())

    # ---- 1. measurements ----
    print("\nC_ell^gg via NaMaster ...")
    t = time.perf_counter()
    meas_gg = compute_cl_gg(
        cat.ra_data[md], cat.dec_data[md], mask_g, nside=nside,
        n_per_bin=8,
    )
    print(f"  {time.perf_counter()-t:.0f}s, n_bins = {len(meas_gg.ell_eff)}")

    print("C_ell^{g-kappa} via NaMaster ...")
    t = time.perf_counter()
    meas_gk = compute_cl_gkappa(
        cat.ra_data[md], cat.dec_data[md],
        mask_g, kappa, mask_kappa, nside=nside, n_per_bin=8,
    )
    print(f"  {time.perf_counter()-t:.0f}s, n_bins = {len(meas_gk.ell_eff)}")

    # ---- 2. forward models ----
    z_grid = np.linspace(z_min + 0.05, z_max - 0.05, 60)
    nz, _ = np.histogram(cat.z_data[md], bins=z_grid, density=True)
    z_centres = 0.5 * (z_grid[:-1] + z_grid[1:])

    print(f"\nLimber forward models at b = {b_fid}, sigma_8 = {sigma8} ...")
    t = time.perf_counter()
    cl_gg_full = np.asarray(cl_gg_limber(
        meas_gg.ell_eff, z_centres, nz, fid, bias=b_fid, sigma8=sigma8))
    cl_gg_smooth = np.asarray(cl_gg_limber_nowiggle(
        meas_gg.ell_eff, z_centres, nz, fid, bias=b_fid, sigma8=sigma8))
    cl_gk_full = np.asarray(cl_gkappa_limber(
        meas_gk.ell_eff, z_centres, nz,
        np.full_like(z_centres, b_fid), fid, sigma8=sigma8))
    cl_gk_smooth = np.asarray(cl_gkappa_limber_nowiggle(
        meas_gk.ell_eff, z_centres, nz,
        np.full_like(z_centres, b_fid), fid, sigma8=sigma8))
    print(f"  {time.perf_counter()-t:.1f}s")
    T_gg = cl_gg_full - cl_gg_smooth
    T_gk = cl_gk_full - cl_gk_smooth
    print(f"  BAO template amplitude (peak): "
          f"|T_gg| = {np.max(np.abs(T_gg)):.3e}, "
          f"|T_gk| = {np.max(np.abs(T_gk)):.3e}")

    # ---- 3. Knox covariance ----
    # use measured C_ell^gg + the empirical kappa-kappa from PR4 alms.
    # for kappa-kappa we re-use the full-sky alm2cl used in the lensing
    # demo but evaluated at meas_gg.ell_eff -- approximation: the
    # observed C_kk is signal + Planck reconstruction noise.
    print("\nKnox covariance ...")
    # shot-noise-corrected C_gg observed (already done in cl_decoupled)
    cl_gg_obs = meas_gg.cl_decoupled
    cl_gk_obs = meas_gk.cl_decoupled
    n_shot_gg = float(meas_gg.n_shot)
    cl_gg_total = cl_gg_obs + n_shot_gg          # signal + shot noise
    # kappa-kappa total = empirical from rotated alm2cl (signal + noise)
    _, _, cl_alm = _build_planck_kappa_eq(nside, PLANCK_DIR)
    cl_kk_total_at_gg = np.interp(meas_gg.ell_eff,
                                     np.arange(len(cl_alm)), cl_alm)
    cl_kk_total_at_gk = np.interp(meas_gk.ell_eff,
                                     np.arange(len(cl_alm)), cl_alm)
    sigma_gg = np.sqrt(2.0 * cl_gg_total ** 2 /
                          ((2 * meas_gg.ell_eff + 1) * f_sky_gg))
    sigma_gk = np.sqrt((cl_gg_total * cl_kk_total_at_gk +
                         cl_gk_obs ** 2) /
                        ((2 * meas_gk.ell_eff + 1) * f_sky_gk))

    # ---- 4. matched filter with marginalised smooth + broadband ----
    # Robust BAO matched filter: fit
    #     cl_obs(ell) = A_smooth * cl_smooth(ell) + c0 + c1 * ell
    #                    + A_BAO * (cl_full - cl_smooth)
    # with A_smooth, c0, c1 as nuisance broadband terms. ``c0``
    # absorbs any residual shot-noise mis-subtraction (white) and
    # ``c1`` absorbs scale-dependent systematics (ell-linear) that
    # mimic a smooth slope. This is the standard BOSS/eBOSS BAO-only
    # recipe specialised to harmonic space.
    ell_min_fit = 30.0
    ell_max_useful = min(3 * 64 - 1, 3 * nside - 1)   # ~ 191
    use_gg = (meas_gg.ell_eff > ell_min_fit) & (meas_gg.ell_eff < ell_max_useful)
    use_gk = (meas_gk.ell_eff > ell_min_fit) & (meas_gk.ell_eff < ell_max_useful)

    def _bao_fit_with_broadband(d, S, T, ell, sigma):
        """Fit d = A_S S + c0 + c1 ell + A_T T; return A_T, sigma_AT,
        SNR, plus the inferred A_S, c0, c1."""
        M = np.column_stack([S, np.ones_like(ell), ell, T])      # (n, 4)
        invs2 = 1.0 / np.maximum(sigma, 1e-50) ** 2
        H = (M * invs2[:, None]).T @ M
        rhs = M.T @ (invs2 * d)
        cov_p = np.linalg.inv(H)
        p_hat = cov_p @ rhs
        A_T = float(p_hat[3]); sA = float(np.sqrt(cov_p[3, 3]))
        return (A_T, sA, A_T / sA,
                float(p_hat[0]), float(p_hat[1]), float(p_hat[2]))

    A_gg, sA_gg, snr_gg, AS_gg, c0_gg, c1_gg = _bao_fit_with_broadband(
        cl_gg_obs[use_gg], cl_gg_smooth[use_gg], T_gg[use_gg],
        meas_gg.ell_eff[use_gg], sigma_gg[use_gg])
    A_gk, sA_gk, snr_gk, AS_gk, c0_gk, c1_gk = _bao_fit_with_broadband(
        cl_gk_obs[use_gk], cl_gk_smooth[use_gk], T_gk[use_gk],
        meas_gk.ell_eff[use_gk], sigma_gk[use_gk])

    # joint (gg+gk): each probe has its own (A_S, c0, c1) nuisance,
    # one shared A_BAO. Total of 7 free parameters.
    n_gg = int(use_gg.sum()); n_gk = int(use_gk.sum())
    d_j = np.concatenate([cl_gg_obs[use_gg], cl_gk_obs[use_gk]])
    sigma_j = np.concatenate([sigma_gg[use_gg], sigma_gk[use_gk]])
    ell_gg = meas_gg.ell_eff[use_gg]; ell_gk = meas_gk.ell_eff[use_gk]
    Mblock = np.zeros((n_gg + n_gk, 7))
    Mblock[:n_gg, 0] = cl_gg_smooth[use_gg]
    Mblock[:n_gg, 1] = 1.0
    Mblock[:n_gg, 2] = ell_gg
    Mblock[n_gg:, 3] = cl_gk_smooth[use_gk]
    Mblock[n_gg:, 4] = 1.0
    Mblock[n_gg:, 5] = ell_gk
    Mblock[:n_gg, 6] = T_gg[use_gg]
    Mblock[n_gg:, 6] = T_gk[use_gk]
    invs2_j = 1.0 / np.maximum(sigma_j, 1e-50) ** 2
    H = (Mblock * invs2_j[:, None]).T @ Mblock
    rhs = Mblock.T @ (invs2_j * d_j)
    cov_p = np.linalg.inv(H)
    p_hat = cov_p @ rhs
    A_j = float(p_hat[6]); sA_j = float(np.sqrt(cov_p[6, 6]))
    snr_j = A_j / sA_j

    print(f"\nharmonic-space BAO matched filter "
          f"(broadband-marginalised: A_smooth + c0 + c1 ell + A_BAO T, "
          f"{ell_min_fit:.0f} < ell < {ell_max_useful}):")
    print(f"  C_ell^gg only:        A_BAO = {A_gg:.2f} +/- {sA_gg:.2f}, "
          f"SNR = {snr_gg:.2f}  (b_eff^2/b_fid^2={AS_gg:.3f}, c0={c0_gg:.2e})")
    print(f"  C_ell^{{g-kappa}} only: A_BAO = {A_gk:.2f} +/- {sA_gk:.2f}, "
          f"SNR = {snr_gk:.2f}  (b_eff/b_fid={AS_gk:.3f}, c0={c0_gk:.2e})")
    print(f"  joint (gg + gk):       A_BAO = {A_j:.2f} +/- {sA_j:.2f}, "
          f"SNR = {snr_j:.2f}")
    # Forecast (if data exactly equals fiducial model with A_BAO=1):
    snr_cr_gg = float(np.sqrt(np.sum((T_gg[use_gg] / sigma_gg[use_gg]) ** 2)))
    snr_cr_gk = float(np.sqrt(np.sum((T_gk[use_gk] / sigma_gk[use_gk]) ** 2)))
    snr_cr_j = float(np.sqrt(snr_cr_gg ** 2 + snr_cr_gk ** 2))
    print(f"\nCramer-Rao forecast (no nuisance marginalisation, "
          f"diagonal Knox sigma):")
    print(f"  CR SNR^A=1: gg = {snr_cr_gg:.2f}, gk = {snr_cr_gk:.2f}, "
          f"joint = {snr_cr_j:.2f}")
    delta_gg = cl_gg_obs - AS_gg * cl_gg_smooth - c0_gg - c1_gg * meas_gg.ell_eff
    delta_gk = cl_gk_obs - AS_gk * cl_gk_smooth - c0_gk - c1_gk * meas_gk.ell_eff

    # ---- 5. figure ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax_gg, ax_gk, ax_snr = axs

    ax_gg.errorbar(meas_gg.ell_eff[~use_gg], delta_gg[~use_gg],
                     yerr=sigma_gg[~use_gg], fmt="o", color="0.7", ms=4,
                     capsize=3, label="excluded")
    ax_gg.errorbar(meas_gg.ell_eff[use_gg], delta_gg[use_gg],
                     yerr=sigma_gg[use_gg], fmt="ok", ms=5, capsize=3,
                     label=r"$C_\ell^{gg, \rm obs} - b^2 C_\ell^{\rm smooth}$")
    ax_gg.plot(meas_gg.ell_eff, T_gg, "C0-", lw=2,
                 label=fr"BAO template ($A=1$)")
    ax_gg.plot(meas_gg.ell_eff, A_gg * T_gg, "C3--", lw=2,
                 label=fr"$A_{{gg}}={A_gg:.2f}\pm{sA_gg:.2f}$ "
                        fr"(SNR$={snr_gg:.1f}$)")
    ax_gg.axhline(0, color="k", lw=0.5)
    ax_gg.set_xscale("log")
    ax_gg.set_xlabel(r"$\ell$"); ax_gg.set_ylabel(r"$\Delta C_\ell^{gg}$")
    ax_gg.set_title(r"$C_\ell^{gg}$ BAO matched filter")
    ax_gg.legend(fontsize=8); ax_gg.grid(alpha=0.3, which="both")

    ax_gk.errorbar(meas_gk.ell_eff[~use_gk], delta_gk[~use_gk],
                     yerr=sigma_gk[~use_gk], fmt="o", color="0.7", ms=4,
                     capsize=3, label="excluded")
    ax_gk.errorbar(meas_gk.ell_eff[use_gk], delta_gk[use_gk],
                     yerr=sigma_gk[use_gk], fmt="ok", ms=5, capsize=3,
                     label=r"$C_\ell^{g\kappa, \rm obs} - b\,C_\ell^{\rm smooth}$")
    ax_gk.plot(meas_gk.ell_eff, T_gk, "C0-", lw=2,
                 label=fr"BAO template ($A=1$)")
    ax_gk.plot(meas_gk.ell_eff, A_gk * T_gk, "C3--", lw=2,
                 label=fr"$A_{{g\kappa}}={A_gk:.2f}\pm{sA_gk:.2f}$ "
                        fr"(SNR$={snr_gk:.1f}$)")
    ax_gk.axhline(0, color="k", lw=0.5)
    ax_gk.set_xscale("log")
    ax_gk.set_xlabel(r"$\ell$"); ax_gk.set_ylabel(r"$\Delta C_\ell^{g\kappa}$")
    ax_gk.set_title(r"$C_\ell^{g\kappa}$ BAO matched filter")
    ax_gk.legend(fontsize=8); ax_gk.grid(alpha=0.3, which="both")

    # cumulative SNR(<ell)
    snr2_gg = np.zeros_like(meas_gg.ell_eff)
    snr2_gg[use_gg] = (T_gg[use_gg] / sigma_gg[use_gg]) ** 2
    snr2_gk = np.zeros_like(meas_gk.ell_eff)
    snr2_gk[use_gk] = (T_gk[use_gk] / sigma_gk[use_gk]) ** 2
    cum_gg = np.sqrt(np.cumsum(snr2_gg))
    cum_gk = np.sqrt(np.cumsum(snr2_gk))
    cum_j = np.sqrt(np.cumsum(snr2_gg + snr2_gk))
    ax_snr.semilogx(meas_gg.ell_eff, cum_gg, "C0-o", ms=4,
                      label=fr"gg, total $A=1$ SNR = {cum_gg[-1]:.2f}")
    ax_snr.semilogx(meas_gk.ell_eff, cum_gk, "C2-s", ms=4,
                      label=fr"g-$\kappa$, total $A=1$ SNR = {cum_gk[-1]:.2f}")
    ax_snr.semilogx(meas_gg.ell_eff, cum_j, "C3-D", ms=4,
                      label=fr"joint, total $A=1$ SNR = {cum_j[-1]:.2f}")
    ax_snr.axvline(ell_max_useful, color="k", lw=0.7, ls=":",
                     label=fr"$\ell_{{\max}}^{{\rm useful}}={ell_max_useful}$")
    ax_snr.set_xlabel(r"$\ell$"); ax_snr.set_ylabel(r"cumulative SNR")
    ax_snr.set_title(r"BAO template-significance budget vs. $\ell$")
    ax_snr.legend(fontsize=8); ax_snr.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "quaia_bao_harmonic.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
