"""Diagnose Gibbs ringing in xi_mask(theta) from sharp lmax cutoff,
and demonstrate Hann/Gaussian tapering of C_ell as the fix.

Plots:
  fig_diag/xi_mask_ringing.png:
    - xi_mask(theta) computed at lmax = 3*nside-1 (sharp cutoff)
    - same with Hann taper from l_taper to lmax
    - same with Gaussian taper at l_smooth
    - reference: deconvolve pixel window separately
  fig_diag/RR_at_pi0.png:
    - integrand chi^2 * xi_mask(theta = rp/chi) for the three tapers,
      at chi_eff = chi(z_med) and pi = 0
    - shows how the ringing in xi_mask propagates to RR(rp)
"""

from __future__ import annotations

import os
import time

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.distance import DistanceCosmo, comoving_distance
from twopt_density.quaia import load_selection_function


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def xi_from_cl(cl, theta, max_l=None):
    """Stable Legendre transform xi(theta) = sum_l (2l+1)/(4pi) C_l P_l(cos t)."""
    if max_l is None:
        max_l = len(cl) - 1
    cos_t = np.cos(theta)
    xi = np.zeros_like(theta)
    P_prev = np.ones_like(cos_t)
    xi += 1.0 / (4 * np.pi) * cl[0] * P_prev
    if max_l < 1:
        return xi
    P_curr = cos_t.copy()
    xi += 3.0 / (4 * np.pi) * cl[1] * P_curr
    for el in range(2, max_l + 1):
        P_next = ((2 * el - 1) * cos_t * P_curr - (el - 1) * P_prev) / el
        xi += (2 * el + 1) / (4 * np.pi) * cl[el] * P_next
        P_prev = P_curr
        P_curr = P_next
    return xi


def hann_taper(ell, l_taper, l_max):
    """Hann (raised-cosine) taper: 1 below l_taper, 0 above l_max,
    smooth (cos^2) transition between."""
    out = np.ones_like(ell, dtype=np.float64)
    above = ell > l_max
    out[above] = 0.0
    in_band = (ell >= l_taper) & (ell <= l_max)
    x = (ell[in_band] - l_taper) / max(l_max - l_taper, 1)
    out[in_band] = np.cos(0.5 * np.pi * x) ** 2
    return out


def gauss_taper(ell, l_smooth):
    """Gaussian beam in ell: exp(-0.5 (l/l_smooth)^2)."""
    return np.exp(-0.5 * (ell / l_smooth) ** 2)


def main():
    sel_path = os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits")
    if not os.path.exists(sel_path):
        raise FileNotFoundError(sel_path)

    print("loading mask ...")
    mask, nside = load_selection_function(sel_path)
    print(f"  NSIDE = {nside}, f_sky = {mask.mean():.3f}")

    lmax = 3 * nside - 1
    print(f"  lmax_sharp = {lmax}")
    cl = hp.anafast(mask, lmax=lmax)
    ell = np.arange(lmax + 1)

    # --- four versions of C_ell ---
    cl_sharp = cl.copy()
    cl_hann = cl * hann_taper(ell, l_taper=int(0.7 * lmax), l_max=lmax)
    cl_gauss = cl * gauss_taper(ell, l_smooth=int(0.5 * lmax))
    # very smooth gaussian at half lmax
    cl_gauss_half = cl * gauss_taper(ell, l_smooth=int(0.3 * lmax))

    theta = np.linspace(1e-6, 0.5, 1200)         # 0 to 28 deg
    xi_sharp = xi_from_cl(cl_sharp, theta)
    xi_hann = xi_from_cl(cl_hann, theta)
    xi_gauss = xi_from_cl(cl_gauss, theta)
    xi_gauss_half = xi_from_cl(cl_gauss_half, theta)

    # --- Fig 1: C_ell with tapers ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_cl, ax_xi_lin, ax_xi_log, ax_xi_zoom = axs.flatten()

    ax_cl.plot(ell, cl_sharp, "k-", lw=1, label="sharp cutoff")
    ax_cl.plot(ell, cl_hann, "C0-", lw=1.5,
                 label=fr"Hann $\ell_{{\rm taper}}=0.7\,\ell_{{\max}}$")
    ax_cl.plot(ell, cl_gauss, "C2-", lw=1.5,
                 label=fr"Gauss $\ell_{{\rm smooth}}=0.5\,\ell_{{\max}}$")
    ax_cl.plot(ell, cl_gauss_half, "C3-", lw=1.5,
                 label=fr"Gauss $\ell_{{\rm smooth}}=0.3\,\ell_{{\max}}$")
    ax_cl.set_yscale("log"); ax_cl.set_xlabel(r"$\ell$")
    ax_cl.set_ylabel(r"$C_\ell^{MM}$")
    ax_cl.set_title(r"Mask $C_\ell$ with various tapers")
    ax_cl.legend(fontsize=8); ax_cl.grid(alpha=0.3, which="both")

    theta_deg = np.rad2deg(theta)
    ax_xi_lin.plot(theta_deg, xi_sharp, "k-", lw=1, label="sharp cutoff (ringing)")
    ax_xi_lin.plot(theta_deg, xi_hann, "C0-", lw=1.5, label="Hann taper")
    ax_xi_lin.plot(theta_deg, xi_gauss, "C2-", lw=1.5, label="Gauss(0.5)")
    ax_xi_lin.plot(theta_deg, xi_gauss_half, "C3-", lw=1.5, label="Gauss(0.3)")
    ax_xi_lin.set_xlabel(r"$\theta$ [deg]"); ax_xi_lin.set_ylabel(r"$\xi_{\rm mask}(\theta)$")
    ax_xi_lin.set_title(r"$\xi_{\rm mask}(\theta)$ on linear scale")
    ax_xi_lin.set_xlim(0, 28)
    ax_xi_lin.legend(fontsize=8); ax_xi_lin.grid(alpha=0.3)

    ax_xi_log.semilogy(theta_deg, np.abs(xi_sharp), "k-", lw=1, label="sharp")
    ax_xi_log.semilogy(theta_deg, np.abs(xi_hann), "C0-", lw=1.5, label="Hann")
    ax_xi_log.semilogy(theta_deg, np.abs(xi_gauss), "C2-", lw=1.5, label="Gauss(0.5)")
    ax_xi_log.semilogy(theta_deg, np.abs(xi_gauss_half), "C3-", lw=1.5,
                          label="Gauss(0.3)")
    ax_xi_log.set_xlabel(r"$\theta$ [deg]")
    ax_xi_log.set_ylabel(r"$|\xi_{\rm mask}(\theta)|$")
    ax_xi_log.set_title(r"$|\xi_{\rm mask}|$ on log scale -- ringing visible at small $\theta$")
    ax_xi_log.set_xlim(0, 10)
    ax_xi_log.legend(fontsize=8); ax_xi_log.grid(alpha=0.3, which="both")

    # zoom on small theta where ringing manifests
    ax_xi_zoom.plot(theta_deg, xi_sharp, "k-", lw=1.2, label="sharp cutoff")
    ax_xi_zoom.plot(theta_deg, xi_hann, "C0-", lw=1.5, label="Hann")
    ax_xi_zoom.plot(theta_deg, xi_gauss, "C2-", lw=1.5, label="Gauss(0.5)")
    ax_xi_zoom.plot(theta_deg, xi_gauss_half, "C3-", lw=1.5, label="Gauss(0.3)")
    ax_xi_zoom.set_xlabel(r"$\theta$ [deg]")
    ax_xi_zoom.set_ylabel(r"$\xi_{\rm mask}(\theta)$")
    ax_xi_zoom.set_title(r"zoom: ringing in sharp-cutoff $\xi_{\rm mask}$")
    ax_xi_zoom.axhline(0, color="k", lw=0.5)
    ax_xi_zoom.set_xlim(0, 4)
    ax_xi_zoom.legend(fontsize=8); ax_xi_zoom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "diag_xi_mask_ringing.png"), dpi=150)
    plt.close(fig)

    # --- Fig 2: propagation to RR(rp) at pi=0 ---
    # at chi(z=1.5) ~ 3500 Mpc/h, theta = rp / chi, rp = 5..200 Mpc/h
    fid = DistanceCosmo(Om=0.31, h=0.68)
    chi_eff = float(np.asarray(comoving_distance(np.asarray([1.5]), fid))[0])
    rp_grid = np.linspace(2.0, 200.0, 800)
    th_at = rp_grid / chi_eff                                 # rad
    integ_sharp = chi_eff ** 2 * np.interp(th_at, theta, xi_sharp)
    integ_hann = chi_eff ** 2 * np.interp(th_at, theta, xi_hann)
    integ_gauss = chi_eff ** 2 * np.interp(th_at, theta, xi_gauss)
    integ_gauss_half = chi_eff ** 2 * np.interp(th_at, theta, xi_gauss_half)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rp_grid, integ_sharp, "k-", lw=1, label="sharp cutoff (ringing visible)")
    ax.plot(rp_grid, integ_hann, "C0-", lw=1.5, label="Hann")
    ax.plot(rp_grid, integ_gauss, "C2-", lw=1.5, label="Gauss(0.5)")
    ax.plot(rp_grid, integ_gauss_half, "C3-", lw=1.5, label="Gauss(0.3)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r_p$ [Mpc/h]")
    ax.set_ylabel(r"$\chi_{\rm eff}^2\,\xi_{\rm mask}(r_p/\chi_{\rm eff})$")
    ax.set_title(r"$RR(r_p)$ integrand at $\chi_{\rm eff}=\chi(z=1.5)$, $\pi=0$ "
                  r"-- ringing $\to$ $RR$ wobbles")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "diag_RR_integrand.png"), dpi=150)
    plt.close(fig)

    print(f"\nwrote diagnostics to {OUT_DIR}")
    print(f"  diag_xi_mask_ringing.png")
    print(f"  diag_RR_integrand.png")
    # numerical summary -- ringing amplitude at small theta
    small_th = (np.rad2deg(theta) > 0.1) & (np.rad2deg(theta) < 1.0)
    ring_amp_sharp = np.std(xi_sharp[small_th]) / max(np.abs(xi_sharp[small_th].mean()), 1e-30)
    ring_amp_hann = np.std(xi_hann[small_th]) / max(np.abs(xi_hann[small_th].mean()), 1e-30)
    print(f"\nrelative wobble of xi_mask(theta) for 0.1 < theta < 1 deg:")
    print(f"  sharp cutoff: {ring_amp_sharp:.2f}")
    print(f"  Hann taper :  {ring_amp_hann:.2f}")


if __name__ == "__main__":
    main()
