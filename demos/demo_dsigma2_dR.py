"""dsigma^2(R)/dR -- variance in spherical shells, two natural estimators.

Demonstrates the two natural kernels for the R-derivative of the
top-hat sphere variance, both implemented in twopt_density.sigma2:

  Option 1 -- analytic derivative kernel  partial K_TH / partial R
  Option 2 -- thick-shell overlap kernel  K_shell(r; R_in, R_out)

Both produce the same physical observable (the change in sigma^2(R)
with respect to R) but with different normalisations and noise
properties:

  Option 1 is bipolar (positive at small r, negative at intermediate)
    and integrates to zero -- cleanest "exact derivative".
  Option 2 is always non-negative and integrates to 1 -- the natural
    "variance in a shell of width R_out - R_in" probe.

Plots:
  fig 1 -- the four kernels K_TH, dK_TH/dR, K_shell at three R values
  fig 2 -- sigma^2(R) and dsigma^2/dR on Quaia G < 20
"""

from __future__ import annotations

import os

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twopt_density.analytic_rr import dr_analytic, rr_analytic
from twopt_density.distance import DistanceCosmo
from twopt_density.projected_xi import _count_pairs_rp_pi, wp_landy_szalay
from twopt_density.quaia import load_quaia, load_selection_function
from twopt_density.sigma2 import (
    dsigma2_dR_from_xi, kernel_shell_3d, kernel_TH_3d,
    kernel_TH_derivative_3d, sigma2_from_xi, sigma2_shell_from_xi,
)


jax.config.update("jax_enable_x64", True)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "quaia")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # --- Fig 1: kernel shapes ---
    R_vals = [10.0, 30.0, 60.0]
    r = np.linspace(0.0, 150.0, 1500)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    ax_K, ax_dK, ax_S = axs

    for R, c in zip(R_vals, ["C0", "C2", "C3"]):
        ax_K.plot(r, kernel_TH_3d(r, R), color=c, lw=1.6,
                    label=fr"$R={R:.0f}$")
        ax_dK.plot(r, kernel_TH_derivative_3d(r, R), color=c, lw=1.6,
                     label=fr"$R={R:.0f}$")
        # thick shell of width 0.2 R centered at R
        h = 0.2 * R
        ax_S.plot(r, kernel_shell_3d(r, R - h, R + h), color=c, lw=1.6,
                    label=fr"$R={R:.0f}$, $\Delta R=0.4\,R$")
    ax_K.set_yscale("log"); ax_K.set_ylim(1e-9, 1e-2)
    ax_K.set_xlabel(r"$r$ [Mpc/h]"); ax_K.set_ylabel(r"$K_{\rm TH}(r; R)$")
    ax_K.set_title(r"top-hat sphere kernel $K_{\rm TH}$")
    ax_K.legend(fontsize=8); ax_K.grid(alpha=0.3, which="both")

    ax_dK.axhline(0, color="k", lw=0.5)
    ax_dK.set_xlabel(r"$r$ [Mpc/h]")
    ax_dK.set_ylabel(r"$\partial K_{\rm TH} / \partial R$")
    ax_dK.set_title(r"derivative kernel $\partial K_{\rm TH}/\partial R$ "
                       r"(bipolar, $\int = 0$)")
    ax_dK.legend(fontsize=8); ax_dK.grid(alpha=0.3)

    ax_S.set_xlabel(r"$r$ [Mpc/h]")
    ax_S.set_ylabel(r"$K_{\rm shell}(r; R_{\rm in}, R_{\rm out})$")
    ax_S.set_title(r"thick-shell overlap kernel "
                      r"($\int = 1$)")
    ax_S.legend(fontsize=8); ax_S.grid(alpha=0.3)

    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, "dsigma2_kernels.png")
    fig.savefig(out1, dpi=140); plt.close(fig)

    # --- Fig 2: dsigma2/dR on Quaia ---
    sel_path = os.path.join(DATA_DIR, "selection_function_NSIDE64_G20.0.fits")
    cat_path = os.path.join(DATA_DIR, "quaia_G20.0.fits")
    if not (os.path.exists(sel_path) and os.path.exists(cat_path)):
        print(f"skipping Quaia panel (missing {cat_path})")
        print(f"wrote {out1}")
        return

    fid = DistanceCosmo(Om=0.31, h=0.68)
    print("loading Quaia ...")
    cat = load_quaia(catalog_path=cat_path, selection_path=sel_path,
                       fid_cosmo=fid, n_random_factor=2, rng_seed=0)
    md = (cat.z_data >= 0.8) & (cat.z_data <= 2.5)
    mr = (cat.z_random >= 0.8) & (cat.z_random <= 2.5)
    mask, nside = load_selection_function(sel_path)

    rng = np.random.default_rng(0)
    n_d = 30_000; n_r = 90_000
    iD = rng.choice(int(md.sum()), n_d, replace=False)
    iR = rng.choice(int(mr.sum()), n_r, replace=False)
    pos_d = cat.xyz_data[np.where(md)[0][iD]]
    pos_r = cat.xyz_random[np.where(mr)[0][iR]]
    z_d = cat.z_data[np.where(md)[0][iD]]
    shift = -np.vstack([pos_d, pos_r]).min(axis=0) + 100.0
    pos_d = pos_d + shift; pos_r = pos_r + shift

    rp_edges = np.concatenate([
        np.logspace(np.log10(2.0), np.log10(40.0), 12),
        np.linspace(50.0, 200.0, 14)[1:],
    ])
    pi_edges = np.linspace(0.0, 200.0, 41)
    rp_c = 0.5 * (rp_edges[1:] + rp_edges[:-1])
    pi_c = 0.5 * (pi_edges[1:] + pi_edges[:-1])

    print("DD pair counts ...")
    DD = _count_pairs_rp_pi(pos_d, pos_d, rp_edges, pi_edges, auto=True,
                              chunk=4000)
    print("analytic RR + DR ...")
    res = rr_analytic(rp_edges, pi_edges, mask, nside, z_d, fid,
                        N_r=10 * n_d)
    cal = wp_landy_szalay(pos_d[: 8000], pos_r[: 24000], rp_edges,
                            pi_max=200.0, n_pi=40)
    cal_a = rr_analytic(rp_edges, cal.pi_edges, mask, nside,
                          z_d[: 8000], fid, N_r=24000)
    calib = float(np.median(cal.RR[cal.RR > 0]
                              / np.maximum(cal_a.RR[cal.RR > 0], 1e-30)))
    RR = calib * res.RR
    DR = dr_analytic(n_d, 10 * n_d, RR)

    # binned LS xi(s) on the (rp, pi) grid via shell averaging in
    # s = sqrt(rp^2 + pi^2)
    s_edges = np.linspace(2.0, 250.0, 36)
    s_c = 0.5 * (s_edges[1:] + s_edges[:-1])
    s2d = np.sqrt(rp_c[:, None] ** 2 + pi_c[None, :] ** 2)
    Ndp = n_d * (n_d - 1) / 2.0; Nrp = (10 * n_d) * (10 * n_d - 1) / 2.0
    DD_n = DD / Ndp; RR_n = RR / Nrp; DR_n = DR / (n_d * 10 * n_d)
    xi = np.zeros_like(s_c)
    for k in range(len(s_c)):
        m = (s2d >= s_edges[k]) & (s2d < s_edges[k + 1])
        if not m.any():
            continue
        SDD = float(np.sum(DD_n[m])); SDR = float(np.sum(DR_n[m]))
        SRR = float(np.sum(RR_n[m]))
        if SRR > 0:
            xi[k] = (SDD - 2 * SDR + SRR) / SRR
    print("xi(s) measured")

    R_grid = np.linspace(8.0, 90.0, 24)
    s2 = sigma2_from_xi(s_c, xi, R_grid, kernel="tophat")
    ds2 = dsigma2_dR_from_xi(s_c, xi, R_grid)
    # finite-difference cross-check
    h = 0.5
    ds2_fd = (sigma2_from_xi(s_c, xi, R_grid + h)
                - sigma2_from_xi(s_c, xi, R_grid - h)) / (2.0 * h)
    # shell-variance at fixed shell width
    shell_widths = [4.0, 8.0]
    shell_vals = {dR: np.array([sigma2_shell_from_xi(s_c, xi, R - dR/2,
                                                            R + dR/2)
                                  for R in R_grid])
                   for dR in shell_widths}

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax_s2, ax_ds2 = axs

    ax_s2.plot(R_grid, s2, "C0o-", ms=5, label=r"$\sigma^2(R)$")
    ax_s2.axhline(0, color="k", lw=0.5)
    ax_s2.set_xlabel(r"$R$ [Mpc/h]")
    ax_s2.set_ylabel(r"$\sigma^2_{\rm TH}(R)$")
    ax_s2.set_title(r"Quaia G$<$20: top-hat sphere variance")
    ax_s2.legend(fontsize=10); ax_s2.grid(alpha=0.3)

    ax_ds2.plot(R_grid, ds2, "C0s-", ms=5, lw=1.6,
                  label=r"option 1: $\int \xi\,\partial K_{\rm TH}/\partial R$")
    ax_ds2.plot(R_grid, ds2_fd, "C2--", lw=1.4,
                  label=r"finite-difference of $\sigma^2$")
    for dR, c in zip(shell_widths, ["C3", "C1"]):
        ax_ds2.plot(R_grid, shell_vals[dR], color=c, marker="^", ms=4,
                       lw=1.2, ls=":",
                       label=fr"option 2: $\sigma^2_{{\rm shell}}(R, "
                              fr"\Delta R={dR:.0f})$")
    ax_ds2.axhline(0, color="k", lw=0.5)
    ax_ds2.set_xlabel(r"$R$ [Mpc/h]")
    ax_ds2.set_ylabel(r"$d\sigma^2/dR$ or $\sigma^2_{\rm shell}$")
    ax_ds2.set_title(r"shell-variance / derivative on Quaia")
    ax_ds2.legend(fontsize=8); ax_ds2.grid(alpha=0.3)

    fig.tight_layout()
    out2 = os.path.join(FIG_DIR, "quaia_dsigma2_dR.png")
    fig.savefig(out2, dpi=140); plt.close(fig)
    print(f"\nwrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
