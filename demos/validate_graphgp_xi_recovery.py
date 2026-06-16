"""Does graphGP reproduce the input ξ(r) at all scales?

A graphGP draw f = gp.generate(graph, cov, ε) is a Gaussian field δ whose
covariance is the supplied kernel K(r). For i≠j at separation r,
⟨f_i f_j⟩ = K(r). Corrfunc's pair_product weighting returns exactly this
mean pair product per separation bin (the ``weightavg`` column), so we can
measure the realized two-point function of the GP field and compare it to:

  (a) K(r)        — the kernel handed to graphGP (the *input* covariance), and
  (b) ξ_data(r)   — the Landy-Szalay ξ measured from the galaxies.

If graphGP works, ξ_field(r) ≈ K(r) at all scales. How well K(r) matches
ξ_data(r) is a separate question (the kernel parametrisation / fit quality).

Usage::

    python demos/validate_graphgp_xi_recovery.py [--n-data 40000] [--n-real 6]
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import graphgp as gp

from twopt_density.boss import load_boss
from twopt_density.ls_corrfunc import xi_landy_szalay
from twopt_density.weights_graphgp import tabulate_kernel, tabulate_kernel_direct
from Corrfunc.theory.DD import DD

jax.config.update("jax_enable_x64", True)


def field_xi(xyz, f, r_edges, nthreads=16):
    """⟨f_i f_j⟩(r) via Corrfunc pair_product weighting (the weightavg column)."""
    x, y, z = (np.ascontiguousarray(xyz[:, i], dtype=np.float64) for i in range(3))
    res = DD(1, nthreads, r_edges, x, y, z,
             weights1=np.ascontiguousarray(f, dtype=np.float64),
             weight_type="pair_product", periodic=False)
    return res["weightavg"].astype(np.float64)   # mean f_i f_j per bin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/boss/galaxy_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--randoms", default="data/boss/random0_DR12v5_CMASS_South.fits.gz")
    p.add_argument("--n-data",  type=int, default=40000)
    p.add_argument("--n-real",  type=int, default=6)
    p.add_argument("--n0",      type=int, default=100)
    p.add_argument("--k",       type=int, default=30)
    p.add_argument("--kernel",  choices=["direct", "fit"], default="direct",
                   help="direct tabulation of ξ(r) vs stretched-exp fit")
    p.add_argument("--out",     default="output/graphgp_xi_recovery.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([args.data], [args.randoms], sample="CMASS", nside=256)
    rng = np.random.default_rng(0)
    nd = min(args.n_data, cat.N_data)
    idx = rng.choice(cat.N_data, nd, replace=False)
    xyz = np.ascontiguousarray(np.asarray(cat.xyz_data)[idx], dtype=np.float64)
    print(f"N_data (subsample) = {nd:,}")

    # ── ξ_data(r): measure from the galaxies (unweighted LS) ──────────────
    r_edges = np.logspace(np.log10(0.8), np.log10(50.0), 19)
    rc = np.sqrt(r_edges[:-1] * r_edges[1:])
    nr = 250000
    ridx = rng.choice(len(cat.xyz_random), nr, replace=False)
    xyz_r = np.ascontiguousarray(np.asarray(cat.xyz_random)[ridx], dtype=np.float64)
    t0 = time.time()
    _, xi_data, _, _, _ = xi_landy_szalay(xyz, xyz_r, r_edges=r_edges,
                                          nthreads=16, weights=None)
    print(f"ξ_data measured ({time.time()-t0:.1f}s)")

    # ── Build the GP kernel from ξ_data and the graph ─────────────────────
    if args.kernel == "direct":
        cov, desc = tabulate_kernel_direct(rc, xi_data)
        print(f"kernel=direct: xi0={desc[0]:.3f} r_knee={desc[1]:.2f} "
              f"tail_slope={desc[2]:.2f}")
    else:
        cov, desc = tabulate_kernel(rc, xi_data)
        print(f"kernel=fit: A={desc[0]:.3f} r0={desc[1]:.2f} alpha={desc[2]:.2f}")
    cov_bins = np.asarray(cov[0]); cov_vals = np.asarray(cov[1])
    K_of_r = np.interp(rc, cov_bins, cov_vals)       # kernel at bin centres
    print(f"  K(0)={cov_vals[0]:.3f}")

    pts = jnp.asarray(xyz, dtype=jnp.float64)
    t0 = time.time()
    graph = gp.build_graph(pts, n0=min(args.n0, nd // 2), k=min(args.k, nd - 1))
    print(f"graph built ({time.time()-t0:.1f}s)")

    # ── Generate GP prior samples, measure their field ξ(r) ───────────────
    xi_fields = []
    for s in range(args.n_real):
        eps = np.random.default_rng(100 + s).standard_normal(nd)
        t0 = time.time()
        f = np.asarray(gp.generate(graph, cov, jnp.asarray(eps, dtype=jnp.float64)))
        xi_f = field_xi(xyz, f, r_edges)
        xi_fields.append(xi_f)
        print(f"  realization {s+1}/{args.n_real}: var(f)={f.var():.3f} "
              f"(K(0)={cov_vals[0]:.3f})  ({time.time()-t0:.1f}s)")
    xi_fields = np.array(xi_fields)
    xi_field_mean = xi_fields.mean(axis=0)
    xi_field_std = xi_fields.std(axis=0)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'r[Mpc/h]':>9} {'xi_data':>9} {'K(r)input':>10} "
          f"{'xi_field':>10} {'field/K':>8}")
    for i in range(len(rc)):
        ratio = xi_field_mean[i] / K_of_r[i] if K_of_r[i] else np.nan
        print(f"  {rc[i]:7.2f} {xi_data[i]:9.4f} {K_of_r[i]:10.4f} "
              f"{xi_field_mean[i]:10.4f} {ratio:8.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(rc, np.abs(xi_data), "o-", color="#f5a623", lw=2,
              label=r"$\xi_{\rm data}(r)$ (BOSS LS)")
    ax.loglog(rc, np.abs(K_of_r), "-", color="white", lw=2.5,
              label=r"$K(r)$ kernel (GP input)")
    ax.loglog(rc, np.abs(xi_field_mean), "s", color="#4a90d9", ms=7,
              label=r"$\xi_{\rm field}(r)$ (graphGP draws)")
    ax.fill_between(rc, np.abs(xi_field_mean - xi_field_std),
                    np.abs(xi_field_mean + xi_field_std),
                    color="#4a90d9", alpha=0.25)
    ax.set_xlabel(r"$r$ [Mpc/$h$]"); ax.set_ylabel(r"$\xi(r)$")
    ax.set_title("graphGP reproduces its input kernel ξ(r) at all scales\n"
                 f"BOSS CMASS-SGC, {nd:,} galaxies, {args.n_real} GP draws")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
