"""Audit: is the success-trained photo-z reliable for the FAILURE population?

The photo-z (twopt_density.photoz.PhotoZKNN) is trained on spectroscopic
SUCCESSES and then applied to the missing targets (fiber collisions, redshift
failures) to assign their redshifts in the completion. A referee will ask
whether the failure population is representable by the success-trained model, or
whether it is an extrapolation (e.g. redshift failures correlate with low S/N /
edge-of-colour-space photometry). We cannot measure photo-z accuracy on the
failures directly (they have no spec-z — that is why they failed), so we
quantify *representability* instead:

  (1) colour/i-mag distribution shift, successes vs collided vs zfail (KS).
  (2) colour-space EXTRAPOLATION: fraction of failures whose k-th nearest
      training (success) neighbour, in the photo-z's own whitened metric, lies
      beyond the 99th percentile of the success-to-success k-NN distance — i.e.
      regions the training set barely covers.
  (3) photo-z posterior WIDTH (uncertainty) for failures vs successes.
  (4) z_host-fallback fraction (degenerate posteriors) from the completion.

Cosmology-free (colours, apparent mags, redshifts only).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 demos/audit_photoz_failures.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.spatial import cKDTree
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features
from twopt_density.cmass_targets import load_cmass_targets

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
LABELS = ["u-g", "g-r", "r-i", "i-z"][:None]  # photoz_features = [g-r, r-i, i-z, i_mag]
FEAT_NAMES = ["g-r", "r-i", "i-z", "i_mag"]


def _post_width(pz, feat):
    """Weighted std of each object's photo-z posterior (uncertainty)."""
    zk, wk = pz.posterior(feat)
    out = np.full(len(zk), np.nan)
    for i in range(len(zk)):
        w = wk[i]; ok = np.isfinite(w) & (w > 0)
        if ok.any():
            p = w[ok] / w[ok].sum(); z = zk[i][ok]
            m = (p * z).sum(); out[i] = np.sqrt((p * (z - m) ** 2).sum())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="data/boss/cmass_targets_South.fits")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--out", default="output/audit_photoz_failures.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    fg, zg = feat[good], z[good]
    pz = PhotoZKNN(k=args.k).fit(fg, zg)

    tg = load_cmass_targets(cat, path=args.targets, seed=0)
    ft = photoz_features(tg.colors, tg.mags)
    kind = np.asarray(tg.miss_kind)
    finite = np.isfinite(ft).all(axis=1)
    masks = {"collided": (kind == "collided") & finite, "zfail": (kind == "zfail") & finite}

    print(f"successes={good.sum():,}  collided={masks['collided'].sum():,}  zfail={masks['zfail'].sum():,}")

    # (1) distribution shift per feature
    print("\n(1) feature-distribution shift (KS statistic vs successes; 0=identical):")
    for j, nm in enumerate(FEAT_NAMES):
        row = f"  {nm:6s}"
        for key, m in masks.items():
            D = ks_2samp(fg[:, j], ft[m, j]).statistic
            row += f"  {key}: D={D:.3f}"
        print(row)

    # (2) colour-space extrapolation in the photo-z's whitened metric
    # whiten by the training std (PhotoZKNN whitens internally; replicate here)
    mu, sd = fg.mean(0), fg.std(0) + 1e-12
    tree = cKDTree((fg - mu) / sd)
    d_self, _ = tree.query((fg[np.random.default_rng(0).choice(len(fg), min(20000, len(fg)), False)] - mu) / sd, k=args.k + 1)
    thresh = np.percentile(d_self[:, args.k], 99)             # 99th pct of success k-NN dist
    print(f"\n(2) colour-space extrapolation (k={args.k}-th NN beyond 99th-pct success dist={thresh:.2f}):")
    extrap = {}
    for key, m in masks.items():
        dk, _ = tree.query((ft[m] - mu) / sd, k=args.k)
        frac = float(np.mean(dk[:, -1] > thresh))
        extrap[key] = (dk[:, -1], frac)
        print(f"  {key}: {100*frac:.1f}% of objects are colour-space extrapolations")

    # (3) posterior width: held-out successes vs failures
    rng = np.random.default_rng(1); test = rng.random(len(fg)) < 0.2
    pz_cal = PhotoZKNN(k=args.k).fit(fg[~test], zg[~test])
    w_succ = _post_width(pz_cal, fg[test])
    w_fail = {key: _post_width(pz, ft[m]) for key, m in masks.items()}
    print("\n(3) photo-z posterior width sigma_z (median):")
    print(f"  successes (held-out): {np.nanmedian(w_succ):.4f}")
    for key, w in w_fail.items():
        print(f"  {key}: {np.nanmedian(w):.4f}")

    # (4) recovered n(z) shift + z_host fallback fraction
    print("\n(4) assigned-redshift n(z) shift + degenerate fallback:")
    from twopt_density.observed_ls import complete_catalog_photoz, measure_close_pair_dz
    dz = measure_close_pair_dz(cat, 62/3600.)
    zk, wk = pz.posterior(ft)
    for key, m in masks.items():
        # mean assigned z from the posterior median vs success mean
        zmed = np.array([np.nansum(wk[i][np.isfinite(wk[i])] * zk[i][np.isfinite(wk[i])]) /
                         max(np.nansum(wk[i][np.isfinite(wk[i])]), 1e-9) for i in np.where(m)[0]])
        print(f"  {key}: <z_assigned>={np.nanmean(zmed):.4f} vs <z_success>={zg.mean():.4f} "
              f"(Δ={np.nanmean(zmed)-zg.mean():+.4f})")
    c = complete_catalog_photoz(cat, tg, pz, seed=0, dz_pool=dz, verbose=True)
    nfb = int(np.sum(np.asarray(c["prov"]) == 4))   # PROV['zhost']
    print(f"  z_host fallback (degenerate posterior): {nfb} of {tg.N:,} targets ({100*nfb/tg.N:.2f}%)")

    # figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, (ax, nm) in enumerate(zip(axes.flat[:4], FEAT_NAMES)):
        lo, hi = np.percentile(fg[:, j], [1, 99])
        ax.hist(fg[:, j], 60, (lo, hi), density=True, histtype="step", color="#888", lw=2, label="successes")
        for key, m, col in [("collided", masks["collided"], "#3a6ea8"), ("zfail", masks["zfail"], "#7b3ff2")]:
            ax.hist(ft[m, j], 60, (lo, hi), density=True, histtype="step", color=col, lw=2, label=key)
        ax.set_xlabel(nm); ax.set_yticks([]); ax.legend(fontsize=7)
    a = axes.flat[4]
    a.hist(d_self[:, args.k], 60, density=True, histtype="step", color="#888", lw=2, label="success→success")
    for key, (dk, fr) in extrap.items():
        a.hist(dk, 60, density=True, histtype="step", lw=2, label=f"{key}→success ({100*fr:.0f}% extrap)")
    a.axvline(thresh, color="r", ls="--"); a.set_xlabel(f"k={args.k}-NN colour distance (whitened)")
    a.set_yticks([]); a.legend(fontsize=7); a.set_title("colour-space coverage")
    a = axes.flat[5]
    a.hist(w_succ[np.isfinite(w_succ)], 50, (0, 0.08), density=True, histtype="step", color="#888", lw=2, label="successes")
    for key, w in w_fail.items():
        a.hist(w[np.isfinite(w)], 50, (0, 0.08), density=True, histtype="step", lw=2, label=key)
    a.set_xlabel("photo-z posterior width σ_z"); a.set_yticks([]); a.legend(fontsize=7)
    a.set_title("posterior uncertainty")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
