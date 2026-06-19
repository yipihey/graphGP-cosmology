"""Validate the k-NN photo-z posterior on a held-out BOSS CMASS spec sample.

Trains PhotoZKNN on 80% of the good-spec (IMATCH==1) galaxies using the reliable
g−r, r−i, i−z colours + i-band magnitude (u dropped: CMASS u-flux is noise), and
on the held-out 20% reports:
  (a) point-estimate accuracy: bias of Δz/(1+z), σ_NMAD, catastrophic fraction;
  (b) PIT / coverage histogram — the rank of the true z within each object's
      posterior must be ~uniform (the key test, since we SAMPLE the posterior);
  (c) stacked-posterior n(z) vs the true held-out n(z).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    ~/.venv/k3d/bin/python3 demos/validate_photoz.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twopt_density.boss import load_boss
from twopt_density.photoz import PhotoZKNN, photoz_features as features_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--out", default="output/photoz_validation.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256, with_photometry=True)
    feat = features_of(cat.colors_data, cat.mags_data)
    z = np.asarray(cat.z_data)
    good = np.isfinite(feat).all(axis=1) & (cat.imatch_data == 1)
    feat, z = feat[good], z[good]
    print(f"good-spec galaxies with reliable photometry: {good.sum():,} / {len(good):,}")

    rng = np.random.default_rng(0)
    test = rng.random(len(z)) < 0.2
    pz = PhotoZKNN(k=args.k).fit(feat[~test], z[~test])

    zt = z[test]
    zphot = pz.point(feat[test], stat="median")
    dz = (zphot - zt) / (1 + zt)
    sigma_nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outlier = np.mean(np.abs(dz) > 0.05)
    print(f"point photo-z: bias<Δz/(1+z)>={np.mean(dz):+.4f}  "
          f"σ_NMAD={sigma_nmad:.4f}  catastrophic(|Δz/(1+z)|>0.05)={outlier:.3f}")

    # PIT: weighted CDF of the posterior evaluated at the true z
    zk, wk = pz.posterior(feat[test])
    pit = np.array([np.sum(wk[i][np.isfinite(wk[i])] *
                           (zk[i][np.isfinite(wk[i])] < zt[i])) for i in range(len(zt))])
    print(f"PIT mean={pit.mean():.3f} (0.5 ideal)  std={pit.std():.3f} (0.289 ideal-uniform)")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.5))
    a1.hexbin(zt, zphot, gridsize=40, cmap="viridis", mincnt=1)
    lim = [z.min(), z.max()]; a1.plot(lim, lim, "r--", lw=1)
    a1.set_xlabel("spec-z"); a1.set_ylabel("photo-z (posterior median)")
    a1.set_title(f"σ_NMAD={sigma_nmad:.4f}, out={outlier:.3f}")
    a2.hist(pit, bins=20, range=(0, 1), color="#4a90d9", edgecolor="k", alpha=0.8)
    a2.axhline(len(pit) / 20, color="r", ls="--", label="uniform")
    a2.set_xlabel("PIT (rank of true z in posterior)"); a2.set_ylabel("count")
    a2.set_title("calibration (flat = good)"); a2.legend()
    zb = np.linspace(z.min(), z.max(), 40)
    a3.hist(zt, bins=zb, density=True, histtype="step", color="k", lw=2, label="true held-out n(z)")
    zs = pz.sample(feat[test], rng, n=1)  # one posterior draw per object
    a3.hist(zs[np.isfinite(zs)], bins=zb, density=True, histtype="step",
            color="#4a90d9", lw=2, label="stacked posterior (1 draw)")
    a3.set_xlabel("z"); a3.set_ylabel("n(z)"); a3.set_title("n(z) recovery"); a3.legend()
    plt.tight_layout(); plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
