# Completion catalogue data model (BOSS CMASS-South)

Systematics-corrected, **equal-weight, cosmology-free** galaxy catalogues: the
galaxies that would have been observed if fiber collisions, redshift failures and
imaging systematics were negligible. Coordinates are observed **RA, Dec, z** — no
fiducial cosmology is baked in (apply your own for 3-D distances). Written by
`demos/write_release_catalogs.py` via `twopt_density.catalog_io.write_release`.

## Products

```
release/
  ensemble/realization_000.fits ... realization_NNN.fits   # the product: sample these
  summary.fits        # convenience: one catalogue with per-object z + uncertainty
  randoms.fits        # matching randoms (sel_map x n(z))
  PROVENANCE.txt      # version + per-class counts
```

**Ensemble** — each file is a complete equal-weight catalogue drawn from the
posterior over the unobserved galaxies. *Use the ensemble for any analysis*: run
your statistic on each realization and take the mean as the estimate and the
across-realization spread as the completion uncertainty (add it in quadrature
with your sample/cosmic-variance error). Validated (see report): the ensemble
recovers truth for w(θ), wp(rp), ξ(s,μ), n(z) and counts-in-cells to ~1–2% on
realistic mocks, and is calibrated (coverage ≈ nominal).

**Summary** — every observed galaxy + every missing target at its fixed position,
with the best-estimate redshift and its uncertainty. The stochastic
WEIGHT_SYSTOT *additions* are NOT in the summary (they are in the ensemble);
instead `WEIGHT_SYSTOT` is provided as a column for users who prefer to weight.

## Columns

### ensemble/realization_*.fits  (HDU `CATALOG`)
| column | type | unit | meaning |
|---|---|---|---|
| RA | float32 | deg | right ascension (observed) |
| DEC | float32 | deg | declination (observed) |
| Z | float32 | — | redshift (spectroscopic for observed; sampled for restored galaxies) |
| PROV | int8 | — | provenance (see codes below) |

### summary.fits  (HDU `SUMMARY`)
| column | type | unit | meaning |
|---|---|---|---|
| RA, DEC | float32 | deg | position |
| Z | float32 | — | best-estimate redshift (spec for observed; ensemble-mean for restored) |
| Z_ERR | float32 | — | redshift uncertainty (0 for observed; ensemble std for restored) |
| PROV | int8 | — | provenance |
| WEIGHT_SYSTOT | float32 | — | imaging-systematic weight (alternative to the ensemble's systot additions) |

### PROV provenance codes
| code | name | meaning |
|---|---|---|
| 0 | observed | observed spectroscopic galaxy (Z is its spec-z) |
| 1 | collided | fiber-collision target restored at its real imaging position; Z = host z + close-pair Δz |
| 2 | zfail | redshift-failure target restored at its real position; Z sampled from the local LOS density × photo-z |
| 3 | systot | WEIGHT_SYSTOT-implied galaxy restored as a local analog (ensemble only) |
| 4 | zhost | redshift fell back to the nearest-neighbour host (degenerate photo-z; rare) |
| 5 | inpaint | mask-hole inpaint transplant (optional hole-free product; off by default) |

## What is and isn't corrected (scope)
- **Corrected:** fiber collisions (w_cp), redshift failures (w_noz), imaging
  systematics (w_systot, density modulation), interior mask holes (optional
  inpaint product).
- **Cosmology-free:** the catalogues carry no fiducial cosmology; redshift-space
  statistics require the analyst's choice of cosmology at measurement time
  (standard practice).
- **Limitations:** redshift failures are a ~1% colour-space extrapolation of the
  success-trained photo-z (small +n(z) bias, documented); sub-arcmin bright-star
  masks below the random-map resolution are not inpainted; the realization spread
  is the completion uncertainty, not cosmic variance.

## Reproducibility
Fixed seeds; environment in `environment.yml`. Regenerate with
`demos/write_release_catalogs.py`. Validation: `demos/mock_truth_recovery.py`
(truth recovery) and `demos/recovery_calibration.py` (coverage).
