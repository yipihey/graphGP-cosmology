# Handoff: cosmology-free observed-space LGCP — drive w(θ) to percent level

**Goal for the next session.** The observed-space (n̂, z) anisotropic LGCP now
reproduces the BOSS CMASS-SGC angular two-point function w(θ) to ~30–45% across
0.06°–2°. Narrow this to **percent level**. *Only then* contact Risa Wechsler
(rwechsler@stanford.edu) and Susan Clark (seclark1@stanford.edu) — a drafted
email is in the session history; do not send until w(θ) is percent-level.

Hard constraints (from Tom): everything in **observed coordinates** — angular
separation Δθ and redshift z. **No fiducial cosmology**, no comoving distances,
no velocity model. Only measured correlations enter, so downstream tasks infer
cosmology unbiased. Catalogs come out as (RA, Dec, z).

---

## Environment / how to run

- Python: `~/.venv/k3d/bin/python3`. GPU: NVIDIA RTX A6000 (46 GB). RAM: 2 TB.
- **graphGP fork** must be on PYTHONPATH (chunked generation + anisotropic
  kernel live only there): `/home/tabel/Projects/graphgp`, branch
  `hierarchical-chunked-generation`. Not yet pushed to a remote fork.
- **Corrfunc** built from source at `~/codes/Corrfunc` (theory.DD + mocks.
  DDtheta_mocks), importable via a `.pth` in the venv. Built with gcc-toolset-11
  and Anaconda's 3.11.5 headers; `_countpairs*.so` symlinked into the package.
- Always set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

One-command reproduction (prints w(θ)/data table + saves a ratio plot):

```bash
PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=16 \
~/.venv/k3d/bin/python3 demos/validate_observed_wtheta.py
```

Runtime ≈ 2–3 min (graph build on ~2.2M candidates + a few field draws + the
Corrfunc passes).

---

## Where the code is

graphGP fork (`~/Projects/graphgp`, branch `hierarchical-chunked-generation`):
- `graphgp/refine.py` — `generate`/`refine` gained `chunk_size` (chunks the
  per-point Vecchia factorization via `lax.map`; byte-identical; bounds GPU
  memory). `compute_cov_matrix` dispatches to the anisotropic path.
- `graphgp/aniso.py` — `AnisotropicCovariance` (2-D bilinear K(Δθ,Δz)),
  `embed_points`, `build_anisotropic_covariance`. (Currently the cosmology
  pipeline uses the *embedded isotropic Matérn* route, not this 2-D object —
  see "what's used" below.)
- `tests/` — 15 tests green (`test_aniso.py`, chunked test in `test_refine.py`).
- `demo_chunked.py`, `demo_anisotropic.py`.

Cosmology repo (`~/Projects/graphGP-cosmology`, `main`, pushed; HEAD `734280a`):
- `twopt_density/observed.py` — the pipeline:
  - `measure_xi_theta_z(catalog, ...)` → ξ(Δθ, Δz), Landy-Szalay, **no cosmology**
    (one 4-D `query_pairs` over tagged data∪window-randoms → DD/DR/RR).
  - `build_observed_kernel(te, ze, xi)` → `(cov, alpha)`. Fits a **sum-of-two-
    Matérns** to the angular profile ln(1+ξ(Δθ,0)); anisotropy via `alpha`
    (embed `(n̂, α·z)`). Returns a standard 1-D `(cov_bins, cov_vals)` kernel.
  - `sample_catalogs_lgcp_observed(catalog, ...)` → list of (ra,dec,z) catalogs.
    LGCP: log-normal intensity `exp(f − σ²/2)`, inhomogeneous-Poisson over
    window candidates, generated with the chunked fork (`chunk_size=50000`).
- `demos/validate_observed_wtheta.py` — the reproduction harness above.
- `demos/validate_graphgp_xi_recovery.py`, `validate_cascade_vs_corrfunc.py`,
  `plot_angular_wtheta.py` (3-D LGCP version), `demo_anisotropic.py`.

## What's used vs available

The pipeline currently uses an **embedded isotropic Matérn** (`(n̂, α·z)` +
1-D Matérn) — PSD by construction, elliptical anisotropy. The fork's 2-D
`AnisotropicCovariance` is available and validated but **not** used by
`observed.py` yet (a directly-tabulated 2-D kernel is not PSD on the real
footprint; see pitfalls). A PSD-projected or richer 2-D kernel is one of the
refinement options below.

---

## Current result (HEAD)

`build_observed_kernel` = sum-of-two-Matérns; n_cand_factor=20 (~2.2 M cand):

```
 theta   w_data   w_LGCP  LGCP/data
 0.061   0.3062   0.1670     0.545
 0.090   0.2298   0.1583     0.689
 0.133   0.1827   0.1409     0.771
 0.197   0.1354   0.1203     0.889
 0.291   0.0948   0.1012     1.068
 0.430   0.0660   0.0761     1.152
 0.636   0.0446   0.0555     1.246
 0.940   0.0273   0.0382     1.397
 1.390   0.0155   0.0214     1.384
 2.056   0.0082   0.0091     1.114
```

Shape: **core (≲0.2°) ~30–45% low**, **wings (0.6–1.4°) ~25–40% high**, good
near 0.2–0.4°. multi_frac ≈ 0.10.

---

## Diagnosed residuals → refinement path to percent level

1. **Core low (θ ≲ 0.2°).** The ξ(Δθ,Δz) measurement's first Δθ bin is ~0.2°
   (n_theta=12 over 2.5°), so sub-0.2° is *extrapolated* by the Matérn.
   Fix: measure ξ(Δθ,Δz) with finer Δθ bins near 0 (log or finer-linear); push
   theta_max down and/or add bins below 0.2°. Also check the LGCP candidate
   density / tiny jitter aren't smoothing the core.
2. **Wings high + core low simultaneously = single-shape kernel can't match the
   power-law w(θ).** Options, in increasing fidelity:
   - tune the two-Matérn fit (it currently least-squares the *angular* profile;
     also fit/penalise the wings, or fit Matérn ν, or add a third component);
   - **per-scale anisotropy**: one global `alpha` assumes the same Δθ/Δz ratio
     at all scales; RSD makes it scale-dependent. Move to a genuinely 2-D PSD
     kernel — use the fork's `AnisotropicCovariance` but *project the measured
     ξ(Δθ,Δz) onto the nearest PSD kernel* (e.g. clip negative eigenvalues of
     the implied operator, or fit a low-rank PSD form) so it doesn't NaN;
   - **fit hyperparameters with graphGP's own likelihood** (`generate_inv` +
     `generate_logdet` + optax/jaxopt) instead of matching ξ — the principled
     "graphGP constructs K from the data" (Tom's framing; Risa's plan §5).
3. **Amplitude / σ².** σ² = ln(1+ξ(0,0)) from the smallest bin; sensitive to
   that bin's noise. Use a robust small-separation estimate; verify the LGCP
   number density and the log-normal `−σ²/2` mean correction.
4. **Validate in 2-D too**, not just the w(θ) projection: compare the catalog's
   ξ(Δθ,Δz) to the input (use `measure_xi_theta_z` on the catalog) so core/wing
   fixes are diagnosed in the plane, not just the LOS integral.

Target: |LGCP/data − 1| ≲ 0.01 across 0.06°–2° (and the ξ(Δθ,Δz) plane).

---

## Pitfalls already hit (don't re-discover)

- **Kernel ξ must be measured UNWEIGHTED.** BOSS `w_data` is the FKP weight
  (mean ≈ 0.25); feeding it as a clustering weight makes `ls_corrfunc` return a
  *negative* ξ(r) → negative-amplitude kernel → everything downstream breaks.
- **morton_cascade ξ ≠ radial ξ(r).** Its per-shell "ξ" is a dyadic-cell
  co-occupancy statistic. Use Corrfunc for ξ(r)/w(θ); cascade for CIC/field
  moments only. (`demos/validate_cascade_vs_corrfunc.py`.)
- **A tabulated 2-D ξ kernel is NOT PSD** on the real (clustered, curved)
  candidate distribution → a single bad Vecchia block makes graphGP set the
  **entire field to NaN** (it does `where(any(isnan), nan, values)`), which the
  sampler's nan-guard then masks into a flat/zero catalog. Use a PSD-by-
  construction kernel (Matérn family) or PSD-project the tabulated one.
- **Don't jitter the catalog off the nside pixel grid.** Candidates and the
  comparison randoms share the nside=256 within-pixel structure (so it cancels,
  w≈0); a large positional jitter on the catalog only → gross flat w(θ)≈const
  artifact. Keep jitter ≪ the smallest measured bin.
- **Kernel far-padding is irrelevant** to the field — Vecchia only uses each
  point's ~30 nearest neighbours (small separations).
- **GPU memory**: the field at ~2 M candidates needs the fork's `chunk_size`
  (50 000 works); the un-chunked path OOMs the A6000.

---

## Deferred outward items (gated on percent-level w(θ))

- Email to Risa & Susan (draft in session history — leads with the validated
  graphGP fork contributions + cosmology-free framing).
- Push the fork to a GitHub fork (`yipihey/graphgp`) and open the upstream PR
  to `stanford-ism/graphgp` for chunked generation + anisotropic kernels.
