# Morton-cascade — session state (current)

User: Tom Abel (KIPAC/Stanford). T³FT cosmological pipeline work.

Library: `/home/claude/work/morton_cascade/`
Outputs go to: `/mnt/user-data/outputs/`
Build: `. $HOME/.cargo/env && cargo build --release` in the library dir.

## Resumption recipe

```bash
cd /home/claude/work/morton_cascade && . $HOME/.cargo/env && cargo test --release 2>&1 | grep "test result"
```

Expected: 4 lines, all "ok":
- `lib`: 52 passed
- `morton-cascade` bin unit-tests: 0 passed
- `integration`: 22 passed
- doctests: 1 passed

Total **75 tests, all green**.

If anything is failing, stop and re-read this file before assuming the failure is meaningful — the previous SESSION_STATE.md was significantly out of date relative to the actual code.

## What has been done (cumulative, as of this session)

### Architecture, settled
- **Count-cascade** (`hier.rs`, `hier_3d.rs`, `hier_nd.rs`, `hier_packed.rs`, `hier_par.rs`, `all_shifts.rs`): dense buffers, stable, untouched.
- **Bit-vector cascade** (`hier_bitvec.rs` + `coord_range.rs`): per-particle bit-planes, depth-first traversal with empty-cell pruning, `l_max` auto from data resolution.
- **Pair (cross-catalog) bit-vector cascade** (`hier_bitvec_pair.rs`): two catalogs sharing a single cell hierarchy via `CoordRange::analyze_pair`. Tracks DD, RR, DR per level. Optional per-point weights. Supports point-list crossover at deep levels. Produces `XiShell` via Landy-Szalay estimator.

### Phase status (corrected from previous session-state document)

The previous SESSION_STATE.md said Phases 1, 2, 3 were "queued." **All three were already implemented** in the working copy when this session resumed. What this session actually did:

1. **Phase 1 retroactive validation + tuning.** Benchmarked the point-list crossover with high-quality (splitmix64) PRNG to avoid the LCG low-bit pathology that polluted earlier numbers. Discovered:
   - The pure bit-vector cascade is fast at every N tested. The doc's "8.3 s at 100k×l_max=16" is not reproducible — current actual is 53 ms.
   - The constant default `crossover_threshold = 64` is **3.9× slower** than pure bit-vec at N=1M (2D, l_max=16) because point-list cost scales with cell count while bit-vec cost is constant per cell visit. The two cross at `count ≈ N / (64·D)`.
   - **Fix applied**: `BitVecCascade::default_crossover_threshold(n) = max(64, n/64)` and same for the pair version with `max(n_d, n_r)`. `build()` now uses this adaptive default. `build_with_threshold()` still takes an explicit value. CLI honors with sentinel `Option<usize>` (None means adaptive).
   - **Validated** at N ∈ {1k, 10k, 50k, 100k, 250k, 500k, 1M}, D ∈ {2, 3}: adaptive default is at or near pure-BV optimum at every N, with no measurable regression at small N.

2. **Phase 2 was already wired** in `hier_bitvec.rs` (PairShell, pair_counts_per_shell, pair_gradient_per_particle) and the CLI (`pairs`, `gradient` subcommands). Not touched.

3. **Phase 3 was implemented but had one failing test.** `hier_bitvec_pair::tests::ls_xi_recovers_clustering_for_clustered_data` was failing because its inline LCG PRNG used `s & mask` for k=10 bits, collapsing 3000 "random" points to only 1024 unique positions (LCG low bits cycle with period ≤ 2^k). The cascade was correctly reporting 2928 coincident pairs in the random catalog. Fixed by switching the test PRNG to high-bits extraction.

### CLI
- `xi` for Landy-Szalay correlation function (data + randoms, optional weights).
- `field-stats` (this session) for per-cell density-field statistics: moments and PDF of δ(c) = W_d/(α·W_r) − 1, W_r-weighted, with footprint cutoff `--w-r-min`. Outputs `field_moments.csv` and `field_pdf.csv`.
- `pairs` for per-shell pair counts.
- `gradient` for per-particle pair-count gradients.
- `--crossover-threshold` is `Option<usize>`; absent → adaptive default.

### `BitVecCascadePair::analyze_field_stats` (this session)
Single-traversal accumulation of W_r-weighted moments of δ at every level, plus optional log-binned PDF. Global α = ΣW_d/ΣW_r; per-cell δ = W_d/(α·W_r) − 1. Cells with W_r ≤ `w_r_min` excluded (outside footprint). 8 unit tests cover: data=randoms gives δ=0 exactly, root identity <δ>=0, var grows with depth for shot noise, w_r_min exclusion, histogram normalization, weighted catalogs, empty-randoms safety. CLI integration test round-trips through the binary.

### Outside-footprint diagnostic (session 4)
Added `n_cells_data_outside` and `sum_w_d_outside` to `DensityFieldStats`. These count cells where `n_d > 0` but `sw_r ≤ w_r_min` — data points falling in regions the random catalog says are outside the surveyed volume. Reported per level as a catalog-quality signal. Visible in CLI output as two new columns of `field_moments.csv`. Unit test verifies counter increments correctly when a contamination cluster is placed outside the random footprint.

### Survey-quality parity benchmark (session 4)
`examples/parity_benchmark.py` validates `field-stats` against direct numpy cell counting on a clustered-data + uniform-randoms-in-octant survey with intentional outside-footprint contamination. **30/30 statistical comparisons agree to 1.8e-11 relative tolerance** (machine precision). Validated metrics: `n_cells_active` (exact integer), `sum_w_r_active` (exact), `mean_delta`, `var_delta`, `n_cells_data_outside` (exact), `sum_w_d_outside` (exact). Corrfunc was the original target reference but doesn't install in this sandbox; direct numpy cell counting on a regular grid is the clean equivalent (in fact arguably better — zero statistical noise from binning). The cascade is now provably correct against an external reference.

### Paired non-dyadic P_N(δ) via sliding cube windows (session 5)
`hier_nd::cascade_pmf_windows_with_randoms` extends the existing
`cascade_with_pmf_windows` (count-cascade, no randoms) to a paired
data + randoms version with full δ machinery. Same sliding-window approach
(three sequential 1D box-filter convolutions in $D$ dimensions) on TWO grids,
then per-cell δ = W_d/(α·W_r) − 1 and W_r-weighted moments. Supports:

- Arbitrary integer cube sides via the existing `s_subshift` upsampling knob —
  not limited to dyadic scales.
- Per-point weights for both catalogs.
- Periodic and non-periodic boundaries.
- Outside-footprint diagnostic (n_windows_data_outside, sum_w_d_outside).
- Optional W_r-weighted PDF of (1+δ) in log-spaced bins.

7 new unit tests cover: data=randoms gives δ=0 exactly, full-box window
gives mean=var=0 algebraically, non-dyadic sides run correctly, s_subshift
gives finer-than-dyadic resolution, the data-outside-footprint diagnostic
works, weighted catalogs preserve identities, non-periodic mode correctly
excludes wrapping windows. One additional test (
`paired_pmf_matches_bitvec_field_stats_at_dyadic_scales`) documents the
intensive-vs-extensive distinction between sliding-window and lattice-aligned
moments.

CLI subcommand `pmf-windows-paired` exposes this with the same flag idioms
as `pmf-windows` plus the `--randoms`, `--weights-data`, `--weights-randoms`,
`--w-r-min`, `--hist-bins`, `--hist-log-min/max` flags from `field-stats`.
Outputs `paired_pmf_moments.csv` and (optionally) `paired_pmf_pdf.csv`.

### Library + doc maintenance
- Pre-existing doctest failure on the LaTeX-ish formula in `BitVecCascadePair::xi_landy_szalay` doc comment — fixed by wrapping in ` ```text ` fence.

### New unit tests added
- `hier_bitvec::tests::default_crossover_threshold_floors_at_64` — boundary cases of adaptive default.
- `hier_bitvec::tests::build_uses_adaptive_default` — confirms `build()` actually picks up the adaptive value.
- `hier_bitvec_pair::tests::pair_default_crossover_uses_max_of_n_d_n_r` — confirms pair version scales by max catalog size.

### Bench / diagnostic examples
All under `examples/`, all gated in `Cargo.toml`:
- `bench_phase1.rs` — full sweep of crossover thresholds at the regimes the old SESSION_STATE flagged as slow. Demonstrates correctness invariance (cells_visited identical across thresholds) and the timing curve.
- `bench_threshold_scan.rs` — timing scan across N for thresh=64 vs pure-BV. Shows the 3.94× regression at N=1M.
- `bench_threshold_sweep.rs` — fine-grained threshold sweep at fixed N to find optima.
- `bench_default_vs_old.rs` — direct old-default-vs-new-default comparison. Headline numbers: 4.47× speedup at 2D N=1M, 2.70× at 3D N=500k.
- `xi_smoke.rs` — end-to-end CLI smoke test (writes binaries, shells out, reads CSV, asserts null).
- `diag_ls_xi.rs`, `diag_rr_only.rs`, `diag_tiny.rs`, `diag_dups.rs`, `diag_dups_good.rs` — PRNG / cascade diagnostics from the LCG-low-bits investigation. Useful as canaries against the same class of bug recurring.

## What's still open

From the original scope:
- **Survey masks/footprints** as first-class plumbing. The cascade already handles arbitrary survey *implicitly* via the random catalog (the standard Landy-Szalay approach), but there is no explicit mask file format and no edge-correction weighting. Design discussion needed before implementing.
- **`pair-counts-cross` CLI** — redundant with `xi` (which already emits raw DD/RR/DR per shell). Skip unless someone needs it standalone.
- **σ²(R) and P_N(V) via the bit-vector cascade**. Currently those statistics still go through the count-cascade `cascade_with_pmf_windows`. Possible to redirect through `hier_bitvec`, but the count-cascade is fine for moderate l_max and the bit-vec advantage is at deep l_max where σ²(R) interpretation gets shot-noise-dominated anyway. Not urgent.
- **RA/Dec/z natively** (Path B with Jacobian-weighted reductions). Long-term; Cartesian remains the working frame.

## Files NOT in the upload tar that the working copy depends on

Important: the original `morton_cascade_lean.tar` Tom uploaded **did not contain**:
- `src/hier_bitvec_pair.rs` (the pair cascade itself).
- `src/coord_range.rs` additions (`analyze_pair`, `from_points_with_range`).
- `src/lib.rs` line 69 (`pub mod hier_bitvec_pair;`).
- The `pairs`, `gradient`, and (now) `xi` subcommands in the CLI.

The working copy in `/home/claude/work/morton_cascade/` has all of these because something in the container persisted them from a prior session. **Before applying any of the patches in `/mnt/user-data/outputs/` to your local tree, confirm those three things exist in your local copy first.**
