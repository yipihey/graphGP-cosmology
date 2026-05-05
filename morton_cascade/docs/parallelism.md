# Parallelism audit

This document audits the use (and non-use) of multi-threading across
`morton_cascade`. Goals: identify race-condition risks, surface
parallelization opportunities, document determinism guarantees, and
note gaps in test coverage.

Last audit: this session, after commit "pooled-aggregate gradients for
higher central moments." Library at 257 lib + 29 integration + 1
doctest tests, all passing.

## 1. Where rayon is used today

### `hier_par.rs` — legacy 2D-threaded cascade

The only file that uses rayon. Provides `bin_to_fine_grid_par` and
`cascade_hierarchical_par`. Self-contained: not called from anywhere
else in the crate, not exercised by any test, not used by any binary
or example.

**Status**: effectively dead code. Mentioned in the lib.rs module
docstring as one of several historical 2D paths. Should be either
re-tested-and-validated or marked deprecated.

**Race-condition analysis** (in case it gets revived):

- `bin_to_fine_grid_par` builds per-thread private grids via
  `par_chunks(...).map(...).collect()` — no shared writes. The reduce
  step uses `par_chunks_mut` over output rows, with each row
  exclusively owned by one task; reads from `partials[..]` are
  immutable. **Safe.**

- `cascade_hierarchical_par` per-level walk uses `par_chunks_mut(m)`
  over `nxt` rows, each row exclusively owned. Per-row partials are
  collected into `Vec<...>` and reduced serially. **Safe.**

**Determinism**: integer reductions (`u128` sums) are associative
regardless of thread order. The single `f64` accumulator
(`sum_sib_var`) is per-row (single-threaded within a row), then
reduced via a serial sweep over the rayon-collected partial vector
(deterministic order). **Bit-exact reproducible across thread counts.**

**Test gap**: zero tests. Should at minimum have an equivalence test
against `hier::cascade_hierarchical_bc` (the serial reference). If we
keep this code, it needs that test before any future change can be
trusted. If we don't, it should go.

### Nothing else

No other file uses rayon. All gradient code, the multi-run aggregator,
the cascade builders for the bit-vector-pair path, the field-stats /
anisotropy / ξ analyses — all single-threaded.

## 2. Where rayon should be used (high-value gaps)

### Per-run loops in `multi_run/mod.rs` — the single biggest win

`CascadeRunner::per_run()` and six downstream loops iterate over
independent runs:

```
for r in &runs {  // runs := self.per_run()
    let stats = r.cascade.analyze_field_stats(cfg);
    /* combine into accumulator */
}
```

Each iteration:
- builds a fresh cascade (the dominant cost, ~20-100ms per run for
  N≈10^4-10^5)
- runs an analysis pass (10-30ms typical)
- in gradient methods: also runs a per-run gradient (additional
  20-50ms typical)

Sites:
- `per_run()` — the prerequisite cascade-build loop, used by all
  others
- `gradient_var_delta_data_run_average`
- `pooled_raw_sums_data` (used by all 4 pooled moment gradient methods)
- `analyze_field_stats`
- `analyze_anisotropy`
- `analyze_cic_pmf`

Plus the per-run cascade build inside each (currently inside the loop;
the analyses re-call `per_run()` internally).

Runs are fully independent: no shared mutable state, no
inter-run dependencies. `BitVecCascadePair`, `TrimmedPoints`, and
`AppliedRun` are all `Send + Sync` (auto-derived; no interior
mutability anywhere in the relevant types). `CascadeRunner::base_*`
fields are immutable shared input.

**Expected speedup**: near-linear up to N_runs threads. A 10-run plan
on an 8-core machine should drop from ~10× single-run time to ~2×
single-run time. This is large.

**Implementation**: replace each `for r in &runs { ... acc += ... }`
with `runs.par_iter().map(|r| ...).collect()` followed by serial
aggregation. The aggregation is small (vector of pooled bin entries
or similar) so the serial finish is negligible.

The post-loop pooling logic (sort by physical_side, sweep adjacent
within tolerance) is also serial, which is fine — it operates on a
small vector (one entry per run-level, typically <100 entries).

### Per-level gradient walks — modest

In `field_stats_gradient.rs` and friends:

```
for l in 0..n_levels {  // typically 7-15
    grads.push(per_level_gradient(...));
}
```

Each level reads the membership index immutably and computes a length-N
output vector. Could be parallelized via `par_iter`, but the level
count caps speedup at ~10×. Less attractive than per-run parallelism
which can have 50+ runs.

**Recommendation**: skip unless profiling shows this is the hot path
post-multi-run-parallelization.

### Membership index build

`CellMembership::build` does a Morton-code computation per particle
plus a sort. The Morton computation is per-particle parallel; the sort
isn't trivially. Could use `par_sort_unstable_by_key`. Per-particle
work is cheap so the speedup is modest unless N >> 10^6.

**Recommendation**: defer until profiling shows this matters.

## 3. Determinism implications

The current single-threaded path is **bit-exact deterministic** —
identical inputs produce identical f64 outputs across runs and
machines. This is a real property worth preserving.

When per-run parallelism is added, determinism depends on what we do
with the per-run results:

- **Vector-of-results pattern** (e.g., `runs.par_iter().map().collect()`
  produces a `Vec<RunResult>` in input order regardless of thread
  scheduling). Subsequent serial aggregation over this ordered vector
  is deterministic. **This is what we should use.**

- **Direct parallel reduce** (e.g., `par_iter().sum()` for f64) is
  NOT deterministic — rayon's reduction tree depends on work
  distribution.

The compensated-summation infrastructure (`compensated_sum.rs`) is
relevant here but solves a different problem (precision of long
sequences), not parallelism determinism.

**Policy**: prefer collect-then-serial-reduce over direct parallel
reductions for any f64 accumulator. Document this in the
implementation notes for new threaded code.

## 4. Test gaps to fix

1. **No `hier_par.rs` tests at all.** Either add equivalence tests
   against the serial path or deprecate.

2. **No "thread-count equivalence" tests.** When we parallelize the
   per-run loops, we should have at least one test that runs the same
   workload with `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=4` and
   asserts bit-exact equality. The collect-then-serial-reduce pattern
   should make this hold.

3. **No micro-benchmarks for multi-run paths.** Have one example
   (`profile_field_stats_gradient.rs`) for single-cascade gradients;
   nothing for multi-run. Should add `profile_multi_run.rs` to
   establish baseline before/after parallelization.

## 5. Python side

22 Python files in the repo. None use multiprocessing, joblib, dask,
or any explicit parallelism. They are:

- single-threaded benchmark drivers (`compare_corrfunc_multi.py`,
  `headline_comparison.py`, etc.)
- plotting / analysis scripts (`pmf_cox_plot.py`, `sigma2_2d.py`)
- ξ-fit examples (`xi_graphgp_fit.py` — this one uses JAX, which
  internally vectorizes but doesn't multi-thread by default)

**Comparison fairness caveat for benchmarks**: the headline cascade
performance numbers in the README compare against Corrfunc, which
internally uses OpenMP. The cascade numbers reported are
single-threaded. This isn't dishonest — the cascade is still faster
above a threshold — but it understates how much faster a parallel
cascade would be, and it should be noted explicitly when the
multi-run parallelization lands.

If any Python script genuinely benefits from parallelism (running many
analyses on a sweep of catalogs, for instance), use `multiprocessing`
or `joblib`. Don't over-engineer with dask unless there's data that
actually doesn't fit.

## 6. Recommendations & priorities

**Do now (this sprint)**:
1. Parallelize the six `for r in &runs` loops in `multi_run/mod.rs`
   via `par_iter()` plus collect-then-serial-aggregate.
2. Parallelize `CascadeRunner::per_run()` itself — the prerequisite
   that all the others rebuild internally. (Or factor the per-run-build
   logic so each parallelized analysis builds its own cascade in
   parallel without going through the shared `per_run` path.)
3. Add a thread-count-equivalence test that runs `analyze_field_stats`
   with `RAYON_NUM_THREADS=1` and a multi-thread setting, asserts
   bit-exact equality.
4. Add `profile_multi_run.rs` to measure speedup.

**Do soon**:
5. Decide on `hier_par.rs`: either add a minimum equivalence test or
   mark deprecated and remove from the public API.
6. Update README to note multi-run parallelism (if it lands) and the
   Corrfunc-vs-cascade comparison caveat.

**Defer**:
7. Per-level gradient parallelism (modest payoff).
8. Membership-index parallel sort (only matters at N > 10^6).
9. GPU port (large investment, separate question).
