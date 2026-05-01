# Implementation Plan: 2pt-Aware Density Estimates

Source document: `twopt_density.pdf` (Tom Abel, 2026-05-01).

This plan turns the document's three-part construction into a concrete set of
modules in this repository. The goal is a pipeline that, given a clustered
point catalog `D` and a random catalog `R`, produces per-point density
weights `{w_i}` whose pairwise statistics reproduce the Landy-Szalay (LS)
two-point correlation function exactly. Once the weights exist, the random
catalog can be discarded and downstream science runs from `{(x_i, w_i)}`
alone.

The two-point estimation backbone is **[Corrfunc]** (Sinha & Garrison 2020),
which gives O(N log N) dual-tree pair counts in serial or OpenMP/MPI. For the
basis-function (SFH) layer we rely on Corrfunc with fine binning, falling back
to **[suave]** (Storey-Fisher) where a true continuous-function estimator is
needed.

[Corrfunc]: https://corrfunc.readthedocs.io
[suave]: https://github.com/kstoreyf/suave


## High-level architecture

```
twopt_density/
    __init__.py
    ls_corrfunc.py          # Part I.0: DD/DR/RR pair counts + LS xi(r) via Corrfunc
    basis.py                # Part II.1: cubic spline / Bessel / compensated bandpass bases
    basis_projection.py     # Part II.2: project Corrfunc fine-bin counts onto a basis
    weights_binned.py       # Part I.3: binned Wiener-filter weights
    weights_basis.py        # Part II.3: basis-form Wiener-filter weights
    weights_graphgp.py      # Part III: Vecchia-approximate GP weights via graphgp
    validate.py             # weighted-DD re-pairing check (Eq. 5 of the doc)
demos/
    demo_part1_binned.py    # synthetic small-N Part I end-to-end
    demo_part2_basis.py     # SFH basis on Quijote sub-catalog
    demo_part3_graphgp.py   # full pipeline on N >= 1e6 halos
```

Existing repo code we can lean on or update:

- `graphGP_cosmo.py:1256 compute_two_point_function` — O(N^2) scipy KDTree
  pair counter. The new `twopt_density.ls_corrfunc.xi_landy_szalay` should
  become the canonical 2pt estimator; the old function is kept as a
  no-Corrfunc fallback.
- `graphGP_cosmo.py` already wires `graphgp.build_graph` and a Wiener-filter-
  style field optimizer, so Part III mostly factors out existing code into a
  reusable function rather than writing it from scratch.


## Part I — Binned LS-consistent weights

### I.0 Pair counts and LS estimator (`ls_corrfunc.py`)

For a periodic box (Quijote-style):

```python
from Corrfunc.theory.DD import DD

r_bins = np.logspace(np.log10(0.1), np.log10(200.0), 76)   # 25 bins/decade
res    = DD(autocorr=1, nthreads=NT, binfile=r_bins,
            X1=x, Y1=y, Z1=z,
            periodic=True, boxsize=L_BOX)
DD_j   = res['npairs']                                     # raw pair counts
RR_j   = N*(N-1) * shell_volumes(r_bins) / L_BOX**3        # analytic
xi_j   = DD_j / RR_j - 1.0                                 # (DR=RR analytically)
```

For a survey geometry (data + random catalogs):

```python
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks       # angular survey case
# or Corrfunc.theory.DD for cartesian survey-like inputs
DD_j = DD(1, NT, r_bins, x, y, z)['npairs']
DR_j = DD(0, NT, r_bins, x, y, z, X2=xR, Y2=yR, Z2=zR)['npairs']
RR_j = DD(1, NT, r_bins, xR, yR, zR)['npairs']

from Corrfunc.utils import convert_3d_counts_to_cf
xi_j = convert_3d_counts_to_cf(N_D, N_D, N_R, N_R,
                                DD_j, DR_j, DR_j, RR_j)
```

`xi_landy_szalay(positions, randoms=None, r_bins=..., box_size=...)` returns
`(r_centers, xi_j, RR_j, DD_j, DR_j)`. `RR_j` is what we will reuse in the
weight equations (Eq. 2 of the doc); we therefore return it explicitly.

### I.1 Local mean-density estimate

The Wiener filter (Eq. 4) needs the local mean density `nbar(x_i)` which
encodes the survey window. Two backends:

- **Periodic box**: `nbar = N_D / L_box^3`, constant.
- **Survey window**: kernel-density estimate from `R` itself, evaluated at
  each `x_i`. Use `scipy.spatial.cKDTree` with a top-hat or Epanechnikov
  kernel of width set to ~5x mean random separation.

This goes in `ls_corrfunc.local_mean_density(positions, randoms, h)`.

### I.2 Covariance assembly (`weights_binned.py`)

For `N_D <= 1e4`, build the dense `C_ij = xi_hat(r_ij)` matrix using the
already-computed `xi_j` and a piecewise-constant or linearly-interpolated
lookup. Add diagonal `N_ii = 1/nbar_i` (Poisson noise). Solve

```
delta_hat = C @ scipy.linalg.cho_solve(cho_factor(C + N), n_minus_nbar)
```

Return `w_i = 1 + delta_hat_i`.

### I.3 Validation (`validate.py`)

Recompute the **weighted** pair count

```
DD_w(r) = sum_{i<k} w_i w_k * 1[r_ik in bin]
```

with Corrfunc's `weights1=` argument (Corrfunc supports per-particle weights
natively via `weight_type='pair_product'`). Verify

```
DD_w(r) / RR(r) - 1 ~= xi_LS(r)
```

within the expected statistical noise. This is exactly Eq. (5) of the doc and
the criterion that decides whether the weight construction is valid.


## Part II — SFH basis-function continuous estimator

### II.1 Bases (`basis.py`)

Three implementations with a common `Basis` interface
`(evaluate(r), inner_product(other), n_basis, support)`:

1. **Cubic B-splines** on a log-r knot vector. Default basis. Smooth,
   locally supported, well-conditioned.
2. **Spherical Bessel modes** `f_alpha(r) = j_0(k_alpha r)`. Each basis
   coefficient `theta_alpha` is a direct estimate of `P(k_alpha)` via the
   Hankel transform. Required for the FFT-free real+Fourier unification
   discussed in Sec. 3.4 of the doc.
3. **Compensated bandpass kernels** `f_alpha(r)` supported on
   `[r_min_alpha, r_max_alpha]` with `integral f(r) r^2 dr = 0`. Dyadic
   cascade so that `theta_l == sibling_xi_l` of the variance cascade
   `sigma^2(R_l) = (1/4) sigma^2(R_{l-1}) + (3/4) xi_bar_{l-1}`.

### II.2 Basis-projected pair counts (`basis_projection.py`)

True SFH requires `DD_alpha = sum_{i<k} f_alpha(r_ik)`, evaluated on each
pair. Two implementation paths, ordered by complexity:

1. **Corrfunc fine-bin projection (default)**. Run Corrfunc once with a
   *fine* bin grid (e.g., 500 log-bins from 0.05 to 250 Mpc/h), then

   ```
   DD_alpha = sum_j  f_alpha(r_j_center) * DD_j
   ```

   This is exact in the limit `dr -> 0`; for the bin widths above the
   integration error is well below sample variance.

2. **suave native (optional)**. The `suave` package implements the
   continuous estimator directly. We will support it as an optional backend
   when `suave` is importable.

`project_pair_counts(corrfunc_counts, basis)` returns `(DD_alpha, DR_alpha,
RR_alpha)` and the normalization `M_alpha = sum_{i<k} f_alpha(r_ik)`
(computable from the data alone, needed for Eq. 8).

The basis coefficients are recovered by least-squares (Eq. 6):
`theta_hat = (F^T F)^-1 F^T y` where `F_{j,alpha} = f_alpha(r_j)` and
`y_j = (DD_j - 2 DR_j + RR_j) / RR_j`.

### II.3 Basis-form Wiener filter (`weights_basis.py`)

Identical to I.2 except `C_ij = sum_alpha theta_hat_alpha f_alpha(r_ij)` is a
smooth function of separation — no bin artifacts. The same `cho_factor`
solver works up to `N_D ~ 1e4` and conjugate-gradient with sparse `C` works
up to `N_D ~ 1e6`.


## Part III — graphgp / Vecchia GP weights

### III.1 Tabulate the kernel

`tabulate_kernel(theta_hat, basis, r_grid)` evaluates the SFH `xi_hat(r)` on
a 1000-point logarithmic r-grid spanning 0.1 to 200 Mpc/h. Adds an optional
diagonal jitter to the implicit C(0) value to handle finite-precision
near-coincidence pairs. This is exactly the format expected by graphgp
(`covariance = (r_grid, xi_grid)`).

### III.2 Build Vecchia graph

```python
import graphgp as gp
graph = gp.build_graph(points, n0=100, k=30)
```

`n0` controls the dense initial block size; `k` controls neighbors per
conditional. Already exposed in this repo via `K_NEIGHBORS=15` and `N0=100`
in `graphGP_cosmo.py`. We can keep those defaults and add a comment that
larger `k` is appropriate when `xi(r)` has long-range structure.

### III.3 Apply inverse Cholesky factor

```python
delta_data = (n - nbar)/nbar         # at the data points
weights    = gp.apply_inverse_cholesky(graph, covariance, delta_data)
```

Cost: `O(N_D * k)` time and memory. Linear scaling to `N_D ~ 1e9`.

### III.4 Drop-in into existing pipeline

The current `graphGP_cosmo.optimize_field` already does an iterative
gradient-based GP MAP. For the 2pt-aware construction we need just the
posterior mean given a *fixed* kernel computed from data. The new
`weights_graphgp.compute_2pt_weights(positions, randoms, basis, **kw)` is
therefore a thin wrapper that: runs Part II → tabulates kernel → calls
graphgp → returns weights. It will live alongside the existing optimizer and
can be selected via a flag in `graphGP_cosmo`.


## Validation strategy

For each of Parts I–III, a separate demo script in `demos/` runs:

1. Generate or load catalog (`synthetic_test`-style GP + Poisson, or
   Quijote halos).
2. Compute LS `xi_hat(r)` via Corrfunc.
3. Solve for `{w_i}`.
4. Recompute weighted-DD pair sum on `{(x_i, w_i)}` only.
5. Plot `xi_recovered(r)` vs `xi_hat(r)`; the two should agree within
   the statistical noise floor (`~ (1+xi)/sqrt(DD_j)`).
6. (Part III only) Plot the GP posterior `delta(x*)` on a slice through
   the box and check it is smooth, peaked at the data clusters, and in the
   right amplitude range.

Acceptance criteria, per part:

| Part | Catalog                | Acceptance criterion                                   |
| ---- | ---------------------- | ------------------------------------------------------ |
| I    | synthetic, N_D = 1e4   | `|xi_w/xi_LS - 1| < 0.1` over the full bin range       |
| II   | synthetic, N_D = 1e5   | same, with ≤15 basis fns vs ≥75 bins                   |
| III  | Quijote, N_D = 5e3 → 1e6 | same, plus runtime grows < linear in `N_D`           |


## Dependencies

Add to `requirements.txt`:

```
Corrfunc>=2.5
graphgp                  # already used by graphGP_cosmo.py, not yet listed
jax                      # already used, not yet listed
optax                    # already used, not yet listed
suave                    # optional, gated behind try/except in basis_projection.py
```


## Roadmap (suggested ordering)

1. **Land Corrfunc-based LS** (`ls_corrfunc.py` + `validate.py` weighted-DD).
   Replaces the slow `compute_two_point_function` and unblocks everything.
2. **Part I weights** with dense Cholesky on the synthetic test in
   `synthetic_test.py`. This validates the entire weight-assignment idea.
3. **Cubic-spline basis + projection** in `basis.py`,
   `basis_projection.py`. Demonstrate <10 basis fns reproducing the binned
   `xi(r)`.
4. **Basis-form weights** with the same synthetic catalog. Show no bin
   artifacts.
5. **graphgp wrapper** in `weights_graphgp.py`. Apply to the 5000-halo
   Quijote sample first, then scale up.
6. **Bessel basis + Hankel transform** for the real-and-Fourier unification
   showcase.
7. **Compensated bandpass cascade** as a stretch goal — connects the
   variance cascade to the Vecchia multiscale graph (Sec. 4.6 of doc).

Each step is independently shippable: every artifact (`xi(r)`, weights,
validation plot) is the same shape across stages, so downstream tools (the
`app.py` Streamlit visualizer in particular) keep working unchanged.
