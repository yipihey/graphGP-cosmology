# Differentiable cascade: math derivations

This document derives the per-particle gradients of every cascade
statistic with respect to the data-catalog weights $\{w_i^d\}$. The
formulas are encoded in `field_stats_gradient.rs`,
`anisotropy_gradient.rs`, and `xi_gradient.rs`; this file exists so
that the derivations are inspectable in one place rather than
scattered across module docstrings.

Notation:
- $i$ ranges over data particles, $j$ over random particles.
- $w_i^d, w_j^r$ are per-particle weights (default 1).
- $W_d(c) = \sum_{i \in c} w_i^d$, $W_r(c) = \sum_{j \in c} w_j^r$ are
  the per-cell weight sums for a cell $c$ at some level.
- $\alpha = (\sum_i w_i^d) / (\sum_j w_j^r)$ — the global mean-density
  ratio used to normalize $\delta$.
- $\delta(c) = W_d(c)/(\alpha W_r(c)) - 1$ — per-cell density contrast.
- $C_\ell$ — the set of cells at level $\ell$.
- For a particle $i$, $c_i^{(\ell)}$ denotes the level-$\ell$ cell
  containing $i$.

Key identity used throughout:
$$\frac{\partial \alpha}{\partial w_i^d} = \frac{1}{\sum_j w_j^r}$$

and:
$$\frac{\partial W_d(c)}{\partial w_i^d} = \mathbb{1}[i \in c]$$

so for any function of $\{W_d(c)\}$ the chain rule via $W_d(c)$ collapses
to a single term: the cell containing $i$.


## 1. Field-statistics: central moments of $\delta$

### 1.1 Forward statistic

For each level $\ell$ define the per-level accumulators:

$$T^{(\ell)} = \sum_{c \in C_\ell} W_r(c)$$
$$S_k^{(\ell)} = \sum_{c \in C_\ell} W_r(c) \, \delta(c)^k \quad (k = 1, 2, 3, 4)$$

Cells with $W_r(c) \le w_{r,\min}$ are excluded (footprint cut).

The k-th raw moment of $\delta$ at level $\ell$ is $A_k = S_k/T$. The
canonical observables are the central moments:

- $m_1 = A_1 = \langle\delta\rangle_{W_r}$ (≈ 0 by α normalization)
- $\mu_2 = A_2 - A_1^2$ (variance, `var_delta`)
- $\mu_3 = A_3 - 3 A_1 A_2 + 2 A_1^3$ (third central moment, `m3_delta`)
- $\mu_4 = A_4 - 4 A_1 A_3 + 6 A_1^2 A_2 - 3 A_1^4$ (fourth, `m4_delta`)
- $S_3 = \mu_3 / \mu_2^2$ (reduced skewness, `s3_delta`)

### 1.2 Cell sensitivity

Differentiating $\delta(c)$ holding $\alpha$ fixed:
$$\frac{\partial \delta(c)}{\partial W_d(c)} = \frac{1}{\alpha W_r(c)}$$

For raw moments:
$$\frac{\partial S_k}{\partial W_d(c)} = W_r(c) \cdot k \delta(c)^{k-1} \cdot \frac{1}{\alpha W_r(c)} = \frac{k \delta(c)^{k-1}}{\alpha}$$
$$\frac{\partial T}{\partial W_d(c)} = 0$$

So $\partial A_k/\partial W_d(c) = k\delta(c)^{k-1}/(\alpha T)$.

### 1.3 Central-moment cell sensitivity (unified formula)

Substituting into the central-moment expansions and simplifying with
$\delta - A_1$ as the natural variable yields a clean unified formula
for all $k \ge 2$:

$$\boxed{\frac{\partial \mu_k}{\partial W_d(c)} = \frac{k}{\alpha T}\left[(\delta(c) - \bar\delta)^{k-1} - \mu_{k-1}\right]}$$

where $\bar\delta \equiv A_1$ and $\mu_1 \equiv 0$ by convention. The
formula reduces to the familiar variance result for $k=2$:
$\partial \mu_2/\partial W_d(c) = 2(\delta(c) - \bar\delta)/(\alpha T)$.

### 1.4 Alpha sensitivity

$$\frac{\partial \delta(c)}{\partial \alpha} = -\frac{W_d(c)}{\alpha^2 W_r(c)} = -\frac{\delta(c) + 1}{\alpha}$$

For $\mu_2$: direct expansion gives $\partial \mu_2/\partial \alpha = -2 \mu_2/\alpha$.

For all $k$: by dimensional argument (since $\delta \propto 1/\alpha$
holding $W_d, W_r$ fixed, $\mu_k \propto \alpha^{-k}$):

$$\boxed{\frac{\partial \mu_k}{\partial \alpha} = -\frac{k \mu_k}{\alpha}}$$

We verify this directly for $k=2, 3$ in code; $k=4$ follows by induction.

### 1.5 Per-particle gradient

Combining cell sensitivity (from §1.3) and α sensitivity (§1.4) via
chain rule and $\partial \alpha/\partial w_i^d = 1/\sum_j w_j^r$:

$$\boxed{\frac{\partial \mu_k}{\partial w_i^d} = \frac{k\left[(\delta(c_i^{(\ell)}) - \bar\delta)^{k-1} - \mu_{k-1}\right]}{\alpha T^{(\ell)}} \;\;-\;\; \frac{k \mu_k^{(\ell)}}{\alpha \cdot \sum_j w_j^r}}$$

The second term is constant across all particles for given $(\ell, k)$.
The first term depends on the particle's containing cell at level $\ell$.

### 1.6 Reduced skewness $S_3 = \mu_3/\mu_2^2$

By chain rule:
$$\frac{\partial S_3}{\partial w_i^d} = \frac{1}{\mu_2^2} \cdot \frac{\partial \mu_3}{\partial w_i^d} - \frac{2 \mu_3}{\mu_2^3} \cdot \frac{\partial \mu_2}{\partial w_i^d}$$

Both terms come from the formulas above.

### 1.7 Implementation

`field_stats_gradient.rs`. Public methods:

```rust
pub fn gradient_var_delta_all_levels(&self, cfg, results) -> FieldStatsGradient
pub fn gradient_m3_delta_all_levels (&self, cfg, results) -> FieldStatsGradient
pub fn gradient_m4_delta_all_levels (&self, cfg, results) -> FieldStatsGradient
pub fn gradient_s3_delta_all_levels (&self, cfg, results) -> FieldStatsGradient
```

Each has an `_aggregate(..., betas: &[f64])` variant returning the
scalar-loss gradient $\partial L/\partial w_i^d$ for $L = \sum_\ell
\beta_\ell \cdot$ (statistic at level $\ell$).

Internally a single helper `per_level_central_moment_gradient(k)`
implements the unified formula for $k=2,3,4$. The S3 gradient uses
a separate chain-rule combination of $\mu_2$ and $\mu_3$ gradients.

### 1.8 Verification

Three properties tested:

1. **Finite-difference parity.** For each particle, perturb its weight
   by $\epsilon$, recompute the moment, compare $(\mu_k(w+\epsilon e_i)
   - \mu_k(w))/\epsilon$ to the analytic gradient. Tolerance scales
   with the moment order ($\epsilon^2$ truncation amplified by $|\delta|^{k-1}$).

2. **Uniform-scaling invariance.** All central moments are invariant
   under $w_i^d \to k w_i^d$ for all $i$ (because $\alpha \to k\alpha$
   and $\delta(c)$ is unchanged). By chain rule:
   $$\sum_i w_i^d \cdot \frac{\partial \mu_k}{\partial w_i^d} = 0$$
   exactly. This is a strong analytical check that catches subtle
   missing terms.

3. **S3 chain-rule consistency.** Direct check that the dedicated S3
   gradient equals $(1/\mu_2^2) \partial \mu_3/\partial w - (2 \mu_3/\mu_2^3) \partial \mu_2/\partial w$.


## 2. Anisotropy: Haar wavelet variances $\langle w_e^2 \rangle$

### 2.1 Forward statistic

At each level $\ell$, for each parent cell $p$ that has all $2^D$
children in footprint, compute the Haar coefficients (one per
non-trivial pattern $e \in \{1, \dots, 2^D - 1\}$):

$$w_e(p) = \frac{1}{2^D} \sum_{\sigma=0}^{2^D - 1} (-1)^{|e \wedge \sigma|} \delta_\sigma(p)$$

where $\delta_\sigma(p)$ is the density contrast of the $\sigma$-th child
of $p$ (a level-($\ell+1$) cell), and $|e \wedge \sigma|$ is the
popcount of the bitwise AND.

The W_r-weighted mean of squared coefficients:

$$\langle w_e^2 \rangle^{(\ell)} = \frac{A_e^{(\ell)}}{T^{(\ell)}}, \quad A_e^{(\ell)} = \sum_p W_r(p) \cdot w_e(p)^2, \quad T^{(\ell)} = \sum_p W_r(p)$$

(Note: $T^{(\ell)}$ here sums over **eligible parents only**, distinct
from field-stats' $T^{(\ell)}$ which sums over all in-footprint cells
at level $\ell$.)

### 2.2 Cell sensitivity

For particle $i$ in child cell $c_{\sigma_i}$ of parent $p_i$:

$$\frac{\partial w_e(p_i)}{\partial w_i^d} = \frac{1}{2^D} (-1)^{|e \wedge \sigma_i|} \cdot \frac{1}{\alpha W_r(c_{\sigma_i})}$$

So:
$$\frac{\partial A_e}{\partial w_i^d} = W_r(p_i) \cdot 2 w_e(p_i) \cdot \frac{(-1)^{|e \wedge \sigma_i|}}{2^D \alpha W_r(c_{\sigma_i})}$$

Critical: the factor $W_r(p_i)$ enters because $A_e$ has $W_r(p)$ as a
multiplier in front of each parent's $w_e(p)^2$. (This factor was
missing in an early version — caught by the uniform-scaling test, see §2.5.)

### 2.3 Alpha sensitivity

Using the cancellation identity $\sum_\sigma (-1)^{|e \wedge \sigma|} = 0$
for $e \neq 0$ (any pair $(\sigma, \sigma \oplus \text{bit}(e))$ has
opposite signs):

$$\frac{\partial w_e(p)}{\partial \alpha} = -\frac{1}{2^D \alpha} \sum_\sigma (-1)^{|e \wedge \sigma|} (\delta_\sigma + 1) = -\frac{w_e(p)}{\alpha}$$

(The constant +1 contributes 0 by the cancellation.)

So $\partial w_e^2/\partial \alpha = -2 w_e^2/\alpha$, and:
$$\frac{\partial \langle w_e^2 \rangle}{\partial \alpha} = -\frac{2 \langle w_e^2 \rangle}{\alpha}$$

(Same dimensional pattern as field-stats.)

### 2.4 Per-particle gradient

$$\boxed{\frac{\partial \langle w_e^2 \rangle^{(\ell)}}{\partial w_i^d} = \frac{2 w_e(p_i) W_r(p_i) (-1)^{|e \wedge \sigma_i|}}{2^D \alpha W_r(c_{\sigma_i}) T^{(\ell)}} \;\;-\;\; \frac{2 \langle w_e^2 \rangle^{(\ell)}}{\alpha \cdot \sum_j w_j^r}}$$

The first term is per-particle (depends on parent $W_r$, child $W_r$,
child position $\sigma$, and pattern $e$). The second is constant
across all particles for given $(\ell, e)$.

### 2.5 Implementation

`anisotropy_gradient.rs`. Public methods:

```rust
pub fn gradient_anisotropy_all_levels(&self, cfg, results) -> AnisotropyGradient
pub fn gradient_anisotropy_aggregate(&self, cfg, results, betas: &[Vec<f64>]) -> Vec<f64>
```

The aggregate API takes `betas[level][pattern]`. Common cosmology
combinations:
- LoS-only: $\beta_{l, 1<<(D-1)} = 1$, rest 0.
- Quadrupole: $\beta_{l, 1<<(D-1)} = 1$, $\beta_{l, 1<<d} = -1/(D-1)$
  for transverse axes — directly recovers $Q_2$.

Walk strategy: iterate parents at level $\ell$ via the random catalog's
membership index (covers all eligible parents — empty parents have
$W_r = 0$ and are ineligible). For each eligible parent: compute
child $W_d, W_r, \delta$ for all $2^D$ children; form $w_e(p)$ per
pattern; distribute the per-particle term.

### 2.6 Verification

Same three-test pattern as field-stats. The uniform-scaling test was
particularly valuable: it caught the missing $W_r(p)$ factor in §2.2
that would have been within FD tolerance.


## 3. Pair counts and Landy-Szalay $\xi$

### 3.1 Forward statistic

The cascade accumulates per-cell weighted self-pair counts:

$$P_d(c) = \frac{W_d(c)^2 - \sum_{i \in c} (w_i^d)^2}{2} = \sum_{i < j \in c} w_i^d w_j^d$$

Cumulative DD at level $\ell$:
$$\text{cum\_dd}(\ell) = \sum_{c \in C_\ell} P_d(c)$$

Cumulative DR (cross-pair):
$$\text{cum\_dr}(\ell) = \sum_{c \in C_\ell} W_d(c) \cdot W_r(c)$$

Per-shell pair counts (shell at level $\ell$ holds pairs sharing a
level-$\ell$ ancestor cell but **not** the level-($\ell+1$) descendant):
$$DD_\ell = \text{cum\_dd}(\ell) - \text{cum\_dd}(\ell + 1)$$
$$DR_\ell = \text{cum\_dr}(\ell) - \text{cum\_dr}(\ell + 1)$$

(For the deepest level: just the cumulative.)

Landy-Szalay estimator:
$$\xi_\ell = \frac{DD_\ell/N_{DD} - 2 DR_\ell/N_{DR} + RR_\ell/N_{RR}}{RR_\ell/N_{RR}}$$

with global normalizations
$N_{DD} = (W_d^2 - \sum_i (w_i^d)^2)/2$,
$N_{DR} = W_d \cdot W_r$,
$N_{RR} = (W_r^2 - \sum_j (w_j^r)^2)/2$
(where $W_d = \sum_i w_i^d$, $W_r = \sum_j w_j^r$ are global totals).

### 3.2 DD / DR per-particle gradient

Per-cell cumulative contribution gradient:
$$\frac{\partial P_d(c)}{\partial w_i^d} = (W_d(c) - w_i^d) \cdot \mathbb{1}[i \in c]$$
$$\frac{\partial (W_d \cdot W_r)(c)}{\partial w_i^d} = W_r(c) \cdot \mathbb{1}[i \in c]$$

So:
$$\frac{\partial \text{cum\_dd}(\ell)}{\partial w_i^d} = W_d(c_i^{(\ell)}) - w_i^d$$
$$\frac{\partial \text{cum\_dr}(\ell)}{\partial w_i^d} = W_r(c_i^{(\ell)})$$

Shell gradients (the $w_i^d$ in DD's cumulative cancels across levels):

$$\boxed{\frac{\partial DD_\ell}{\partial w_i^d} = W_d(c_i^{(\ell)}) - W_d(c_i^{(\ell+1)})}$$

$$\boxed{\frac{\partial DR_\ell}{\partial w_i^d} = W_r(c_i^{(\ell)}) - W_r(c_i^{(\ell+1)})}$$

Geometric meaning: $\partial DD_\ell/\partial w_i^d$ is the (weighted)
count of particles paired with $i$ at separations in shell $\ell$ —
its level-$\ell$-cell partners minus its level-($\ell+1$)-cell partners.

(For unit weights this is exactly the count of "shell-$\ell$ partners"
of particle $i$. Verified by the Euler-theorem identity:
$\sum_i w_i^d \cdot \partial DD_\ell/\partial w_i^d = 2 DD_\ell$
because DD is degree-2 homogeneous in $\{w^d\}$.)

### 3.3 $\xi$ per-particle gradient

Define $f_{DD} = DD_\ell/N_{DD}$, $f_{DR} = DR_\ell/N_{DR}$,
$f_{RR} = RR_\ell/N_{RR}$. Then $\xi_\ell = (f_{DD} - 2 f_{DR} + f_{RR})/f_{RR}$.

For data weights, $\partial RR_\ell/\partial w_i^d = 0$ and
$\partial N_{RR}/\partial w_i^d = 0$, so $f_{RR}$ is invariant under
data-weight perturbation. Other normalizations:

$$\frac{\partial N_{DD}}{\partial w_i^d} = W_d - w_i^d$$
$$\frac{\partial N_{DR}}{\partial w_i^d} = W_r$$

Quotient rule:
$$\frac{\partial f_{DD}}{\partial w_i^d} = \frac{\partial DD_\ell/\partial w_i^d - f_{DD} \cdot (W_d - w_i^d)}{N_{DD}}$$
$$\frac{\partial f_{DR}}{\partial w_i^d} = \frac{\partial DR_\ell/\partial w_i^d - f_{DR} \cdot W_r}{N_{DR}}$$

And:
$$\boxed{\frac{\partial \xi_\ell}{\partial w_i^d} = \frac{1}{f_{RR}}\left[\frac{\partial f_{DD}}{\partial w_i^d} - 2 \frac{\partial f_{DR}}{\partial w_i^d}\right]}$$

(Defined when $f_{RR} > 0$ and all normalizations are positive.)

### 3.4 Implementation

`xi_gradient.rs`. Public types and methods:

```rust
pub struct XiGradient {
    pub dd_grads: Vec<Vec<f64>>,  // [shell][particle]
    pub dr_grads: Vec<Vec<f64>>,
    pub xi_grads: Vec<Vec<f64>>,  // empty for shells where xi is undefined
}

impl<const D: usize> BitVecCascadePair<D> {
    pub fn gradient_xi_data_all_shells(&self, stats, shells) -> XiGradient;
    pub fn gradient_xi_data_aggregate(&self, stats, shells, betas) -> Vec<f64>;
}
```

Walk strategy: precompute per-particle $W_d$ and $W_r$ at every cascade
level (one walk over the membership indices). Each shell's gradient is
then a per-particle subtraction of consecutive levels. ξ gradient
combines DD/DR gradients with the global normalization derivatives.

### 3.5 Verification

Five tests:

1. **DD finite-difference parity.**
2. **DR finite-difference parity.**
3. **ξ finite-difference parity** (the chain-rule combination of the above).
4. **Euler theorem on DD:** $\sum_i w_i^d \cdot \partial DD_\ell/\partial w_i^d = 2 DD_\ell$
   (DD is degree-2 homogeneous in $\{w^d\}$).
5. **Euler theorem on DR:** $\sum_i w_i^d \cdot \partial DR_\ell/\partial w_i^d = DR_\ell$
   (DR is degree-1 homogeneous in $\{w^d\}$).
6. **ξ uniform-scaling invariance:** $\sum_i w_i^d \cdot \partial \xi_\ell/\partial w_i^d = 0$
   (ξ is dimensionless under uniform scaling).
7. **Aggregate ↔ per-shell identity.**

Tests 4 and 5 (Euler identities) are particularly strong — they pin
the gradient to within f64 precision, not just to FD tolerance.


## 4. Cell membership: the structural primitive

The gradients above all reduce to "for each particle $i$, find its
level-$\ell$ cell and read its $W_d, W_r$." That lookup is provided
by `cell_membership.rs` via Morton-sorted permutation indices.

### 4.1 Morton sort + binary search

For each particle compute its level-$L_{\max}$ Morton code
(interleaved per-axis bits, `D` bits per level, level 1 in the most-
significant chunk). Sort particles by code. The defining property of
Morton ordering: **particles sharing an ancestor cell at any level are
contiguous in the sorted permutation**, because the level-$\ell$ cell
ID is the top $\ell \cdot D$ bits of the level-$L_{\max}$ code.

Lookup `members(level, cell_id)` = binary-search the sorted code array
for the contiguous range with the matching prefix. $O(\log N)$ per
query, $O(N)$ memory.

### 4.2 Sparse iteration

`non_empty_cells_at(level)` walks the sorted code array, grouping
consecutive entries by their level-$\ell$ prefix. Yields only
non-empty cells — sparse-friendly.

This is the workhorse for the gradient implementations: each gradient
walks the membership index once per level, computing per-cell $W_d, W_r$
(by summing weights over the cell's members), and crediting the
per-particle term.


## 5. Cost analysis

For a cascade of depth $L_{\max}$ with $N_d$ data particles:

| Operation | Cost | Memory |
|---|---|---|
| Membership build | $O(N_d \log N_d)$ | $O(N_d)$ per catalog |
| Field-stats forward | $O(N_d L_{\max})$ | $O(L_{\max})$ |
| Field-stats gradient (per moment) | $O(N_d L_{\max})$ | $O(N_d L_{\max})$ all_levels, $O(N_d)$ aggregate |
| Anisotropy forward | $O(N_d L_{\max} 2^D)$ | $O(L_{\max} 2^D)$ |
| Anisotropy gradient | $O(N_p L_{\max} (2^D)^2)$ | $O(L_{\max} 2^D N_d)$ all_levels |
| ξ pair-count forward | $O(N_d L_{\max})$ | $O(L_{\max})$ |
| ξ gradient | $O(N_d L_{\max})$ | $O(N_d L_{\max})$ |

Per-gradient cost is **same order** as one forward pass. Profiling
(at $N=10^5$) shows gradient time at 0.16-0.47x the forward time,
because the gradient walks only non-empty cells (via membership
index), while the forward pass descends the full cascade tree.


## 6. Random-weight gradients

Variance gradient with respect to random weights is implemented; higher
moments and ξ random-weight gradients deferred (math derivation is
straightforward extension of the same pattern, but algebra is denser).

### 6.1 Why random-weight gradients are harder

For data-weight gradients, $T^{(\ell)} = \sum_c W_r(c)$ doesn't depend
on $w_i^d$, so the chain rule via $W_d(c)$ is local: only the cell
containing $i$ contributes. For random-weight gradients, $W_r(c)$
appears in many places:

- denominator of $\delta(c)$,
- the global $\alpha = \sum w^d / \sum w^r$,
- $T^{(\ell)}$ (denominator of $A_k$),
- footprint cutoff threshold (treat as discrete),
- $N_{RR}$, $N_{DR}$, $RR_\ell$ for ξ,
- anisotropy parent eligibility.

Cleanest decomposition: total derivative = path-through-α + path-through-local-$W_r(c_j)$.

### 6.2 Path 1: through α

$\partial \alpha / \partial w_j^r = -\alpha / \sum_j w_j^r$.

From §1.4: $\partial \mu_k / \partial \alpha = -k \mu_k / \alpha$.

So path-1 contribution:
$$\frac{\partial \mu_k}{\partial \alpha} \cdot \frac{\partial \alpha}{\partial w_j^r} = \frac{k \mu_k}{\sum_j w_j^r}$$

Same value for every random particle.

### 6.3 Path 2: through local $W_r(c_j)$

Let $u = W_r(c_j^{(\ell)})$ for a random particle $j$ in cell $c_j$ at
level $\ell$. Holding $\alpha$ and other cells fixed:

$$\frac{\partial \delta(c_j)}{\partial u} = -\frac{W_d(c_j)}{\alpha u^2} = -\frac{\delta_j + 1}{u}$$

Cell-$j$'s contributions to $S_k$ and $T$:
$$\frac{\partial S_k}{\partial u} = \delta_j^k + u \cdot k \delta_j^{k-1} \cdot \left(-\frac{\delta_j + 1}{u}\right) = -(k-1)\delta_j^k - k\delta_j^{k-1}$$
$$\frac{\partial T}{\partial u} = 1$$

For $k=1$: $\partial S_1/\partial u = -1$.
For $k=2$: $\partial S_2/\partial u = -\delta_j^2 - 2\delta_j$.

### 6.4 Variance: $\mu_2 = A_2 - A_1^2$

$$\frac{\partial A_k}{\partial u} = \frac{1}{T}\left[\frac{\partial S_k}{\partial u} - A_k\right]$$

For $k=1$: $\partial A_1/\partial u = -(1 + \bar\delta)/T$.
For $k=2$: $\partial A_2/\partial u = (-\delta_j^2 - 2\delta_j - A_2)/T$.

Combining:
$$\frac{\partial \mu_2}{\partial u} = \frac{1}{T}\left[-\delta_j^2 - 2\delta_j + 2\bar\delta + 2\bar\delta^2 - A_2\right]$$

Using $A_2 = \mu_2 + \bar\delta^2$ and rewriting around $d_j = \delta_j - \bar\delta$:
$$\frac{\partial \mu_2}{\partial u} = -\frac{1}{T}\left[d_j^2 + 2(1 + \bar\delta) d_j + \mu_2\right]$$

### 6.5 Total per-particle gradient

$$\boxed{\frac{\partial \mu_2}{\partial w_j^r} = \frac{2 \mu_2}{\sum_j w_j^r} - \frac{1}{T^{(\ell)}}\left[d_j^2 + 2(1 + \bar\delta) d_j + \mu_2\right]}$$

where $d_j = \delta(c_j^{(\ell)}) - \bar\delta$ if $c_j$ is in footprint;
the local term is omitted otherwise (cell isn't part of $T_\ell$ or $S_k$
in the forward pass).

### 6.6 Verification — Euler theorem

Variance is invariant under uniform scaling of all random weights
($w_j^r \to k w_j^r$ implies $\alpha \to \alpha/k$ and $W_r(c) \to k W_r(c)$,
so $\delta(c) = W_d/(\alpha W_r)$ is unchanged). By Euler's theorem on
homogeneous functions:
$$\sum_j w_j^r \cdot \frac{\partial \mu_2}{\partial w_j^r} = 0$$

Direct check using the formula:
- First term sums to $\sum_j w_j^r \cdot 2\mu_2/\sum w_r = 2\mu_2$.
- Second term, grouping by cell: $\sum_c W_r(c)/T \cdot [d_c^2 + 2(1+\bar\delta) d_c + \mu_2] = \mu_2 + 0 + \mu_2 = 2\mu_2$ (since $\sum_c W_r(c) d_c^2/T = \mu_2$, $\sum_c W_r(c) d_c/T = 0$).
- Total: $2\mu_2 - 2\mu_2 = 0$. ✓

This is the strict invariant verified in
`gradient_var_random_uniform_scaling_invariant`.

### 6.7 Implementation

`field_stats_gradient.rs`. Public methods:

```rust
pub fn gradient_var_delta_random_all_levels(&self, cfg, results)
    -> RandomWeightFieldStatsGradient
pub fn gradient_var_delta_random_aggregate(&self, cfg, results, betas)
    -> Vec<f64>
```

Output type:
```rust
pub struct RandomWeightFieldStatsGradient {
    pub random_weight_grads: Vec<Vec<f64>>,  // [level][random_particle_idx]
}
```

Internally `per_level_variance_random_gradient` walks the random-cell
membership index, computing per-cell δ and distributing the local term
to each cell's random particles. The global α-term is added uniformly
to every random particle.

### 6.8 Higher-moment random-weight gradients (k = 3, 4)

The path-1 (α) contribution generalizes trivially:
$$\text{path 1: } \frac{k \mu_k}{\sum_j w_j^r}$$

For path 2, recompute $\partial \mu_k / \partial u$ where $u = W_r(c_j)$
using the cell sensitivities from §6.3. Symbolic verification (sympy)
gives a clean unified formula valid for $k = 2, 3, 4$:

$$\boxed{T \cdot \frac{\partial \mu_k}{\partial u} = -(k-1) d_j^k - k(1+\bar\delta) d_j^{k-1} + k(1+\bar\delta) \mu_{k-1} - \mu_k}$$

where $d_j = \delta_j - \bar\delta$ and $\mu_1 \equiv 0$. Cross-checks:
- $k=2$: $-d_j^2 - 2(1+\bar\delta) d_j - \mu_2$ ✓ (matches §6.4).
- $k=3$: $-2 d_j^3 - 3(1+\bar\delta) d_j^2 + 3(1+\bar\delta) \mu_2 - \mu_3$.
- $k=4$: $-3 d_j^4 - 4(1+\bar\delta) d_j^3 + 4(1+\bar\delta) \mu_3 - \mu_4$.

Combined per-particle gradient:
$$\boxed{\frac{\partial \mu_k}{\partial w_j^r} = \frac{k \mu_k}{\sum_j w_j^r} - \frac{1}{T^{(\ell)}}\left[(k{-}1) d_j^k + k(1+\bar\delta) d_j^{k-1} - k(1+\bar\delta) \mu_{k-1} + \mu_k\right]}$$

The reduced skewness $S_3 = \mu_3 / \mu_2^2$ uses the same chain rule
as the data-weight case: $\partial S_3 / \partial w_j^r = (1/\mu_2^2)
\partial \mu_3 / \partial w_j^r - (2 \mu_3/\mu_2^3) \partial \mu_2 / \partial w_j^r$.

Public methods: `gradient_{m3,m4,s3}_delta_random_{all_levels,aggregate}`.
Implementation shares `per_level_central_moment_random_gradient(k)` for k ∈ {2, 3, 4}.

### 6.9 ξ random-weight gradients

Per-cell pair counts and their random-weight derivatives:

| Quantity | Random-weight derivative |
|----------|--------------------------|
| $DD_\ell$ | $0$ (depends only on $\{w^d\}$) |
| $\partial \text{cum\_dr}(\ell)/\partial w_j^r$ | $W_d(c_j^{(\ell)})$ |
| $\partial \text{cum\_rr}(\ell)/\partial w_j^r$ | $W_r(c_j^{(\ell)}) - w_j^r$ |
| $\partial DR_\ell/\partial w_j^r$ | $W_d(c_j^{(\ell)}) - W_d(c_j^{(\ell+1)})$ |
| $\partial RR_\ell/\partial w_j^r$ | $W_r(c_j^{(\ell)}) - W_r(c_j^{(\ell+1)})$ |

Symmetric in shape to the data-weight DR/DD pair (with $w_j^r$ canceling
across consecutive levels in the cumulative just like $w_i^d$ does).

For the ξ gradient: with $\partial DD/\partial w_j^r = 0$ and
$\partial N_{DD}/\partial w_j^r = 0$, $f_{DD}$ has zero random-weight
derivative. The remaining chain through $f_{DR}$, $f_{RR}$:

$$\frac{\partial f_{DR}}{\partial w_j^r} = \frac{\partial DR/\partial w_j^r - f_{DR} \cdot W_{d,\text{total}}}{N_{DR}}$$
$$\frac{\partial f_{RR}}{\partial w_j^r} = \frac{\partial RR/\partial w_j^r - f_{RR} \cdot (W_{r,\text{total}} - w_j^r)}{N_{RR}}$$

$$\boxed{\frac{\partial \xi_\ell}{\partial w_j^r} = \frac{1}{f_{RR}^2}\left[(2 f_{DR} - f_{DD}) \frac{\partial f_{RR}}{\partial w_j^r} - 2 f_{RR} \frac{\partial f_{DR}}{\partial w_j^r}\right]}$$

(derived from $\xi = f_{DD}/f_{RR} - 2 f_{DR}/f_{RR} + 1$ with
$\partial f_{DD}/\partial w_j^r = 0$).

Public types and methods: `XiRandomGradient` (with `dr_grads`,
`rr_grads`, `xi_grads`), `gradient_xi_random_{all_shells,aggregate}`.

Verification: Euler theorem yields exact f64-precision invariants:
- DR (degree-1 in $\{w^r\}$): $\sum_j w_j^r \cdot \partial DR/\partial w_j^r = DR$.
- RR (degree-2): $\sum_j w_j^r \cdot \partial RR/\partial w_j^r = 2 RR$.
- ξ (degree-0): $\sum_j w_j^r \cdot \partial \xi/\partial w_j^r = 0$.

### 6.10 Anisotropy random-weight gradient

This is the algebra-heaviest piece because $W_r$ enters in many places:
the child cell's $W_r(c_\sigma)$ (via $\delta_\sigma$ denominator), the
parent cell's $W_r(p) = \sum_\sigma W_r(c_\sigma)$ (multiplier on $w_e^2$
in $A_e$ and a term in $T^{(\ell)}$), and α (global).

For random particle $j$ in level-$(\ell+1)$ child cell $c_{\sigma_j}$ of
level-$\ell$ parent $p_j$, define $u = W_r(c_{\sigma_j})$. The
parent's $W_r$ depends on $u$ as $\partial u_p / \partial u = 1$ (since
$u_p = \sum_\sigma W_r(c_\sigma)$ and only one term involves $u$).

**Path 2** (holding α and other cells' $W_r$ fixed). Only $p_j$'s
contribution to $A_e$ and $T$ has $u$-dependence:

- $\partial \delta_{\sigma_j}/\partial u = -(\delta_{\sigma_j}+1)/u$
- $\partial w_e(p_j)/\partial u = -(-1)^{|e \wedge \sigma_j|}(\delta_{\sigma_j}+1)/(2^D u)$
- $\partial[u_p \cdot w_e(p_j)^2]/\partial u = w_e(p_j)^2 + u_p \cdot 2 w_e \cdot \partial w_e/\partial u$
- $\partial T/\partial u = 1$

Combining via $\partial(A/T)/\partial u$:
$$\frac{\partial \langle w_e^2 \rangle}{\partial u} = \frac{1}{T}\left[w_e(p_j)^2 - \langle w_e^2 \rangle - \frac{2 u_p w_e(p_j) (-1)^{|e \wedge \sigma_j|} (\delta_{\sigma_j} + 1)}{2^D u}\right]$$

**Path 1** (through α): $\partial\langle w_e^2\rangle/\partial \alpha = -2\langle w_e^2 \rangle/\alpha$ (§2.3); chain via $\partial\alpha/\partial w_j^r = -\alpha/\sum w_r$ gives $+2\langle w_e^2 \rangle/\sum w_r$.

**Combined per-particle gradient**:
$$\boxed{\frac{\partial \langle w_e^2 \rangle^{(\ell)}}{\partial w_j^r} = \frac{2 \langle w_e^2 \rangle^{(\ell)}}{\sum w_r} + \frac{1}{T^{(\ell)}}\left[w_e(p_j)^2 - \langle w_e^2 \rangle^{(\ell)} - \frac{2 W_r(p_j) w_e(p_j) (-1)^{|e \wedge \sigma_j|} (\delta_{\sigma_j} + 1)}{2^D W_r(c_{\sigma_j})}\right]}$$

(Path 2 included only when $p_j$ is eligible — all $2^D$ children in
footprint AND parent in footprint. Otherwise only path 1 applies.)

Public types and methods: `AnisotropyRandomGradient`,
`gradient_anisotropy_random_{all_levels,aggregate}`.

Verification: $\langle w_e^2 \rangle$ invariant under uniform scaling
of $\{w^r\}$, so $\sum_j w_j^r \cdot \partial \langle w_e^2 \rangle / \partial w_j^r = 0$
exactly. This invariant catches any missing factor in the parent-W_r
or path-2 algebra (just as it caught the missing $W_r(p)$ in the
data-weight case in commit 10).

### 6.11 Implementation summary

| Statistic | Data-weight | Random-weight |
|-----------|-------------|---------------|
| $\mu_2$ (var)   | ✅ | ✅ |
| $\mu_3, \mu_4, S_3$ | ✅ | ✅ |
| $\langle w_e^2 \rangle$ (anisotropy) | ✅ | ✅ |
| DD pair counts  | ✅ | $\equiv 0$ trivially |
| DR pair counts  | ✅ | ✅ |
| RR pair counts  | $\equiv 0$ trivially | ✅ |
| $\xi_{LS}$      | ✅ | ✅ |

All gradients verified by:
1. Finite-difference parity (per-particle FD vs analytic).
2. Euler-theorem invariants (uniform-scaling identities, exact to f64).
3. Aggregate ↔ per-bin combination identity.
4. Where applicable: chain-rule consistency (S3 from m2/m3 gradients).

## 7. Multi-run gradients

The multi-run aggregator (`CascadeRunner`) runs $N$ cascades on
shifted/resized views of one base catalog. Each per-cascade gradient
lives in **cascade-particle-index space**, which is generally not the
same as the original-catalog-index space (due to clipping in isolated
mode and re-orderings).

For multi-run gradients to compose, we need a mapping from
cascade-particle-index back to original-catalog-index per run.

### 7.1 Index mapping primitive

`AppliedRun` and `RunResult` carry `original_d_indices: Vec<u32>` and
`original_r_indices: Vec<u32>`. Entry `k` is the index in the runner's
base catalog that became the $k$-th particle in that run's cascade.
Particles clipped in isolated mode don't appear in the mapping at all
(so they receive zero gradient contribution from that run, which is
the correct semantics — they didn't participate).

### 7.2 Lifting per-cascade gradients

`CascadeRunner::lift_gradient_d_to_original(cascade_grad, mapping, n_base)`
returns a length-`n_base` vector with each `cascade_grad[k]` placed at
position `mapping[k]`. Original particles not in the mapping get value 0.

### 7.3 Composition: linear case (run average)

For statistics that are linear in per-cascade outputs (e.g., the average
across runs of var_delta at a chosen level):

$$\bar V = \frac{1}{N} \sum_{r=1}^{N} V_r, \quad
\frac{\partial \bar V}{\partial w_i^d} = \frac{1}{N} \sum_{r=1}^{N} \text{lift}_r\!\left(\frac{\partial V_r}{\partial w_i^d}\right)$$

`CascadeRunner::gradient_var_delta_data_run_average(cfg, level)`
implements this — runs each per-cascade gradient, lifts, averages.

### 7.4 Composition: pooled-aggregate case

`CascadeRunner::analyze_field_stats` produces a *pooled* aggregate: it
sums raw $S_k$, $T$ across runs at the same physical scale, then
re-derives central moments from the pooled sums. The gradient of this
pooled var_delta w.r.t. an original data weight is implemented in
`CascadeRunner::gradient_var_delta_data_pooled`.

**Math.** Pooled raw sums are linear in per-run sums:
$T^\text{pool} = \sum_r T^{(r)}$, $S_k^\text{pool} = \sum_r S_k^{(r)}$.
Pooled variance is:
$$\mu_2^\text{pool} = \frac{S_2^\text{pool}}{T^\text{pool}} - \left(\frac{S_1^\text{pool}}{T^\text{pool}}\right)^2$$

Differentiating directly w.r.t. an original data weight $w_i^d$:
$$\frac{\partial \mu_2^\text{pool}}{\partial w_i^d} = \frac{1}{T^\text{pool}}\left[\partial S_2^\text{pool}_i - 2 m_1^\text{pool} \cdot \partial S_1^\text{pool}_i\right]$$

where $\partial S_k^\text{pool}_i = \sum_r \text{lift}_r(\partial S_k^{(r)}_i)$
and $\partial T^\text{pool} = 0$ for data weights (T sums random
weights only).

**Per-run raw-sum gradients.** The primitive
`BitVecCascadePair::gradient_raw_sums_data_all_levels` returns
per-particle $(\partial S_1, \partial S_2, \partial S_3, \partial S_4)$
at every cascade level. Unified formula for $k = 1, 2, 3, 4$:

$$\frac{\partial S_k^{(\ell)}}{\partial w_i^d} = \mathbb{1}[c_i^{(\ell)} \in \text{footprint}] \cdot \frac{k \, \delta(c_i^{(\ell)})^{k-1}}{\alpha} \;-\; \frac{k\,(S_k^{(\ell)} + S_{k-1}^{(\ell)})}{\alpha \cdot \sum_j w_j^r}$$

(with $S_0 \equiv T$). The local term applies to particles in
in-footprint cells; the global $\alpha$-term applies uniformly.

**Pooled chain rules** for higher central moments. Using
$A_k = S_k^\text{pool}/T^\text{pool}$, $m_1 = A_1$:

$$\frac{\partial \mu_2^\text{pool}}{\partial w_i^d} = \frac{1}{T^\text{pool}}\left[\partial S_2 - 2 m_1 \partial S_1\right]$$

$$\frac{\partial \mu_3^\text{pool}}{\partial w_i^d} = \frac{1}{T^\text{pool}}\left[\partial S_3 - 3 m_1 \partial S_2 - 3(\mu_2 - m_1^2) \partial S_1\right]$$

$$\frac{\partial \mu_4^\text{pool}}{\partial w_i^d} = \frac{1}{T^\text{pool}}\left[\partial S_4 - 4 m_1 \partial S_3 + 6 m_1^2 \partial S_2 - 4(\mu_3 + m_1^3) \partial S_1\right]$$

For $S_3 = \mu_3/\mu_2^2$:

$$\frac{\partial S_3^\text{pool}}{\partial w_i^d} = \frac{1}{\mu_2^2} \frac{\partial \mu_3^\text{pool}}{\partial w_i^d} - \frac{2 \mu_3}{\mu_2^3} \frac{\partial \mu_2^\text{pool}}{\partial w_i^d}$$

(All quantities on the right at the pooled bin's pooled values.)

**Sanity checks.** Combining the per-cascade $(\partial S_1, \partial S_2)$
via the variance chain rule recovers `gradient_var_delta_all_levels`
to f64 precision — verified by `raw_sum_gradient_combines_to_var_gradient`.
Higher moments combine analogously, verified by
`raw_sum_gradient_combines_to_higher_moment_gradients`.

**Implementation strategy** (in `CascadeRunner::gradient_var_delta_data_pooled`):

1. Walk all per-run cascades. For each (run, level), compute per-particle
   $(\partial S_1, \partial S_2)$ via `gradient_raw_sums_data_all_levels`.
2. Lift each gradient vector from cascade-particle-index to original-
   particle-index space via `lift_gradient_d_to_original` and the run's
   `original_d_indices` mapping.
3. Bin entries by `physical_side` matching `analyze_field_stats`'s
   sort-and-sweep logic.
4. Within each bin, sum lifted $(\partial S_1, \partial S_2)$ across
   contributing runs.
5. Combine via $\partial \mu_2^\text{pool}_i = (1/T^\text{pool})
   (\partial S_2^\text{pool}_i - 2 m_1^\text{pool} \partial S_1^\text{pool}_i)$
   per bin.

Returns `PooledFieldStatsGradient` with `bin_grads[bin_idx][original_particle_idx]`
and `bin_sides[bin_idx]` matching the order of `AggregatedFieldStats::by_side`.

**Verification**: 4 tests including (a) single-run reduction (pooled
gradient with `just_base` plan equals per-cascade variance gradient at
the matching level), (b) uniform-scaling invariance ($\sum_i w_i \cdot
\partial \mu_2^\text{pool} / \partial w_i = 0$ exactly), (c) end-to-end
finite-difference parity with multiple shifted runs (perturb base
weight, recompute pooled var via full `analyze_field_stats`, compare
slope), (d) aggregate-scalar API consistency.

### 7.5 Implemented and deferred

**Implemented**:

- **Field-stats moments** (`var_delta`, `m3_delta`, `m4_delta`,
  `s3_delta`): single shared internal helper `pooled_raw_sums_data`
  produces per-bin pooled raw sums plus lifted
  $(\partial S_1, \ldots, \partial S_4)$; each public moment method
  applies its specific chain rule.

- **Anisotropy** (`gradient_anisotropy_data_pooled`): per-pattern,
  per-axis, and LoS-quadrupole gradients. The pooled mean is
  $\overline{w_e^2}^\text{pool} = \Sigma_r A_e^{(r)} / \Sigma_r T^{(r)}$
  where $A_e^{(r)} = T^{(r)} \cdot \overline{w_e^2}^{(r)}$ is the
  per-run raw accumulator. Since $T^{(r)} = $ sum_w_r_parents depends
  only on random weights, $\partial T^{(r)}/\partial w_i^d = 0$ and
  the chain rule reduces to lifting per-run mean gradients (scaled by
  $T^{(r)}$ to recover raw-accumulator gradients) and dividing by
  pooled $T$. No new primitive needed; reuses
  `gradient_anisotropy_all_levels`.

- **ξ (Landy-Szalay)** (`gradient_xi_data_pooled`): per-(resize-group,
  shell), per-particle gradients. Per resize group, pooled pair
  counts add linearly across shifts: $X^\text{pool}_\ell = \Sigma_r X^{(r)}_\ell$
  for $X \in \{DD, DR, RR\}$. The forward
  `pool_xi_resize_group` uses **count-based** pooled normalization
  $N_{DD}^\text{pool} = n_d^\text{pool}(n_d^\text{pool}-1)/2$ where
  $n_d^\text{pool} = \Sigma_r n_d^{(r)}$ (particle count, not weight
  sum). Since count-based normalization doesn't depend on data
  weights, the chain rule simplifies:
  
  $$\frac{\partial \xi^\text{pool}}{\partial w_i^d} = \frac{1}{f_{RR}^\text{pool}}\left[\frac{1}{N_{DD}^\text{pool}} \sum_r \text{lift}_r\!\left(\frac{\partial DD^{(r)}}{\partial w_i^d}\right) - \frac{2}{N_{DR}^\text{pool}} \sum_r \text{lift}_r\!\left(\frac{\partial DR^{(r)}}{\partial w_i^d}\right)\right]$$
  
  Per-run pair-count gradients $\partial DD^{(r)}/\partial w_i^d$ come
  from `gradient_xi_data_all_shells`. Note: the count-based pooled
  formula does NOT inherit the uniform-scaling invariance that the
  weight-based single-cascade formula has — an artifact of the
  forward semantics that the gradient correctly reflects.

- **Random-weight pooled gradients (symmetric coverage)**:
  `gradient_{var,m3,m4,s3}_delta_random_pooled`,
  `gradient_anisotropy_random_pooled`, and `gradient_xi_random_pooled`,
  with aggregate-scalar variants for each. The random-weight side has
  one extra wrinkle: $T^\text{pool} = \Sigma_r T^{(r)}$ depends on
  random weights ($\partial T^{(r)}/\partial w_j^r \neq 0$), so the
  chain rule for pooled $\mu_k$ picks up an extra $\partial T$ term:

  $$\frac{\partial A_k^\text{pool}}{\partial w_j^r} = \frac{1}{T^\text{pool}}\!\left[\frac{\partial S_k^\text{pool}}{\partial w_j^r} - A_k^\text{pool} \cdot \frac{\partial T^\text{pool}}{\partial w_j^r}\right]$$

  with the same algebraic combinations for $\mu_2, \mu_3, \mu_4, S_3$
  applied on top. New raw-sum primitive
  `gradient_raw_sums_random_all_levels` returns
  $(\partial T, \partial S_1, \ldots, \partial S_4)$ per random
  particle per level. For anisotropy, the pooled gradient uses the
  existing per-mean primitive plus a new $\partial T$ helper
  (`gradient_anisotropy_t_random_all_levels`) and converts to raw-
  accumulator gradients via $\partial A_e^{(r)} = T^{(r)} \cdot
  \partial \overline{w_e^2}^{(r)} + \overline{w_e^2}^{(r)} \cdot \partial T^{(r)}$.
  For ξ, the count-based pooled normalization is independent of random
  weights AND $\partial DD/\partial w_j^r = 0$, leaving only $\partial DR$
  and $\partial RR$ contributions.

**Status**: the differentiable-cascade gradient surface is now
**structurally complete**. All three observables (field-stats
moments, anisotropy, ξ) have gradients in both per-cascade and
multi-run-pooled modes, in both data-weight and random-weight
directions. Aggregate-scalar variants exist for each.

## 8. Forward-mode and Hessian-vector products

The reverse-mode infrastructure described above already materializes
the per-particle Jacobian for each pooled observable. Forward-mode
directional derivatives — Jacobian-vector products $J v$ — are
therefore one row-dot-product per output:

$$\left(J v\right)_b = \sum_i \frac{\partial f_b}{\partial w_i} v_i$$

The `jvp` module exposes one `jvp_*` free function per pooled
gradient type, plus a Hessian-vector product helper:

$$H v \approx \frac{g(w + \epsilon v) - g(w - \epsilon v)}{2 \epsilon}$$

This is forward-FD over the reverse-mode gradient — two gradient
evaluations, no Hessian materialized. With $\epsilon = 10^{-6}$
typical HVP relative error is $\sim 10^{-9}$ on smooth losses.
Verified by exact agreement on a synthetic quadratic and by
second-order FD agreement on cascade losses.

The example `optimize_data_weights_newton_cg.rs` demonstrates this
in a Newton-CG loop on the same target as the Adam example: ~$10^{13}\times$
loss reduction in 25 outer iterations vs Adam's $\sim 10^4\times$ in 60.

## 9. Cell pair-separation kernels (variance ↔ ξ)

For a cell of volume $V$, the cell-averaged correlation function is

$$\bar\xi_V = \int_0^{r_\max} K_V(r) \xi(r) dr$$

where $K_V(r)$ is the probability density that two uniform random
points in $V$ are at separation $r$. Discretized on the cascade's
shells:

$$\bar\xi_V \approx \sum_s K_V(\bar r_s) \cdot \xi_s \cdot \Delta r_s$$

The `cell_kernels` module provides:
- `AnalyticBoxKernel1D` — closed-form triangular density for 1D segments
- `MonteCarloKernel::unit_cube::<D>(n_pairs, n_bins, seed)` — MC-
  tabulated kernel for any D-dim cube. $10^5$ pair samples give
  sub-percent kernel accuracy in milliseconds.
- `variance_from_xi(group, kernel, cell_side)` — the integration
- `compute_kernel_betas_for_group(group, kernel, cell_side)` —
  returns `betas` weights such that
  $\bar\xi_V = \sum_s \beta_s \cdot \xi_s$

**Differentiability comes free.** Because $\bar\xi_V$ is a linear
combination of pooled-ξ shell values with weights independent of
$w$, the gradient is one call to the existing aggregate API:

$$\frac{\partial \bar\xi_V}{\partial w_i} = \sum_s K_V(\bar r_s) \Delta r_s \cdot \frac{\partial \xi_s}{\partial w_i}$$

```rust
let betas: Vec<Vec<f64>> = agg.by_resize.iter().enumerate()
    .map(|(g, group)| {
        if g == target_group {
            compute_kernel_betas_for_group(group, &kernel, cell_side)
        } else {
            vec![0.0; group.shells.len()]
        }
    }).collect();
let grad_var_v = runner.gradient_xi_data_pooled_aggregate(scale_tol, &betas);
```

End-to-end FD parity test
(`variance_from_xi_gradient_via_aggregate_matches_finite_difference`)
verifies this works.

**Why no closed-form 2D/3D kernels?** The pair-separation density
inside a 2D square or 3D cube has known closed forms (Philip 2007,
Mathai 1999) but they are piecewise (2 pieces in 2D, 3 in 3D) with
`arcsin`/`arccos` in the middle pieces. MC kernels are accurate
enough at the cascade's shell granularity, computed once and
cached. Users wanting the analytic forms can use the MC kernel as a
verification reference.

**Use cases**:
- Cross-check between the cascade's two variance estimators
  (`var_delta` directly vs kernel-from-ξ). Disagreement signals
  systematics (cascade shell-structure semantics, shot-noise
  estimation, mode mismatch).
- Predicting $\sigma^2(V)$ from a theoretical $\xi(r)$ without
  running a separate variance walk.
- Differentiable cell-averaged correlation functions for parameter-
  fitting workflows (gradient comes free).
