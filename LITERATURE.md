# Closest constructions in the literature

The per-particle baseline implemented in
``twopt_density/weights_pair_counts.py`` decomposes the global
Landy-Szalay/Davis-Peebles two-point estimator into per-particle, per-bin
contributions. After a literature search the closest matches in the
**spatial-statistics** and **cosmology** literatures are summarized below.

## Direct precursor (spatial statistics, very close match)

### Getis & Franklin (1987) — Local K function / neighbourhood density

> A. Getis and J. Franklin, *Second-order neighbourhood analysis of mapped
> point patterns*, Ecology **68**, 473 (1987).

For each point ``i`` in a pattern, define

```
L_i(r) = sqrt( (a / (n - 1)) * sum_{j != i} 1[|x_i - x_j| <= r] )
```

with edge corrections. The ``localK``/``localL`` family in R's
``spatstat`` package implements this directly; the manual notes
explicitly that "L_i(r) can be interpreted as one of the summands that
contributes to the global estimate of the L function." That is exactly
the property our ``per_particle_pair_counts(positions, r_edges)``
exploits, with
``mean_i b_i^(j) / E[b^(j)] = DD_j / RR_j - 1 = xi_simple(r_j)``.

What the Getis-Franklin construction does NOT have explicitly: the
Davis-Peebles per-particle ratio
``b_DD_i^(j) N_R / (b_DR_i^(j) N_D) - 1`` for a survey window. The
spatial-stats community's "intensity-reweighted" estimators
(Shaw & Møller 2021; Baddeley et al.) reach the same place by a
different route -- they normalise the local sum by an estimated point
intensity at each location. Our DR-corrected form is the cleanest cosmology
analog because the random catalog already provides a non-parametric
intensity estimate.

### Cressie & Collins (2001) — Product-density LISA

> N. Cressie and L. Collins, *Analysis of spatial point patterns using
> bundles of product density LISA functions*, J. Agric. Biol. Environ.
> Stat. **6**, 118 (2001).

Generalises the LISA framework (Anselin 1995) to point processes: define
a per-event statistic whose sum equals (a known multiple of) the global
product density / pair correlation function. The aggregation modes in
``aggregate_weights`` (``RR``, ``RR_xi``) are LISA-style summaries of
these per-event functions. The direct cosmology analog appears not to be
named in the cosmology literature.

### Anselin (1995) — LISA framework (the conceptual umbrella)

> L. Anselin, *Local Indicators of Spatial Association — LISA*,
> Geographical Analysis **27**, 93 (1995).

Defines a LISA as any local statistic ``L_i`` whose sum is proportional
to a corresponding global statistic. Our per-particle, per-bin
``delta_i^(j)`` is a LISA for the simple correlation estimator: the
arithmetic mean over particles equals ``xi_simple(r_j)``. The
``b_DR``-weighted mean is the corresponding LISA for Davis-Peebles.


## Cosmology literature: same goal, different implementation

### Davis & Peebles (1983) — global DD/DR

> M. Davis and P. J. E. Peebles, *A survey of galaxy redshifts. V. The
> two-point position and velocity correlations*, ApJ **267**, 465 (1983).

The global ``DD/DR - 1`` estimator we are decomposing per particle. Our
window-aware per-particle quantity
``b_DD_i^(j) N_R / (b_DR_i^(j) N_D) - 1`` is the natural pointwise
analog whose ``b_DR``-weighted average reproduces the global form.

### Landy & Szalay (1993) — the gold-standard estimator we recover

> S. D. Landy and A. S. Szalay, *Bias and variance of angular correlation
> functions*, ApJ **412**, 64 (1993).

LS adds the ``+ RR`` term for window-robustness at the price of a small
bias offset. Our identity test compared per-particle DP to LS and
agrees to within ~5% across the strongly-clustered range; the gap is
exactly the LS-vs-DP normalisation difference, not a per-particle
construction artefact.

### Marked correlation function (Beisbart, Sheth, Skibba, Connolly)

> R. Skibba, R. K. Sheth, A. J. Connolly, R. Scranton, *The
> luminosity-weighted "marked" correlation function*, MNRAS **369**,
> 68 (2006).

Closest cosmology cousin in spirit: each galaxy carries a "mark" m_i
(luminosity, colour, mass) and the marked correlation
``M(r) = (1 + W(r)) / (1 + xi(r))``
is computed from
``WW(r) = sum_{i<k} m_i m_k 1[r_ik in B_j]``.
**Difference**: the marked statistic uses externally-supplied marks. Our
construction *derives* the marks from the same pair counts already
needed for the LS estimator; no external information.

### DTFE / Voronoi tessellation (Schaap & van de Weygaert 2000+)

> W. E. Schaap and R. van de Weygaert, *Continuous fields and discrete
> samples: reconstruction through Delaunay tessellations*, A&A **363**,
> L29 (2000).

Per-galaxy density via the inverse volume of the contiguous Voronoi
cell. The cleanest existing way to attach a single density to each
data point. **Difference**: the Voronoi/Delaunay density is a
geometric estimator that does not directly link to the two-point
estimator; ours is built such that the per-particle quantities aggregate
into ``xi_LS``.

### Smoothed-particle / kernel density estimator

A KDE-based local overdensity at each particle. This is exactly what
``twopt_density.weights_binned.kde_overdensity`` implements as the data
input to the Wiener filter -- effectively the same quantity the GAMA
collaboration uses as ``Sigma_5``-style environment. **Difference**: a
single-scale KDE drops the multi-bin information; our per-particle
matrix ``delta_pp[i, j]`` keeps the full scale-dependence.

### kNN-CDF (Banerjee & Abel 2021)

> A. Banerjee and T. Abel, *Nearest neighbour distributions: New
> statistical measures for cosmological clustering*, MNRAS **500**,
> 5479 (2021).

Per-particle distances to the k-th nearest neighbour, aggregated as a
CDF. Conceptually parallel: a per-particle quantity that aggregates to a
clustering summary. **Difference**: kNN-CDF is sensitive to all higher
N-point functions through a different summary statistic; the per-
particle pair count is targeted at exactly reproducing the LS two-point
estimator.

### FKP weights (Feldman, Kaiser, Peacock 1994)

> H. Feldman, N. Kaiser, J. Peacock, *Power-spectrum analysis of
> three-dimensional redshift surveys*, ApJ **426**, 23 (1994).

``w_FKP(x) = 1 / (1 + n(x) P_0)``: a per-galaxy weight derived from the
local mean density and a target power-spectrum amplitude, optimised to
minimise the variance of ``P(k)``. **Difference**: FKP weights minimise
estimator variance for a *known* clustering amplitude; our weights
*reproduce* the measured clustering when fed back through the pair sum.
The two have completely different design objectives.


## What appears genuinely new

The combination of (a) a per-particle, per-bin density estimator
``delta_i^(j)`` that aggregates *exactly* to the binned LS pair count
and (b) a window-aware Davis-Peebles form that uses ``b_DR_i^(j)`` as a
non-parametric local intensity estimate is essentially the
Getis-Franklin / Cressie-Collins LISA construction transplanted to a
cosmology pipeline. The local-K body of work in spatial statistics has
not been imported into galaxy clustering analyses to our knowledge.

Two natural follow-ups suggested by the literature:

1. **Bundles** of LISA functions (Cressie & Collins 2001): instead of a
   single scalar weight per particle, classify particles by the *shape*
   of their full ``delta_i^(j)`` vector across bins. Hierarchical
   clustering on these vectors gives an empirical environment
   classification (cluster core / filament / void) using only the
   pair-count outputs already computed.
2. **Intensity-reweighted form** (Shaw & Møller 2021) for cases
   where the random catalog density is itself uncertain or anisotropic;
   replaces ``b_DR_i`` with a kernel-density estimate of the survey
   intensity at each data point.


## Extension to three-point: how would a 3pt estimate change the weights?

The LISA framework extends naturally to higher-order correlations,
though we have not found this written out for point processes in either
the spatial-statistics or the cosmology literature.

### The 3pt LISA primitive

For each particle ``i`` and triangle bin ``alpha`` (with side lengths
``(s_1, s_2, s_3)`` in some tolerance), define

```
T_i^(alpha) = # of triangles in shape alpha that contain i as a vertex
```

with the natural per-particle overdensity

```
delta_i^(3, alpha) = T_i^(alpha) / E[T^(alpha)] - 1.
```

The expectation under uniform Poisson is

```
E[T^(alpha)] = (N - 1)(N - 2) * V_T(alpha) / V_box^2,
```

where ``V_T(alpha)`` is the geometric volume of valid triangle
configurations at fixed first vertex (computable analytically for
simple shapes; estimable via the random catalog otherwise).

The 3pt analog of the LISA identity is

```
mean_i delta_i^(3, alpha) = DDD_alpha / RRR_alpha - 1 = zeta_simple(alpha).
```

We verified this numerically in ``demos/demo_3pt_lisa.py``: on a
3000-point clustered toy with an equilateral bin at ``s ~ 8 +- 2`` Mpc,
the per-particle mean and the global ``DDD/RRR - 1`` agree to machine
precision (1e-12).

### Window-aware form

Replace the uniform-Poisson expectation with the random-catalog version.
Two natural per-particle 3pt analogs of LS:

```
delta_i^(3, alpha)_DP  =  T_DDD_i N_R^2 / (T_DRR_i N_D^2) - 1     (Davis-Peebles)
delta_i^(3, alpha)_LS  =  ( T_DDD_i - 3 T_DDR_i + 3 T_DRR_i - T_RRR_i ) / T_RRR_i_anchored
```

The DP form is the cleanest and most directly analogous to the 2pt
construction; the full LS form requires per-particle DDR, DRR, RRR
which costs an extra factor of ``N_R / N_D`` per term.

### How the weights change

The weight assignment problem from the doc (Eq. 2)

```
sum_{i<k} w_i w_k 1[r_ik in B_j] = (1 + xi_j) RR_j     (j = 1..N_b)
```

is augmented by 3pt constraints

```
sum_{i<j<k} w_i w_j w_k 1[(s_1,s_2,s_3) in alpha]
                          = (1 + xi + xi + xi + zeta_alpha) RRR_alpha    (alpha = 1..N_T).
```

Linearised in ``delta_i = w_i - 1``, the 2pt and 3pt linear constraints
share the same unknowns:

```
sum_i delta_i b_i^(j)        = -[ correction term at order xi^2 ]
sum_i delta_i T_i^(alpha)    = -[ correction term at order xi*zeta + ... ]
```

with coefficients ``b_i^(j)`` (per-particle pair count) and
``T_i^(alpha)`` (per-particle triangle count). Writing this as a stacked
linear system

```
[ B  ] delta = r_2pt
[ T  ]       = r_3pt
```

with ``B`` of shape ``(N_b, N_D)`` and ``T`` of shape ``(N_T, N_D)``,
we add ``N_T`` rows. The solution space has dimension ``N_D - N_b - N_T``.
For ``N_T`` ~ 100 triangle bins and ``N_D`` ~ 5000 the system remains
massively under-determined, but each constraint **tightens** the
admissible weights.

### Aggregation at the LISA level

The LISA-style scalar weight per particle generalises to a
multi-order combination

```
delta_i = (sum_j a_j delta_i^(2,j) + sum_alpha b_alpha delta_i^(3,alpha))
          / (sum_j a_j + sum_alpha b_alpha).
```

The natural choice ``a_j = RR_j`` and ``b_alpha = RRR_alpha`` weights
by the volume of each scale (Poisson-noise optimal), automatically
balancing 2pt and 3pt contributions according to their statistical
power. Particles in **filaments** -- which have moderate 2pt
overdensity but high 3pt overdensity (because their three-point shape
preferentially contains aligned triples) -- get weights enhanced
relative to the pure 2pt aggregate.

### Cost

Per-particle pair counting at moderate ``r_max`` is ``O(N log N)``.
Triangle counting per particle is ``O(N * k_pair^2)`` where ``k_pair`` is
the typical number of pair-neighbors per particle inside the triangle
shell. For a Quijote-like 5000-halo catalog with 30 neighbors per
particle in a 10-Mpc shell that's about ``5000 * 900 = 4.5 * 10^6``
operations -- still sub-second. For ``N = 1e5`` with the same density
it's ``1e5 * 900 = 1e8`` ops, a few seconds. Beyond ``N ~ 1e6`` the
cost becomes uncomfortable and one would want to integrate the
3pt LISA into the same multipole / FFT-based fast bispectrum estimators
already used in cosmology (Slepian & Eisenstein 2015; Sugiyama+ 2019).

### Status in the literature

As far as we found:

- **2pt LISA on point processes** is well-developed: Cressie & Collins
  (2001) product-density LISA, Getis-Franklin local K function,
  ``spatstat::localK``.
- **3pt LISA on point processes** does not appear named in either
  the spatial-statistics or cosmology literature. The closest cosmology
  cousin is **Slepian & Eisenstein (2015)**'s ``O(N^2)`` 3pt estimator
  via spherical harmonics; that is a global compression but conceptually
  preserves a per-particle structure that could be relabeled as a 3pt
  LISA.

So this is genuinely an open direction; the demo verifying the identity
is the seed for a real implementation.


## Faster than triangle counts: anisotropy-preserving 2nd-order estimators

The 3pt construction above costs ``O(N * k_pair^2)`` per particle (the
inner loop runs over pairs of pair-neighbors). If the goal is to add
**shape / anisotropy information** without paying full triangle cost,
several existing estimators in cosmology and spatial statistics achieve
this at ``O(N * k_pair)`` -- the same as the 2pt LISA. They are still
2nd-order in counts (i.e. "two-point", not "three-point"), but they
carry tensorial / directional structure that a scalar pair count
cannot.

### Per-particle pair multipoles (cosmology: Hamilton 1992)

The standard quadrupole / hexadecapole moments of the 2pt correlation
function ``xi(r, mu) = sum_L xi_L(r) * P_L(mu)`` decompose the pair
distribution into Legendre multipoles. The natural per-particle LISA at
multipole ``L`` is

```
b_i^(j, L) = sum_{k != i, r_ik in B_j} P_L(mu_ik),
```

with ``mu_ik`` the cosine of the angle between ``x_k - x_i`` and a
chosen axis (line of sight, principal axis of local neighborhood, or
any preferred direction). The standard scalar count is the ``L=0``
moment. ``L=2`` carries the dominant anisotropy signal:

```
mean_i b_i^(j, 2) / mean_i b_i^(j, 0)   =   xi_2(r_j),
```

the quadrupole of the LS estimator. Verified numerically in
``demos/demo_pair_quadrupole.py``: an isotropic clustered toy gives
``xi_2(r) ~ 0`` (noise floor 0.02), while a z-squashed pancake catalog
gives ``xi_2 ~ -0.25`` peaked at the pancake scale. The cost is the
same as the scalar count -- one extra ``np.add.at`` accumulating
``P_L(mu)`` instead of ``1.0``.

Compared to triangle counts:

| primitive             | order | what it captures           | cost            |
| --------------------- | ----- | -------------------------- | --------------- |
| ``b_i^(j, 0)``        | 2pt   | scalar density at scale r  | O(N k_pair)     |
| ``b_i^(j, 2)``        | 2pt   | + alignment / quadrupole   | O(N k_pair)     |
| ``b_i^(j, 4)``        | 2pt   | + hexadecapole anisotropy  | O(N k_pair)     |
| ``T_i^(alpha)``       | 3pt   | + triangle shape structure | O(N k_pair^2)   |

For the price of a couple more ``np.add.at`` calls on the same pair
list one gains the full angular-multipole expansion at every scale.
Each multipole gives an additional LISA primitive that can be
aggregated into the per-particle weight:

```
delta_i = a_0 delta_i^(0) + a_2 delta_i^(2) + a_4 delta_i^(4) + ...
```

Particles whose local neighborhood is preferentially aligned (e.g., in
filaments) get ``delta_i^(2)`` flagged separately from their scalar
overdensity ``delta_i^(0)``.

### Counts-in-cells skewness (Peebles 1980, Bouchet+ 1992, Bernardeau)

For non-Gaussianity per se -- the 3rd-order moment of the smoothed
density field ``S_3 = <delta_R^3> / <delta_R^2>^2`` -- counts-in-cells
on a regular grid is ``O(N)`` for binning plus moment evaluation, so
it costs *less* than even the 2pt LISA for a fixed smoothing scale.
With anisotropic cells (cylindrical along the LOS, ellipsoidal
oriented to a chosen tensor) the moments carry directional information
without pair enumeration.

| primitive                   | what it captures            | cost           |
| --------------------------- | --------------------------- | -------------- |
| spherical CIC ``S_3``       | scalar 3rd-order amplitude  | O(N)           |
| cylindrical / oriented CIC  | + LOS / axial anisotropy    | O(N) per shape |
| skew spectra (Munshi+ 2022) | shape-dependent bispectrum  | O(N log N)     |

The 3rd-order CIC variants miss the pair-resolved scale information
that the LISA primitives give, but they carry the dominant
non-Gaussian and anisotropic signals at constant cost.

### Per-particle local inertia tensor (Hahn et al. 2007 cosmic-web)

For each particle ``i``, the symmetric tensor

```
I_i = sum_{k in shell} (x_k - x_i) (x_k - x_i)^T  /  |x_k - x_i|^2
```

is the moment-of-inertia of pair-separation directions. Its
**eigenvalues** describe local shape (sphere / pancake / filament /
void) and its **eigenvectors** the orientation. This is exactly what
``graphGP_cosmo.py`` already computes from the GP-reconstructed density
field via the Hessian (Hahn et al. 2007; Forero-Romero et al. 2009);
the per-particle pair-tensor form is the discrete-sample analog and
is what ``b_i^(j, L=2)`` captures bin by bin.

### Bottom line for "skewness with anisotropy"

If the goal is **fast 2nd-order estimators that carry shape /
anisotropy information**, the cleanest answer is per-particle pair
multipoles ``b_i^(j, L)`` for ``L = 0, 2, (4)``. Same cost as the
existing pair-count baseline; identity ``mean_i b_i^(j, L) / b_i^(j, 0)``
recovers the standard ``xi_L(r)`` multipole; verified numerically.

If the goal is **fast 3rd-order non-Gaussianity** (genuine skewness),
counts-in-cells with anisotropic cell shapes (Bouchet+ 1992) is the
cheapest path, costing ``O(N)`` per cell shape. For the per-particle
analog, our 3pt LISA remains O(N k_pair^2); skew-spectrum methods like
Munshi et al. (2022, *Mon. Not. R. Astron. Soc.* **513**, 4309) hit
``O(N log N)`` via FFT but lose the per-particle decomposition.

### Additional references

- T. Hamilton, *Measuring omega and the real correlation function from
  the redshift correlation function*, ApJ **385**, L5 (1992):
  introduces the multipole decomposition of xi(r, mu).
- O. Hahn, C. M. Carollo, C. Porciani, A. Dekel, *The properties of
  cosmic web environments*, MNRAS **375**, 489 (2007).
- D. Munshi, R. Lee, et al., *Skew-spectra for galaxy
  clustering*, arXiv:2107.10765, 2107.13533.
- F. Bouchet, R. Schaeffer, M. Davis, *Skewness, variance and 3-point
  function*, ApJ **383**, 19 (1991): CIC moments + skewness.
- J. Forero-Romero et al., *Cosmic web classification*, MNRAS **396**,
  1815 (2009).


## Trace-tidal split: per-particle local tensors from pair counts

The natural unification of the multipole and Hahn-style cosmic-web
constructions: for each particle ``i`` and bin ``j`` build the
symmetric ``3x3`` pair tensor

```
T_i^(j)_{ab} = sum_{k != i, r_ik in B_j}  n_ik^a * n_ik^b,     n_ik = (x_k - x_i)/|...|
```

Trace-tidal decomposition:

```
Tr(T_i^(j))                 = b_i^(j)            scalar pair count = density (LISA L=0)
T_i^(j) - Tr(T_i^(j))/3 * I = traceless tensor   local tidal / anisotropy at scale r_j
```

The trace recovers the existing 2pt LISA primitive. The traceless part
encodes the L=2 angular structure as a full ``3x3`` tensor (six
independent components) -- containing the same information as the three
multipoles ``b_i^(j, L=2)`` along three orthogonal axes, but in a
coordinate-free form.

### Eigendecomposition: Hahn-style cosmic-web class per particle, per scale

The eigenvalue *pattern* of the traceless tensor classifies the local
cosmic-web type without requiring a continuous density-field
reconstruction:

| eigenvalue pattern (traceless, sum = 0)         | geometry          |
| ----------------------------------------------- | ----------------- |
| 1 large ``+``, 2 ``~equal -``                   | filament          |
| 1 large ``-``, 2 ``~equal +``                   | sheet / pancake   |
| 3 distinct (``+``, ``~0``, ``-``)               | triaxial          |
| all near 0                                      | isotropic         |

This is the discrete-sample analog of the smoothed-Hessian
classification of Hahn et al. (2007) and Forero-Romero et al. (2009),
but computed **directly from per-particle pair counts** at
``O(N * k_pair)``. No density-field smoothing or Hessian computation
required.

Numerically verified in ``demos/demo_pair_tensor.py``: on three toy
catalogs with the same density profile but different blob shapes,

```
catalog              mean traceless eigenvalues (j ~ 10 Mpc)    interpretation
-------------------  ----------------------------------------    --------------
isotropic blobs      (-0.005, +0.002, +0.003)                    no preferred axis
z-pancakes (sheets)  (-0.104, +0.049, +0.055)                    sheet  along z (eigvec)
z-filaments          (-0.154, -0.126, +0.279)                    filament along z
```

The ``z`` direction lights up unambiguously in both anisotropic
geometries with the correct sign pattern.

### Connection to the existing GP-Hessian classifier

``graphGP_cosmo.py`` computes a per-halo Hessian of the GP-reconstructed
density field and classifies cosmic-web environment from its eigenvalues.
That construction needs the full GP regression first; the per-particle
pair tensor here gets the same eigenvalue structure directly from
``query_pairs`` plus a 3x3 outer-product accumulation. Same physics,
~100x cheaper, and trivially scale-resolved (one tensor per bin).

### Multi-scale tidal vector field per particle

Stacking T_traceless across bins gives a ``(N_D, n_bins, 3, 3)`` array
-- a per-particle, multi-scale "tidal-tensor field". Each entry is six
free numbers, so the storage is moderate (``6 N_D n_bins`` floats:
``~7 MB`` for ``N_D = 5000`` and ``n_bins = 30``). With this in hand:

- Eigenvector orientation as a function of scale tracks how the local
  filament/sheet axis evolves between scales (a "tidal renormalisation
  flow" per particle).
- Frobenius norm ``||T_traceless^(j)||_F`` measures the magnitude of
  anisotropy at each scale, naturally complementing the scalar
  overdensity ``delta_i^(j)``.
- Comparison to galaxy spins / shapes / angular momenta gives a
  per-particle intrinsic-alignment statistic (Lee & Pen 2002; Codis et
  al. 2018) -- but with the tidal axes derived without any continuous
  field reconstruction.

### LISA aggregation including tidal information

The scalar weight per particle generalises naturally to a
density-plus-tidal combination

```
w_i = 1 + a * delta_i + b * Tr(M T_traceless^i)
```

with ``M`` a fixed 3x3 mask tensor selecting a preferred direction
(LOS for redshift-space, intrinsic axis for IA studies, etc.) and
``b`` a scalar tuning the relative weight of the anisotropy. This
gives weights that distinguish two equally-overdense particles by
the orientation of their local clustering -- something the scalar
LISA cannot do.

### What's new vs the literature

- **Hahn+ 2007 / Forero-Romero+ 2009**: tidal tensor from smoothed
  density Hessian. Continuous-field construction; one tensor per Eulerian
  cell, one smoothing scale.
- **Pichon-Codis et al.**: tidal tensor from a Voronoi tessellation +
  potential reconstruction. Discrete but expensive.
- **Schmittfull et al. (FAST-PT)**: bispectrum decomposition into
  tidal-shear-density operators. Global, not per-particle.

The per-particle pair tensor T_i^(j) appears not to have been written
down in this exact form. It is the natural meeting point of the LISA
framework (per-event statistic that aggregates to a global summary)
and Hahn-style tidal classification (eigenvalues of a local tensor
classify cosmic web). Same cost as a scalar pair count.

### Additional references

- B. Pichon, C. Codis, et al., *Spin alignment of dark matter halos
  in filaments and walls*, MNRAS **427**, 3320 (2012).
- M. Schmittfull, T. Baldauf, M. Zaldarriaga, *Iterative initial
  condition reconstruction*, PRD **96**, 023505 (2017).
- J. Lee and U.-L. Pen, *Theory of intrinsic galaxy spin alignments*,
  ApJ **567**, L111 (2002).
- S. Codis et al., *Spin alignment of dark matter haloes around
  cosmic filaments*, MNRAS **481**, 4753 (2018).


## References (URLs)

- Getis & Franklin (1987): https://esajournals.onlinelibrary.wiley.com/doi/10.2307/1938452
- Anselin (1995): https://onlinelibrary.wiley.com/doi/10.1111/j.1538-4632.1995.tb00338.x
- Cressie & Collins (2001): https://link.springer.com/article/10.1198/108571101300325292
- Davis & Peebles (1983): https://ui.adsabs.harvard.edu/abs/1983ApJ...267..465D/abstract
- Landy & Szalay (1993): https://ui.adsabs.harvard.edu/abs/1993ApJ...412...64L/abstract
- Skibba, Sheth, Connolly, Scranton (2006): https://academic.oup.com/mnras/article/369/1/68/1051594
- Schaap & van de Weygaert (2000): https://www.astro.rug.nl/~weygaert/tim1publication/dtfeaaletter.pdf
- Banerjee & Abel (2021): https://academic.oup.com/mnras/article/500/4/5479/5996202
- Feldman, Kaiser, Peacock (1994): https://ui.adsabs.harvard.edu/abs/1994ApJ...426...23F/abstract
- Shaw & Møller (2021), intensity-reweighted: https://arxiv.org/abs/2004.00527
- ``spatstat`` ``localK``/``getis`` documentation: https://rdrr.io/cran/spatstat.explore/man/localK.html
