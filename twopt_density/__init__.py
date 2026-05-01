"""2pt-aware density estimation.

Per the plan in ``IMPLEMENTATION_PLAN.md`` and the source document
``twopt_density.pdf``: build per-point density weights ``{w_i}`` whose
weighted-DD pair sum reproduces the Landy-Szalay two-point estimator.

Modules
-------
ls_corrfunc       Pair counts and LS xi(r) via Corrfunc.
basis             Cubic-spline / Bessel / compensated-bandpass bases.
basis_projection  Project Corrfunc fine-bin counts onto a basis.
weights_binned    Part I: binned Wiener-filter weights.
weights_basis     Part II: basis-form Wiener-filter weights.
weights_graphgp   Part III: Vecchia-approximate GP weights.
validate          Weighted-DD re-pairing check.
"""

from .ls_corrfunc import xi_landy_szalay, local_mean_density
