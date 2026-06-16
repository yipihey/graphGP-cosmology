"""2pt-aware density estimation and lightcone-native clustering primitives.

Modules
-------
ls_corrfunc       Pair counts and LS xi(r) via Corrfunc.
basis             Cubic-spline / Bessel / compensated-bandpass bases.
basis_projection  Project Corrfunc fine-bin counts onto a basis.
weights_binned    Part I: binned Wiener-filter weights.
weights_basis     Part II: basis-form Wiener-filter weights.
weights_graphgp   Part III: Vecchia-approximate GP weights.
validate          Weighted-DD re-pairing check.
knn_cdf           Joint angular kNN-CDF P_>=k(theta; z_q, z_n) primitive
                  (lightcone_native_v3.pdf, T. Abel 2026).
"""

from .ls_corrfunc import xi_landy_szalay, local_mean_density
from .knn_cdf import joint_knn_cdf, KnnCdfResult
from .twoMRS import TwoMRSCatalog, load_2mrs, make_mock_2mrs
from .cf4 import CF4Catalog, load_cf4, make_mock_cf4
from .boss import BOSSCatalog, load_boss, make_mock_boss
from .density_field import DensityFieldResult, sample_posterior_density_field
from .weight_budget import WeightBudget, weight_budget
from .knn_derived import (
    mean_count, cic_pmf, cic_moments, sigma2_clust,
    rsd_first_moment, xi_dp, xi_ls, xi_diag, jackknife_cov,
)
from .knn_analytic_rr import (
    random_queries_from_selection_function, analytic_rr_cube,
)
