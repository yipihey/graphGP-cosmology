"""Per-galaxy weight budget decomposition.

Answers "what fraction of each galaxy's total weight correction comes from
each observational effect?", e.g. fiber collisions, dust attenuation,
angular incompleteness, FKP statistical down-weighting, and the intrinsic
density field.

Usage::

    from twopt_density.weight_budget import weight_budget

    cat = load_boss(...)           # BOSSCatalog with individual components
    result = sample_posterior_density_field(cat, ...)
    budget = weight_budget(cat, result)

    print(budget.summary())
    # mean log-fractions:
    #   density        0.12 ± 0.04  (FKP density field 1+δ)
    #   fiber_collision 0.35 ± 0.18  (WEIGHT_CP)
    #   dust_imaging    0.29 ± 0.11  (WEIGHT_SYSTOT)
    #   redshift_failure 0.18 ± 0.06 (WEIGHT_NOZ)
    #   fkp_statistical  0.06 ± 0.02 (WEIGHT_FKP)

    # Per-galaxy log-fractions as dict of arrays:
    fracs = budget.log_fractions()

Decomposition convention
------------------------
All contributions are multiplicative: w_total = Π_k w_k.
The "weight budget" decomposes log(w_total) = Σ_k log(w_k) into
positive contributions (each log(w_k) is taken as |log(w_k)| if signed).

The sign is preserved in ``budget.log_deltas`` (log(w_k)); the
``log_fractions`` method normalises the absolute values to sum to 1.
This handles cases where a component is < 1 (e.g. FKP down-weights dense
regions relative to their shot-noise level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class WeightBudget:
    """Per-galaxy weight budget for one survey + posterior density field.

    ``components`` maps a human-readable effect name to a ``(N_D,)`` array
    of per-galaxy multiplicative weights from that effect alone.

    ``log_fractions()`` returns the fraction of |log(w_total)| attributable
    to each component, so the numbers sum to 1 per galaxy.
    """
    components: Dict[str, np.ndarray]    # name → (N_D,) weight arrays (>0)
    N_data: int

    # ── derived ──────────────────────────────────────────────────────────

    def log_deltas(self) -> Dict[str, np.ndarray]:
        """Signed log(w_k) for each component — the additive budget."""
        return {k: np.log(np.clip(v, 1e-30, None)) for k, v in self.components.items()}

    def log_fractions(self) -> Dict[str, np.ndarray]:
        """Fraction of |log(w_total)| from each component, per galaxy.

        Values are non-negative and sum to 1 per galaxy.
        """
        ld = self.log_deltas()
        abs_ld = {k: np.abs(v) for k, v in ld.items()}
        total = sum(abs_ld.values()) + 1e-30
        return {k: v / total for k, v in abs_ld.items()}

    def w_total(self) -> np.ndarray:
        """Product of all components — the total per-galaxy weight."""
        out = np.ones(self.N_data)
        for v in self.components.values():
            out = out * v
        return out

    def summary(self, percentiles: tuple = (16, 50, 84)) -> str:
        """Print table of mean log-fraction per component.

        Returns a formatted string with columns:
            effect | mean log-frac | std | pXX% values
        """
        fracs = self.log_fractions()
        lines = ["Per-galaxy weight budget decomposition",
                 "=" * 60,
                 f"{'Effect':<22}  {'mean':>6}  {'std':>6}  "
                 f"  {'p16':>6}  {'p50':>6}  {'p84':>6}"]
        lines.append("-" * 60)
        for name, f in sorted(fracs.items(), key=lambda x: -x[1].mean()):
            m = f.mean()
            s = f.std()
            pvals = np.percentile(f, list(percentiles))
            lines.append(
                f"{name:<22}  {m:6.3f}  {s:6.3f}  "
                f"  {pvals[0]:6.3f}  {pvals[1]:6.3f}  {pvals[2]:6.3f}"
            )
        lines.append("-" * 60)
        lines.append(f"{'w_total (mean / std)':<22}  "
                     f"{self.w_total().mean():6.3f}  "
                     f"{self.w_total().std():6.3f}")
        return "\n".join(lines)

    def table(self) -> dict:
        """Return summary stats as a plain dict (for DataFrames / JSON)."""
        fracs = self.log_fractions()
        out = {}
        for name, f in fracs.items():
            p16, p50, p84 = np.percentile(f, [16, 50, 84])
            out[name] = {
                "mean_frac": float(f.mean()),
                "std_frac": float(f.std()),
                "p16": float(p16), "p50": float(p50), "p84": float(p84),
                "mean_w": float(self.components[name].mean()),
            }
        return out


# ──────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────

def weight_budget(
    catalog,
    density_result,
    *,
    extra: Optional[Dict[str, np.ndarray]] = None,
) -> WeightBudget:
    """Build a WeightBudget from a catalog + DensityFieldResult.

    Automatically extracts known observational components for each survey
    type, then adds the density-field contribution (1+δ from FKP estimator).

    Parameters
    ----------
    catalog
        TwoMRSCatalog, CF4Catalog, BOSSCatalog, or any catalog with sel_map.
    density_result
        DensityFieldResult from sample_posterior_density_field.
    extra
        Optional dict of additional named weight arrays to include.

    Returns
    -------
    WeightBudget
    """
    N = catalog.N_data
    comps: Dict[str, np.ndarray] = {}

    # ── 1. Density field (1+δ_FKP) — common to all surveys ───────────
    w_density = density_result.data_weights()   # posterior mean 1+δ
    comps["density_field"] = np.clip(w_density, 1e-6, None)

    # ── 2. Angular completeness — from the HealPIX selection map ─────
    sel_map = getattr(catalog, "sel_map", None)
    nside   = getattr(catalog, "nside", 64)
    if sel_map is not None and len(sel_map) > 0:
        try:
            import healpy as hp
            theta = np.radians(90.0 - catalog.dec_data)
            phi   = np.radians(catalog.ra_data)
            ipix  = hp.ang2pix(nside, theta, phi)
            w_comp = np.clip(sel_map[ipix], 1e-6, 1.0)
            comps["angular_completeness"] = w_comp
        except ImportError:
            pass

    # ── 3. BOSS-specific components ───────────────────────────────────
    from .boss import BOSSCatalog
    if isinstance(catalog, BOSSCatalog):
        if catalog.w_sys_data is not None:
            comps["dust_imaging"]       = np.clip(catalog.w_sys_data, 1e-6, None)
        if catalog.w_noz_data is not None:
            comps["redshift_failure"]   = np.clip(catalog.w_noz_data, 1e-6, None)
        if catalog.w_cp_data is not None:
            comps["fiber_collision"]    = np.clip(catalog.w_cp_data,  1e-6, None)
        if catalog.w_fkp_data is not None:
            comps["fkp_statistical"]    = np.clip(catalog.w_fkp_data, 1e-6, None)

    # ── 4. 2MRS-specific components ───────────────────────────────────
    from .twoMRS import TwoMRSCatalog
    if isinstance(catalog, TwoMRSCatalog):
        if catalog.ebv_data is not None:
            # K-band dust: A_K ≈ 0.31 × E(B-V)  (Cardelli et al. 1989)
            # Dust attenuates flux → galaxy missing if near K_s limit.
            # Completeness correction weight: 1/10^(0.4 × A_K) i.e. weight
            # inflates galaxies that were harder to detect through dust.
            A_K = 0.31 * np.clip(catalog.ebv_data, 0.0, None)
            comps["dust_attenuation"] = 1.0 / np.clip(
                10.0 ** (0.4 * A_K), 1e-6, None
            )
        if catalog.w_data is not None and not np.allclose(catalog.w_data, 1.0):
            comps["ks_completeness"] = np.clip(catalog.w_data, 1e-6, None)

    # ── 5. CF4-specific components ────────────────────────────────────
    from .cf4 import CF4Catalog
    if isinstance(catalog, CF4Catalog):
        if catalog.sigma_mu_data is not None:
            # Optimal PV weight: w ∝ 1/σ_μ² (Hamilton 1997 / Watkins et al.)
            # Normalised to mean = 1 so it represents relative up/down-weighting.
            sigma_safe = np.clip(catalog.sigma_mu_data, 0.01, None)
            w_pv = 1.0 / sigma_safe ** 2
            comps["peculiar_velocity_precision"] = w_pv / w_pv.mean()

    # ── 6. Any user-supplied extras ───────────────────────────────────
    if extra:
        comps.update(extra)

    return WeightBudget(components=comps, N_data=N)
