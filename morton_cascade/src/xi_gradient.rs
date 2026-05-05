// xi_gradient.rs
//
// Backward-mode gradient of pair-count statistics (DD, DR, ξ_LS) with
// respect to per-particle data weights. For each cascade shell ℓ and
// each data particle i, this computes ∂DD_ℓ/∂w_i^d, ∂DR_ℓ/∂w_i^d,
// and ∂ξ_ℓ/∂w_i^d.
//
// Math derivation (commit 11 design notes):
//
// Per-cell weighted pair count.
//   P_d(c) = (W_d(c)² − Σ_{i∈c} (w_i^d)²) / 2
//
// (This is the canonical weighted self-pair count, equal to
// Σ_{i<j∈c} w_i w_j. The cascade accumulates this directly; see
// hier_bitvec_pair.rs:863.)
//
// Cumulative cell sum at level ℓ.
//   cum_dd(ℓ) = Σ_{c∈C_ℓ} P_d(c)
//   cum_dr(ℓ) = Σ_{c∈C_ℓ} W_d(c) · W_r(c)
//
// Per-particle cumulative gradient (i in cell c at level ℓ):
//   ∂P_d(c)/∂w_i^d = W_d(c) − w_i^d
//   ∂(W_d·W_r)(c)/∂w_i^d = W_r(c)
//
// Shell pair count is the difference of consecutive cumulative levels:
//   DD_ℓ = cum_dd(ℓ) − cum_dd(ℓ+1)
//   DR_ℓ = cum_dr(ℓ) − cum_dr(ℓ+1)
//
// So the per-particle gradient at shell ℓ:
//   ∂DD_ℓ/∂w_i^d = (W_d(c_i^(ℓ)) − w_i^d) − (W_d(c_i^(ℓ+1)) − w_i^d)
//                = W_d(c_i^(ℓ)) − W_d(c_i^(ℓ+1))
//   ∂DR_ℓ/∂w_i^d = W_r(c_i^(ℓ)) − W_r(c_i^(ℓ+1))
//
// (For the deepest shell with no ℓ+1: just the first term.)
//
// Geometric meaning: ∂DD_ℓ/∂w_i^d is the count of pairs that include
// particle i at separation in shell ℓ — i's level-ℓ-cell partners minus
// its level-(ℓ+1)-cell partners (the latter are at finer separations).
//
// Landy-Szalay ξ.
//   ξ_ℓ = (DD_ℓ/N_DD − 2 DR_ℓ/N_DR + RR_ℓ/N_RR) / (RR_ℓ/N_RR)
// with global normalizations
//   N_DD = (W_d_total² − Σ_i (w_i^d)²) / 2
//   N_DR = W_d_total · W_r_total
//   N_RR = (W_r_total² − Σ_j (w_j^r)²) / 2  [does not depend on w^d]
//
// Per-particle ∂N_DD/∂w_i^d = W_d_total − w_i^d
// Per-particle ∂N_DR/∂w_i^d = W_r_total
// Per-particle ∂RR_ℓ/∂w_i^d = 0  (depends only on randoms)
//
// Define f_DD = DD_ℓ/N_DD, f_DR = DR_ℓ/N_DR, f_RR = RR_ℓ/N_RR. Then:
//   ∂ξ_ℓ/∂w_i^d = (1/f_RR) · [ ∂f_DD/∂w_i^d − 2 ∂f_DR/∂w_i^d ]
// where
//   ∂f_DD/∂w_i^d = (∂DD_ℓ/∂w_i^d − f_DD · (W_d_total − w_i^d)) / N_DD
//   ∂f_DR/∂w_i^d = (∂DR_ℓ/∂w_i^d − f_DR · W_r_total) / N_DR
//
// Implementation cost: O(N_d · L_max) for the per-particle walk —
// same order as one forward pair-count pass.

use crate::cell_membership::{CellMembership, WhichCatalog};
use crate::hier_bitvec_pair::{BitVecCascadePair, PairLevelStats, XiShell};

/// Per-particle gradients of DD, DR, and ξ at every cascade shell
/// with respect to the data-catalog weights.
///
/// Indexed `*_grads[shell][particle_idx]`. Each shell corresponds to
/// one entry in the input pair-count statistics (matching the order
/// of [`BitVecCascadePair::xi_landy_szalay`]'s output).
#[derive(Clone, Debug)]
pub struct XiGradient {
    /// `dd_grads[shell][i]` = ∂DD_shell / ∂w_i^d.
    pub dd_grads: Vec<Vec<f64>>,
    /// `dr_grads[shell][i]` = ∂DR_shell / ∂w_i^d.
    pub dr_grads: Vec<Vec<f64>>,
    /// `xi_grads[shell][i]` = ∂ξ_shell / ∂w_i^d. Empty for shells
    /// where ξ is undefined (RR_ℓ = 0 or N_RR = 0).
    pub xi_grads: Vec<Vec<f64>>,
}

impl<const D: usize> BitVecCascadePair<D> {
    /// Per-particle gradients of DD, DR, and ξ at every cascade shell
    /// with respect to data-catalog weights.
    ///
    /// Requires the pair-count statistics from `analyze` and the
    /// shell-level ξ_LS results from `xi_landy_szalay`.
    ///
    /// Cost: O(N_d · L_max). Memory: O(N_d · L_max).
    pub fn gradient_xi_data_all_shells(
        &self,
        #[allow(unused_variables)]
        stats: &[PairLevelStats],
        shells: &[XiShell],
    ) -> XiGradient {
        let n_d = self.n_d();
        let n_shells = shells.len();

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        // Pre-compute per-particle (level, W_d-of-cell, W_r-of-cell)
        // table once. For each level ℓ in 0..n_levels, we need the
        // W_d and W_r of the cell containing each particle. Since the
        // membership index iterates non-empty cells, we walk those
        // and credit the W_d/W_r to the contained particles.
        let n_levels = self.l_max() + 1;  // includes root level 0
        let mut per_particle_wd = vec![vec![0.0_f64; n_d]; n_levels];
        let mut per_particle_wr = vec![vec![0.0_f64; n_d]; n_levels];
        for level in 0..n_levels {
            // Data-side walk: per-particle W_d at this level.
            for (cell_id, members_d) in mem_d.non_empty_cells_at(level) {
                let w_d_c: f64 = match self.weights_d() {
                    Some(w) => members_d.iter().map(|&i| w[i as usize]).sum(),
                    None => members_d.len() as f64,
                };
                // Look up matching random cell to get W_r at this level.
                let members_r = mem_r.members(level, cell_id);
                let w_r_c: f64 = match self.weights_r() {
                    Some(w) => members_r.iter().map(|&i| w[i as usize]).sum(),
                    None => members_r.len() as f64,
                };
                for &i in members_d {
                    per_particle_wd[level][i as usize] = w_d_c;
                    per_particle_wr[level][i as usize] = w_r_c;
                }
            }
        }

        // DD/DR gradients at each shell.
        // The cascade's xi_landy_szalay defines shell at level ℓ as
        // pairs sharing a level-ℓ cell but NOT the level-(ℓ+1) cell.
        // PairLevelStats indexes l by stats[l].level. Match shells to
        // stats by level.
        let mut dd_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
        let mut dr_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
        for shell in shells {
            let l = shell.level;
            let mut dd_g = vec![0.0_f64; n_d];
            let mut dr_g = vec![0.0_f64; n_d];
            // Coarse contribution: W_d(c_i^(l)) and W_r(c_i^(l)).
            if l < n_levels {
                for i in 0..n_d {
                    dd_g[i] = per_particle_wd[l][i];
                    dr_g[i] = per_particle_wr[l][i];
                }
            }
            // Subtract finer contribution if shell isn't the deepest.
            let l_finer = l + 1;
            if l_finer < n_levels {
                for i in 0..n_d {
                    dd_g[i] -= per_particle_wd[l_finer][i];
                    dr_g[i] -= per_particle_wr[l_finer][i];
                }
            }
            dd_grads.push(dd_g);
            dr_grads.push(dr_g);
        }

        // ξ gradient via chain rule.
        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        // N_DD = (W_d² − Σw_d²) / 2
        let sum_w_d_sq: f64 = match self.weights_d() {
            Some(w) => w.iter().map(|&v| v * v).sum(),
            None => self.n_d() as f64,
        };
        let n_dd_norm = if total_w_d > 0.0 {
            0.5 * (total_w_d * total_w_d - sum_w_d_sq)
        } else { 0.0 };
        let n_dr_norm = total_w_d * total_w_r;
        // N_RR = (W_r² − Σw_r²) / 2 — used as f_RR's normalizer.
        let sum_w_r_sq: f64 = match self.weights_r() {
            Some(w) => w.iter().map(|&v| v * v).sum(),
            None => self.n_r() as f64,
        };
        let n_rr_norm = if total_w_r > 0.0 {
            0.5 * (total_w_r * total_w_r - sum_w_r_sq)
        } else { 0.0 };

        let mut xi_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
        for (s_idx, shell) in shells.iter().enumerate() {
            let dd = shell.dd;
            let dr = shell.dr;
            let rr = shell.rr;
            // ξ undefined if RR == 0 or any normalization is zero.
            if rr <= 0.0 || n_rr_norm <= 0.0 || n_dd_norm <= 0.0 || n_dr_norm <= 0.0 {
                xi_grads.push(Vec::new());
                continue;
            }
            let f_dd = dd / n_dd_norm;
            let f_dr = dr / n_dr_norm;
            let f_rr = rr / n_rr_norm;
            let inv_f_rr = 1.0 / f_rr;
            // ∂f_DD/∂w_i^d = (∂DD/∂w_i^d − f_DD · (W_d_total − w_i^d)) / N_DD
            // ∂f_DR/∂w_i^d = (∂DR/∂w_i^d − f_DR · W_r_total) / N_DR
            let mut g = vec![0.0_f64; n_d];
            for i in 0..n_d {
                let wi = match self.weights_d() {
                    Some(w) => w[i],
                    None => 1.0,
                };
                let d_dd = dd_grads[s_idx][i];
                let d_dr = dr_grads[s_idx][i];
                let d_f_dd = (d_dd - f_dd * (total_w_d - wi)) / n_dd_norm;
                let d_f_dr = (d_dr - f_dr * total_w_r) / n_dr_norm;
                g[i] = inv_f_rr * (d_f_dd - 2.0 * d_f_dr);
            }
            xi_grads.push(g);
        }

        XiGradient { dd_grads, dr_grads, xi_grads }
    }

    /// Aggregate-scalar gradient for ξ: given per-shell betas, return
    /// `∂L / ∂w_i^d` for the scalar loss `L = Σ_ℓ β_ℓ · ξ_ℓ`.
    ///
    /// Shells where ξ is undefined contribute 0 regardless of β.
    pub fn gradient_xi_data_aggregate(
        &self,
        stats: &[PairLevelStats],
        shells: &[XiShell],
        betas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(betas.len(), shells.len(),
            "betas length {} != shells length {}", betas.len(), shells.len());
        let full = self.gradient_xi_data_all_shells(stats, shells);
        let n_d = self.n_d();
        let mut out = vec![0.0_f64; n_d];
        for (s, &b) in betas.iter().enumerate() {
            if b == 0.0 { continue; }
            if full.xi_grads[s].is_empty() { continue; }
            for (i, &g) in full.xi_grads[s].iter().enumerate() {
                out[i] += b * g;
            }
        }
        out
    }
}

/// Per-particle gradients of DR, RR, and ξ at every cascade shell
/// with respect to the **random-catalog** weights.
///
/// Indexed `*_grads[shell][random_particle_idx]`. DD is omitted from
/// this struct because `∂DD_ℓ / ∂w_j^r ≡ 0` (DD depends only on data
/// weights).
#[derive(Clone, Debug)]
pub struct XiRandomGradient {
    /// `dr_grads[shell][j]` = ∂DR_shell / ∂w_j^r.
    pub dr_grads: Vec<Vec<f64>>,
    /// `rr_grads[shell][j]` = ∂RR_shell / ∂w_j^r.
    pub rr_grads: Vec<Vec<f64>>,
    /// `xi_grads[shell][j]` = ∂ξ_shell / ∂w_j^r. Empty for shells
    /// where ξ is undefined (RR_ℓ = 0 or N_RR = 0).
    pub xi_grads: Vec<Vec<f64>>,
}

impl<const D: usize> BitVecCascadePair<D> {
    /// Per-particle gradients of DR, RR, and ξ at every cascade shell
    /// with respect to **random-catalog** weights.
    ///
    /// Math (see `docs/differentiable_cascade.md` §6.11):
    ///
    /// ```text
    ///   ∂DR_ℓ / ∂w_j^r = W_d(c_j^(ℓ)) − W_d(c_j^(ℓ+1))
    ///   ∂RR_ℓ / ∂w_j^r = W_r(c_j^(ℓ)) − W_r(c_j^(ℓ+1))
    ///   ∂ξ_ℓ  / ∂w_j^r via chain rule through f_DD = DD/N_DD,
    ///                                          f_DR = DR/N_DR,
    ///                                          f_RR = RR/N_RR
    /// ```
    ///
    /// (DD random-weight gradient is identically zero and not returned.)
    ///
    /// Cost: O(N_r · L_max). Memory: O(N_r · L_max) for all_shells.
    pub fn gradient_xi_random_all_shells(
        &self,
        #[allow(unused_variables)]
        stats: &[PairLevelStats],
        shells: &[XiShell],
    ) -> XiRandomGradient {
        let n_r = self.n_r();
        let n_shells = shells.len();

        let mem_d = CellMembership::build(self, WhichCatalog::Data);
        let mem_r = CellMembership::build(self, WhichCatalog::Randoms);

        // Pre-compute per-RANDOM-particle (W_d, W_r) at every cascade
        // level. Mirrors the data-weight pre-compute but indexed over
        // random particles.
        let n_levels = self.l_max() + 1;
        let mut per_random_wd = vec![vec![0.0_f64; n_r]; n_levels];
        let mut per_random_wr = vec![vec![0.0_f64; n_r]; n_levels];
        for level in 0..n_levels {
            for (cell_id, members_r) in mem_r.non_empty_cells_at(level) {
                let w_r_c: f64 = match self.weights_r() {
                    Some(w) => members_r.iter().map(|&j| w[j as usize]).sum(),
                    None => members_r.len() as f64,
                };
                let members_d = mem_d.members(level, cell_id);
                let w_d_c: f64 = match self.weights_d() {
                    Some(w) => members_d.iter().map(|&i| w[i as usize]).sum(),
                    None => members_d.len() as f64,
                };
                for &j in members_r {
                    per_random_wd[level][j as usize] = w_d_c;
                    per_random_wr[level][j as usize] = w_r_c;
                }
            }
        }

        // Per-shell DR and RR gradients via difference of consecutive
        // cumulative levels.
        let mut dr_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
        let mut rr_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
        for shell in shells {
            let l = shell.level;
            let mut dr_g = vec![0.0_f64; n_r];
            let mut rr_g = vec![0.0_f64; n_r];
            if l < n_levels {
                for j in 0..n_r {
                    dr_g[j] = per_random_wd[l][j];
                    rr_g[j] = per_random_wr[l][j];
                }
            }
            let l_finer = l + 1;
            if l_finer < n_levels {
                for j in 0..n_r {
                    dr_g[j] -= per_random_wd[l_finer][j];
                    rr_g[j] -= per_random_wr[l_finer][j];
                }
            }
            dr_grads.push(dr_g);
            rr_grads.push(rr_g);
        }

        // ξ gradient via chain rule. ∂DD/∂w_j^r ≡ 0 and ∂N_DD/∂w_j^r ≡ 0
        // so ∂f_DD/∂w_j^r = 0 — the DD path drops out entirely.
        let total_w_d = self.sum_w_d();
        let total_w_r = self.sum_w_r();
        let sum_w_d_sq: f64 = match self.weights_d() {
            Some(w) => w.iter().map(|&v| v * v).sum(),
            None => self.n_d() as f64,
        };
        let n_dd_norm = if total_w_d > 0.0 {
            0.5 * (total_w_d * total_w_d - sum_w_d_sq)
        } else { 0.0 };
        let n_dr_norm = total_w_d * total_w_r;
        let sum_w_r_sq: f64 = match self.weights_r() {
            Some(w) => w.iter().map(|&v| v * v).sum(),
            None => self.n_r() as f64,
        };
        let n_rr_norm = if total_w_r > 0.0 {
            0.5 * (total_w_r * total_w_r - sum_w_r_sq)
        } else { 0.0 };

        let mut xi_grads: Vec<Vec<f64>> = Vec::with_capacity(n_shells);
        for (s_idx, shell) in shells.iter().enumerate() {
            let dd = shell.dd;
            let dr = shell.dr;
            let rr = shell.rr;
            if rr <= 0.0 || n_rr_norm <= 0.0 || n_dd_norm <= 0.0 || n_dr_norm <= 0.0 {
                xi_grads.push(Vec::new());
                continue;
            }
            let f_dd = dd / n_dd_norm;
            let f_dr = dr / n_dr_norm;
            let f_rr = rr / n_rr_norm;
            let inv_f_rr_sq = 1.0 / (f_rr * f_rr);
            // ξ = f_DD/f_RR − 2 f_DR/f_RR + 1
            // ∂ξ/∂w_j^r = (1/f_RR²) · [(2 f_DR − f_DD) ∂f_RR − 2 f_RR ∂f_DR]
            let mut g = vec![0.0_f64; n_r];
            for j in 0..n_r {
                let wj = match self.weights_r() {
                    Some(w) => w[j],
                    None => 1.0,
                };
                let d_dr = dr_grads[s_idx][j];
                let d_rr = rr_grads[s_idx][j];
                let d_f_dr = (d_dr - f_dr * total_w_d) / n_dr_norm;
                let d_f_rr = (d_rr - f_rr * (total_w_r - wj)) / n_rr_norm;
                g[j] = inv_f_rr_sq * ((2.0 * f_dr - f_dd) * d_f_rr - 2.0 * f_rr * d_f_dr);
            }
            xi_grads.push(g);
        }

        XiRandomGradient { dr_grads, rr_grads, xi_grads }
    }

    /// Aggregate-scalar gradient for ξ (random-weight): given per-shell
    /// betas, returns `∂L/∂w_j^r` for `L = Σ_ℓ β_ℓ · ξ_ℓ`.
    pub fn gradient_xi_random_aggregate(
        &self,
        stats: &[PairLevelStats],
        shells: &[XiShell],
        betas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(betas.len(), shells.len(),
            "betas length {} != shells length {}", betas.len(), shells.len());
        let full = self.gradient_xi_random_all_shells(stats, shells);
        let n_r = self.n_r();
        let mut out = vec![0.0_f64; n_r];
        for (s, &b) in betas.iter().enumerate() {
            if b == 0.0 { continue; }
            if full.xi_grads[s].is_empty() { continue; }
            for (j, &g) in full.xi_grads[s].iter().enumerate() {
                out[j] += b * g;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord_range::{CoordRange, TrimmedPoints};

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn make_uniform(n: usize, bits: u32, seed: u64) -> Vec<[u64; 3]> {
        let mut s = seed;
        let mask = (1u64 << bits) - 1;
        (0..n)
            .map(|_| [splitmix64(&mut s) & mask, splitmix64(&mut s) & mask, splitmix64(&mut s) & mask])
            .collect()
    }

    fn build_pair(
        pts_d: Vec<[u64; 3]>, pts_r: Vec<[u64; 3]>,
        wd: Option<Vec<f64>>, wr: Option<Vec<f64>>,
    ) -> BitVecCascadePair<3> {
        let range = CoordRange::analyze_pair(&pts_d, &pts_r);
        let td = TrimmedPoints::from_points_with_range(pts_d, range.clone());
        let tr = TrimmedPoints::from_points_with_range(pts_r, range);
        BitVecCascadePair::<3>::build_full(td, tr, wd, wr, None, 64)
    }

    #[test]
    fn dd_gradient_finite_difference_parity() {
        // FD parity for ∂DD_ℓ/∂w_i^d. Linear in w under perturbation,
        // so analytical gradient should match FD to high precision.
        let bits = 4u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform(n_d, bits, 11444);
        let pts_r = make_uniform(n_r, bits, 11555);
        let wd_base: Vec<f64> = (0..n_d).map(|i| 0.7 + 0.3 * (i as f64 / n_d as f64)).collect();
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let stats_base = pair_base.analyze();
        let shells_base = pair_base.xi_landy_szalay(&stats_base);
        let grad = pair_base.gradient_xi_data_all_shells(&stats_base, &shells_base);

        // Pick a shell with measurable DD.
        let test_shell = shells_base.iter().enumerate()
            .find(|(_, s)| s.dd > 0.5)
            .map(|(i, _)| i);
        let test_shell = match test_shell {
            Some(s) => s,
            None => return,
        };

        let eps = 1e-5;
        let dd_base = shells_base[test_shell].dd;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze();
            let shells_pert = pair_pert.xi_landy_szalay(&stats_pert);
            let dd_pert = shells_pert[test_shell].dd;
            let fd_grad = (dd_pert - dd_base) / eps;
            let an_grad = grad.dd_grads[test_shell][i];
            // DD is linear in w_i (quadratic ∝ w_i times other w's),
            // so FD truncation is small. Tolerance 1e-3 absolute.
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-3,
                "shell {} particle {}: FD {} vs analytic {} (diff {})",
                test_shell, i, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn dr_gradient_finite_difference_parity() {
        let bits = 4u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform(n_d, bits, 12444);
        let pts_r = make_uniform(n_r, bits, 12555);
        let wd_base: Vec<f64> = (0..n_d).map(|i| 0.7 + 0.3 * (i as f64 / n_d as f64)).collect();
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let stats_base = pair_base.analyze();
        let shells_base = pair_base.xi_landy_szalay(&stats_base);
        let grad = pair_base.gradient_xi_data_all_shells(&stats_base, &shells_base);

        let test_shell = shells_base.iter().enumerate()
            .find(|(_, s)| s.dr > 0.5)
            .map(|(i, _)| i);
        let test_shell = match test_shell {
            Some(s) => s,
            None => return,
        };

        let eps = 1e-5;
        let dr_base = shells_base[test_shell].dr;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze();
            let shells_pert = pair_pert.xi_landy_szalay(&stats_pert);
            let dr_pert = shells_pert[test_shell].dr;
            let fd_grad = (dr_pert - dr_base) / eps;
            let an_grad = grad.dr_grads[test_shell][i];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-3,
                "shell {} particle {}: FD {} vs analytic {} (diff {})",
                test_shell, i, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn xi_gradient_finite_difference_parity() {
        // FD parity for ∂ξ_ℓ/∂w_i^d. The chain-rule combination of
        // DD/DR derivatives + N_DD, N_DR derivatives — the most
        // composite test in this module.
        let bits = 4u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform(n_d, bits, 13444);
        let pts_r = make_uniform(n_r, bits, 13555);
        let wd_base: Vec<f64> = (0..n_d).map(|i| 0.7 + 0.3 * (i as f64 / n_d as f64)).collect();
        let wr = vec![1.0_f64; n_r];

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd_base.clone()), Some(wr.clone()));
        let stats_base = pair_base.analyze();
        let shells_base = pair_base.xi_landy_szalay(&stats_base);
        let grad = pair_base.gradient_xi_data_all_shells(&stats_base, &shells_base);

        // Pick a shell with valid ξ.
        let test_shell = shells_base.iter().enumerate()
            .find(|(_, s)| s.rr > 0.5 && s.xi_ls.is_finite())
            .map(|(i, _)| i);
        let test_shell = match test_shell {
            Some(s) => s,
            None => return,
        };

        let eps = 1e-5;
        let xi_base = shells_base[test_shell].xi_ls;

        for i in 0..n_d {
            let mut wd_pert = wd_base.clone();
            wd_pert[i] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd_pert), Some(wr.clone()));
            let stats_pert = pair_pert.analyze();
            let shells_pert = pair_pert.xi_landy_szalay(&stats_pert);
            let xi_pert = shells_pert[test_shell].xi_ls;
            let fd_grad = (xi_pert - xi_base) / eps;
            let an_grad = grad.xi_grads[test_shell][i];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "shell {} particle {}: FD {} vs analytic {} (diff {})",
                test_shell, i, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn xi_gradient_uniform_scaling_invariant() {
        // Landy-Szalay ξ is invariant under uniform scaling of all
        // data weights (DD scales as k², N_DD scales as k², so f_DD
        // is invariant; similarly f_DR invariant since k cancels).
        // ⇒ Σ_i w_i · ∂ξ_ℓ/∂w_i = 0 by chain rule.
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 14777);
        let pts_r = make_uniform(n_r, 5, 14888);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 40.0).collect();
        let pair = build_pair(pts_d, pts_r, Some(weights_d.clone()), None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_data_all_shells(&stats, &shells);

        for (s_idx, gs) in grad.xi_grads.iter().enumerate() {
            if gs.is_empty() { continue; }
            let weighted_sum: f64 = gs.iter().zip(weights_d.iter())
                .map(|(g, w)| g * w).sum();
            assert!(weighted_sum.abs() < 1e-9,
                "shell {}: Σ w_i · ∂ξ/∂w_i = {} (should be ≈ 0)",
                s_idx, weighted_sum);
        }
    }

    #[test]
    fn dd_gradient_sums_to_2_dd() {
        // For DD: Σ_i w_i · ∂DD/∂w_i = 2·DD (Euler's theorem on
        // homogeneous functions: DD is degree-2 homogeneous in {w^d}).
        // This is a strong analytic invariant.
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 15777);
        let pts_r = make_uniform(n_r, 5, 15888);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 40.0).collect();
        let pair = build_pair(pts_d, pts_r, Some(weights_d.clone()), None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_data_all_shells(&stats, &shells);

        for (s_idx, shell) in shells.iter().enumerate() {
            if shell.dd <= 0.0 { continue; }
            let weighted_sum: f64 = grad.dd_grads[s_idx].iter().zip(weights_d.iter())
                .map(|(g, w)| g * w).sum();
            let expected = 2.0 * shell.dd;
            let rel = (weighted_sum - expected).abs() / expected.max(1e-12);
            assert!(rel < 1e-9,
                "shell {}: Σ w_i · ∂DD/∂w_i = {} vs 2·DD = {} (rel diff {})",
                s_idx, weighted_sum, expected, rel);
        }
    }

    #[test]
    fn dr_gradient_sums_to_dr() {
        // For DR: Σ_i w_i · ∂DR/∂w_i = DR (degree-1 homogeneous in {w^d}).
        let n_d = 40;
        let n_r = 120;
        let pts_d = make_uniform(n_d, 5, 16777);
        let pts_r = make_uniform(n_r, 5, 16888);
        let weights_d: Vec<f64> = (0..n_d).map(|i| 0.5 + (i as f64) / 40.0).collect();
        let pair = build_pair(pts_d, pts_r, Some(weights_d.clone()), None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_data_all_shells(&stats, &shells);

        for (s_idx, shell) in shells.iter().enumerate() {
            if shell.dr <= 0.0 { continue; }
            let weighted_sum: f64 = grad.dr_grads[s_idx].iter().zip(weights_d.iter())
                .map(|(g, w)| g * w).sum();
            let expected = shell.dr;
            let rel = (weighted_sum - expected).abs() / expected.max(1e-12);
            assert!(rel < 1e-9,
                "shell {}: Σ w_i · ∂DR/∂w_i = {} vs DR = {} (rel diff {})",
                s_idx, weighted_sum, expected, rel);
        }
    }

    #[test]
    fn xi_gradient_aggregate_matches_per_shell() {
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, 5, 17777);
        let pts_r = make_uniform(n_r, 5, 17888);
        let pair = build_pair(pts_d, pts_r, None, None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_data_all_shells(&stats, &shells);

        let betas: Vec<f64> = (0..shells.len())
            .map(|s| if s % 2 == 0 { 1.0 } else { -0.5 } * (s as f64 + 1.0))
            .collect();
        let agg = pair.gradient_xi_data_aggregate(&stats, &shells, &betas);

        let mut expected = vec![0.0_f64; n_d];
        for s in 0..shells.len() {
            if grad.xi_grads[s].is_empty() { continue; }
            for (i, &g) in grad.xi_grads[s].iter().enumerate() {
                expected[i] += betas[s] * g;
            }
        }
        for i in 0..n_d {
            assert!((agg[i] - expected[i]).abs() < 1e-12,
                "particle {}: aggregate {} vs Σ β_s grad_s {}",
                i, agg[i], expected[i]);
        }
    }

    // -----------------------------------------------------------------
    // Commit 12b: ξ random-weight gradients
    // -----------------------------------------------------------------

    #[test]
    fn dr_random_gradient_finite_difference_parity() {
        // FD parity for ∂DR_ℓ/∂w_j^r. DR is symmetric in data/random
        // weights so this exercises the same machinery as the
        // data-weight DR test, just in random-weight space.
        let bits = 4u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform(n_d, bits, 28444);
        let pts_r = make_uniform(n_r, bits, 28555);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let stats_base = pair_base.analyze();
        let shells_base = pair_base.xi_landy_szalay(&stats_base);
        let grad = pair_base.gradient_xi_random_all_shells(&stats_base, &shells_base);

        let test_shell = shells_base.iter().enumerate()
            .find(|(_, s)| s.dr > 0.5)
            .map(|(i, _)| i);
        let test_shell = match test_shell {
            Some(s) => s,
            None => return,
        };

        let eps = 1e-5;
        let dr_base = shells_base[test_shell].dr;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze();
            let shells_pert = pair_pert.xi_landy_szalay(&stats_pert);
            let dr_pert = shells_pert[test_shell].dr;
            let fd_grad = (dr_pert - dr_base) / eps;
            let an_grad = grad.dr_grads[test_shell][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-3,
                "shell {} random j={}: FD {} vs analytic {} (diff {})",
                test_shell, j, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn rr_random_gradient_finite_difference_parity() {
        let bits = 4u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform(n_d, bits, 29444);
        let pts_r = make_uniform(n_r, bits, 29555);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let stats_base = pair_base.analyze();
        let shells_base = pair_base.xi_landy_szalay(&stats_base);
        let grad = pair_base.gradient_xi_random_all_shells(&stats_base, &shells_base);

        let test_shell = shells_base.iter().enumerate()
            .find(|(_, s)| s.rr > 0.5)
            .map(|(i, _)| i);
        let test_shell = match test_shell {
            Some(s) => s,
            None => return,
        };

        let eps = 1e-5;
        let rr_base = shells_base[test_shell].rr;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze();
            let shells_pert = pair_pert.xi_landy_szalay(&stats_pert);
            let rr_pert = shells_pert[test_shell].rr;
            let fd_grad = (rr_pert - rr_base) / eps;
            let an_grad = grad.rr_grads[test_shell][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 1e-3,
                "shell {} random j={}: FD {} vs analytic {} (diff {})",
                test_shell, j, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn xi_random_gradient_finite_difference_parity() {
        // FD parity for ∂ξ_ℓ/∂w_j^r. Tests the chain-rule combination
        // of DR/RR random-weight derivatives + N_DR, N_RR derivatives.
        let bits = 4u32;
        let n_d = 25;
        let n_r = 75;
        let pts_d = make_uniform(n_d, bits, 30444);
        let pts_r = make_uniform(n_r, bits, 30555);
        let wd = vec![1.0_f64; n_d];
        let wr_base: Vec<f64> = (0..n_r).map(|j| 0.7 + 0.3 * (j as f64 / n_r as f64)).collect();

        let pair_base = build_pair(pts_d.clone(), pts_r.clone(),
                                   Some(wd.clone()), Some(wr_base.clone()));
        let stats_base = pair_base.analyze();
        let shells_base = pair_base.xi_landy_szalay(&stats_base);
        let grad = pair_base.gradient_xi_random_all_shells(&stats_base, &shells_base);

        let test_shell = shells_base.iter().enumerate()
            .find(|(_, s)| s.rr > 0.5 && s.xi_ls.is_finite())
            .map(|(i, _)| i);
        let test_shell = match test_shell {
            Some(s) => s,
            None => return,
        };

        let eps = 1e-5;
        let xi_base = shells_base[test_shell].xi_ls;

        for j in 0..n_r {
            let mut wr_pert = wr_base.clone();
            wr_pert[j] += eps;
            let pair_pert = build_pair(pts_d.clone(), pts_r.clone(),
                                       Some(wd.clone()), Some(wr_pert));
            let stats_pert = pair_pert.analyze();
            let shells_pert = pair_pert.xi_landy_szalay(&stats_pert);
            let xi_pert = shells_pert[test_shell].xi_ls;
            let fd_grad = (xi_pert - xi_base) / eps;
            let an_grad = grad.xi_grads[test_shell][j];
            let abs_diff = (fd_grad - an_grad).abs();
            assert!(abs_diff < 5e-3,
                "shell {} random j={}: FD {} vs analytic {} (diff {})",
                test_shell, j, fd_grad, an_grad, abs_diff);
        }
    }

    #[test]
    fn dr_random_gradient_sums_to_dr() {
        // Euler theorem: DR is degree-1 homogeneous in {w^r}, so
        // Σ_j w_j^r · ∂DR/∂w_j^r = DR exactly.
        let n_d = 40;
        let n_r = 100;
        let pts_d = make_uniform(n_d, 5, 31777);
        let pts_r = make_uniform(n_r, 5, 31888);
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.5 + (j as f64) / 50.0).collect();
        let pair = build_pair(pts_d, pts_r, None, Some(weights_r.clone()));
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_random_all_shells(&stats, &shells);

        for (s_idx, shell) in shells.iter().enumerate() {
            if shell.dr <= 0.0 { continue; }
            let weighted_sum: f64 = grad.dr_grads[s_idx].iter().zip(weights_r.iter())
                .map(|(g, w)| g * w).sum();
            let expected = shell.dr;
            let rel = (weighted_sum - expected).abs() / expected.max(1e-12);
            assert!(rel < 1e-9,
                "shell {}: Σ w_j^r · ∂DR/∂w_j^r = {} vs DR = {} (rel diff {})",
                s_idx, weighted_sum, expected, rel);
        }
    }

    #[test]
    fn rr_random_gradient_sums_to_2_rr() {
        // Euler theorem: RR is degree-2 homogeneous in {w^r}.
        let n_d = 40;
        let n_r = 100;
        let pts_d = make_uniform(n_d, 5, 32777);
        let pts_r = make_uniform(n_r, 5, 32888);
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.5 + (j as f64) / 50.0).collect();
        let pair = build_pair(pts_d, pts_r, None, Some(weights_r.clone()));
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_random_all_shells(&stats, &shells);

        for (s_idx, shell) in shells.iter().enumerate() {
            if shell.rr <= 0.0 { continue; }
            let weighted_sum: f64 = grad.rr_grads[s_idx].iter().zip(weights_r.iter())
                .map(|(g, w)| g * w).sum();
            let expected = 2.0 * shell.rr;
            let rel = (weighted_sum - expected).abs() / expected.max(1e-12);
            assert!(rel < 1e-9,
                "shell {}: Σ w_j^r · ∂RR/∂w_j^r = {} vs 2·RR = {} (rel diff {})",
                s_idx, weighted_sum, expected, rel);
        }
    }

    #[test]
    fn xi_random_gradient_uniform_scaling_invariant() {
        // ξ is invariant under uniform scaling of random weights:
        // α → α/k, all W_r(c) → k W_r(c), DD invariant, DR scales
        // by k (cancelled by N_DR scaling by k), RR scales by k²
        // (cancelled by N_RR ≈ k² scaling). So ξ unchanged, hence
        // Σ_j w_j^r · ∂ξ/∂w_j^r = 0 exactly.
        let n_d = 40;
        let n_r = 100;
        let pts_d = make_uniform(n_d, 5, 33777);
        let pts_r = make_uniform(n_r, 5, 33888);
        let weights_r: Vec<f64> = (0..n_r).map(|j| 0.5 + (j as f64) / 50.0).collect();
        let pair = build_pair(pts_d, pts_r, None, Some(weights_r.clone()));
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_random_all_shells(&stats, &shells);

        for (s_idx, gs) in grad.xi_grads.iter().enumerate() {
            if gs.is_empty() { continue; }
            let weighted_sum: f64 = gs.iter().zip(weights_r.iter())
                .map(|(g, w)| g * w).sum();
            assert!(weighted_sum.abs() < 1e-9,
                "shell {}: Σ w_j^r · ∂ξ/∂w_j^r = {} (should be ≈ 0)",
                s_idx, weighted_sum);
        }
    }

    #[test]
    fn xi_random_aggregate_matches_per_shell() {
        let n_d = 30;
        let n_r = 90;
        let pts_d = make_uniform(n_d, 5, 34777);
        let pts_r = make_uniform(n_r, 5, 34888);
        let pair = build_pair(pts_d, pts_r, None, None);
        let stats = pair.analyze();
        let shells = pair.xi_landy_szalay(&stats);
        let grad = pair.gradient_xi_random_all_shells(&stats, &shells);

        let betas: Vec<f64> = (0..shells.len())
            .map(|s| if s % 2 == 0 { 1.0 } else { -0.5 } * (s as f64 + 1.0))
            .collect();
        let agg = pair.gradient_xi_random_aggregate(&stats, &shells, &betas);

        let mut expected = vec![0.0_f64; n_r];
        for s in 0..shells.len() {
            if grad.xi_grads[s].is_empty() { continue; }
            for (j, &g) in grad.xi_grads[s].iter().enumerate() {
                expected[j] += betas[s] * g;
            }
        }
        for j in 0..n_r {
            assert!((agg[j] - expected[j]).abs() < 1e-12,
                "random j={}: aggregate {} vs Σ β_s grad_s {}",
                j, agg[j], expected[j]);
        }
    }
}
