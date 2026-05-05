// compensated_sum.rs
//
// Neumaier compensated summation. A drop-in for f64 accumulators
// that recovers ~full f64 precision regardless of summation order
// or magnitude spread, at the cost of ~4× the floating-point
// operations per add.
//
// Use case in this crate: per-cell weighted sums in cascade analysis.
// When per-particle weights span wide dynamic range (e.g., FKP weights
// from ~1 to ~1e4 in the same catalog), the naive `let mut s = 0.0;
// for w in weights { s += w; }` accumulator loses ~log10(max/min) digits
// of precision per add. Neumaier defends against this with no
// algorithmic restructuring of the caller — just swap `f64` for
// `CompensatedSum` where the accumulator lives.

/// Neumaier compensated sum. Maintains a running total plus a
/// compensation term that captures bits lost at each addition.
/// Final result is recovered by `.value()`.
///
/// Compared to Kahan summation, Neumaier additionally handles the case
/// where `|x| > |sum|` (large term added to small accumulator), so it
/// is robust to ANY summation order.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompensatedSum {
    sum: f64,
    compensation: f64,
}

impl CompensatedSum {
    /// Construct a new accumulator initialized to zero.
    #[inline]
    pub const fn new() -> Self {
        Self { sum: 0.0, compensation: 0.0 }
    }

    /// Construct from an initial value.
    #[inline]
    pub const fn from_value(v: f64) -> Self {
        Self { sum: v, compensation: 0.0 }
    }

    /// Add `x` to the accumulator. Constant-time, ~4 flops.
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;
        // Capture the bits lost: depends on which of |sum|, |x| is larger.
        if self.sum.abs() >= x.abs() {
            // sum dominates: x's low bits are lost in t = sum + x
            self.compensation += (self.sum - t) + x;
        } else {
            // x dominates: sum's low bits are lost in t = sum + x
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// Recover the compensated value: sum + accumulated compensation.
    #[inline]
    pub fn value(&self) -> f64 {
        self.sum + self.compensation
    }

    /// Reset to zero in-place.
    #[inline]
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
    }
}

impl std::ops::AddAssign<f64> for CompensatedSum {
    #[inline]
    fn add_assign(&mut self, x: f64) {
        self.add(x);
    }
}

impl From<CompensatedSum> for f64 {
    #[inline]
    fn from(c: CompensatedSum) -> f64 {
        c.value()
    }
}

/// Sum a slice of f64 values with Neumaier compensation.
/// Convenience wrapper around `CompensatedSum`.
#[inline]
pub fn neumaier_sum(xs: &[f64]) -> f64 {
    let mut s = CompensatedSum::new();
    for &x in xs { s.add(x); }
    s.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sum_is_zero() {
        let s = CompensatedSum::new();
        assert_eq!(s.value(), 0.0);
        assert_eq!(neumaier_sum(&[]), 0.0);
    }

    #[test]
    fn single_value_round_trips() {
        let s = CompensatedSum::from_value(3.14159);
        assert_eq!(s.value(), 3.14159);
    }

    #[test]
    fn agrees_with_naive_for_benign_input() {
        // Same-order-of-magnitude positive values: naive and Neumaier
        // agree to f64 precision.
        let xs: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.001).collect();
        let naive: f64 = xs.iter().sum();
        let neumaier = neumaier_sum(&xs);
        assert!((naive - neumaier).abs() < 1e-12,
            "benign-input drift: naive={}, neumaier={}", naive, neumaier);
    }

    #[test]
    fn beats_naive_on_pathological_input() {
        // Classic "1 + 1e16 - 1e16" case: naive returns 0, Neumaier
        // returns 1.
        let mut s = CompensatedSum::new();
        s.add(1.0);
        s.add(1e16);
        s.add(-1e16);
        let neumaier = s.value();
        let naive = 1.0_f64 + 1e16 - 1e16;
        assert_eq!(naive, 0.0, "naive should fail this test (precondition)");
        assert!((neumaier - 1.0).abs() < 1e-12,
            "Neumaier should recover 1.0, got {}", neumaier);
    }

    #[test]
    fn handles_large_then_small() {
        // The Neumaier-vs-Kahan distinguishing case: add large term
        // BEFORE small terms.
        let mut s = CompensatedSum::new();
        s.add(1e16);
        for _ in 0..1000 { s.add(1.0); }
        s.add(-1e16);
        let result = s.value();
        let expected = 1000.0;
        assert!((result - expected).abs() < 1e-9,
            "large-then-small case: got {}, expected {}", result, expected);
    }

    #[test]
    fn handles_small_then_large() {
        // Reverse order. Both should work.
        let mut s = CompensatedSum::new();
        for _ in 0..1000 { s.add(1.0); }
        s.add(1e16);
        s.add(-1e16);
        let result = s.value();
        let expected = 1000.0;
        assert!((result - expected).abs() < 1e-9,
            "small-then-large case: got {}, expected {}", result, expected);
    }

    #[test]
    fn add_assign_operator_works() {
        let mut s = CompensatedSum::new();
        s += 1.0;
        s += 2.0;
        s += 3.0;
        assert!((s.value() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn many_alternating_signs_recovers_exact() {
        // Sum of (i, -i) pairs should be exactly 0; with Neumaier this
        // works regardless of summation order.
        let mut s = CompensatedSum::new();
        for i in 1..=1000 {
            s.add(i as f64);
        }
        for i in 1..=1000 {
            s.add(-(i as f64));
        }
        let result = s.value();
        assert!(result.abs() < 1e-12,
            "alternating signs should give 0, got {}", result);
    }

    #[test]
    fn neumaier_sum_helper_matches_loop() {
        let xs: Vec<f64> = (0..500).map(|i| (i as f64).sin()).collect();
        let helper = neumaier_sum(&xs);
        let mut manual = CompensatedSum::new();
        for &x in &xs { manual.add(x); }
        assert_eq!(helper, manual.value());
    }

    #[test]
    fn wide_dynamic_range_weights_recover_correct_total() {
        // Realistic FKP-style scenario: weights span ~5 orders of magnitude
        // and we want the exact sum.
        let mut weights: Vec<f64> = Vec::new();
        for i in 0..1000 {
            // Half the weights are ~1, half are ~1e5
            weights.push(if i % 2 == 0 { 1.0 } else { 1e5 });
        }
        let expected = 500.0 + 500.0 * 1e5;  // exact integer math
        let neumaier = neumaier_sum(&weights);
        let naive: f64 = weights.iter().sum();
        // Both should be very close here (well-behaved sum), but Neumaier
        // is provably exact to f64 precision.
        assert!((neumaier - expected).abs() < 1e-6,
            "Neumaier on FKP-style: got {}, expected {} (diff {})",
            neumaier, expected, neumaier - expected);
        // Naive will agree here too — pathology requires near-cancellation.
        // Test is mostly that the helper integrates with realistic data
        // shapes without numerical surprise.
        let _ = naive;
    }
}
