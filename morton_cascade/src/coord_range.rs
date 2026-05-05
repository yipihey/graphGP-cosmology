// coord_range.rs
//
// Pre-cascade analysis of point coordinates: identify which bits are actually
// used and trim away dead bits. This lets the cascade run only as deep as the
// data actually supports.
//
// Two cases handled:
//   - Trailing dead bits (e.g. coords are quantized at coarse resolution but
//     stored in u32): right-shift to drop them.
//   - Leading dead bits (e.g. coords use only a fraction of the bit-width's
//     range): cap the maximum cascade level at the highest meaningful bit.
//
// Per-axis: the effective range is determined by the OR of all coordinates
// (which bits are ever set). The lowest set bit of that OR gives bit_min;
// the highest gives bit_max.
//
// For uniformly populated coords spanning [0, 2^k), bit_min=0 and bit_max=k.
// For coords that are all multiples of 2^j, bit_min=j (we can right-shift).
// For all-zero coords on an axis (degenerate dataset), bit_min and bit_max
// are both reported as 0 and effective_bits=0.

/// The bit range actually used by point coordinates along each axis.
///
/// `bit_min[d]` is the lowest set bit position across all points (axis d).
/// `bit_max[d]` is the position one past the highest set bit (so a value of 16
/// means bits 0..15 may be set).
/// `effective_bits[d] = bit_max[d] - bit_min[d]`.
///
/// For an axis with all zero coords, `bit_max=0`, `bit_min=0`, `effective_bits=0`.
#[derive(Clone, Debug)]
pub struct CoordRange<const D: usize> {
    pub bit_min: [u32; D],
    pub bit_max: [u32; D],
    pub effective_bits: [u32; D],
}

impl<const D: usize> CoordRange<D> {
    /// Scan `pts` to determine which bits are populated on each axis.
    ///
    /// O(N) work; one pass through the input.
    pub fn analyze(pts: &[[u64; D]]) -> Self {
        let mut bit_or = [0u64; D];
        let mut nonzero_seen = [false; D];
        for p in pts {
            for d in 0..D {
                bit_or[d] |= p[d];
                if p[d] != 0 { nonzero_seen[d] = true; }
            }
        }
        let mut bit_min = [0u32; D];
        let mut bit_max = [0u32; D];
        let mut effective_bits = [0u32; D];
        for d in 0..D {
            if nonzero_seen[d] {
                bit_min[d] = bit_or[d].trailing_zeros();
                bit_max[d] = 64 - bit_or[d].leading_zeros();
                effective_bits[d] = bit_max[d] - bit_min[d];
            } else {
                bit_min[d] = 0;
                bit_max[d] = 0;
                effective_bits[d] = 0;
            }
        }
        Self { bit_min, bit_max, effective_bits }
    }

    /// Maximum cascade depth this dataset can meaningfully support.
    ///
    /// Returns the largest `effective_bits[d]` across axes. A cascade with
    /// `l_max` larger than this would just be subdividing cells whose contents
    /// are already at single-bit resolution along the limiting axis.
    pub fn max_supported_l_max(&self) -> u32 {
        let mut m = 0;
        for d in 0..D {
            if self.effective_bits[d] > m {
                m = self.effective_bits[d];
            }
        }
        m
    }

    /// Smallest `effective_bits` across axes (the limiting-resolution axis).
    pub fn min_effective_bits(&self) -> u32 {
        let mut m = u32::MAX;
        for d in 0..D {
            m = m.min(self.effective_bits[d]);
        }
        if m == u32::MAX { 0 } else { m }
    }

    /// Compute a coordinate range that covers two point sets jointly: the
    /// per-axis bit-OR is taken over the concatenation of both inputs, so cells
    /// at every level line up between the two catalogs after trimming with
    /// this shared range.
    ///
    /// Use this when you need data and randoms (or two arbitrary catalogs) to
    /// share a single cell hierarchy, e.g. for cross pair counting.
    pub fn analyze_pair(a: &[[u64; D]], b: &[[u64; D]]) -> Self {
        let mut bit_or = [0u64; D];
        let mut nonzero_seen = [false; D];
        for p in a.iter().chain(b.iter()) {
            for d in 0..D {
                bit_or[d] |= p[d];
                if p[d] != 0 { nonzero_seen[d] = true; }
            }
        }
        let mut bit_min = [0u32; D];
        let mut bit_max = [0u32; D];
        let mut effective_bits = [0u32; D];
        for d in 0..D {
            if nonzero_seen[d] {
                bit_min[d] = bit_or[d].trailing_zeros();
                bit_max[d] = 64 - bit_or[d].leading_zeros();
                effective_bits[d] = bit_max[d] - bit_min[d];
            } else {
                bit_min[d] = 0;
                bit_max[d] = 0;
                effective_bits[d] = 0;
            }
        }
        Self { bit_min, bit_max, effective_bits }
    }

    /// Construct a coordinate range that represents a box of side `2^bits[d]`
    /// along each axis, regardless of where any actual points happen to fall.
    ///
    /// Use this for periodic-box analyses where the box geometry — not the
    /// data extent — defines the cascade resolution. Points should already
    /// live in `[0, 2^bits[d])` on each axis when trimmed against this range.
    ///
    /// Each `bits[d]` must be ≤ 63.
    pub fn for_box_bits(bits: [u32; D]) -> Self {
        let mut bit_min = [0u32; D];
        let mut bit_max = [0u32; D];
        let mut effective_bits = [0u32; D];
        for d in 0..D {
            assert!(bits[d] <= 63,
                "for_box_bits: bits[{}] = {} exceeds u64 width (63)", d, bits[d]);
            bit_min[d] = 0;
            bit_max[d] = bits[d];
            effective_bits[d] = bits[d];
        }
        Self { bit_min, bit_max, effective_bits }
    }
}

/// Coordinates with leading dead bits trimmed off (right-shifted by bit_min).
///
/// After trimming, coordinates fit in `effective_bits[d]` bits along axis `d`.
/// Use these for binning into a cascade with `l_max <= max_supported_l_max()`.
#[derive(Clone, Debug)]
pub struct TrimmedPoints<const D: usize> {
    pub points: Vec<[u64; D]>,
    pub range: CoordRange<D>,
}

impl<const D: usize> TrimmedPoints<D> {
    /// Build trimmed coordinates from input. The input is consumed.
    ///
    /// Each output coord is `input[d] >> bit_min[d]`, putting the lowest
    /// meaningful bit at position 0.
    pub fn from_points(pts: Vec<[u64; D]>) -> Self {
        let range = CoordRange::analyze(&pts);
        let mut points = pts;
        for p in points.iter_mut() {
            for d in 0..D {
                p[d] >>= range.bit_min[d];
            }
        }
        Self { points, range }
    }

    /// Build trimmed coordinates using an externally-supplied range. Used when
    /// two catalogs must share a single coordinate frame (e.g. data + randoms
    /// for cross pair counting). The supplied range MUST cover every coord in
    /// `pts`; in debug builds we assert this.
    pub fn from_points_with_range(pts: Vec<[u64; D]>, range: CoordRange<D>) -> Self {
        let mut points = pts;
        for p in points.iter_mut() {
            for d in 0..D {
                debug_assert!(
                    range.bit_min[d] == 0
                        || (p[d] & ((1u64 << range.bit_min[d]) - 1)) == 0,
                    "point coord {} on axis {} has bits below bit_min {} set; \
                     supplied range does not cover this catalog",
                    p[d], d, range.bit_min[d]
                );
                p[d] >>= range.bit_min[d];
            }
        }
        Self { points, range }
    }

    /// Number of points.
    pub fn len(&self) -> usize { self.points.len() }
    pub fn is_empty(&self) -> bool { self.points.is_empty() }

    /// Convert to u16 coords for use with the existing cascade APIs.
    ///
    /// The trimmed coords are left-aligned into u16: bit at position
    /// `effective_bits[d] - 1` becomes bit 15. Coords that exceed the u16
    /// range after trimming (effective_bits[d] > 16) are truncated.
    ///
    /// Returns the u16 points and the per-axis bit-shift applied.
    pub fn to_u16(&self) -> (Vec<[u16; D]>, [u32; D]) {
        let mut shifts = [0u32; D];
        for d in 0..D {
            // Place high bit of effective range at u16 bit 15 (i.e. bit 15
            // of the u16 should hold the topmost data bit).
            let eb = self.range.effective_bits[d];
            shifts[d] = if eb >= 16 { eb - 16 } else { 0 };
        }
        let pts_u16: Vec<[u16; D]> = self.points.iter().map(|p| {
            let mut out = [0u16; D];
            for d in 0..D {
                let v = p[d] >> shifts[d];
                let eb = self.range.effective_bits[d];
                if eb < 16 {
                    // Left-shift so highest data bit lands at bit 15
                    out[d] = ((v << (16 - eb)) & 0xFFFF) as u16;
                } else {
                    out[d] = (v & 0xFFFF) as u16;
                }
            }
            out
        }).collect();
        (pts_u16, shifts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_dead_bits_detected() {
        // Coords are all multiples of 1024 (low 10 bits zero)
        let pts: Vec<[u64; 2]> = vec![
            [1024, 2048],
            [3072, 1024],
            [5120, 4096],
            [2048, 3072],
        ];
        let r = CoordRange::analyze(&pts);
        assert_eq!(r.bit_min[0], 10);
        assert_eq!(r.bit_min[1], 10);
        // bit_max[0] = position past topmost set bit
        // OR of (1024, 3072, 5120, 2048) = 1024 | 3072 | 5120 | 2048
        //   = 0x400 | 0xC00 | 0x1400 | 0x800 = 0x1C00 (binary 1110000000000)
        // Highest set bit is 12, so bit_max = 13.
        assert_eq!(r.bit_max[0], 13);
        assert_eq!(r.effective_bits[0], 3);
    }

    #[test]
    fn leading_dead_bits_detected() {
        // Coords use only the low 8 bits even though stored as u64
        let pts: Vec<[u64; 3]> = (0..200u64)
            .map(|i| [i, (i * 37) % 200, (i * 17) % 200])
            .collect();
        let r = CoordRange::analyze(&pts);
        for d in 0..3 {
            assert_eq!(r.bit_min[d], 0);
            assert!(r.bit_max[d] <= 8);
            assert!(r.effective_bits[d] <= 8);
        }
    }

    #[test]
    fn all_zero_axis_handled() {
        let pts: Vec<[u64; 2]> = vec![[5, 0], [3, 0], [7, 0]];
        let r = CoordRange::analyze(&pts);
        assert_eq!(r.effective_bits[1], 0);
        assert!(r.effective_bits[0] >= 1);
    }

    #[test]
    fn trim_right_shifts_correctly() {
        let pts: Vec<[u64; 2]> = vec![
            [1024, 2048],
            [3072, 1024],
            [5120, 4096],
        ];
        let trimmed = TrimmedPoints::from_points(pts);
        // bit_min was 10, so values should be divided by 1024
        assert_eq!(trimmed.points[0], [1, 2]);
        assert_eq!(trimmed.points[1], [3, 1]);
        assert_eq!(trimmed.points[2], [5, 4]);
        assert_eq!(trimmed.range.bit_min, [10, 10]);
    }

    #[test]
    fn max_supported_depth() {
        let pts: Vec<[u64; 3]> = vec![
            [0, 0, 0],
            [1023, 0xFFFF, 0xFF],   // axis 0: 10 bits, axis 1: 16 bits, axis 2: 8 bits
        ];
        let r = CoordRange::analyze(&pts);
        assert_eq!(r.max_supported_l_max(), 16);     // axis 1 sets the max
        assert_eq!(r.min_effective_bits(), 8);       // axis 2 is the bottleneck
    }

    #[test]
    fn to_u16_preserves_high_bits() {
        // 8-bit data should end up left-aligned in u16: value 1 -> bit 15 set
        let pts: Vec<[u64; 1]> = vec![[1], [128], [255]];
        let trimmed = TrimmedPoints::from_points(pts);
        // bit_min = 0 (all values have bit 0 maybe set), bit_max = 8, effective = 8
        assert_eq!(trimmed.range.effective_bits[0], 8);
        let (u16_pts, shifts) = trimmed.to_u16();
        assert_eq!(shifts[0], 0);
        // value 1 left-shifted by (16-8)=8 -> 256
        assert_eq!(u16_pts[0][0], 256);
        // value 128 -> 32768
        assert_eq!(u16_pts[1][0], 32768);
        // value 255 -> 65280
        assert_eq!(u16_pts[2][0], 65280);
    }
}
