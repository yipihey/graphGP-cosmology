// morton-cascade CLI
//
// Driven by command-line flags. Reads a point file (binary little-endian f64
// arrays of shape [N, D]), writes cascade output as CSV.
//
// Subcommands:
//   cascade     Basic per-level stats (mean, var, dvar, sigma^2)
//   pmf         Per-level cell-count histograms
//   tpcf        Two-point correlation xi(r) at axis lags
//   fingerprint Run cascade plus optional Poisson-reference comparison
//
// Run `morton-cascade --help` for a full reference.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use morton_cascade::{
    L_MAX, L_MAX_3D,
    hier, hier_3d, hier_packed,
    hier_nd::cascade_nd,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// Argument parsing
// ============================================================================

#[derive(Debug)]
struct CommonArgs {
    input: PathBuf,
    output: PathBuf,
    dim: usize,
    box_size: f64,
    s_subshift: usize,
    periodic: bool,
    packed: bool,           // 2D only
    quiet: bool,
}

#[derive(Debug)]
enum Subcommand {
    Cascade(CommonArgs),
    Pmf(CommonArgs),
    PmfWindows {
        common: CommonArgs,
        points_per_decade: f64,
        side_min: usize,
        side_max: Option<usize>,
        explicit_sides: Option<Vec<usize>>,
    },
    Tpcf {
        common: CommonArgs,
        max_lag_level: usize,
    },
    Fingerprint {
        common: CommonArgs,
        with_poisson_ref: bool,
        n_real: usize,
        #[allow(dead_code)]
        n_pts: Option<usize>,    // optional: use first N from file (for repeated draws -- not yet)
    },
    Pairs {
        common: CommonArgs,
        max_depth: Option<usize>,
        crossover_threshold: Option<usize>,
    },
    Gradient {
        common: CommonArgs,
        target_level: usize,
    },
    Xi {
        common: CommonArgs,
        randoms: PathBuf,
        weights_data: Option<PathBuf>,
        weights_randoms: Option<PathBuf>,
        max_depth: Option<usize>,
        crossover_threshold: Option<usize>,
    },
    FieldStats {
        common: CommonArgs,
        randoms: PathBuf,
        weights_data: Option<PathBuf>,
        weights_randoms: Option<PathBuf>,
        max_depth: Option<usize>,
        crossover_threshold: Option<usize>,
        w_r_min: f64,
        hist_bins: usize,
        hist_log_min: f64,
        hist_log_max: f64,
    },
    Anisotropy {
        common: CommonArgs,
        randoms: PathBuf,
        weights_data: Option<PathBuf>,
        weights_randoms: Option<PathBuf>,
        max_depth: Option<usize>,
        crossover_threshold: Option<usize>,
        w_r_min: f64,
    },
    Scattering {
        common: CommonArgs,
        randoms: PathBuf,
        weights_data: Option<PathBuf>,
        weights_randoms: Option<PathBuf>,
        max_depth: Option<usize>,
        crossover_threshold: Option<usize>,
        w_r_min: f64,
    },
    PmfWindowsPaired {
        common: CommonArgs,
        randoms: PathBuf,
        weights_data: Option<PathBuf>,
        weights_randoms: Option<PathBuf>,
        points_per_decade: f64,
        side_min: usize,
        side_max: Option<usize>,
        explicit_sides: Option<Vec<usize>>,
        w_r_min: f64,
        hist_bins: usize,
        hist_log_min: f64,
        hist_log_max: f64,
    },
    MultiRun {
        common: CommonArgs,
        statistic: MultiRunStatistic,
        boundary: MultiRunBoundary,
        randoms: Option<PathBuf>,
        weights_data: Option<PathBuf>,
        weights_randoms: Option<PathBuf>,
        n_shifts: usize,
        shift_magnitude: f64,
        shift_seed: u64,
        resize_factors: Vec<f64>,
        resize_points_per_decade: f64,
        resize_min_scale: f64,
        bin_tol: f64,
        w_r_min: f64,
        max_depth: Option<usize>,
        cic_max_bins: usize,
        cic_skip_zero_bins: bool,
        compensated_sums: bool,
        xi_fit_basis: Option<String>,
        xi_fit_knots: usize,
        xi_fit_r_min: Option<f64>,
        xi_fit_r_max: Option<f64>,
        xi_fit_window: String,
        xi_fit_weighting: String,
        xi_fit_eval_n: usize,
    },
    AngularKnnCdf(AngularKnnCdfArgs),
}

/// Arguments for the `angular-knn-cdf` subcommand. All array inputs
/// are little-endian f64 / i64 binary files. Output is a directory
/// of binary cubes plus `meta.json`.
#[derive(Debug)]
struct AngularKnnCdfArgs {
    query_data: PathBuf,
    query_z: PathBuf,
    neigh_data: PathBuf,
    neigh_z: PathBuf,
    weights_neigh: Option<PathBuf>,
    chord_radii: PathBuf,
    z_q_edges: PathBuf,
    z_n_edges: PathBuf,
    region_labels: Option<PathBuf>,
    n_regions: usize,
    k_max: usize,
    self_exclude: bool,
    query_targetid: Option<PathBuf>,
    neigh_targetid: Option<PathBuf>,
    output: PathBuf,
    quiet: bool,
    diagonal_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiRunStatistic {
    FieldStats,
    Anisotropy,
    CicPmf,
    Xi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiRunBoundary {
    Periodic,
    Isolated,
}

const HELP: &str = "morton-cascade — hierarchical cell-count cascade for spatial points

USAGE:
    morton-cascade <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    cascade        Basic per-level stats: mean, variance, Schur residual, sigma^2(R)
    pmf            Cell-count probability mass functions per dyadic level (2D only)
    pmf-windows    P_N(V) at log-spaced cube-window volumes (any dimension 2..=5)
    tpcf           Two-point correlation xi(r) at axis-aligned log-spaced lags (2D/3D)
    fingerprint    Cascade plus optional Poisson-reference comparison, multi-realization
                   summary suitable for the cascade_plot.py visualization
    pairs          Per-shell pair counts via bit-vector cascade (any dim 2..=5).
                   Auto-sets cascade depth from data resolution; supports going
                   to bit-width-many levels. Headline output: cube-shell pair
                   counts at all scales in one pass.
    gradient       Per-particle pair-count gradient at a chosen tree level.
                   Returns N values: grad[i] = number of points sharing a cell
                   with point i at that level (delta-function form).
    xi             Two-catalog Landy-Szalay correlation function ξ(r) via the
                   bit-vector pair cascade (any dim 2..=5). Requires both a data
                   catalog (--input) and a randoms catalog (--randoms) drawn
                   from the same survey volume / footprint. Optional per-point
                   weights via --weights-data / --weights-randoms (binary f64,
                   one weight per point).
    field-stats    Density-field cell-count statistics: per-level moments and
                   PDF of the density contrast δ(c) = W_d(c) / (α · W_r(c)) − 1,
                   weighted by W_r (effective volume). The randoms encode the
                   survey selection function (footprint, fiber completeness,
                   imaging systematics, dust correction, etc.) — moments and
                   P(δ) are then natively edge-corrected. Cells with W_r below
                   --w-r-min are excluded as outside-footprint.
    anisotropy     Cell-wavelet axis-aligned anisotropy moments (3D only).
                   At each dyadic scale, decomposes the 8 children of each
                   parent cell into 7 Haar-wavelet detail coefficients along
                   x, y, z, face-diagonals, and body-diagonal. Reports the
                   per-axis variance of each coefficient and the quadrupole
                   moment Q_2 = <w_z²> − ½(<w_x²> + <w_y²>) which probes
                   line-of-sight (z-axis) versus transverse anisotropy.
                   Native cascade observable; not equivalent to HIPSTER's
                   P_ℓ(k) Legendre multipoles, but related — both probe
                   axis-direction-dependent clustering.
    scattering     Second-order Haar wavelet scattering coefficients (3D only).
                   Computes <|w_{e_2}^(l_2)[ |w_{e_1}^(l_1)| ]|²> over every
                   pair of cascade levels (l_1, l_2 < l_1) and axis directions
                   (e_1, e_2 ∈ {x, y, z}). Captures non-Gaussian structure
                   complementary to the bispectrum: how local fluctuations at
                   one scale themselves cluster at coarser scales. Following
                   Mallat (2012); Bruna & Mallat (2013). One row per (l_1, l_2)
                   pair, with first-order coefficients for context.
    pmf-windows-paired
                   Paired-cube-window P_N(δ) at arbitrary integer cube sizes
                   (not just dyadic). Same δ machinery as `field-stats` but on
                   sliding cube windows — gives finer scale resolution. Supports
                   --side-min / --side-max / --points-per-decade for log-spaced
                   side selection, OR --explicit-sides for user-chosen sides.
    multi-run      Run a chosen statistic across many shifted/resized cascades
                   and aggregate by physical cell side. Lets you fill in
                   intermediate scales (between dyadic cascade levels) via
                   resize factors, and reduce shot noise via random shifts.
                   Choose --statistic field-stats|anisotropy|cic-pmf and
                   --boundary periodic|isolated. Periodic mode synthesises
                   randoms from the box volume; isolated mode requires
                   --randoms. Use --n-shifts N --shift-magnitude M for
                   shift-bootstrap error bars; --resize-points-per-decade N
                   or --resize-factors a,b,c for non-dyadic scales. Outputs
                   a CSV per statistic plus per-run diagnostics.

GLOBAL OPTIONS (apply to every subcommand):
    -i, --input <FILE>        Input file: binary little-endian f64, shape [N, D].
                              Required.
    -o, --output <DIR>        Output directory for CSV files. Default: ./cascade_out
    -d, --dim <D>             Dimension: 2, 3, 4, or 5. Default: 2.
    -L, --box-size <L>        Domain size: input coords assumed to be in [0, L)^D.
                              Default: 1.0.
    -s, --subshift <S>        Sub-shift refinement levels. Default: 1.
    --non-periodic            Use non-periodic boundary conditions. Default: periodic.
    --packed                  Use adaptive-width packed cascade (2D only, ~2x faster).
    -q, --quiet               Suppress progress output.
    -h, --help                Show this help and exit.
    -V, --version             Show version and exit.

SUBCOMMAND OPTIONS:
    tpcf:
        --max-lag-level <L>   Compute TPCF up to this tree level (default: cascade L_MAX).
    pmf-windows:
        --points-per-decade <F>   Volume-decade density of window sizes. Default: 5.
        --side-min <N>            Smallest cube side in fine-grid units. Default: 1.
        --side-max <N>            Largest cube side. Default: M/2 where M = 2^(L_max+s_sub).
        --windows <CSV>           Explicit comma-separated list of integer sides. Overrides
                                  the log-spaced selection if provided.
    fingerprint:
        --no-poisson          Skip the Poisson reference cascade.
        --n-realizations <N>  Number of realizations to average. Default: 1.
                              (>1 currently requires re-runs over the same input file
                              with a different seed argument; not yet wired.)
    pairs:
        --max-depth <N>           Cap the cascade depth. Default: data-supported max
                                  (= max effective bits across axes after trimming).
        --crossover-threshold <N> Switch from bit-vec to point-list when cell count <= N.
                                  Default: 64. Lower = use point-list earlier (faster
                                  for sparse data, slightly slower for dense).
    gradient:
        --target-level <N>        Tree level at which to compute the gradient. Required.
    xi:
        --randoms <FILE>          Random catalog file (same f64 [N, D] format as --input).
                                  Required.
        --weights-data <FILE>     Optional per-data-point weights (binary f64, length N_d).
        --weights-randoms <FILE>  Optional per-random-point weights (binary f64, length N_r).
        --max-depth <N>           Cap the cascade depth. Default: data-supported max.
        --crossover-threshold <N> Switch from bit-vec to point-list when both cell
                                  counts <= N. Default: 64.
    field-stats:
        --randoms <FILE>          Random catalog file. Required.
        --weights-data <FILE>     Optional per-data-point weights.
        --weights-randoms <FILE>  Optional per-random-point weights.
        --w-r-min <F>             Exclude cells with summed random weight <= this
                                  (outside footprint). Default: 0.0.
        --hist-bins <N>           Number of histogram bins for P(δ). Default: 50;
                                  pass 0 to skip the histogram.
        --hist-log-min <F>        Histogram covers log10(1+δ) ∈ [min, max].
        --hist-log-max <F>        Defaults: -3.0 to +3.0 (overdensities 1e-3 to 1e3).
        --max-depth <N>           Cap the cascade depth.
        --crossover-threshold <N> Override adaptive crossover threshold default.
    anisotropy:
        --randoms <FILE>          Random catalog file. Required. (3D only)
        --weights-data <FILE>     Optional per-data-point weights.
        --weights-randoms <FILE>  Optional per-random-point weights.
        --w-r-min <F>             Exclude cells with summed random weight <= this
                                  (outside footprint). Default: 0.0.
        --max-depth <N>           Cap the cascade depth.
        --crossover-threshold <N> Override adaptive crossover threshold default.
        Output: field_anisotropy.csv with per-axis wavelet variances and
        the quadrupole moment Q_2 at every dyadic level.
    multi-run:
        --statistic <NAME>        Which aggregator to run. One of:
                                  field-stats, anisotropy, cic-pmf, xi. Required.
                                  For `xi`: shifts pool DD/RR/DR within each
                                  resize group (shift-bootstrap variance per
                                  shell); resize groups stay separate so
                                  downstream tools can fit a continuous
                                  ξ(r) basis to combine across shell windows.
        --boundary <MODE>         periodic (no randoms; cosmological boxes) or
                                  isolated (with --randoms; survey-style with
                                  rescaling-as-clipping). Default: periodic.
        --randoms <FILE>          Random catalog. Required for --boundary isolated.
        --weights-data <FILE>     Optional per-data-point weights.
        --weights-randoms <FILE>  Optional per-random-point weights (isolated).
        --n-shifts <N>            Number of random shifts of the box origin
                                  (each gives an independent estimate at
                                  the same physical scales). Default: 0.
        --shift-magnitude <F>     Fixed length of each random offset, in
                                  box-fraction units. Default: 0.25.
        --shift-seed <N>          Seed for shift offset generation. Default: 42.
        --resize-factors <CSV>    Explicit list of box-rescaling factors
                                  (e.g. 0.5,0.707,1.0). Each scale is
                                  combined with every shift offset (cartesian
                                  product). Default: empty (no resizing).
        --resize-points-per-decade <F>
                                  Alternative to --resize-factors: log-spaced
                                  resize factors at this density per decade
                                  in volume V. Set to 0 to disable. Default: 0.
        --resize-min-scale <F>    Smallest resize factor when using
                                  --resize-points-per-decade. Default: 0.5.
        --bin-tol <F>             Relative tolerance for grouping per-run
                                  results into bins by physical side.
                                  Default: 0.001 (0.1%).
        --w-r-min <F>             Outside-footprint cutoff (field-stats /
                                  anisotropy only). Default: 0.0.
        --max-depth <N>           Cap the cascade depth.
        --cic-skip-zero-bins      For --statistic cic-pmf only: omit
                                  histogram-bin rows with zero count and
                                  zero density. Useful for sparse
                                  long-tail distributions; the default
                                  emits every bin so reshape-to-matrix
                                  is straightforward in pandas / R.
        --compensated-sums        Use Neumaier compensated summation in
                                  the outer cell-aggregator accumulators
                                  (currently field-stats and anisotropy).
                                  Costs ~4× the per-cell flops in the
                                  hot path; recovers ~full f64 precision
                                  regardless of summation order or
                                  magnitude spread. Useful when
                                  per-particle weights span wide dynamic
                                  range (e.g., FKP weights crossing
                                  multiple decades) or when δ-moments
                                  would suffer catastrophic cancellation
                                  across many cells. Off by default;
                                  benign data is unaffected by the flag.
        --xi-fit-basis <NAME>     For --statistic xi only: optional
                                  Storey-Fisher-Hogg-style continuous
                                  basis fit on top of the raw shell
                                  measurements. Combines resize groups
                                  by treating each cascade shell as a
                                  top-hat-windowed measurement of ξ(r)
                                  and fitting the underlying smooth
                                  function. Supported basis: linear-bsplines.
        --xi-fit-knots <N>        Number of basis knots in log r. Default: 16.
        --xi-fit-r-min <F>        Lower edge of basis range in trimmed
                                  units. Default: smallest shell window.
        --xi-fit-r-max <F>        Upper edge of basis range. Default:
                                  largest shell window.
        --xi-fit-window <KIND>    Pair-density weighting w(r) inside
                                  each shell: euclidean (r^(D-1)) or
                                  empirical-rr (uses cascade RR_sum;
                                  more correct for surveys with edges).
                                  Default: empirical-rr.
        --xi-fit-weighting <KIND> Measurement weighting in the LS fit:
                                  shift-bootstrap, shift-bootstrap-poisson,
                                  or uniform. Default: shift-bootstrap-poisson.
        --xi-fit-eval-n <N>       Output grid size for fitted ξ(r) CSV.
                                  Default: 100.
        Output: multi_run_<statistic>.csv (one row per physical-side bin)
        and multi_run_diagnostics.csv (one row per run, with
        footprint_coverage for the isolated case). With --xi-fit-basis,
        also multi_run_xi_fit_coefs.csv (basis coefficients) and
        multi_run_xi_evaluated.csv (fitted ξ(r) on log-spaced grid).

INPUT FORMAT:
    Binary little-endian f64 array of shape [N, D], C-order (point-major).
    From numpy:   pts.astype('<f8').tofile('pts.bin')
    From Julia:   write(\"pts.bin\", reinterpret(UInt8, vec(pts')))   # pts is N x D

OUTPUT FORMAT:
    CSV files written to <output> directory. See README for column schemas.

EXAMPLES:
    # Basic 2D cascade
    morton-cascade cascade -i pts.bin -L 256.0 -o out/

    # 3D cascade with TPCF
    morton-cascade tpcf -i pts.bin -d 3 -L 100.0 -o out/

    # Fingerprint comparison with Poisson reference
    morton-cascade fingerprint -i pts.bin -L 256.0 -o out/

    # 2D MCMC inner loop using packed cascade
    morton-cascade cascade -i pts.bin -L 256.0 --packed -o out/

    # 3D Landy-Szalay correlation function with data + randoms
    morton-cascade xi -i data.bin --randoms randoms.bin -d 3 -L 256.0 -o out/

    # Edge-corrected per-cell density-field statistics & P(δ)
    morton-cascade field-stats -i data.bin --randoms randoms.bin -d 3 -L 256.0 -o out/
";

fn parse_args() -> Result<Subcommand, String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        return Err(format!("missing subcommand. Run `{} --help` for usage.", argv[0]));
    }

    // Handle --help / --version before subcommand parsing
    for a in &argv[1..] {
        match a.as_str() {
            "-h" | "--help" => { print!("{}", HELP); std::process::exit(0); }
            "-V" | "--version" => { println!("morton-cascade {}", VERSION); std::process::exit(0); }
            _ => {}
        }
    }

    let sub = argv[1].clone();
    let rest = &argv[2..];

    // Early-dispatch for `angular-knn-cdf` because it has a custom
    // argument set that does NOT share the global CommonArgs flags
    // (no -L, -d, -i; multiple binary inputs instead).
    if sub == "angular-knn-cdf" {
        return parse_angular_knn_cdf_args(rest);
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut dim: usize = 2;
    let mut box_size: f64 = 1.0;
    let mut s_subshift: usize = 1;
    let mut periodic = true;
    let mut packed = false;
    let mut quiet = false;
    let mut max_lag_level: Option<usize> = None;
    let mut with_poisson_ref = true;
    let mut n_realizations: usize = 1;
    let mut points_per_decade: f64 = 5.0;
    let mut side_min: usize = 1;
    let mut side_max: Option<usize> = None;
    let mut explicit_sides: Option<Vec<usize>> = None;
    let mut max_depth: Option<usize> = None;
    let mut crossover_threshold: Option<usize> = None;
    let mut target_level: Option<usize> = None;
    let mut randoms_path: Option<PathBuf> = None;
    let mut weights_data_path: Option<PathBuf> = None;
    let mut weights_randoms_path: Option<PathBuf> = None;
    let mut w_r_min: f64 = 0.0;
    let mut hist_bins: usize = 50;
    let mut hist_log_min: f64 = -3.0;
    let mut hist_log_max: f64 = 3.0;

    // multi-run subcommand options
    let mut mr_statistic: Option<String> = None;
    let mut mr_boundary: Option<String> = None;
    let mut mr_n_shifts: usize = 0;
    let mut mr_shift_magnitude: f64 = 0.25;
    let mut mr_shift_seed: u64 = 42;
    let mut mr_resize_factors: Vec<f64> = Vec::new();
    let mut mr_resize_points_per_decade: f64 = 0.0;
    let mut mr_resize_min_scale: f64 = 0.5;
    let mut mr_bin_tol: f64 = 0.001;
    let mut mr_cic_max_bins: usize = 2_000_000;
    let mut mr_cic_skip_zero_bins: bool = false;
    let mut mr_compensated_sums: bool = false;
    // multi-run xi-fit options (continuous-function fit, commit 2)
    let mut mr_xi_fit_basis: Option<String> = None;
    let mut mr_xi_fit_knots: usize = 16;
    let mut mr_xi_fit_r_min: Option<f64> = None;
    let mut mr_xi_fit_r_max: Option<f64> = None;
    let mut mr_xi_fit_window: String = "empirical-rr".to_string();
    let mut mr_xi_fit_weighting: String = "shift-bootstrap-poisson".to_string();
    let mut mr_xi_fit_eval_n: usize = 100;

    let mut i = 0;
    while i < rest.len() {
        let arg = &rest[i];
        let next = || -> Result<&str, String> {
            rest.get(i + 1).map(|s| s.as_str())
                .ok_or_else(|| format!("flag `{}` requires a value", arg))
        };
        match arg.as_str() {
            "-i" | "--input"     => { input = Some(PathBuf::from(next()?)); i += 2; }
            "-o" | "--output"    => { output = Some(PathBuf::from(next()?)); i += 2; }
            "-d" | "--dim"       => { dim = next()?.parse().map_err(|e| format!("--dim: {}", e))?; i += 2; }
            "-L" | "--box-size"  => { box_size = next()?.parse().map_err(|e| format!("--box-size: {}", e))?; i += 2; }
            "-s" | "--subshift"  => { s_subshift = next()?.parse().map_err(|e| format!("--subshift: {}", e))?; i += 2; }
            "--non-periodic"     => { periodic = false; i += 1; }
            "--packed"           => { packed = true; i += 1; }
            "-q" | "--quiet"     => { quiet = true; i += 1; }
            "--max-lag-level"    => { max_lag_level = Some(next()?.parse().map_err(|e| format!("--max-lag-level: {}", e))?); i += 2; }
            "--no-poisson"       => { with_poisson_ref = false; i += 1; }
            "--n-realizations"   => { n_realizations = next()?.parse().map_err(|e| format!("--n-realizations: {}", e))?; i += 2; }
            "--points-per-decade" => { points_per_decade = next()?.parse().map_err(|e| format!("--points-per-decade: {}", e))?; i += 2; }
            "--side-min"         => { side_min = next()?.parse().map_err(|e| format!("--side-min: {}", e))?; i += 2; }
            "--side-max"         => { side_max = Some(next()?.parse().map_err(|e| format!("--side-max: {}", e))?); i += 2; }
            "--windows"          => {
                let csv = next()?;
                let parsed: Result<Vec<usize>, _> = csv.split(',').map(|s| s.trim().parse::<usize>()).collect();
                explicit_sides = Some(parsed.map_err(|e| format!("--windows parse error: {}", e))?);
                i += 2;
            }
            "--max-depth"        => { max_depth = Some(next()?.parse().map_err(|e| format!("--max-depth: {}", e))?); i += 2; }
            "--crossover-threshold" => { crossover_threshold = Some(next()?.parse().map_err(|e| format!("--crossover-threshold: {}", e))?); i += 2; }
            "--target-level"     => { target_level = Some(next()?.parse().map_err(|e| format!("--target-level: {}", e))?); i += 2; }
            "--randoms"          => { randoms_path = Some(PathBuf::from(next()?)); i += 2; }
            "--weights-data"     => { weights_data_path = Some(PathBuf::from(next()?)); i += 2; }
            "--weights-randoms"  => { weights_randoms_path = Some(PathBuf::from(next()?)); i += 2; }
            "--w-r-min"          => { w_r_min = next()?.parse().map_err(|e| format!("--w-r-min: {}", e))?; i += 2; }
            "--hist-bins"        => { hist_bins = next()?.parse().map_err(|e| format!("--hist-bins: {}", e))?; i += 2; }
            "--hist-log-min"     => { hist_log_min = next()?.parse().map_err(|e| format!("--hist-log-min: {}", e))?; i += 2; }
            "--hist-log-max"     => { hist_log_max = next()?.parse().map_err(|e| format!("--hist-log-max: {}", e))?; i += 2; }
            "--statistic"        => { mr_statistic = Some(next()?.to_string()); i += 2; }
            "--boundary"         => { mr_boundary = Some(next()?.to_string()); i += 2; }
            "--n-shifts"         => { mr_n_shifts = next()?.parse().map_err(|e| format!("--n-shifts: {}", e))?; i += 2; }
            "--shift-magnitude"  => { mr_shift_magnitude = next()?.parse().map_err(|e| format!("--shift-magnitude: {}", e))?; i += 2; }
            "--shift-seed"       => { mr_shift_seed = next()?.parse().map_err(|e| format!("--shift-seed: {}", e))?; i += 2; }
            "--resize-factors"   => {
                let csv = next()?;
                let parsed: Result<Vec<f64>, _> = csv.split(',').map(|s| s.trim().parse::<f64>()).collect();
                mr_resize_factors = parsed.map_err(|e| format!("--resize-factors parse error: {}", e))?;
                i += 2;
            }
            "--resize-points-per-decade" => { mr_resize_points_per_decade = next()?.parse().map_err(|e| format!("--resize-points-per-decade: {}", e))?; i += 2; }
            "--resize-min-scale" => { mr_resize_min_scale = next()?.parse().map_err(|e| format!("--resize-min-scale: {}", e))?; i += 2; }
            "--bin-tol"          => { mr_bin_tol = next()?.parse().map_err(|e| format!("--bin-tol: {}", e))?; i += 2; }
            "--cic-max-bins"     => { mr_cic_max_bins = next()?.parse().map_err(|e| format!("--cic-max-bins: {}", e))?; i += 2; }
            "--cic-skip-zero-bins" => { mr_cic_skip_zero_bins = true; i += 1; }
            "--compensated-sums" => { mr_compensated_sums = true; i += 1; }
            "--xi-fit-basis"     => { mr_xi_fit_basis = Some(next()?.to_string()); i += 2; }
            "--xi-fit-knots"     => { mr_xi_fit_knots = next()?.parse().map_err(|e| format!("--xi-fit-knots: {}", e))?; i += 2; }
            "--xi-fit-r-min"     => { mr_xi_fit_r_min = Some(next()?.parse().map_err(|e| format!("--xi-fit-r-min: {}", e))?); i += 2; }
            "--xi-fit-r-max"     => { mr_xi_fit_r_max = Some(next()?.parse().map_err(|e| format!("--xi-fit-r-max: {}", e))?); i += 2; }
            "--xi-fit-window"    => { mr_xi_fit_window = next()?.to_string(); i += 2; }
            "--xi-fit-weighting" => { mr_xi_fit_weighting = next()?.to_string(); i += 2; }
            "--xi-fit-eval-n"    => { mr_xi_fit_eval_n = next()?.parse().map_err(|e| format!("--xi-fit-eval-n: {}", e))?; i += 2; }
            _ => return Err(format!("unknown flag: {}", arg)),
        }
    }

    let input = input.ok_or_else(|| "missing required flag --input".to_string())?;
    let output = output.unwrap_or_else(|| PathBuf::from("./cascade_out"));

    if !(2..=5).contains(&dim) {
        return Err(format!("--dim must be in 2..=5, got {}", dim));
    }
    if box_size <= 0.0 {
        return Err(format!("--box-size must be > 0, got {}", box_size));
    }
    if packed && dim != 2 {
        return Err("--packed is only available for 2D currently".to_string());
    }

    let common = CommonArgs {
        input, output, dim, box_size, s_subshift, periodic, packed, quiet,
    };

    match sub.as_str() {
        "cascade"  => Ok(Subcommand::Cascade(common)),
        "pmf"      => {
            if dim != 2 {
                return Err("`pmf` subcommand currently 2D-only; use `pmf-windows` for higher D".to_string());
            }
            Ok(Subcommand::Pmf(common))
        }
        "pmf-windows" => Ok(Subcommand::PmfWindows {
            common,
            points_per_decade,
            side_min,
            side_max,
            explicit_sides,
        }),
        "pmf-windows-paired" => Ok(Subcommand::PmfWindowsPaired {
            common,
            randoms: randoms_path
                .ok_or_else(|| "--randoms is required for `pmf-windows-paired`".to_string())?,
            weights_data: weights_data_path,
            weights_randoms: weights_randoms_path,
            points_per_decade,
            side_min,
            side_max,
            explicit_sides,
            w_r_min,
            hist_bins,
            hist_log_min,
            hist_log_max,
        }),
        "tpcf"     => Ok(Subcommand::Tpcf {
            common,
            max_lag_level: max_lag_level.unwrap_or(L_MAX),
        }),
        "fingerprint" => Ok(Subcommand::Fingerprint {
            common,
            with_poisson_ref,
            n_real: n_realizations,
            n_pts: None,
        }),
        "pairs" => Ok(Subcommand::Pairs {
            common,
            max_depth,
            crossover_threshold,
        }),
        "gradient" => Ok(Subcommand::Gradient {
            common,
            target_level: target_level
                .ok_or_else(|| "--target-level is required for `gradient` subcommand".to_string())?,
        }),
        "xi" => Ok(Subcommand::Xi {
            common,
            randoms: randoms_path
                .ok_or_else(|| "--randoms is required for `xi` subcommand".to_string())?,
            weights_data: weights_data_path,
            weights_randoms: weights_randoms_path,
            max_depth,
            crossover_threshold,
        }),
        "field-stats" => Ok(Subcommand::FieldStats {
            common,
            randoms: randoms_path
                .ok_or_else(|| "--randoms is required for `field-stats` subcommand".to_string())?,
            weights_data: weights_data_path,
            weights_randoms: weights_randoms_path,
            max_depth,
            crossover_threshold,
            w_r_min,
            hist_bins,
            hist_log_min,
            hist_log_max,
        }),
        "anisotropy" => Ok(Subcommand::Anisotropy {
            common,
            randoms: randoms_path
                .ok_or_else(|| "--randoms is required for `anisotropy` subcommand".to_string())?,
            weights_data: weights_data_path,
            weights_randoms: weights_randoms_path,
            max_depth,
            crossover_threshold,
            w_r_min,
        }),
        "scattering" => Ok(Subcommand::Scattering {
            common,
            randoms: randoms_path
                .ok_or_else(|| "--randoms is required for `scattering` subcommand".to_string())?,
            weights_data: weights_data_path,
            weights_randoms: weights_randoms_path,
            max_depth,
            crossover_threshold,
            w_r_min,
        }),
        "multi-run" => {
            let stat_str = mr_statistic
                .ok_or_else(|| "--statistic is required for `multi-run` (one of: \
                                field-stats, anisotropy, cic-pmf, xi)".to_string())?;
            let statistic = match stat_str.as_str() {
                "field-stats" => MultiRunStatistic::FieldStats,
                "anisotropy"  => MultiRunStatistic::Anisotropy,
                "cic-pmf"     => MultiRunStatistic::CicPmf,
                "xi"          => MultiRunStatistic::Xi,
                other => return Err(format!(
                    "unknown --statistic `{}` (must be field-stats|anisotropy|cic-pmf|xi)",
                    other)),
            };
            // Boundary defaults to periodic (the cosmological-box use case).
            let bdy_str = mr_boundary.unwrap_or_else(|| "periodic".to_string());
            let boundary = match bdy_str.as_str() {
                "periodic" => MultiRunBoundary::Periodic,
                "isolated" => MultiRunBoundary::Isolated,
                other => return Err(format!(
                    "unknown --boundary `{}` (must be periodic|isolated)", other)),
            };
            if boundary == MultiRunBoundary::Isolated && randoms_path.is_none() {
                return Err("--boundary isolated requires --randoms".to_string());
            }
            if mr_n_shifts == 0 && mr_resize_factors.is_empty()
                && mr_resize_points_per_decade <= 0.0
            {
                eprintln!("note: multi-run with no shifts and no resizings — \
                          will produce a single-run aggregated result \
                          (equivalent to running the statistic once).");
            }
            Ok(Subcommand::MultiRun {
                common,
                statistic,
                boundary,
                randoms: randoms_path,
                weights_data: weights_data_path,
                weights_randoms: weights_randoms_path,
                n_shifts: mr_n_shifts,
                shift_magnitude: mr_shift_magnitude,
                shift_seed: mr_shift_seed,
                resize_factors: mr_resize_factors,
                resize_points_per_decade: mr_resize_points_per_decade,
                resize_min_scale: mr_resize_min_scale,
                bin_tol: mr_bin_tol,
                w_r_min,
                max_depth,
                cic_max_bins: mr_cic_max_bins,
                cic_skip_zero_bins: mr_cic_skip_zero_bins,
                compensated_sums: mr_compensated_sums,
                xi_fit_basis: mr_xi_fit_basis,
                xi_fit_knots: mr_xi_fit_knots,
                xi_fit_r_min: mr_xi_fit_r_min,
                xi_fit_r_max: mr_xi_fit_r_max,
                xi_fit_window: mr_xi_fit_window,
                xi_fit_weighting: mr_xi_fit_weighting,
                xi_fit_eval_n: mr_xi_fit_eval_n,
            })
        }
        _ => Err(format!("unknown subcommand: {}. Try --help.", sub)),
    }
}

// ============================================================================
// angular-knn-cdf: argument parsing and execution
// ============================================================================

fn parse_angular_knn_cdf_args(rest: &[String]) -> Result<Subcommand, String> {
    let mut query_data: Option<PathBuf> = None;
    let mut query_z: Option<PathBuf> = None;
    let mut neigh_data: Option<PathBuf> = None;
    let mut neigh_z: Option<PathBuf> = None;
    let mut weights_neigh: Option<PathBuf> = None;
    let mut chord_radii: Option<PathBuf> = None;
    let mut z_q_edges: Option<PathBuf> = None;
    let mut z_n_edges: Option<PathBuf> = None;
    let mut region_labels: Option<PathBuf> = None;
    let mut n_regions: usize = 0;
    let mut k_max: usize = 0;
    let mut self_exclude = false;
    let mut query_targetid: Option<PathBuf> = None;
    let mut neigh_targetid: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut quiet = false;
    let mut diagonal_only = false;

    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].as_str();
        let need_val = |idx: usize, key: &str| -> Result<String, String> {
            if idx + 1 >= rest.len() {
                Err(format!("`{}` requires a value", key))
            } else { Ok(rest[idx + 1].clone()) }
        };
        match a {
            "--query-data" => { query_data = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--query-z"    => { query_z    = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--neigh-data" => { neigh_data = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--neigh-z"    => { neigh_z    = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--weights-neigh" => { weights_neigh = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--chord-radii"   => { chord_radii   = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--z-q-edges"     => { z_q_edges     = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--z-n-edges"     => { z_n_edges     = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--region-labels" => { region_labels = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--n-regions"     => { n_regions = need_val(i, a)?.parse()
                .map_err(|e| format!("--n-regions: {}", e))?; i += 2; }
            "--k-max"         => { k_max     = need_val(i, a)?.parse()
                .map_err(|e| format!("--k-max: {}", e))?; i += 2; }
            "--self-exclude"  => { self_exclude = true; i += 1; }
            "--query-targetid" => { query_targetid = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "--neigh-targetid" => { neigh_targetid = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "-o" | "--output"  => { output = Some(PathBuf::from(need_val(i, a)?)); i += 2; }
            "-q" | "--quiet"   => { quiet = true; i += 1; }
            "--diagonal-only"  => { diagonal_only = true; i += 1; }
            _ => return Err(format!("unknown flag for `angular-knn-cdf`: {}", a)),
        }
    }

    let args = AngularKnnCdfArgs {
        query_data: query_data.ok_or("--query-data is required")?.to_path_buf(),
        query_z: query_z.ok_or("--query-z is required")?.to_path_buf(),
        neigh_data: neigh_data.ok_or("--neigh-data is required")?.to_path_buf(),
        neigh_z: neigh_z.ok_or("--neigh-z is required")?.to_path_buf(),
        weights_neigh,
        chord_radii: chord_radii.ok_or("--chord-radii is required")?.to_path_buf(),
        z_q_edges: z_q_edges.ok_or("--z-q-edges is required")?.to_path_buf(),
        z_n_edges: z_n_edges.ok_or("--z-n-edges is required")?.to_path_buf(),
        region_labels,
        n_regions,
        k_max,
        self_exclude,
        query_targetid,
        neigh_targetid,
        output: output.ok_or("--output is required")?.to_path_buf(),
        quiet,
        diagonal_only,
    };
    Ok(Subcommand::AngularKnnCdf(args))
}

fn read_f64_vec(path: &PathBuf) -> Result<Vec<f64>, String> {
    let mut buf = Vec::new();
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("opening `{}`: {}", path.display(), e))?;
    f.read_to_end(&mut buf)
        .map_err(|e| format!("reading `{}`: {}", path.display(), e))?;
    if buf.len() % 8 != 0 {
        return Err(format!("`{}` size ({}) not a multiple of 8", path.display(), buf.len()));
    }
    let n = buf.len() / 8;
    let mut out = vec![0f64; n];
    for i in 0..n {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[i * 8..(i + 1) * 8]);
        out[i] = f64::from_le_bytes(bytes);
    }
    Ok(out)
}

fn read_i64_vec(path: &PathBuf) -> Result<Vec<i64>, String> {
    let mut buf = Vec::new();
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("opening `{}`: {}", path.display(), e))?;
    f.read_to_end(&mut buf)
        .map_err(|e| format!("reading `{}`: {}", path.display(), e))?;
    if buf.len() % 8 != 0 {
        return Err(format!("`{}` size ({}) not a multiple of 8", path.display(), buf.len()));
    }
    let n = buf.len() / 8;
    let mut out = vec![0i64; n];
    for i in 0..n {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[i * 8..(i + 1) * 8]);
        out[i] = i64::from_le_bytes(bytes);
    }
    Ok(out)
}

fn write_f64_vec(path: &PathBuf, data: &[f64]) -> Result<(), String> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for v in data { buf.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(path, &buf)
        .map_err(|e| format!("writing `{}`: {}", path.display(), e))
}

fn write_i64_vec(path: &PathBuf, data: &[i64]) -> Result<(), String> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for v in data { buf.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(path, &buf)
        .map_err(|e| format!("writing `{}`: {}", path.display(), e))
}

fn run_angular_knn_cdf(args: AngularKnnCdfArgs) -> Result<(), String> {
    use morton_cascade::angular_knn_cdf::angular_knn_cdf_3d;

    if !args.quiet { eprintln!("loading inputs ..."); }
    let q_flat = read_f64_vec(&args.query_data)?;
    if q_flat.len() % 3 != 0 {
        return Err(format!("query-data length {} not divisible by 3", q_flat.len()));
    }
    let n_q = q_flat.len() / 3;
    let mut query_pts: Vec<[f64; 3]> = Vec::with_capacity(n_q);
    for i in 0..n_q {
        query_pts.push([q_flat[i*3], q_flat[i*3+1], q_flat[i*3+2]]);
    }
    let query_z = read_f64_vec(&args.query_z)?;
    if query_z.len() != n_q {
        return Err(format!("query-z length {} != n_q {}", query_z.len(), n_q));
    }

    let n_flat = read_f64_vec(&args.neigh_data)?;
    if n_flat.len() % 3 != 0 {
        return Err(format!("neigh-data length {} not divisible by 3", n_flat.len()));
    }
    let n_n = n_flat.len() / 3;
    let mut neigh_pts: Vec<[f64; 3]> = Vec::with_capacity(n_n);
    for i in 0..n_n {
        neigh_pts.push([n_flat[i*3], n_flat[i*3+1], n_flat[i*3+2]]);
    }
    let neigh_z = read_f64_vec(&args.neigh_z)?;
    if neigh_z.len() != n_n {
        return Err(format!("neigh-z length {} != n_n {}", neigh_z.len(), n_n));
    }

    let weights_neigh = match &args.weights_neigh {
        Some(p) => Some(read_f64_vec(p)?),
        None => None,
    };
    let chord_radii = read_f64_vec(&args.chord_radii)?;
    let z_q_edges = read_f64_vec(&args.z_q_edges)?;
    let z_n_edges = read_f64_vec(&args.z_n_edges)?;
    let region_labels = match &args.region_labels {
        Some(p) => Some(read_i64_vec(p)?),
        None => None,
    };
    let query_tid = match &args.query_targetid {
        Some(p) => Some(read_i64_vec(p)?),
        None => None,
    };
    let neigh_tid = match &args.neigh_targetid {
        Some(p) => Some(read_i64_vec(p)?),
        None => None,
    };

    if !args.quiet {
        eprintln!("  N_q={}, N_n={}, n_theta={}, n_z_q={}, n_z_n={}, k_max={}, \
                   weighted={}, regions={}, self_exclude={}",
                  n_q, n_n, chord_radii.len(),
                  z_q_edges.len() - 1, z_n_edges.len() - 1, args.k_max,
                  weights_neigh.is_some(), args.n_regions, args.self_exclude);
        eprintln!("running angular-knn-cdf ...");
    }

    let t0 = std::time::Instant::now();
    let cubes = angular_knn_cdf_3d(
        &query_pts, &query_z,
        query_tid.as_deref(),
        &neigh_pts, &neigh_z,
        neigh_tid.as_deref(),
        weights_neigh.as_deref(),
        &chord_radii, &z_q_edges, &z_n_edges,
        region_labels.as_deref(),
        args.n_regions,
        args.k_max,
        args.self_exclude,
        args.diagonal_only,
    );
    let elapsed = t0.elapsed().as_secs_f64();
    if !args.quiet {
        eprintln!("  done in {:.2}s", elapsed);
    }

    std::fs::create_dir_all(&args.output)
        .map_err(|e| format!("creating `{}`: {}", args.output.display(), e))?;

    write_i64_vec(&args.output.join("H_geq_k.bin"), &cubes.h_geq_k)?;
    write_f64_vec(&args.output.join("sum_n.bin"), &cubes.sum_n)?;
    write_f64_vec(&args.output.join("sum_n2.bin"), &cubes.sum_n2)?;
    // Higher moments p=3, p=4 (note v4_1 §6) — feeds skewness S₃ and
    // kurtosis S₄ via Eq. (13–14, 16) in derived helpers.
    write_f64_vec(&args.output.join("sum_n3.bin"), &cubes.sum_n3)?;
    write_f64_vec(&args.output.join("sum_n4.bin"), &cubes.sum_n4)?;
    write_i64_vec(&args.output.join("N_q.bin"), &cubes.n_q_per_zq)?;
    if let Some(h_pr) = &cubes.h_geq_k_per_region {
        write_i64_vec(&args.output.join("H_geq_k_per_region.bin"), h_pr)?;
    }
    if let Some(s1_pr) = &cubes.sum_n_per_region {
        write_f64_vec(&args.output.join("sum_n_per_region.bin"), s1_pr)?;
    }
    if let Some(s2_pr) = &cubes.sum_n2_per_region {
        write_f64_vec(&args.output.join("sum_n2_per_region.bin"), s2_pr)?;
    }
    if let Some(s3_pr) = &cubes.sum_n3_per_region {
        write_f64_vec(&args.output.join("sum_n3_per_region.bin"), s3_pr)?;
    }
    if let Some(s4_pr) = &cubes.sum_n4_per_region {
        write_f64_vec(&args.output.join("sum_n4_per_region.bin"), s4_pr)?;
    }
    if let Some(nq_pr) = &cubes.n_q_per_region {
        write_i64_vec(&args.output.join("N_q_per_region.bin"), nq_pr)?;
    }

    // meta.json with shapes for the Python wrapper.
    let meta = format!(
        "{{\
\"n_theta\":{},\"n_z_q\":{},\"n_z_n\":{},\"k_max\":{},\"n_regions\":{},\
\"has_per_region\":{},\"is_diagonal\":{},\"has_higher_moments\":true,\
\"elapsed_s\":{}\
}}",
        cubes.n_theta, cubes.n_z_q, cubes.n_z_n, cubes.k_max, cubes.n_regions,
        cubes.h_geq_k_per_region.is_some(), cubes.is_diagonal, elapsed,
    );
    std::fs::write(args.output.join("meta.json"), meta)
        .map_err(|e| format!("writing meta.json: {}", e))?;

    Ok(())
}

// ============================================================================
// Point loading
// ============================================================================

fn load_points_f64(path: &PathBuf, dim: usize) -> Result<Vec<f64>, String> {
    let mut buf = Vec::new();
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("opening `{}`: {}", path.display(), e))?;
    f.read_to_end(&mut buf)
        .map_err(|e| format!("reading `{}`: {}", path.display(), e))?;
    if buf.len() % 8 != 0 {
        return Err(format!("file size ({}) not a multiple of 8 bytes", buf.len()));
    }
    let n_floats = buf.len() / 8;
    if n_floats % dim != 0 {
        return Err(format!(
            "file holds {} f64 values, not divisible by dim={} (got {} remainder)",
            n_floats, dim, n_floats % dim
        ));
    }
    let mut out = vec![0.0f64; n_floats];
    for i in 0..n_floats {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[i*8..(i+1)*8]);
        out[i] = f64::from_le_bytes(bytes);
    }
    Ok(out)
}

fn scale_to_u16(x: f64, box_size: f64) -> u16 {
    // Wrap into [0, box_size) with periodic mapping, then scale to [0, 2^16)
    let xw = x.rem_euclid(box_size);
    let v = (xw / box_size * 65536.0).floor();
    v.clamp(0.0, 65535.0) as u16
}

/// Scale a f64 into a 32-bit-resolution u64 in [0, 2^32). Used for bit-vec
/// cascade which can go to higher resolution than the legacy u16 path.
fn scale_to_u32_in_u64(x: f64, box_size: f64) -> u64 {
    let xw = x.rem_euclid(box_size);
    let v = (xw / box_size * 4294967296.0).floor();
    v.clamp(0.0, 4294967295.0) as u64
}

fn pack_nd_u64<const D: usize>(flat: &[f64], box_size: f64) -> Vec<[u64; D]> {
    flat.chunks_exact(D)
        .map(|c| {
            let mut p = [0u64; D];
            for d in 0..D { p[d] = scale_to_u32_in_u64(c[d], box_size); }
            p
        })
        .collect()
}

fn pack_2d(flat: &[f64], box_size: f64) -> Vec<(u16, u16)> {
    flat.chunks_exact(2)
        .map(|c| (scale_to_u16(c[0], box_size), scale_to_u16(c[1], box_size)))
        .collect()
}

fn pack_3d(flat: &[f64], box_size: f64) -> Vec<(u16, u16, u16)> {
    flat.chunks_exact(3)
        .map(|c| (
            scale_to_u16(c[0], box_size),
            scale_to_u16(c[1], box_size),
            scale_to_u16(c[2], box_size),
        ))
        .collect()
}

fn pack_nd<const D: usize>(flat: &[f64], box_size: f64) -> Vec<[u16; D]> {
    flat.chunks_exact(D)
        .map(|c| {
            let mut p = [0u16; D];
            for d in 0..D { p[d] = scale_to_u16(c[d], box_size); }
            p
        })
        .collect()
}

// ============================================================================
// Subcommand implementations
// ============================================================================

fn run_cascade(args: CommonArgs) -> Result<(), String> {
    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} points (dim={}) from {}", n_pts, args.dim, args.input.display());
    }
    std::fs::create_dir_all(&args.output)
        .map_err(|e| format!("creating output directory `{}`: {}", args.output.display(), e))?;
    let stats_path = args.output.join("level_stats.csv");
    let mut f = std::fs::File::create(&stats_path)
        .map_err(|e| format!("creating `{}`: {}", stats_path.display(), e))?;
    writeln!(f, "level,R_tree,n_cells,mean,var,dvar,sigma2_field")
        .map_err(stringify)?;

    match args.dim {
        2 => {
            let pts = pack_2d(&flat, args.box_size);
            let stats = if args.packed {
                let hint = hier_packed::predict_max_counts_2d(n_pts as u64, 4);
                let (s, _, _) = hier_packed::cascade_adaptive(&pts, args.s_subshift, args.periodic, Some(hint));
                s
            } else {
                let (s, _) = hier::cascade_hierarchical_bc(&pts, args.s_subshift, args.periodic);
                s
            };
            for (l, st) in stats.iter().enumerate() {
                let r_tree = (1usize << (L_MAX - l)) as f64;
                let s2 = if st.mean > 1e-12 { (st.var - st.mean) / (st.mean * st.mean) } else { 0.0 };
                writeln!(f, "{},{},{},{:.10e},{:.10e},{:.10e},{:.10e}",
                    l, r_tree, st.n_cells_total, st.mean, st.var, st.dvar, s2).map_err(stringify)?;
            }
        }
        3 => {
            let pts = pack_3d(&flat, args.box_size);
            let (stats, _, _) = hier_3d::cascade_3d_with_tpcf(&pts, args.s_subshift, args.periodic, &[]);
            for (l, st) in stats.iter().enumerate() {
                let r_tree = (1usize << (L_MAX_3D - l)) as f64;
                let s2 = if st.mean > 1e-12 { (st.var - st.mean) / (st.mean * st.mean) } else { 0.0 };
                writeln!(f, "{},{},{},{:.10e},{:.10e},{:.10e},{:.10e}",
                    l, r_tree, st.n_cells_total, st.mean, st.var, st.dvar, s2).map_err(stringify)?;
            }
        }
        4 => run_cascade_nd::<4>(&flat, &args, &mut f, 4)?,
        5 => run_cascade_nd::<5>(&flat, &args, &mut f, 3)?,
        _ => unreachable!(),
    }

    if !args.quiet {
        eprintln!("Wrote {}", stats_path.display());
    }
    Ok(())
}

fn run_cascade_nd<const D: usize>(
    flat: &[f64], args: &CommonArgs, f: &mut std::fs::File, l_max: usize,
) -> Result<(), String> {
    let pts = pack_nd::<D>(flat, args.box_size);
    let (stats, _) = cascade_nd::<D>(&pts, l_max, args.s_subshift, args.periodic);
    for (l, st) in stats.iter().enumerate() {
        let r_tree = (1usize << (l_max - l)) as f64;
        let s2 = if st.mean > 1e-12 { (st.var - st.mean) / (st.mean * st.mean) } else { 0.0 };
        writeln!(f, "{},{},{},{:.10e},{:.10e},{:.10e},{:.10e}",
            l, r_tree, st.n_cells_total, st.mean, st.var, st.dvar, s2).map_err(stringify)?;
    }
    Ok(())
}

fn run_pmf(args: CommonArgs) -> Result<(), String> {
    assert_eq!(args.dim, 2, "pmf currently 2D-only");
    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / 2;
    if !args.quiet {
        eprintln!("Loaded {} points from {}", n_pts, args.input.display());
    }
    let pts = pack_2d(&flat, args.box_size);
    std::fs::create_dir_all(&args.output).map_err(stringify)?;

    let (stats, _, pmfs) = hier::cascade_with_pmf(&pts, args.s_subshift, args.periodic);

    // Per-level stats
    let stats_path = args.output.join("level_stats.csv");
    let mut sf = std::fs::File::create(&stats_path).map_err(stringify)?;
    writeln!(sf, "level,R_tree,n_cells,mean,var,dvar,sigma2_field,skew,kurt").map_err(stringify)?;
    for (l, st) in stats.iter().enumerate() {
        let r_tree = (1usize << (L_MAX - l)) as f64;
        let s2 = if st.mean > 1e-12 { (st.var - st.mean) / (st.mean * st.mean) } else { 0.0 };
        let pmf = pmfs.iter().find(|p| p.level == l);
        let (sk, ku) = pmf.map(|p| (p.skew, p.kurt)).unwrap_or((0.0, 0.0));
        writeln!(sf, "{},{},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.6e},{:.6e}",
            l, r_tree, st.n_cells_total, st.mean, st.var, st.dvar, s2, sk, ku).map_err(stringify)?;
    }

    // Histograms
    let pmf_path = args.output.join("pmfs.csv");
    let mut pf = std::fs::File::create(&pmf_path).map_err(stringify)?;
    writeln!(pf, "level,R_tree,n_total,count,frequency").map_err(stringify)?;
    for p in &pmfs {
        for (k, &h) in p.histogram.iter().enumerate() {
            if h > 0 {
                writeln!(pf, "{},{},{},{},{}", p.level, p.r_tree, p.n_total, k, h).map_err(stringify)?;
            }
        }
    }
    if !args.quiet {
        eprintln!("Wrote {} and {}", stats_path.display(), pmf_path.display());
    }
    Ok(())
}

fn run_pmf_windows(
    args: CommonArgs,
    points_per_decade: f64,
    side_min: usize,
    side_max: Option<usize>,
    explicit_sides: Option<Vec<usize>>,
) -> Result<(), String> {
    use morton_cascade::hier_nd::{cascade_with_pmf_windows, log_spaced_window_sides};
    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} points (dim={}) from {}", n_pts, args.dim, args.input.display());
    }
    std::fs::create_dir_all(&args.output).map_err(stringify)?;

    // Effective fine-grid M = 2^(L_max + s_subshift). L_max defaults: L_MAX (2D) or
    // L_MAX_3D (3D). For D=4,5 we use the same L_MAX_3D = 7 fallback to keep buffer
    // sizes manageable; user can shrink box via --side-max.
    let l_max_eff = match args.dim {
        2 => L_MAX,
        3 => L_MAX_3D,
        4 => 4,    // M = 32 -> M^4 = 1M
        5 => 3,    // M = 16 -> M^5 = 1M
        _ => unreachable!(),
    };
    let m_eff: usize = 1 << (l_max_eff + args.s_subshift);
    let max_default = m_eff / 2;
    let smax = side_max.unwrap_or(max_default).min(m_eff);

    let sides = if let Some(s) = explicit_sides {
        let mut s = s;
        s.sort_unstable();
        s.dedup();
        s.into_iter().filter(|&k| k >= 1 && k <= m_eff).collect()
    } else {
        log_spaced_window_sides(side_min.max(1), smax, args.dim, points_per_decade)
    };

    if sides.is_empty() {
        return Err("no valid window sides selected (check --side-min, --side-max)".to_string());
    }
    if !args.quiet {
        eprintln!("Computing P_N(V) at {} window sides (k = {} ... {})",
                  sides.len(), sides[0], sides.last().unwrap());
    }

    let pmfs = match args.dim {
        2 => {
            let pts = pack_nd::<2>(&flat, args.box_size);
            cascade_with_pmf_windows::<2>(&pts, l_max_eff, args.s_subshift, args.periodic, &sides)
        }
        3 => {
            let pts = pack_nd::<3>(&flat, args.box_size);
            cascade_with_pmf_windows::<3>(&pts, l_max_eff, args.s_subshift, args.periodic, &sides)
        }
        4 => {
            let pts = pack_nd::<4>(&flat, args.box_size);
            cascade_with_pmf_windows::<4>(&pts, l_max_eff, args.s_subshift, args.periodic, &sides)
        }
        5 => {
            let pts = pack_nd::<5>(&flat, args.box_size);
            cascade_with_pmf_windows::<5>(&pts, l_max_eff, args.s_subshift, args.periodic, &sides)
        }
        _ => unreachable!(),
    };

    // Write level_stats.csv with per-window moments
    let stats_path = args.output.join("level_stats_windows.csv");
    let mut sf = std::fs::File::create(&stats_path).map_err(stringify)?;
    writeln!(sf, "window_side,volume_tree,n_total,mean,var,sigma2_field,skew,kurt").map_err(stringify)?;
    for p in &pmfs {
        let s2 = if p.mean > 1e-12 { (p.var - p.mean) / (p.mean * p.mean) } else { 0.0 };
        writeln!(sf, "{},{:.10e},{},{:.10e},{:.10e},{:.10e},{:.6e},{:.6e}",
            p.window_side, p.volume_tree, p.n_total, p.mean, p.var, s2, p.skew, p.kurt
        ).map_err(stringify)?;
    }

    // Write pmfs_windows.csv with full histograms
    let pmf_path = args.output.join("pmfs_windows.csv");
    let mut pf = std::fs::File::create(&pmf_path).map_err(stringify)?;
    writeln!(pf, "window_side,volume_tree,n_total,count,frequency").map_err(stringify)?;
    for p in &pmfs {
        for (n, &h) in p.histogram.iter().enumerate() {
            if h > 0 {
                writeln!(pf, "{},{:.10e},{},{},{}",
                    p.window_side, p.volume_tree, p.n_total, n, h).map_err(stringify)?;
            }
        }
    }

    if !args.quiet {
        eprintln!("Wrote {} and {}", stats_path.display(), pmf_path.display());
    }
    Ok(())
}

fn run_pmf_windows_paired(
    args: CommonArgs,
    randoms_path: PathBuf,
    weights_data_path: Option<PathBuf>,
    weights_randoms_path: Option<PathBuf>,
    points_per_decade: f64,
    side_min: usize,
    side_max: Option<usize>,
    explicit_sides: Option<Vec<usize>>,
    w_r_min: f64,
    hist_bins: usize,
    hist_log_min: f64,
    hist_log_max: f64,
) -> Result<(), String> {
    use morton_cascade::hier_nd::{cascade_pmf_windows_with_randoms,
        log_spaced_window_sides, PairedPmfConfig};

    let flat_d = load_points_f64(&args.input, args.dim)?;
    let n_d = flat_d.len() / args.dim;
    let flat_r = load_points_f64(&randoms_path, args.dim)?;
    let n_r = flat_r.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} data points and {} randoms (dim={})", n_d, n_r, args.dim);
    }

    let weights_d = match &weights_data_path {
        Some(p) => Some(load_weights(p, n_d)?), None => None,
    };
    let weights_r = match &weights_randoms_path {
        Some(p) => Some(load_weights(p, n_r)?), None => None,
    };

    std::fs::create_dir_all(&args.output).map_err(stringify)?;

    let l_max_eff = match args.dim {
        2 => L_MAX,
        3 => L_MAX_3D,
        4 => 4,
        5 => 3,
        _ => unreachable!(),
    };
    let m_eff: usize = 1 << (l_max_eff + args.s_subshift);
    let smax = side_max.unwrap_or(m_eff / 2).min(m_eff);

    let sides = if let Some(s) = explicit_sides {
        let mut s = s;
        s.sort_unstable();
        s.dedup();
        s.into_iter().filter(|&k| k >= 1 && k <= m_eff).collect()
    } else {
        log_spaced_window_sides(side_min.max(1), smax, args.dim, points_per_decade)
    };
    if sides.is_empty() {
        return Err("no valid window sides selected (check --side-min, --side-max)".to_string());
    }
    if !args.quiet {
        eprintln!("Computing P_N(δ) at {} window sides (k = {} ... {})",
                  sides.len(), sides[0], sides.last().unwrap());
    }

    let cfg = PairedPmfConfig { w_r_min, hist_bins, hist_log_min, hist_log_max };

    macro_rules! run_for_dim {
        ($D:literal) => {{
            let pts_d = pack_nd::<$D>(&flat_d, args.box_size);
            let pts_r = pack_nd::<$D>(&flat_r, args.box_size);
            cascade_pmf_windows_with_randoms::<$D>(
                &pts_d, weights_d.as_deref(),
                &pts_r, weights_r.as_deref(),
                l_max_eff, args.s_subshift, args.periodic, &sides, &cfg)
        }};
    }
    let stats = match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => unreachable!(),
    };

    let mom_path = args.output.join("paired_pmf_moments.csv");
    let mut mf = std::fs::File::create(&mom_path).map_err(stringify)?;
    writeln!(mf,
        "window_side,volume_fine,n_windows_total,n_windows_active,sum_w_r_active,\
         mean_delta,var_delta,m3_delta,m4_delta,s3_delta,min_delta,max_delta,\
         n_windows_data_outside,sum_w_d_outside")
        .map_err(stringify)?;
    for st in &stats {
        writeln!(mf,
            "{},{:.10e},{},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{},{:.10e}",
            st.window_side, st.volume_fine, st.n_windows_total, st.n_windows_active,
            st.sum_w_r_active, st.mean_delta, st.var_delta, st.m3_delta, st.m4_delta,
            st.s3_delta, st.min_delta, st.max_delta,
            st.n_windows_data_outside, st.sum_w_d_outside)
            .map_err(stringify)?;
    }

    if hist_bins > 0 {
        let pdf_path = args.output.join("paired_pmf_pdf.csv");
        let mut pf = std::fs::File::create(&pdf_path).map_err(stringify)?;
        writeln!(pf, "window_side,bin_lo_log,bin_hi_log,delta_lo,delta_hi,density,underflow,overflow")
            .map_err(stringify)?;
        for st in &stats {
            for (k, &density) in st.hist_density.iter().enumerate() {
                let lo = st.hist_bin_edges[k];
                let hi = st.hist_bin_edges[k + 1];
                let delta_lo = 10.0_f64.powf(lo) - 1.0;
                let delta_hi = 10.0_f64.powf(hi) - 1.0;
                writeln!(pf,
                    "{},{:.6e},{:.6e},{:.6e},{:.6e},{:.10e},{:.10e},{:.10e}",
                    st.window_side, lo, hi, delta_lo, delta_hi, density,
                    st.hist_underflow_w_r, st.hist_overflow_w_r)
                    .map_err(stringify)?;
            }
        }
        if !args.quiet {
            eprintln!("Wrote {} and {}", mom_path.display(), pdf_path.display());
        }
    } else if !args.quiet {
        eprintln!("Wrote {}", mom_path.display());
    }
    Ok(())
}

fn run_tpcf(args: CommonArgs, max_lag_level: usize) -> Result<(), String> {
    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} points (dim={}) from {}", n_pts, args.dim, args.input.display());
    }
    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let tpcf_path = args.output.join("tpcf.csv");
    let mut f = std::fs::File::create(&tpcf_path).map_err(stringify)?;
    writeln!(f, "level,k,r_tree,r_fine,smoothing_h_fine,xi,n_pairs").map_err(stringify)?;

    match args.dim {
        2 => {
            let pts = pack_2d(&flat, args.box_size);
            let lag_levels: Vec<usize> = (1..=max_lag_level.min(L_MAX)).collect();
            let (_, _, tpcf) = hier::cascade_hierarchical_with_tpcf(&pts, args.s_subshift, args.periodic, &lag_levels);
            for tp in &tpcf {
                writeln!(f, "{},{},{},{},{},{:.10e},{}",
                    tp.level, tp.k, tp.r_tree, tp.r_fine, tp.smoothing_h_fine, tp.xi, tp.n_pairs).map_err(stringify)?;
            }
        }
        3 => {
            let pts = pack_3d(&flat, args.box_size);
            let lag_levels: Vec<usize> = (1..=max_lag_level.min(L_MAX_3D)).collect();
            let (_, _, tpcf) = hier_3d::cascade_3d_with_tpcf(&pts, args.s_subshift, args.periodic, &lag_levels);
            for tp in &tpcf {
                writeln!(f, "{},{},{},{},{},{:.10e},{}",
                    tp.level, tp.k, tp.r_tree, tp.r_fine, tp.smoothing_h_fine, tp.xi, tp.n_pairs).map_err(stringify)?;
            }
        }
        _ => return Err(format!("tpcf currently 2D/3D only, got dim={}", args.dim)),
    }
    if !args.quiet {
        eprintln!("Wrote {}", tpcf_path.display());
    }
    Ok(())
}

fn run_pairs(
    args: CommonArgs,
    max_depth: Option<usize>,
    crossover_threshold: Option<usize>,
) -> Result<(), String> {
    use morton_cascade::coord_range::TrimmedPoints;
    use morton_cascade::hier_bitvec::BitVecCascade;

    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} points (dim={}) from {}", n_pts, args.dim, args.input.display());
    }
    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let stats_path = args.output.join("pairs.csv");
    let mut f = std::fs::File::create(&stats_path).map_err(stringify)?;
    writeln!(f, "level,cell_side_trimmed,cell_side_phys,r_inner_phys,r_outer_phys,n_pairs,cumulative_pairs").map_err(stringify)?;

    // Dispatch on dim. The cascade uses u32-resolution coords packed into u64.
    macro_rules! run_for_dim {
        ($D:literal) => {{
            let pts = pack_nd_u64::<$D>(&flat, args.box_size);
            let trimmed = TrimmedPoints::<$D>::from_points(pts);
            let thresh = crossover_threshold.unwrap_or_else(||
                BitVecCascade::<$D>::default_crossover_threshold(n_pts));
            if !args.quiet {
                eprintln!("Trimmed coord ranges (effective bits per axis): {:?}",
                    trimmed.range.effective_bits);
                let auto_l = trimmed.range.max_supported_l_max();
                eprintln!("Auto cascade depth: {}", auto_l);
                eprintln!("Crossover threshold: {}{}", thresh,
                    if crossover_threshold.is_none() { " (adaptive default)" } else { " (user)" });
            }
            let casc = BitVecCascade::<$D>::build_with_threshold(trimmed, max_depth, thresh);
            let stats = casc.analyze();
            let shells = casc.pair_counts_per_shell(&stats);
            // Conversion factor: trimmed unit -> physical box units.
            // After packing to u32 then trimming, the trimmed-unit step corresponds
            // to box_size / 2^max_eff in physical units (since the input box_size
            // maps to 2^32 raw, trimmed range is 2^max_eff).
            let max_eff = casc.points.range.max_supported_l_max() as usize;
            let trimmed_to_phys = args.box_size / (1u64 << max_eff) as f64;
            for (s, ls) in shells.iter().zip(stats.iter()) {
                let cell_phys = s.cell_side_trimmed * trimmed_to_phys;
                let r_inner_phys = s.r_inner_trimmed * trimmed_to_phys;
                let r_outer_phys = s.r_outer_trimmed * trimmed_to_phys;
                writeln!(f, "{},{:.10e},{:.10e},{:.10e},{:.10e},{},{}",
                    s.level, s.cell_side_trimmed, cell_phys, r_inner_phys, r_outer_phys,
                    s.n_pairs, ls.cumulative_pairs).map_err(stringify)?;
            }
        }};
    }
    match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => return Err(format!("unsupported dim {}", args.dim)),
    }

    if !args.quiet {
        eprintln!("Wrote {}", stats_path.display());
    }
    Ok(())
}

fn load_weights(path: &PathBuf, expected_n: usize) -> Result<Vec<f64>, String> {
    let mut buf = Vec::new();
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("opening weights `{}`: {}", path.display(), e))?;
    f.read_to_end(&mut buf)
        .map_err(|e| format!("reading weights `{}`: {}", path.display(), e))?;
    if buf.len() != expected_n * 8 {
        return Err(format!(
            "weights file `{}` has {} bytes, expected {} (= {} f64 values)",
            path.display(), buf.len(), expected_n * 8, expected_n));
    }
    let mut out = vec![0.0f64; expected_n];
    for i in 0..expected_n {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[i*8..(i+1)*8]);
        out[i] = f64::from_le_bytes(bytes);
    }
    Ok(out)
}

fn run_xi(
    args: CommonArgs,
    randoms_path: PathBuf,
    weights_data_path: Option<PathBuf>,
    weights_randoms_path: Option<PathBuf>,
    max_depth: Option<usize>,
    crossover_threshold: Option<usize>,
) -> Result<(), String> {
    use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
    use morton_cascade::hier_bitvec_pair::BitVecCascadePair;

    let flat_d = load_points_f64(&args.input, args.dim)?;
    let n_d = flat_d.len() / args.dim;
    let flat_r = load_points_f64(&randoms_path, args.dim)?;
    let n_r = flat_r.len() / args.dim;

    if !args.quiet {
        eprintln!("Loaded {} data points (dim={}) from {}", n_d, args.dim, args.input.display());
        eprintln!("Loaded {} random points from {}", n_r, randoms_path.display());
    }

    let weights_d = match &weights_data_path {
        Some(p) => Some(load_weights(p, n_d)?),
        None => None,
    };
    let weights_r = match &weights_randoms_path {
        Some(p) => Some(load_weights(p, n_r)?),
        None => None,
    };
    if !args.quiet {
        if weights_d.is_some() { eprintln!("Loaded data weights from {}", weights_data_path.as_ref().unwrap().display()); }
        if weights_r.is_some() { eprintln!("Loaded random weights from {}", weights_randoms_path.as_ref().unwrap().display()); }
    }

    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let xi_path = args.output.join("xi_landy_szalay.csv");
    let mut f = std::fs::File::create(&xi_path).map_err(stringify)?;
    writeln!(f, "level,cell_side_trimmed,cell_side_phys,r_inner_phys,r_outer_phys,dd,rr,dr,xi_ls,cumulative_dd,cumulative_rr,cumulative_dr").map_err(stringify)?;

    macro_rules! run_for_dim {
        ($D:literal) => {{
            let pts_d = pack_nd_u64::<$D>(&flat_d, args.box_size);
            let pts_r = pack_nd_u64::<$D>(&flat_r, args.box_size);
            // Joint range so cells line up between data and randoms.
            let range = CoordRange::analyze_pair(&pts_d, &pts_r);
            let td = TrimmedPoints::<$D>::from_points_with_range(pts_d, range.clone());
            let tr = TrimmedPoints::<$D>::from_points_with_range(pts_r, range.clone());
            let thresh = crossover_threshold.unwrap_or_else(||
                BitVecCascadePair::<$D>::default_crossover_threshold(n_d, n_r));
            if !args.quiet {
                eprintln!("Joint trim eff_bits per axis: {:?}", td.range.effective_bits);
                let auto_l = td.range.max_supported_l_max();
                eprintln!("Auto cascade depth: {}", auto_l);
                eprintln!("Crossover threshold: {}{}", thresh,
                    if crossover_threshold.is_none() { " (adaptive default)" } else { " (user)" });
            }
            let pair = BitVecCascadePair::<$D>::build_full(
                td, tr, weights_d, weights_r, max_depth, thresh);
            let stats = pair.analyze();
            let shells = pair.xi_landy_szalay(&stats);
            let max_eff = pair.data.range.max_supported_l_max() as usize;
            let trimmed_to_phys = args.box_size / (1u64 << max_eff) as f64;
            for (s, ls) in shells.iter().zip(stats.iter()) {
                let cell_phys = s.cell_side_trimmed * trimmed_to_phys;
                let r_inner_phys = s.r_inner_trimmed * trimmed_to_phys;
                let r_outer_phys = s.r_outer_trimmed * trimmed_to_phys;
                writeln!(f,
                    "{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}",
                    s.level, s.cell_side_trimmed, cell_phys, r_inner_phys, r_outer_phys,
                    s.dd, s.rr, s.dr, s.xi_ls,
                    ls.cumulative_dd, ls.cumulative_rr, ls.cumulative_dr).map_err(stringify)?;
            }
        }};
    }
    match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => return Err(format!("unsupported dim {}", args.dim)),
    }

    if !args.quiet {
        eprintln!("Wrote {}", xi_path.display());
    }
    Ok(())
}

fn run_field_stats(
    args: CommonArgs,
    randoms_path: PathBuf,
    weights_data_path: Option<PathBuf>,
    weights_randoms_path: Option<PathBuf>,
    max_depth: Option<usize>,
    crossover_threshold: Option<usize>,
    w_r_min: f64,
    hist_bins: usize,
    hist_log_min: f64,
    hist_log_max: f64,
) -> Result<(), String> {
    use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
    use morton_cascade::hier_bitvec_pair::{BitVecCascadePair, FieldStatsConfig};

    let flat_d = load_points_f64(&args.input, args.dim)?;
    let n_d = flat_d.len() / args.dim;
    let flat_r = load_points_f64(&randoms_path, args.dim)?;
    let n_r = flat_r.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} data points (dim={}) from {}", n_d, args.dim, args.input.display());
        eprintln!("Loaded {} random points from {}", n_r, randoms_path.display());
    }

    let weights_d = match &weights_data_path {
        Some(p) => Some(load_weights(p, n_d)?),
        None => None,
    };
    let weights_r = match &weights_randoms_path {
        Some(p) => Some(load_weights(p, n_r)?),
        None => None,
    };
    if !args.quiet {
        if weights_d.is_some() { eprintln!("Loaded data weights from {}", weights_data_path.as_ref().unwrap().display()); }
        if weights_r.is_some() { eprintln!("Loaded random weights from {}", weights_randoms_path.as_ref().unwrap().display()); }
    }

    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let mom_path = args.output.join("field_moments.csv");
    let mut mf = std::fs::File::create(&mom_path).map_err(stringify)?;
    writeln!(mf,
        "level,cell_side_trimmed,cell_side_phys,n_cells_active,sum_w_r_active,\
         mean_delta,var_delta,m3_delta,m4_delta,s3_delta,min_delta,max_delta,\
         n_cells_data_outside,sum_w_d_outside")
        .map_err(stringify)?;

    let pdf_path = args.output.join("field_pdf.csv");
    let mut pf = if hist_bins > 0 {
        let mut f = std::fs::File::create(&pdf_path).map_err(stringify)?;
        writeln!(f, "level,cell_side_trimmed,cell_side_phys,bin_lo,bin_hi,\
                     log10_one_plus_delta_lo,log10_one_plus_delta_hi,density,underflow,overflow")
            .map_err(stringify)?;
        Some(f)
    } else { None };

    let cfg = FieldStatsConfig {
        w_r_min, hist_bins, hist_log_min, hist_log_max,
        ..Default::default()
    };

    macro_rules! run_for_dim {
        ($D:literal) => {{
            let pts_d = pack_nd_u64::<$D>(&flat_d, args.box_size);
            let pts_r = pack_nd_u64::<$D>(&flat_r, args.box_size);
            let range = CoordRange::analyze_pair(&pts_d, &pts_r);
            let td = TrimmedPoints::<$D>::from_points_with_range(pts_d, range.clone());
            let tr = TrimmedPoints::<$D>::from_points_with_range(pts_r, range.clone());
            let thresh = crossover_threshold.unwrap_or_else(||
                BitVecCascadePair::<$D>::default_crossover_threshold(n_d, n_r));
            if !args.quiet {
                eprintln!("Joint trim eff_bits per axis: {:?}", td.range.effective_bits);
                let auto_l = td.range.max_supported_l_max();
                eprintln!("Auto cascade depth: {}", auto_l);
                eprintln!("Crossover threshold: {}{}", thresh,
                    if crossover_threshold.is_none() { " (adaptive default)" } else { " (user)" });
                eprintln!("Footprint cutoff w_r_min: {}", w_r_min);
                eprintln!("PDF bins: {} over log10(1+δ) in [{}, {}]",
                    hist_bins, hist_log_min, hist_log_max);
            }
            let pair = BitVecCascadePair::<$D>::build_full(
                td, tr, weights_d, weights_r, max_depth, thresh);
            let stats = pair.analyze_field_stats(&cfg);
            let max_eff = pair.data.range.max_supported_l_max() as usize;
            let trimmed_to_phys = args.box_size / (1u64 << max_eff) as f64;
            for st in &stats {
                let cell_phys = st.cell_side_trimmed * trimmed_to_phys;
                writeln!(mf,
                    "{},{:.10e},{:.10e},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{},{:.10e}",
                    st.level, st.cell_side_trimmed, cell_phys, st.n_cells_active,
                    st.sum_w_r_active, st.mean_delta, st.var_delta, st.m3_delta,
                    st.m4_delta, st.s3_delta, st.min_delta, st.max_delta,
                    st.n_cells_data_outside, st.sum_w_d_outside)
                    .map_err(stringify)?;
                if let Some(ref mut f) = pf {
                    for (k, &density) in st.hist_density.iter().enumerate() {
                        let lo = st.hist_bin_edges[k];
                        let hi = st.hist_bin_edges[k + 1];
                        // Δ at bin centre via 10^(log)−1
                        let centre = 0.5 * (lo + hi);
                        let one_plus = 10.0_f64.powf(centre);
                        let delta_centre = one_plus - 1.0;
                        // Bin lo/hi in δ space (not log) for downstream convenience
                        let delta_lo = 10.0_f64.powf(lo) - 1.0;
                        let delta_hi = 10.0_f64.powf(hi) - 1.0;
                        let _ = delta_centre;  // currently unused; might emit if desired
                        writeln!(f,
                            "{},{:.10e},{:.10e},{:.10e},{:.10e},{:.6e},{:.6e},{:.10e},{:.10e},{:.10e}",
                            st.level, st.cell_side_trimmed, cell_phys,
                            delta_lo, delta_hi, lo, hi, density,
                            st.hist_underflow_w_r, st.hist_overflow_w_r)
                            .map_err(stringify)?;
                    }
                }
            }
        }};
    }
    match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => return Err(format!("unsupported dim {}", args.dim)),
    }

    if !args.quiet {
        eprintln!("Wrote {}", mom_path.display());
        if hist_bins > 0 { eprintln!("Wrote {}", pdf_path.display()); }
    }
    Ok(())
}

fn run_anisotropy(
    args: CommonArgs,
    randoms_path: PathBuf,
    weights_data_path: Option<PathBuf>,
    weights_randoms_path: Option<PathBuf>,
    max_depth: Option<usize>,
    crossover_threshold: Option<usize>,
    w_r_min: f64,
) -> Result<(), String> {
    use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
    use morton_cascade::hier_bitvec_pair::{BitVecCascadePair, FieldStatsConfig};

    let flat_d = load_points_f64(&args.input, args.dim)?;
    let n_d = flat_d.len() / args.dim;
    let flat_r = load_points_f64(&randoms_path, args.dim)?;
    let n_r = flat_r.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} data points (dim={}) from {}", n_d, args.dim, args.input.display());
        eprintln!("Loaded {} random points from {}", n_r, randoms_path.display());
    }

    let weights_d = match &weights_data_path {
        Some(p) => Some(load_weights(p, n_d)?),
        None => None,
    };
    let weights_r = match &weights_randoms_path {
        Some(p) => Some(load_weights(p, n_r)?),
        None => None,
    };

    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let aniso_path = args.output.join("field_anisotropy.csv");
    let mut af = std::fs::File::create(&aniso_path).map_err(stringify)?;

    let cfg = FieldStatsConfig { w_r_min, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0, ..Default::default() };

    // Run a single dimension. Macro because monomorphization on a const
    // generic D needs a literal; matches the pattern used by other CLI
    // subcommands here.
    macro_rules! run_for_dim {
        ($D:literal) => {{
            let pts_d = pack_nd_u64::<$D>(&flat_d, args.box_size);
            let pts_r = pack_nd_u64::<$D>(&flat_r, args.box_size);
            let range = CoordRange::analyze_pair(&pts_d, &pts_r);
            let td = TrimmedPoints::<$D>::from_points_with_range(pts_d, range.clone());
            let tr = TrimmedPoints::<$D>::from_points_with_range(pts_r, range.clone());
            let thresh = crossover_threshold.unwrap_or_else(||
                BitVecCascadePair::<$D>::default_crossover_threshold(n_d, n_r));
            if !args.quiet {
                eprintln!("Joint trim eff_bits per axis: {:?}", td.range.effective_bits);
                eprintln!("Auto cascade depth: {}", td.range.max_supported_l_max());
                eprintln!("Crossover threshold: {}{}", thresh,
                    if crossover_threshold.is_none() { " (adaptive default)" } else { " (user)" });
                eprintln!("Footprint cutoff w_r_min: {}", w_r_min);
                eprintln!("LoS convention: last axis (axis index {})", $D - 1);
            }
            let pair = BitVecCascadePair::<$D>::build_full(
                td, tr, weights_d.clone(), weights_r.clone(), max_depth, thresh);
            let stats = pair.analyze_anisotropy(&cfg);

            let max_eff = pair.data.range.max_supported_l_max() as usize;
            let trimmed_to_phys = args.box_size / (1u64 << max_eff) as f64;

            // CSV header: D-generic. We write all 2^D - 1 non-trivial wavelet
            // patterns (label `w2_p<bits>` for pattern bits like 001, 011, 111),
            // plus the D axis-aligned, plus the LoS quadrupole and reduced.
            let mut header = String::from(
                "level,cell_side_trimmed,cell_side_phys,n_parents,sum_w_r_parents");
            for d in 0..$D {
                header.push_str(&format!(",w2_axis_{}", d));
            }
            for e in 1..(1u32 << $D) {
                header.push_str(&format!(",w2_p{:0width$b}", e, width = $D as usize));
            }
            header.push_str(",quadrupole_los,reduced_quadrupole_los");
            writeln!(af, "{}", header).map_err(stringify)?;

            for st in &stats {
                let cell_phys = st.cell_side_trimmed * trimmed_to_phys;
                let mut line = format!(
                    "{},{:.10e},{:.10e},{},{:.10e}",
                    st.level, st.cell_side_trimmed, cell_phys, st.n_parents, st.sum_w_r_parents);
                for d in 0..$D {
                    line.push_str(&format!(",{:.10e}", st.mean_w_squared_for_axis(d)));
                }
                for e in 1..(1usize << $D) {
                    line.push_str(&format!(",{:.10e}", st.mean_w_squared_pattern(e)));
                }
                line.push_str(&format!(",{:.10e},{:.10e}",
                    st.quadrupole_los, st.reduced_quadrupole_los));
                writeln!(af, "{}", line).map_err(stringify)?;
            }
        }};
    }

    match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => return Err(format!(
            "anisotropy currently supports dim ∈ {{2,3,4,5}}; got {}",
            args.dim)),
    }

    if !args.quiet {
        eprintln!("Wrote {}", aniso_path.display());
    }
    Ok(())
}

fn run_scattering(
    args: CommonArgs,
    randoms_path: PathBuf,
    weights_data_path: Option<PathBuf>,
    weights_randoms_path: Option<PathBuf>,
    max_depth: Option<usize>,
    crossover_threshold: Option<usize>,
    w_r_min: f64,
) -> Result<(), String> {
    use morton_cascade::coord_range::{CoordRange, TrimmedPoints};
    use morton_cascade::hier_bitvec_pair::{BitVecCascadePair, FieldStatsConfig};

    if args.dim != 3 {
        return Err(format!("scattering is 3D-only; got dim={}", args.dim));
    }

    let flat_d = load_points_f64(&args.input, args.dim)?;
    let n_d = flat_d.len() / args.dim;
    let flat_r = load_points_f64(&randoms_path, args.dim)?;
    let n_r = flat_r.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} data points (dim=3) from {}", n_d, args.input.display());
        eprintln!("Loaded {} random points from {}", n_r, randoms_path.display());
    }

    let weights_d = match &weights_data_path {
        Some(p) => Some(load_weights(p, n_d)?),
        None => None,
    };
    let weights_r = match &weights_randoms_path {
        Some(p) => Some(load_weights(p, n_r)?),
        None => None,
    };

    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let scat_path = args.output.join("field_scattering.csv");
    let mut sf = std::fs::File::create(&scat_path).map_err(stringify)?;
    // Header: per row, give (l_fine, l_coarse), cell sides, parent counts, sum_w_r,
    // 3 first-order coefficients (one per axis e_1), and 9 second-order
    // coefficients indexed by (e_1, e_2) ∈ {x, y, z}².
    writeln!(sf,
        "level_fine,level_coarse,cell_side_fine_phys,cell_side_coarse_phys,\
         n_parents_coarse,sum_w_r_parents,\
         first_x,first_y,first_z,\
         s2_x_x,s2_x_y,s2_x_z,\
         s2_y_x,s2_y_y,s2_y_z,\
         s2_z_x,s2_z_y,s2_z_z")
        .map_err(stringify)?;

    let cfg = FieldStatsConfig { w_r_min, hist_bins: 0, hist_log_min: -3.0, hist_log_max: 3.0, ..Default::default() };

    let pts_d = pack_nd_u64::<3>(&flat_d, args.box_size);
    let pts_r = pack_nd_u64::<3>(&flat_r, args.box_size);
    let range = CoordRange::analyze_pair(&pts_d, &pts_r);
    let td = TrimmedPoints::<3>::from_points_with_range(pts_d, range.clone());
    let tr = TrimmedPoints::<3>::from_points_with_range(pts_r, range.clone());
    let thresh = crossover_threshold.unwrap_or_else(||
        BitVecCascadePair::<3>::default_crossover_threshold(n_d, n_r));
    if !args.quiet {
        eprintln!("Joint trim eff_bits per axis: {:?}", td.range.effective_bits);
        eprintln!("Auto cascade depth: {}", td.range.max_supported_l_max());
        eprintln!("Crossover threshold: {}{}", thresh,
            if crossover_threshold.is_none() { " (adaptive default)" } else { " (user)" });
        eprintln!("Footprint cutoff w_r_min: {}", w_r_min);
        eprintln!("Computing 1st + 2nd-order Haar scattering coefficients...");
    }
    let pair = BitVecCascadePair::<3>::build_full(
        td, tr, weights_d, weights_r, max_depth, thresh);
    let stats = pair.analyze_scattering_2nd_order(&cfg);

    let max_eff = pair.data.range.max_supported_l_max() as usize;
    let trimmed_to_phys = args.box_size / (1u64 << max_eff) as f64;

    for st in &stats {
        let cell_fine = st.cell_side_fine_trimmed * trimmed_to_phys;
        let cell_coarse = st.cell_side_coarse_trimmed * trimmed_to_phys;
        let f = &st.first_order;
        let s = &st.second_order;
        writeln!(sf,
            "{},{},{:.10e},{:.10e},{},{:.10e},\
             {:.10e},{:.10e},{:.10e},\
             {:.10e},{:.10e},{:.10e},\
             {:.10e},{:.10e},{:.10e},\
             {:.10e},{:.10e},{:.10e}",
            st.level_fine, st.level_coarse, cell_fine, cell_coarse,
            st.n_parents_coarse, st.sum_w_r_parents,
            f[0], f[1], f[2],
            s[0][0], s[0][1], s[0][2],
            s[1][0], s[1][1], s[1][2],
            s[2][0], s[2][1], s[2][2])
            .map_err(stringify)?;
    }

    if !args.quiet {
        eprintln!("Wrote {} ({} (l_fine, l_coarse) entries)",
            scat_path.display(), stats.len());
    }
    Ok(())
}

fn run_gradient(args: CommonArgs, target_level: usize) -> Result<(), String> {
    use morton_cascade::coord_range::TrimmedPoints;
    use morton_cascade::hier_bitvec::BitVecCascade;

    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} points (dim={}) from {}", n_pts, args.dim, args.input.display());
    }
    std::fs::create_dir_all(&args.output).map_err(stringify)?;
    let path = args.output.join("gradient.csv");
    let mut f = std::fs::File::create(&path).map_err(stringify)?;
    writeln!(f, "particle,gradient_pair_count").map_err(stringify)?;

    macro_rules! run_for_dim {
        ($D:literal) => {{
            let pts = pack_nd_u64::<$D>(&flat, args.box_size);
            let trimmed = TrimmedPoints::<$D>::from_points(pts);
            let casc = BitVecCascade::<$D>::build(trimmed, None);
            if target_level > casc.l_max {
                return Err(format!(
                    "--target-level {} exceeds cascade depth {} (set by data resolution)",
                    target_level, casc.l_max));
            }
            let grad = casc.pair_gradient_per_particle(target_level);
            for (i, g) in grad.iter().enumerate() {
                writeln!(f, "{},{}", i, g).map_err(stringify)?;
            }
        }};
    }
    match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => return Err(format!("unsupported dim {}", args.dim)),
    }

    if !args.quiet {
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

fn run_fingerprint(args: CommonArgs, with_poisson_ref: bool, n_real: usize) -> Result<(), String> {
    if args.dim != 2 {
        return Err("fingerprint currently 2D-only (matches cascade_plot.py)".to_string());
    }
    if n_real != 1 {
        eprintln!("warning: --n-realizations > 1 not yet wired; running 1 realization");
    }
    let flat = load_points_f64(&args.input, args.dim)?;
    let n_pts = flat.len() / 2;
    if !args.quiet {
        eprintln!("Loaded {} points from {}", n_pts, args.input.display());
    }
    let pts = pack_2d(&flat, args.box_size);
    std::fs::create_dir_all(&args.output).map_err(stringify)?;

    // Cascade with PMFs
    let (stats, _, pmfs) = hier::cascade_with_pmf(&pts, args.s_subshift, args.periodic);
    let lag_levels: Vec<usize> = (1..=L_MAX).collect();
    let (_, _, tpcf) = hier::cascade_hierarchical_with_tpcf(&pts, args.s_subshift, args.periodic, &lag_levels);

    write_fingerprint_files(&args.output, "cox", &stats, &pmfs, &tpcf)?;

    // Spatial map at level 5
    let l_map = 5usize;
    let n_per_axis = 1usize << l_map;
    let h_l = (1u32 << 16) / (n_per_axis as u32);
    let mut grid = vec![0u32; n_per_axis * n_per_axis];
    for &(x, y) in &pts {
        let cx = (x as u32 / h_l) as usize;
        let cy = (y as u32 / h_l) as usize;
        grid[cy * n_per_axis + cx] += 1;
    }
    let map_path = args.output.join(format!("spatial_map_l{}.csv", l_map));
    let mut mf = std::fs::File::create(&map_path).map_err(stringify)?;
    writeln!(mf, "cy,cx,count").map_err(stringify)?;
    for cy in 0..n_per_axis {
        for cx in 0..n_per_axis {
            writeln!(mf, "{},{},{}", cy, cx, grid[cy * n_per_axis + cx]).map_err(stringify)?;
        }
    }

    if with_poisson_ref {
        // Poisson reference with same N: deterministic placement of N points
        // uniformly across the box using a low-discrepancy sequence so we don't
        // need an RNG dependency. (For statistical noise floor estimation, the
        // user should run multi-realization separately.)
        if !args.quiet { eprintln!("Building Poisson reference with same N..."); }
        let mut ref_pts: Vec<(u16, u16)> = Vec::with_capacity(n_pts);
        // Halton sequence (base 2 and 3) — quasi-random uniform fill
        for i in 0..n_pts {
            let h2 = halton(i + 1, 2);
            let h3 = halton(i + 1, 3);
            ref_pts.push((
                (h2 * 65536.0).clamp(0.0, 65535.0) as u16,
                (h3 * 65536.0).clamp(0.0, 65535.0) as u16,
            ));
        }
        let (rstats, _, rpmfs) = hier::cascade_with_pmf(&ref_pts, args.s_subshift, args.periodic);
        let (_, _, rtpcf) = hier::cascade_hierarchical_with_tpcf(&ref_pts, args.s_subshift, args.periodic, &lag_levels);
        write_fingerprint_files(&args.output, "ref", &rstats, &rpmfs, &rtpcf)?;
    }

    if !args.quiet {
        eprintln!("Wrote fingerprint files to {}", args.output.display());
        eprintln!("Next: python3 cascade_plot.py  (or cascade_plot_v3.py for the multi-realization view)");
    }
    Ok(())
}

fn write_fingerprint_files(
    dir: &std::path::Path,
    suffix: &str,
    stats: &[morton_cascade::LevelStats],
    pmfs: &[hier::PmfLevel],
    tpcf: &[hier::TpcfPoint],
) -> Result<(), String> {
    let stats_path = dir.join(format!("level_stats_{}.csv", suffix));
    let mut sf = std::fs::File::create(&stats_path).map_err(stringify)?;
    writeln!(sf, "realization,level,R_tree,n_cells,mean,var,dvar,sigma2_field,skew,kurt").map_err(stringify)?;
    for (l, st) in stats.iter().enumerate() {
        let r_tree = (1usize << (L_MAX - l)) as f64;
        let s2 = if st.mean > 1e-12 { (st.var - st.mean) / (st.mean * st.mean) } else { 0.0 };
        let pmf = pmfs.iter().find(|p| p.level == l);
        let (sk, ku) = pmf.map(|p| (p.skew, p.kurt)).unwrap_or((0.0, 0.0));
        writeln!(sf, "0,{},{},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.6e},{:.6e}",
            l, r_tree, st.n_cells_total, st.mean, st.var, st.dvar, s2, sk, ku).map_err(stringify)?;
    }

    let pmf_path = dir.join(format!("pmfs_{}.csv", suffix));
    let mut pf = std::fs::File::create(&pmf_path).map_err(stringify)?;
    writeln!(pf, "realization,level,R_tree,n_total,count,frequency").map_err(stringify)?;
    for p in pmfs {
        for (k, &h) in p.histogram.iter().enumerate() {
            if h > 0 {
                writeln!(pf, "0,{},{},{},{},{}", p.level, p.r_tree, p.n_total, k, h).map_err(stringify)?;
            }
        }
    }

    let tpcf_path = dir.join(format!("tpcf_{}.csv", suffix));
    let mut tf = std::fs::File::create(&tpcf_path).map_err(stringify)?;
    writeln!(tf, "realization,level,k,r_tree,smoothing_h_fine,xi").map_err(stringify)?;
    for tp in tpcf {
        writeln!(tf, "0,{},{},{},{},{:.10e}",
            tp.level, tp.k, tp.r_tree, tp.smoothing_h_fine, tp.xi).map_err(stringify)?;
    }
    Ok(())
}

// ============================================================================
// multi-run subcommand
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_multi_run(
    args: CommonArgs,
    statistic: MultiRunStatistic,
    boundary: MultiRunBoundary,
    randoms_path: Option<PathBuf>,
    weights_data_path: Option<PathBuf>,
    weights_randoms_path: Option<PathBuf>,
    n_shifts: usize,
    shift_magnitude: f64,
    shift_seed: u64,
    resize_factors: Vec<f64>,
    resize_points_per_decade: f64,
    resize_min_scale: f64,
    bin_tol: f64,
    w_r_min: f64,
    max_depth: Option<usize>,
    cic_max_bins: usize,
    cic_skip_zero_bins: bool,
    compensated_sums: bool,
    xi_fit_basis: Option<String>,
    xi_fit_knots: usize,
    xi_fit_r_min: Option<f64>,
    xi_fit_r_max: Option<f64>,
    xi_fit_window: String,
    xi_fit_weighting: String,
    xi_fit_eval_n: usize,
) -> Result<(), String> {
    use morton_cascade::hier_bitvec_pair::{FieldStatsConfig, CicPmfConfig};
    use morton_cascade::multi_run::{CascadeRunPlan, CascadeRunner, CascadeRunSpec};

    let flat_d = load_points_f64(&args.input, args.dim)?;
    let n_d = flat_d.len() / args.dim;
    if !args.quiet {
        eprintln!("Loaded {} data points (dim={}) from {}",
                  n_d, args.dim, args.input.display());
    }

    let weights_d_vec = match &weights_data_path {
        Some(p) => Some(load_weights(p, n_d)?),
        None => None,
    };

    // Isolated mode reads randoms; periodic mode does not.
    let (flat_r, _n_r, weights_r_vec) = match boundary {
        MultiRunBoundary::Isolated => {
            let path = randoms_path.as_ref()
                .ok_or_else(|| "--randoms is required for --boundary isolated".to_string())?;
            let f = load_points_f64(path, args.dim)?;
            let n = f.len() / args.dim;
            if !args.quiet {
                eprintln!("Loaded {} random points from {}", n, path.display());
            }
            let w = match &weights_randoms_path {
                Some(p) => Some(load_weights(p, n)?),
                None => None,
            };
            (Some(f), n, w)
        }
        MultiRunBoundary::Periodic => (None, 0, None),
    };

    std::fs::create_dir_all(&args.output).map_err(stringify)?;

    // The CLI accepts box-size in physical units; the cascade works in
    // u64 trimmed coords. We pack with the user's box size and use
    // box_bits derived from the trimmed range.
    if !args.quiet {
        eprintln!("Plan: n_shifts={} shift_magnitude={} resize_factors={} \
                   resize_per_decade={}",
                  n_shifts, shift_magnitude, resize_factors.len(),
                  resize_points_per_decade);
        eprintln!("Statistic: {:?}, Boundary: {:?}, bin_tol: {}",
                  statistic, boundary, bin_tol);
    }

    // The build-plan helper stays D-generic via a macro, since CascadeRunPlan
    // is parameterised by D.
    macro_rules! build_plan {
        ($D:literal) => {{
            let mut plans: Vec<CascadeRunPlan<$D>> = Vec::new();

            let shift_plan = if n_shifts > 0 {
                CascadeRunPlan::<$D>::random_offsets(
                    n_shifts, shift_magnitude, shift_seed)
            } else {
                CascadeRunPlan::<$D>::just_base()
            };
            plans.push(shift_plan);

            let resize_plan: Option<CascadeRunPlan<$D>> = if !resize_factors.is_empty() {
                let mut runs = Vec::with_capacity(resize_factors.len());
                for (i, &s) in resize_factors.iter().enumerate() {
                    if s <= 0.0 {
                        return Err(format!(
                            "--resize-factors[{}] = {} must be > 0", i, s));
                    }
                    runs.push((format!("resize_{}", i),
                               CascadeRunSpec::<$D>::resize(s)));
                }
                Some(CascadeRunPlan { runs, intended_boundary: None })
            } else if resize_points_per_decade > 0.0 {
                Some(CascadeRunPlan::<$D>::log_spaced_resizings(
                    resize_min_scale, 1.0, resize_points_per_decade))
            } else {
                None
            };

            if let Some(rp) = resize_plan {
                let composed = CascadeRunPlan::compose(&plans[0], &rp);
                plans[0] = composed;
            }
            plans.into_iter().next().unwrap()
        }};
    }

    macro_rules! pack_data_and_box {
        ($D:literal) => {{
            let pts_d = pack_nd_u64::<$D>(&flat_d, args.box_size);
            let pts_r: Vec<[u64; $D]> = match &flat_r {
                Some(f) => pack_nd_u64::<$D>(f, args.box_size),
                None => Vec::new(),
            };
            // For the runner box_bits we need the joint-trimmed range of
            // data + randoms; periodic-mode users are responsible for
            // making sure all points are inside [0, box_size)^D.
            use morton_cascade::coord_range::CoordRange;
            let range = if pts_r.is_empty() {
                CoordRange::<$D>::analyze_pair(&pts_d, &[])
            } else {
                CoordRange::<$D>::analyze_pair(&pts_d, &pts_r)
            };
            let mut box_bits = [0u32; $D];
            for d in 0..$D {
                box_bits[d] = range.effective_bits[d].max(1);
            }
            (pts_d, pts_r, box_bits)
        }};
    }

    macro_rules! run_for_dim {
        ($D:literal) => {{
            let (pts_d, pts_r, box_bits) = pack_data_and_box!($D);
            let plan = build_plan!($D);
            if !args.quiet {
                eprintln!("Box bits: {:?}, plan size: {} runs",
                          box_bits, plan.len());
            }

            let runner = match boundary {
                MultiRunBoundary::Periodic => {
                    CascadeRunner::<$D>::new_periodic(
                        pts_d, weights_d_vec.clone(), box_bits, plan)
                }
                MultiRunBoundary::Isolated => {
                    CascadeRunner::<$D>::new_isolated(
                        pts_d, weights_d_vec.clone(),
                        pts_r, weights_r_vec.clone(),
                        box_bits, plan)
                }
            };
            let runner = if let Some(d) = max_depth {
                runner.with_l_max(d)
            } else {
                runner
            };

            // Trimmed→physical scaling matches the rest of the CLI:
            // trimmed coord 1 = physical box_size / 2^max_eff.
            let max_eff = box_bits.iter().copied().max().unwrap_or(0) as usize;
            let trimmed_to_phys = args.box_size / (1u64 << max_eff) as f64;

            // Diagnostics file is the same regardless of statistic.
            let diag_path = args.output.join("multi_run_diagnostics.csv");
            let mut df = std::fs::File::create(&diag_path).map_err(stringify)?;
            writeln!(df, "name,scale,offset,footprint_coverage,n_levels")
                .map_err(stringify)?;

            match statistic {
                MultiRunStatistic::FieldStats => {
                    let cfg = FieldStatsConfig {
                        w_r_min, hist_bins: 0,
                        hist_log_min: -3.0, hist_log_max: 3.0,
                        compensated_sums: compensated_sums,
                    };
                    let agg = runner.analyze_field_stats(&cfg, bin_tol);
                    write_field_stats_csv(&args, &agg, trimmed_to_phys)?;
                    write_diagnostics(&mut df, &agg.per_run_diagnostics)?;
                }
                MultiRunStatistic::Anisotropy => {
                    let cfg = FieldStatsConfig {
                        w_r_min, hist_bins: 0,
                        hist_log_min: -3.0, hist_log_max: 3.0,
                        compensated_sums: compensated_sums,
                    };
                    let agg = runner.analyze_anisotropy(&cfg, bin_tol);
                    write_anisotropy_csv::<$D>(&args, &agg, trimmed_to_phys)?;
                    write_diagnostics(&mut df, &agg.per_run_diagnostics)?;
                }
                MultiRunStatistic::CicPmf => {
                    let cfg = CicPmfConfig { max_bins: cic_max_bins };
                    let agg = runner.analyze_cic_pmf(&cfg, bin_tol);
                    write_cic_pmf_csv(&args, &agg, trimmed_to_phys,
                        cic_skip_zero_bins)?;
                    write_diagnostics(&mut df, &agg.per_run_diagnostics)?;
                }
                MultiRunStatistic::Xi => {
                    // bin_tol here is reused as the scale-grouping tolerance.
                    // Reasonable defaults give the user no surprises.
                    let agg = runner.analyze_xi(bin_tol);
                    write_xi_csv(&args, &agg, trimmed_to_phys)?;
                    write_diagnostics(&mut df, &agg.per_run_diagnostics)?;
                    // Optional continuous fit.
                    if let Some(basis_kind) = &xi_fit_basis {
                        run_xi_continuous_fit(
                            &args, &agg, trimmed_to_phys,
                            basis_kind, xi_fit_knots,
                            xi_fit_r_min, xi_fit_r_max,
                            &xi_fit_window, &xi_fit_weighting,
                            xi_fit_eval_n, $D)?;
                    }
                }
            }
            if !args.quiet {
                eprintln!("Wrote {}", diag_path.display());
            }
        }};
    }

    match args.dim {
        2 => run_for_dim!(2),
        3 => run_for_dim!(3),
        4 => run_for_dim!(4),
        5 => run_for_dim!(5),
        _ => return Err(format!("multi-run supports dim ∈ {{2,3,4,5}}; got {}",
                                args.dim)),
    }
    Ok(())
}

fn write_field_stats_csv<const D: usize>(
    args: &CommonArgs,
    agg: &morton_cascade::multi_run::AggregatedFieldStats<D>,
    trimmed_to_phys: f64,
) -> Result<(), String> {
    let path = args.output.join("multi_run_field_stats.csv");
    let mut f = std::fs::File::create(&path).map_err(stringify)?;
    writeln!(f, "physical_side,physical_side_phys,n_runs,sum_w_r_total,\
                 n_cells_active_total,mean_delta,var_delta,m3_delta,m4_delta,\
                 s3_delta,min_delta,max_delta,mean_delta_arv,var_delta_arv,\
                 n_cells_data_outside_total,sum_w_d_outside_total")
        .map_err(stringify)?;
    for b in &agg.by_side {
        writeln!(f, "{:.10e},{:.10e},{},{:.10e},{},{:.10e},{:.10e},{:.10e},\
                     {:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{},{:.10e}",
            b.physical_side, b.physical_side * trimmed_to_phys,
            b.n_contributing_runs, b.sum_w_r_total,
            b.n_cells_active_total,
            b.mean_delta, b.var_delta, b.m3_delta, b.m4_delta, b.s3_delta,
            b.min_delta, b.max_delta,
            b.mean_delta_across_run_var, b.var_delta_across_run_var,
            b.n_cells_data_outside_total, b.sum_w_d_outside_total
        ).map_err(stringify)?;
    }
    if !args.quiet {
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

fn write_anisotropy_csv<const D: usize>(
    args: &CommonArgs,
    agg: &morton_cascade::multi_run::AggregatedAnisotropyStats<D>,
    trimmed_to_phys: f64,
) -> Result<(), String> {
    let path = args.output.join("multi_run_anisotropy.csv");
    let mut f = std::fs::File::create(&path).map_err(stringify)?;
    let mut header = String::from(
        "physical_side,physical_side_phys,n_runs,sum_w_r_parents_total,n_parents_total");
    for d in 0..D { header.push_str(&format!(",w2_axis_{}", d)); }
    for e in 1..(1u32 << D) {
        header.push_str(&format!(",w2_p{:0width$b}", e, width = D as usize));
    }
    header.push_str(",quadrupole_los,reduced_quadrupole_los,quadrupole_los_arv");
    writeln!(f, "{}", header).map_err(stringify)?;

    for b in &agg.by_side {
        let mut line = format!(
            "{:.10e},{:.10e},{},{:.10e},{}",
            b.physical_side, b.physical_side * trimmed_to_phys,
            b.n_contributing_runs, b.sum_w_r_parents_total,
            b.n_parents_total);
        for d in 0..D {
            let v = b.mean_w_squared_axis.get(d).copied().unwrap_or(0.0);
            line.push_str(&format!(",{:.10e}", v));
        }
        for e in 1..(1usize << D) {
            let v = b.mean_w_squared_by_pattern.get(e).copied().unwrap_or(0.0);
            line.push_str(&format!(",{:.10e}", v));
        }
        line.push_str(&format!(",{:.10e},{:.10e},{:.10e}",
            b.quadrupole_los, b.reduced_quadrupole_los,
            b.quadrupole_los_across_run_var));
        writeln!(f, "{}", line).map_err(stringify)?;
    }
    if !args.quiet {
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

fn write_cic_pmf_csv<const D: usize>(
    args: &CommonArgs,
    agg: &morton_cascade::multi_run::AggregatedCicPmf<D>,
    trimmed_to_phys: f64,
    skip_zero_bins: bool,
) -> Result<(), String> {
    // CIC PMF is special: each bin has a histogram of variable length.
    // Default emits one row per (side, count_bin) entry up through the
    // largest observed bin, including zeros (so reshape-to-matrix is
    // trivial in pandas / R). With --cic-skip-zero-bins we omit the
    // zero-count bins for a much sparser file when distributions have
    // long zero tails.
    let path = args.output.join("multi_run_cic_pmf.csv");
    let mut f = std::fs::File::create(&path).map_err(stringify)?;
    writeln!(f, "physical_side,physical_side_phys,n_runs,n_cells_visited_total,\
                 n_cells_total,count_bin,histogram_count,histogram_density,\
                 mean,var,skew,kurt,mean_arv")
        .map_err(stringify)?;
    for b in &agg.by_side {
        let n_bins = b.histogram_counts.len().max(b.histogram_density.len());
        for k in 0..n_bins {
            let c = b.histogram_counts.get(k).copied().unwrap_or(0);
            let p = b.histogram_density.get(k).copied().unwrap_or(0.0);
            if skip_zero_bins && c == 0 && p == 0.0 { continue; }
            writeln!(f, "{:.10e},{:.10e},{},{},{},{},{},{:.10e},{:.10e},\
                         {:.10e},{:.10e},{:.10e},{:.10e}",
                b.physical_side, b.physical_side * trimmed_to_phys,
                b.n_contributing_runs, b.n_cells_visited_total,
                b.n_cells_total, k, c, p,
                b.mean, b.var, b.skew, b.kurt, b.mean_across_run_var
            ).map_err(stringify)?;
        }
    }
    if !args.quiet {
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

fn write_xi_csv<const D: usize>(
    args: &CommonArgs,
    agg: &morton_cascade::multi_run::AggregatedXi<D>,
    trimmed_to_phys: f64,
) -> Result<(), String> {
    // One row per (resize_group, shell). Resize groups stay separate
    // by design (different scales probe different shell volumes), so
    // downstream tools should index by (scale, level) when fitting a
    // continuous-function basis to combine across resizes.
    let path = args.output.join("multi_run_xi_raw.csv");
    let mut f = std::fs::File::create(&path).map_err(stringify)?;
    writeln!(f, "scale,n_shifts,level,r_center,r_center_phys,\
                 r_half_width,r_half_width_phys,\
                 dd_sum,rr_sum,dr_sum,n_d_sum,n_r_sum,\
                 xi_naive,xi_shift_bootstrap_var")
        .map_err(stringify)?;
    for g in &agg.by_resize {
        for s in &g.shells {
            writeln!(f, "{:.10e},{},{},{:.10e},{:.10e},{:.10e},{:.10e},\
                         {:.10e},{:.10e},{:.10e},{},{},{:.10e},{:.10e}",
                g.scale, g.n_shifts, s.level,
                s.r_center, s.r_center * trimmed_to_phys,
                s.r_half_width, s.r_half_width * trimmed_to_phys,
                s.dd_sum, s.rr_sum, s.dr_sum,
                s.n_d_sum, s.n_r_sum,
                s.xi_naive, s.xi_shift_bootstrap_var
            ).map_err(stringify)?;
        }
    }
    if !args.quiet {
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_xi_continuous_fit<const D: usize>(
    args: &CommonArgs,
    agg: &morton_cascade::multi_run::AggregatedXi<D>,
    trimmed_to_phys: f64,
    basis_kind: &str,
    n_knots: usize,
    r_min_user: Option<f64>,
    r_max_user: Option<f64>,
    window_kind: &str,
    weighting_kind: &str,
    eval_n: usize,
    dim: usize,
) -> Result<(), String> {
    use morton_cascade::multi_run::xi_continuous::{
        XiBasis, XiWindowWeighting, XiMeasurementWeighting,
        fit_xi_continuous,
    };

    // Resolve r_min / r_max defaults from the AggregatedXi if user
    // didn't supply them. Use the smallest finite-width shell as r_min
    // and the largest as r_max.
    let (data_r_min, data_r_max) = {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0_f64;
        for g in &agg.by_resize {
            for s in &g.shells {
                if s.r_half_width <= 0.0 { continue; }
                let l = s.r_center - s.r_half_width;
                let h = s.r_center + s.r_half_width;
                if l > 0.0 && l < lo { lo = l; }
                if h > hi { hi = h; }
            }
        }
        (lo, hi)
    };
    if !data_r_min.is_finite() || data_r_max <= 0.0 {
        return Err("xi-fit: no usable shells (all degenerate)".to_string());
    }
    let r_min = r_min_user.unwrap_or(data_r_min);
    let r_max = r_max_user.unwrap_or(data_r_max);
    if r_min >= r_max {
        return Err(format!("xi-fit: r_min ({}) must be < r_max ({})", r_min, r_max));
    }

    let basis = match basis_kind {
        "linear-bsplines" => XiBasis::LinearBSplineLogR {
            n_knots, r_min, r_max,
        },
        other => return Err(format!(
            "unknown --xi-fit-basis `{}` (supported: linear-bsplines)",
            other)),
    };
    let window = match window_kind {
        "euclidean"     => XiWindowWeighting::EuclideanRPow { d: dim },
        "empirical-rr"  => XiWindowWeighting::EmpiricalRR { fallback_d: dim },
        other => return Err(format!(
            "unknown --xi-fit-window `{}` (supported: euclidean, empirical-rr)",
            other)),
    };
    let weighting = match weighting_kind {
        "shift-bootstrap"          => XiMeasurementWeighting::ShiftBootstrap,
        "shift-bootstrap-poisson"  => XiMeasurementWeighting::ShiftBootstrapPlusPoisson,
        "uniform"                  => XiMeasurementWeighting::Uniform,
        other => return Err(format!(
            "unknown --xi-fit-weighting `{}` (supported: shift-bootstrap, \
             shift-bootstrap-poisson, uniform)", other)),
    };

    if !args.quiet {
        eprintln!("xi-fit: basis={} n_knots={} r_min={:.4e} r_max={:.4e} \
                   window={} weighting={}",
            basis_kind, n_knots, r_min, r_max, window_kind, weighting_kind);
    }

    let fit = fit_xi_continuous(agg, &basis, window, weighting)?;

    if !args.quiet {
        eprintln!("xi-fit: n_meas_used={}, n_basis={}, reduced_chi2={:.4}",
            fit.n_measurements_used, fit.n_basis, fit.reduced_chi_squared);
    }

    // Coefficient CSV: one row per basis function with knot location
    // and coefficient value + 1σ from the diagonal of the cov matrix.
    let coef_path = args.output.join("multi_run_xi_fit_coefs.csv");
    let mut cf = std::fs::File::create(&coef_path).map_err(stringify)?;
    writeln!(cf, "knot_index,knot_log_r,knot_r,knot_r_phys,coef,coef_sigma")
        .map_err(stringify)?;
    let log_min = r_min.ln();
    let log_max = r_max.ln();
    let du = (log_max - log_min) / (n_knots as f64 - 1.0).max(1.0);
    for n in 0..fit.n_basis {
        let log_r = log_min + n as f64 * du;
        let r = log_r.exp();
        let sigma = fit.coef_cov[n][n].max(0.0).sqrt();
        writeln!(cf, "{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}",
            n, log_r, r, r * trimmed_to_phys, fit.coefs[n], sigma)
            .map_err(stringify)?;
    }
    if !args.quiet {
        eprintln!("Wrote {}", coef_path.display());
    }

    // Evaluation CSV: log-spaced grid r in [r_min, r_max] with fitted
    // ξ(r) and 1σ uncertainty from coef covariance.
    let eval_path = args.output.join("multi_run_xi_evaluated.csv");
    let mut ef = std::fs::File::create(&eval_path).map_err(stringify)?;
    writeln!(ef, "r,r_phys,xi_fit,xi_fit_sigma").map_err(stringify)?;
    for k in 0..eval_n {
        let log_r = log_min + (k as f64) * (log_max - log_min) / (eval_n as f64 - 1.0).max(1.0);
        let r = log_r.exp();
        let xi = fit.evaluate(r);
        let sigma = fit.sigma_at(r);
        writeln!(ef, "{:.10e},{:.10e},{:.10e},{:.10e}",
            r, r * trimmed_to_phys, xi, sigma).map_err(stringify)?;
    }
    if !args.quiet {
        eprintln!("Wrote {}", eval_path.display());
    }
    Ok(())
}

fn write_diagnostics<const D: usize>(
    f: &mut std::fs::File,
    diags: &[morton_cascade::multi_run::RunDiagnostic<D>],
) -> Result<(), String> {
    for d in diags {
        // Format offset as comma-separated within a single quoted field
        let off_str: String = d.spec.offset_frac.iter()
            .map(|v| format!("{:.6}", v))
            .collect::<Vec<_>>().join(";");
        writeln!(f, "{},{:.6},{},{:.6},{}",
            d.name, d.spec.scale, off_str, d.footprint_coverage, d.n_levels
        ).map_err(stringify)?;
    }
    Ok(())
}

fn halton(mut i: usize, base: usize) -> f64 {
    let mut f = 1.0 / base as f64;
    let mut r = 0.0;
    while i > 0 {
        r += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    r
}

fn stringify<E: std::fmt::Display>(e: E) -> String { e.to_string() }

// ============================================================================
// Entry point
// ============================================================================

fn main() -> ExitCode {
    let sub = match parse_args() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}", e);
            eprintln!("Run `morton-cascade --help` for usage.");
            return ExitCode::from(2);
        }
    };

    let result = match sub {
        Subcommand::Cascade(c) => run_cascade(c),
        Subcommand::Pmf(c) => run_pmf(c),
        Subcommand::PmfWindows { common, points_per_decade, side_min, side_max, explicit_sides } =>
            run_pmf_windows(common, points_per_decade, side_min, side_max, explicit_sides),
        Subcommand::Tpcf { common, max_lag_level } => run_tpcf(common, max_lag_level),
        Subcommand::Fingerprint { common, with_poisson_ref, n_real, .. } =>
            run_fingerprint(common, with_poisson_ref, n_real),
        Subcommand::Pairs { common, max_depth, crossover_threshold } =>
            run_pairs(common, max_depth, crossover_threshold),
        Subcommand::Gradient { common, target_level } =>
            run_gradient(common, target_level),
        Subcommand::Xi { common, randoms, weights_data, weights_randoms, max_depth, crossover_threshold } =>
            run_xi(common, randoms, weights_data, weights_randoms, max_depth, crossover_threshold),
        Subcommand::FieldStats { common, randoms, weights_data, weights_randoms,
            max_depth, crossover_threshold, w_r_min, hist_bins, hist_log_min, hist_log_max } =>
            run_field_stats(common, randoms, weights_data, weights_randoms,
                max_depth, crossover_threshold,
                w_r_min, hist_bins, hist_log_min, hist_log_max),
        Subcommand::Anisotropy { common, randoms, weights_data, weights_randoms,
            max_depth, crossover_threshold, w_r_min } =>
            run_anisotropy(common, randoms, weights_data, weights_randoms,
                max_depth, crossover_threshold, w_r_min),
        Subcommand::Scattering { common, randoms, weights_data, weights_randoms,
            max_depth, crossover_threshold, w_r_min } =>
            run_scattering(common, randoms, weights_data, weights_randoms,
                max_depth, crossover_threshold, w_r_min),
        Subcommand::PmfWindowsPaired { common, randoms, weights_data, weights_randoms,
            points_per_decade, side_min, side_max, explicit_sides,
            w_r_min, hist_bins, hist_log_min, hist_log_max } =>
            run_pmf_windows_paired(common, randoms, weights_data, weights_randoms,
                points_per_decade, side_min, side_max, explicit_sides,
                w_r_min, hist_bins, hist_log_min, hist_log_max),
        Subcommand::MultiRun { common, statistic, boundary, randoms,
            weights_data, weights_randoms, n_shifts, shift_magnitude, shift_seed,
            resize_factors, resize_points_per_decade, resize_min_scale,
            bin_tol, w_r_min, max_depth, cic_max_bins, cic_skip_zero_bins,
            compensated_sums,
            xi_fit_basis, xi_fit_knots, xi_fit_r_min, xi_fit_r_max,
            xi_fit_window, xi_fit_weighting, xi_fit_eval_n } =>
            run_multi_run(common, statistic, boundary, randoms,
                weights_data, weights_randoms, n_shifts, shift_magnitude, shift_seed,
                resize_factors, resize_points_per_decade, resize_min_scale,
                bin_tol, w_r_min, max_depth, cic_max_bins, cic_skip_zero_bins,
                compensated_sums,
                xi_fit_basis, xi_fit_knots, xi_fit_r_min, xi_fit_r_max,
                xi_fit_window, xi_fit_weighting, xi_fit_eval_n),
        Subcommand::AngularKnnCdf(args) => run_angular_knn_cdf(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}
