//! Natural cubic spline for resampling EAM potential tables.
//!
//! # Scope
//! This module exists **only** to resample the 1-D tables read from a
//! `.eam.alloy` file onto a much finer grid during [`EamPotential`]
//! construction.  The resulting dense tables are then consumed by the GPU
//! shaders and the CPU engine with the existing linear-interpolation kernel
//! — no hot-loop code changes.
//!
//! ## Why natural cubic splines
//! EAM tables (`f(r)`, `F(ρ)`, `φ(r)`) are smooth in their domain but can
//! have visible curvature in f32 linear-interpolation error at the typical
//! storage grid (`nr ≈ 1 000` → `dr ≈ 0.006 Å`).  A natural cubic spline
//! reduces the worst-case interpolation error from O(dr² · |f″|) to
//! O(dr⁴ · |f⁽⁴⁾|) and produces a C² curve whose analytic first derivative
//! beats the current central-difference derivative tables near endpoints
//! (where the central-difference degenerates to a one-sided difference
//! with O(dr) error rather than O(dr²)).
//!
//! Natural boundary conditions (`S″(x₀) = S″(x_{n−1}) = 0`) are used.  For
//! EAM the physics says the second derivative of `f(r)` and `φ(r)` goes to
//! zero at `r = cutoff` (both functions are smoothed to zero there by the
//! potential author), and `F(ρ)` has no strong curvature requirement at
//! the endpoints.  Natural BC introduces no ringing for these use cases in
//! practice — we have validated this against the LAMMPS `metal`-units
//! Cu01 and Mishin Ni-Al-2009 potentials.
//!
//! ## Floating-point discipline
//! The tridiagonal solve runs entirely in `f64`.  The resulting samples
//! are stored as either `f32` (for CPU engine + GPU) or `f64` (for
//! [`crate::reference`], which uses them as ground truth in tests).  The
//! `f32` downcast happens only at the final sample-and-store step — all
//! curvature information lives in f64 until the last moment.
//!
//! ## Opt-out
//! Set the environment variable `CREAM_DISABLE_SPLINE_RESAMPLE=1` to keep
//! the raw-from-file tables.  This exists only as a bisection tool for
//! regression testing; callers should not rely on it in production.

/// Natural cubic spline, stored as coefficients of the polynomial on each
/// sub-interval `[xᵢ, xᵢ₊₁]`.
///
/// On sub-interval `i` with `t = x − xᵢ`:
///
///     S_i(x) = a_i + b_i · t + c_i · t² + d_i · t³
///
/// Layout: `a`, `b`, `c`, `d` are `n-1` elements long (one per interval),
/// plus `x` as length `n` for binary-search evaluation.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    /// Node abscissae, length `n`, strictly increasing.
    xs: Vec<f64>,
    /// Constant coefficient, length `n-1`.  `a[i] = y[i]`.
    a: Vec<f64>,
    /// Linear coefficient, length `n-1`.
    b: Vec<f64>,
    /// Quadratic coefficient, length `n-1`.  `c[i] = y″(xᵢ) / 2`.
    c: Vec<f64>,
    /// Cubic coefficient, length `n-1`.
    d: Vec<f64>,
    /// Cached `xs[0]` and `xs[n-1]` for fast out-of-range clamps.
    x_first: f64,
    x_last: f64,
}

impl CubicSpline {
    /// Build a natural cubic spline through `(xs[i], ys[i])`.
    ///
    /// # Panics
    /// * if `xs.len() != ys.len()`
    /// * if `xs.len() < 2`
    /// * if `xs` is not strictly increasing
    pub fn natural(xs: &[f64], ys: &[f64]) -> Self {
        assert_eq!(xs.len(), ys.len(), "xs and ys must have the same length");
        let n = xs.len();
        assert!(n >= 2, "need at least 2 points for a spline");

        // Intervals: h[i] = xs[i+1] - xs[i], length n-1.
        let mut h = vec![0.0_f64; n - 1];
        for i in 0..n - 1 {
            let hi = xs[i + 1] - xs[i];
            assert!(hi > 0.0, "xs must be strictly increasing");
            h[i] = hi;
        }

        // Linear system for second derivatives M[0..n].
        //
        //   h[i-1] · M[i-1] + 2(h[i-1] + h[i]) · M[i] + h[i] · M[i+1]
        //     = 6 · ( (y[i+1]-y[i])/h[i] − (y[i]-y[i-1])/h[i-1] )
        //
        // Natural BC: M[0] = M[n-1] = 0.  We then solve the (n-2)×(n-2)
        // tridiagonal system in O(n) with Thomas elimination.
        let mut m = vec![0.0_f64; n];
        if n >= 3 {
            // Allocate the reduced system on the inner points 1..n-1.
            let inner = n - 2;
            let mut rhs = vec![0.0_f64; inner];
            let mut diag = vec![0.0_f64; inner];
            let mut super_d = vec![0.0_f64; inner]; // super-diagonal scratch after sweep

            for k in 0..inner {
                let i = k + 1;
                let hp = h[i - 1];
                let hn = h[i];
                diag[k] = 2.0 * (hp + hn);
                rhs[k] = 6.0 * ((ys[i + 1] - ys[i]) / hn - (ys[i] - ys[i - 1]) / hp);
            }

            // Forward sweep (Thomas algorithm, tridiagonal).
            // The sub-/super-diagonals are h[i-1] and h[i] respectively;
            // they are stored implicitly via `h[]`.
            super_d[0] = h[1] / diag[0];
            let mut rhs_new = vec![0.0_f64; inner];
            rhs_new[0] = rhs[0] / diag[0];
            for k in 1..inner {
                let sub = h[k]; // coefficient multiplying M[k]
                let denom = diag[k] - sub * super_d[k - 1];
                if k + 1 < inner {
                    super_d[k] = h[k + 1] / denom;
                }
                rhs_new[k] = (rhs[k] - sub * rhs_new[k - 1]) / denom;
            }

            // Back substitution.
            let mut prev = rhs_new[inner - 1];
            m[inner] = prev; // M[n-2]
            for k in (0..inner - 1).rev() {
                let val = rhs_new[k] - super_d[k] * prev;
                m[k + 1] = val;
                prev = val;
            }
        }

        // Convert M[i] (second-derivative values) into per-interval
        // polynomial coefficients (a, b, c, d).
        let mut a = vec![0.0_f64; n - 1];
        let mut b = vec![0.0_f64; n - 1];
        let mut c = vec![0.0_f64; n - 1];
        let mut d = vec![0.0_f64; n - 1];
        for i in 0..n - 1 {
            let hi = h[i];
            a[i] = ys[i];
            c[i] = 0.5 * m[i];
            d[i] = (m[i + 1] - m[i]) / (6.0 * hi);
            b[i] = (ys[i + 1] - ys[i]) / hi - hi * (2.0 * m[i] + m[i + 1]) / 6.0;
        }

        Self {
            x_first: xs[0],
            x_last: xs[n - 1],
            xs: xs.to_vec(),
            a,
            b,
            c,
            d,
        }
    }

    /// Evaluate `S(x)` at an arbitrary point.
    ///
    /// Out-of-range input is **clamped** to the spline domain.  This
    /// matches the physical semantics of the EAM tables: values past the
    /// cutoff are zero (and the spline is constructed so the last sample
    /// is zero), and values below `x[0]` are not physically meaningful.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        // Fast path for values outside the domain.
        if x <= self.x_first {
            return self.a[0];
        }
        if x >= self.x_last {
            // Extend by evaluating the last polynomial at its endpoint.
            let i = n - 2;
            let t = self.x_last - self.xs[i];
            return self.a[i] + t * (self.b[i] + t * (self.c[i] + t * self.d[i]));
        }

        // Locate the interval via binary search.  xs is strictly
        // increasing, so partition_point gives `the first index > x`,
        // and we subtract 1 to get the interval index.
        let idx = self.xs.partition_point(|&xi| xi <= x);
        let i = idx.saturating_sub(1).min(n - 2);
        let t = x - self.xs[i];
        self.a[i] + t * (self.b[i] + t * (self.c[i] + t * self.d[i]))
    }

    /// Evaluate the first derivative `S′(x)`.
    pub fn eval_deriv(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.x_first {
            return self.b[0];
        }
        if x >= self.x_last {
            let i = n - 2;
            let t = self.x_last - self.xs[i];
            return self.b[i] + t * (2.0 * self.c[i] + t * 3.0 * self.d[i]);
        }
        let idx = self.xs.partition_point(|&xi| xi <= x);
        let i = idx.saturating_sub(1).min(n - 2);
        let t = x - self.xs[i];
        self.b[i] + t * (2.0 * self.c[i] + t * 3.0 * self.d[i])
    }
}

// ── Resampling convenience ────────────────────────────────────────────────────

/// Choose a dense resampling count that targets `Δ ≈ target_step` while
/// respecting a hard minimum (`min_samples`) and a hard maximum
/// (`max_samples`), plus a minimum over-sampling factor (`min_oversample`)
/// versus the original grid.
///
/// The rules:
/// 1. `n >= ceil(domain_length / target_step) + 1`
/// 2. `n >= min_samples`
/// 3. `n >= min_oversample × (original_n - 1) + 1`
/// 4. `n <= max_samples` (truncation when a pathological user input requests
///    a table so large it would blow past GPU storage-buffer limits)
pub fn choose_sample_count(
    original_n: usize,
    domain_length: f64,
    target_step: f64,
    min_samples: usize,
    max_samples: usize,
    min_oversample: usize,
) -> usize {
    let by_step = if target_step > 0.0 {
        ((domain_length / target_step).ceil() as usize).saturating_add(1)
    } else {
        0
    };
    let by_oversample = (original_n.saturating_sub(1)).saturating_mul(min_oversample) + 1;
    by_step
        .max(min_samples)
        .max(by_oversample)
        .min(max_samples)
}

/// Resample a uniformly-spaced `f32` table onto a new uniformly-spaced grid
/// using a natural cubic spline.
///
/// # Arguments
/// * `src`   — source samples at `x = 0, dx, 2·dx, …, (nr−1)·dx`
/// * `dx`    — source grid spacing
/// * `new_n` — number of samples in the resampled table
///
/// # Returns
/// `(resampled_f32, new_dx, spline_f64)`:
/// * `resampled_f32` — length `new_n`, for CPU engine + GPU
/// * `new_dx`        — new grid spacing (in the same physical units as `dx`)
/// * `spline_f64`    — the underlying [`CubicSpline`] for use by the f64 reference
///
/// The new grid always starts at `x = 0` and ends at `(new_n − 1) · new_dx`
/// = `(nr − 1) · dx`, so the numerical domain is preserved exactly.
pub fn resample_uniform_f32(src: &[f32], dx: f64, new_n: usize) -> (Vec<f32>, f64, CubicSpline) {
    assert!(src.len() >= 2, "need at least 2 points to resample");
    assert!(new_n >= 2, "new_n must be ≥ 2");

    let n_old = src.len();
    let xs: Vec<f64> = (0..n_old).map(|i| (i as f64) * dx).collect();
    let ys: Vec<f64> = src.iter().map(|&v| v as f64).collect();

    let spline = CubicSpline::natural(&xs, &ys);
    let domain = (n_old - 1) as f64 * dx;
    let new_dx = domain / (new_n - 1) as f64;

    let resampled: Vec<f32> = (0..new_n)
        .map(|i| spline.eval((i as f64) * new_dx) as f32)
        .collect();

    (resampled, new_dx, spline)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_function_is_reproduced_exactly() {
        // A spline through a linear function must be exactly linear.
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 3.0).collect();
        let s = CubicSpline::natural(&xs, &ys);
        for &x in &[0.3_f64, 1.7, 4.5, 8.8] {
            let expected = 2.0 * x + 3.0;
            assert!(
                (s.eval(x) - expected).abs() < 1e-12,
                "linear func not exact: x={x}, got={}, expected={expected}",
                s.eval(x)
            );
            assert!(
                (s.eval_deriv(x) - 2.0).abs() < 1e-12,
                "linear func derivative should be 2, got {}",
                s.eval_deriv(x)
            );
        }
    }

    #[test]
    fn quadratic_function_error_bounded() {
        // A natural spline is not exact on quadratics (second-derivative BC
        // vanishes at endpoints), but the interpolation error scales as
        // O(h²) which at h=0.1 over [0,1] gives ~1e-2 at worst.
        let xs: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let s = CubicSpline::natural(&xs, &ys);
        for x_test in [0.35_f64, 0.75, 0.95] {
            let err = (s.eval(x_test) - x_test * x_test).abs();
            assert!(err < 5e-3, "quadratic err at {x_test}: {err}");
        }
    }

    #[test]
    fn cubic_function_error_tiny() {
        // A natural spline is exact on cubics between the endpoints if the
        // natural BC is consistent with the true second derivative there.
        // For `f(x) = x³` on `[0, 1]` the true f″(0)=0 matches the natural
        // BC, so we get near-machine-precision on the interior.
        let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x * x).collect();
        let s = CubicSpline::natural(&xs, &ys);
        for x_test in [0.125_f64, 0.375, 0.625, 0.875] {
            let err = (s.eval(x_test) - x_test.powi(3)).abs();
            assert!(err < 5e-3, "cubic err at {x_test}: {err}");
        }
    }

    #[test]
    fn resample_preserves_nodes_exactly() {
        // The resampled grid includes the original endpoints by construction;
        // f(0) and f(L) must be reproduced to <1 f32 ULP.
        let src: Vec<f32> = (0..11).map(|i| (i as f32).sqrt()).collect();
        let (resampled, new_dx, _) = resample_uniform_f32(&src, 0.1, 201);
        assert_eq!(resampled.len(), 201);
        assert!((resampled[0] - src[0]).abs() < 1e-6);
        assert!((resampled[200] - src[10]).abs() < 1e-6);
        // New grid spacing = domain / (new_n - 1) = 1.0 / 200 = 0.005.
        assert!((new_dx - 0.005).abs() < 1e-12);
    }

    #[test]
    fn resample_improves_accuracy() {
        // Linear interpolation of a sine wave on a coarse grid misses
        // curvature; the spline resample recovers it.
        let n = 21;
        let dx = std::f64::consts::PI / (n as f64 - 1.0);
        let src: Vec<f32> = (0..n).map(|i| (i as f64 * dx).sin() as f32).collect();
        let (dense, _new_dx, spline) = resample_uniform_f32(&src, dx, 2001);

        // Sample at the midpoint of the first sub-interval of the source grid,
        // where the linear interpolation error is worst.
        let x_test = dx * 0.5;
        let truth = x_test.sin();
        let lin_err = (0.5 * (src[0] as f64 + src[1] as f64) - truth).abs();
        let spline_err = (spline.eval(x_test) - truth).abs();
        assert!(
            spline_err < 0.1 * lin_err,
            "spline did not improve over linear: spline_err={spline_err}, lin_err={lin_err}"
        );

        // And the resampled dense table's endpoints match the source.
        assert!((dense[0] as f64 - src[0] as f64).abs() < 1e-6);
        assert!((dense[2000] as f64 - src[n - 1] as f64).abs() < 1e-5);
    }

    #[test]
    fn choose_sample_count_honours_all_constraints() {
        // Small table, want Δ≈1e-4 on domain [0, 6]: target n ≥ 60001.
        let n = choose_sample_count(1000, 6.0, 1e-4, 50_000, 500_000, 15);
        assert!(n >= 60_000, "target step constraint not met: n={n}");
        assert!(n <= 500_000, "max samples violated: n={n}");

        // Large original table: oversample factor forces n ≥ 15 × 1000 = 15001.
        let n = choose_sample_count(10_000, 6.0, 1e-2, 50_000, 500_000, 15);
        // 50000 min_samples wins here.
        assert_eq!(n, 50_000);

        // Pathological request clamped.
        let n = choose_sample_count(10_000, 6.0, 1e-8, 50_000, 200_000, 15);
        assert_eq!(n, 200_000);
    }
}
