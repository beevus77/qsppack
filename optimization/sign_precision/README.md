# Fixed-delta precision experiment

The target is the bounded odd approximation to 1 on `[0.002,1]`, with oddness
supplying the negative interval. The earlier figures and polynomial library
are preserved. New constructions use the same float64 Remez algorithm, initial
exchange points, root grids, and numerical settings as the preceding studies.

The experiment verifies minimum degrees at epsilon=1e-5 and 1e-6 and combines
them with the existing epsilon=1e-4 and 1e-7 minima. An affine least-squares fit
`degree = a*ln(1/epsilon) + b` to those four points supplies the first degree
for a direct epsilon=1e-10 attempt. There are no intermediate 1e-8 or 1e-9
Remez constructions. A subsequent correction, if needed, uses the measured
error of that first attempt and aims slightly below epsilon. Precision probes
are distinguished from verified exact minimum degrees in the plot.

## Artifacts and reproduction

- `degree_vs_epsilon.png` and `.pdf`: degree plot and logarithmic fit.
- `results.json`: the four verified minimum-degree brackets.
- `log_fit.json`: model coefficients, residuals, and extrapolated degree.
- `fit_<degree>.json`: newly constructed coefficients, extrema, errors,
  numerical audit results, stage timings, and peak resident memory. Fits are
  retained even if an audit fails.
- `precision_probes.json`: concise records of the direct 1e-10 attempts.
- `attempts.json`: process wall times, stdout, stderr, and timeout information.
- `spotcheck_<degree>.json`, when requested: 80-digit evaluations at selected
  extrema of the stored float64 polynomial. These are diagnostic checks, not
  higher-precision reoptimization or a new full-domain certificate.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qsppack-sign-mpl python optimization/sign_precision.py
```

The run resumes saved fits. `--plots-only` regenerates completed results.
`--spotcheck DEGREE` runs the optional diagnostic using `mpmath` (available
in the experiment environment). The subprocess timeout is 90 minutes and the
degree ceiling is 12001. Those are experiment budgets, not algorithmic limits.

## What consumes memory

`ConstrainedRemezFitter._design` materializes
`cos(omega[:,None] * orders[None,:])`. `_polynomial_derivative` similarly
materializes grid-by-order sine arrays and elementwise products. The stress
settings use a root grid of approximately 12d points and `(d+1)/2` odd orders.
One float64 grid-by-order array therefore occupies approximately `48*d^2`
bytes. Evaluation can hold two such arrays at once, before other allocations.
At degree 11000 that is about 5.8 GB per array and about 11.6 GB for two arrays.

The Remez exchange matrix has only approximately `(d/2)^2` entries, about
`2*d^2` bytes; it is not the largest allocation at these degrees. Its dense
least-squares factorization still costs substantial time and workspace.
`check_feasibility` obtains roots of the degree-d-1 derivative using
`numpy.polynomial.chebyshev.chebroots`, which builds a dense colleague matrix
and solves an eigenvalue problem. It is invoked for both raw and scaled
coefficients, so these feasibility checks are another major cost.

The `exchange` and `metrics` timers are separate. The `root_search` timer
overlaps `exchange` and must not be added to it when totaling runtime. The
independent audit has its own timer. Peak resident memory covers the entire
worker process. Chunked/Clenshaw evaluation could reduce the largest transient
arrays; that optimization is not applied in this experiment.

## What limits numerical precision

There is no single experimentally established smallest epsilon for every
delta and degree. The exact existing code thresholds are:

- `_roots` skips an interval if every sampled derivative magnitude is below
  1e-10 and treats derivative samples below 1e-12 as zero. These are derivative
  thresholds, not direct approximation-error thresholds.
- Remez's exchange error allowance is 2e-14 in this experiment. Candidate
  errors are obtained by subtracting a float64 polynomial value from 1.
- The independent audit allows ripple spread up to
  `max(5e-13, 2e-5*error)` and uses 5e-13 for the magnitude and error-identity
  checks. Consequently, passing an audit at a tiny epsilon does not imply the
  same *relative* ripple accuracy as in the larger-error experiments.
- A minimum-degree decision requires
  `E_d + 5e-13 < epsilon < E_(d-2) - 5e-13`. This criterion can fail even when
  both polynomials and their errors have been accurately computed.

From the measured near-exponential error decay, neighboring odd degrees at
delta=0.002 improve the error by approximately 0.4%. Near epsilon=1e-10 the
expected difference is only about 4e-13, which is smaller than the combined
1e-12 margin required by the current exact-degree decision rule. Thus the
existing degree certificate is expected to become limiting before RAM in
this particular experiment. That is a limit of the current numerical decision
rule, not a theorem that the mathematical optimum cannot be resolved.

The underlying calculations also have finite precision: dense cosine
evaluation, the float64 least-squares solve, differentiation/root finding,
and subtraction of polynomial values close to 1 can all contribute error.
Float64 epsilon is 2.22e-16. On this ARM machine NumPy `longdouble` also has a
52-bit fraction and provides no extra precision. Resolving smaller ripples or
neighboring-degree differences may require a genuinely higher-precision
solve/evaluation path as well as scale-aware root and convergence criteria.
Simply reducing audit tolerances would not establish the missing accuracy.
