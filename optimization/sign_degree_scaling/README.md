# Odd sign approximation: degree scaling

The experiment uses `qsppack.remez.ConstrainedRemezFitter` to approximate 1 on
`[delta, 1]` with an odd polynomial. Oddness supplies the approximation to -1 on
`[-1, -delta]`. The plotted error is the supremum over these two fitting
intervals; there is no accuracy requirement inside the transition gap.
The final polynomial must satisfy `|P(x)| <= 1` throughout `[-1, 1]`.

## Figures and data

- `degree_vs_delta.png` / `.pdf`: nine gap endpoints from 0.3 down to 0.02,
  at errors 1e-4 and 1e-7. Both axes are logarithmic. Includes the requested
  inverse-square-root reference and inverse-delta references.
- `degree_vs_epsilon.png` / `.pdf`: thirteen errors from 1e-2 down to 1e-8,
  at gap endpoints 0.2, 0.1, and 0.05. Both axes are logarithmic. Each dashed
  logarithmic reference is normalized to its measured curve at epsilon=1e-5.
- `equioscillation.png` / `.pdf`: a representative approximation, its signed
  raw error, and its bounded error, with independently computed extrema marked.
- `minimum_degrees.json`: each minimum odd degree, its error, the failing
  preceding odd degree and error, and scaling/alternation diagnostics.
- `fits.json`: all Remez evaluations, including Chebyshev coefficients,
  extrema, convergence status, scaling factors, and ripple spreads.

Open markers flag polynomials that require rescaling. All plotted polynomials
are the bounded variants; the raw errors and coefficients remain in the data.

The measured log-log slopes of degree versus delta are -0.997 at epsilon=1e-4
and -1.001 at epsilon=1e-7 over the sampled range. These support an inverse-delta
dependence. Representative minimum odd degrees are:

| delta | epsilon=1e-4 | epsilon=1e-7 |
| --- | ---: | ---: |
| 0.1 | 87 | 153 |
| 0.05 | 173 | 305 |
| 0.025 | 343 | 609 |
| 0.02 | 429 | 761 |

The representative degree-87 polynomial has bounded error 8.37002783e-5;
degree 85 fails with error 1.03407240e-4. Its scaling divisor is
1.0000418518906515, and all 45 required alternating extrema are present.
The raw ripple magnitude spread is approximately 1.48e-14.

## Why scaling still yields the bounded optimum here

Write the unconstrained odd minimax approximation as `R`, with uniform error
`e` relative to 1. It alternates between `1-e` and `1+e` on the positive fitting
interval. In every audited fit, all positive derivative roots lie in that
interval, and `R` has no interior extrema in the gap. Thus its full-domain
maximum magnitude is `1+e`.

QSPPACK's rescaling gives `P = R/(1+e)` and bounded error

`E = 2e/(1+e)`.

Its error `1-P` alternates between 0 and E. Equivalently, its error about the
midpoint `1-E/2` alternates symmetrically with amplitude `E/2`. An error relative
to 1 that alternates above and below zero is incompatible with the upper bound
`P <= 1` on the positive fitting interval.

This scaling preserves optimality for this particular constant-target problem.
If a bounded odd competitor had error `E' < E`, its values on the positive fit
interval would lie in `[1-E', 1]`. Dividing that competitor by `1-E'/2` would
produce an odd approximation to 1 with error at most `E'/(2-E') < e`, contradicting
unconstrained minimax optimality. This argument depends on the independently
verified feasibility of the scaled R; it does not justify scaling arbitrary
constrained Remez problems.

## Minimum-degree and oscillation verification

The search is binary in the integer index k of odd degree `d=2k+1`, with cached
evaluations. Each selected degree passes the requested error, and `d-2` fails.
The odd polynomial spaces are nested, so these two checks establish the
minimum odd degree once the minimax property is verified.

For `d=2m-1`, the odd basis has m coefficients. The script requires exactly
`m+1` alternating, equal-magnitude raw error extrema, including the two fitting
endpoints. On `[delta,1]`, the odd basis is a Haar space: dividing any nonzero
`x Q(x^2)` by positive x reduces its zeros to those of a degree-at-most-m-1
polynomial. Thus the alternation condition certifies the raw minimax fit.

An independent Chebyshev derivative evaluation and Brent root solver find the
`m-1` positive derivative roots. Even parity of the derivative supplies their
negative counterparts, accounting for its entire degree `2m-2`. All roots must
lie outside the transition gap. This makes endpoint/root evaluations sufficient
for continuous-domain errors and magnitude bounds, rather than relying on a
sampled error grid. These are floating-point numerical certificates, not
interval-arithmetic proofs.

The raw and midpoint-centered bounded ripple spreads must be no more than
`max(5e-13, 2e-5 * error)`. Bounds and the raw-to-bounded error identity are
checked to 5e-13. Both final search endpoints must be separated from the
requested epsilon by more than 5e-13. Remez must report convergence with no
rank-deficient exchange systems. All coefficients of even order are exactly
zero by construction.

Initial exchange points are Chebyshev-distributed in `x^2`, using
`sqrt(delta^2 + (1-delta^2)*sin(theta)^2)` for uniform theta in `[0,pi/2]`.
This avoids rank-deficient intermediate exchanges observed with the default
uniform angular seed at small target errors. The library is otherwise unchanged.

## Interpreting the gap reference

For an odd polynomial, the substitution `P(x)=x Q(x^2)` transforms the positive
interval into `[delta^2,1]`. This matters when interpreting an inverse-square-root
gap law: `1/sqrt(delta^2)` is `1/delta` in the original x coordinate. The plots
include the requested `1/sqrt(delta)` reference explicitly so that its different
slope can be compared with the measured degrees. The logarithmic-error
references are normalized visual guides, not rigorous degree bounds.

## Reproduction

From the repository root, with the project's NumPy, SciPy, and Matplotlib
dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qsppack-sign-mpl python optimization/sign_degree_scaling.py
```

The script caches fits and writes results after each search point. To regenerate
only the figures, add `--plots-only`. To run a fresh independent experiment,
use `--output /tmp/qsppack-sign-fresh`. Exact last digits can vary with numerical
library versions; the stored error margins and oscillation checks document the
precision relevant to the degree decisions.
