# Small-gap Remez stress test

These outputs are separate from `../sign_degree_scaling/`; the original plots
and data are preserved. The same QSPPACK implementation and independent
alternation/extrema audit are used. No library algorithms were changed.

The requested gap endpoints run from 1e-2 through 1e-8, with an additional 0.005
point near the practical budget boundary. The fixed errors are 1e-4 and 1e-7.
A second sweep varies epsilon from 1e-2 through 1e-8 at gaps 0.01, 0.005, 0.001.

Each individual Remez fit is isolated in a subprocess with a 90-second timeout.
We impose a degree cap of 2001 to keep dense matrix and root-grid allocations
bounded. This cap is an experiment budget, **not** a demonstrated hard limit of
the algorithm. The root grid has at least 12d points; one derivative-evaluation
array alone contains approximately 6d^2 double-precision values. Larger degrees
also require dense least-squares solves and full-degree eigenvalue solves for
the library's feasibility checks.

## Interpreting outputs

- `degree_vs_delta_stress.png` and `.pdf` span the full requested delta range.
  Open circles are measured minimum degrees. Triangles report only that the
  minimum exceeds 2001, when the degree-2001 minimax error is independently
  verified to fail the requested epsilon. Crosses, if present, mean a numerical
  check or timeout prevented a certificate. Dashed curves extrapolate the
  measured minimum at delta=0.01 as C/delta. They are not Remez results.
- `degree_vs_epsilon_stress.png` and `.pdf` show the smaller-gap accuracy sweep,
  with the same distinction between measured minima and lower bounds.
- `results.json` records every requested point's status and its supporting
  error/bracket or failure message. `fits.json` retains every successful fit's
  coefficients, extrema, ripple spreads, scaling factors, and timings.
- `fixed_degree_error.png` and `.pdf` show actual degree-2001 errors across
  the small gaps, with no extrapolation. Crosses distinguish diagnostic errors
  from the fits that pass every strict audit check.

The delta=1e-4, degree-2001 fit failed the original absolute 5e-13 check of the
raw-to-bounded minimax error identity. A separate diagnostic rerun found that
Remez converged in five exchanges and passed the root-count/alternation checks:
the raw ripple spread was 3.61e-12, and the identity residual was -1.38e-12.
This is a strict audit-tolerance failure, **not** a Remez convergence failure.
Its bounded error was approximately 0.838899. We retain the failure marker
and the original tolerance rather than silently relaxing the check.
`failure_diagnostics.json` records that rerun's measured quantities.
At delta=1e-8, the same strict check failed with a residual of 5.34e-13, just
above its 5e-13 tolerance. Remez converged in six exchanges, with all 1002
alternating extrema and a raw ripple spread of 7.86e-13. Its bounded error was
0.99998313895. Both diagnostics can be reproduced with
`python optimization/sign_degree_stress.py --diagnose 1e-4 1e-8` under the same
environment variables as the main run.

Every measured minimum has a passing degree d and a failing degree d-2. Every
successful fit passes the same numerical alternation, root-count, and full-domain
magnitude-bound checks described in the original experiment's README. Scaling
is flagged explicitly. No result beyond the cap is labeled a computed optimum.

The new verified minima include:

| delta | epsilon | Minimum odd degree |
| --- | --- | ---: |
| 0.01 | 1e-4 | 857 |
| 0.01 | 1e-7 | 1521 |
| 0.01 | 1e-8 | 1745 |
| 0.005 | 1e-4 | 1713 |

At delta=0.005, degree 2001 has bounded error 2.20143e-5 and therefore misses
epsilon=1e-7. At delta=0.001 the same degree has error 0.124552. The algorithm
can still converge and equioscillate at extremely small gaps when the degree
is held fixed; the resulting accuracy degrades toward error 1.

Extrapolating the measured delta=0.01 constants to delta=1e-8 predicts roughly
857 million degrees for epsilon=1e-4 and 1.521 billion for epsilon=1e-7.
These are unverified degree estimates. Their dense exchange matrices alone
would occupy approximately 1.47 and 4.63 exabytes, respectively (decimal units),
before other arrays and solver workspace. This experiment therefore does not
compute minimum-degree polynomials at every requested tiny gap.

## A rigorous reason the smallest gaps require enormous degrees

For an odd degree-d polynomial bounded by 1 on [-1,1], Bernstein's inequality
gives `|P'(x)| <= d/sqrt(1-x^2)`. Since P(0)=0 and P(delta)>=1-epsilon,
integration over [0,delta] yields

`d >= (1-epsilon)/arcsin(delta)`.

This is our deduction from the standard derivative inequality, stated for
example in [Nayak's dissertation, Fact A.2.3](https://www.math.uwaterloo.ca/~anayak/papers/Nayak99b.pdf).
The plotted analytic bound uses epsilon=1e-4 and is therefore valid for both
fixed-error curves. Unlike the C/delta extrapolations, it is a necessary bound
independent of the observed Remez results. It already forces approximately
100 million degrees at delta=1e-8. It is not a tight epsilon-dependent estimate.

Thus binary search reduces the number of fits but cannot avoid the cost of an
individual enormous-degree fit. Merely holding the dense exchange matrix
requires approximately `2d^2` bytes, before solver workspace and derivative
grids. A billion-degree fit needs roughly two exabytes for that matrix alone.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qsppack-sign-mpl python optimization/sign_degree_stress.py
```

The run resumes cached fits and completed result rows. `--plots-only`
regenerates the stress-test figures without altering the original figures.
