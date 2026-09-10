# Remez degree sweep through 7999

This study restricts delta to `[0.001,0.1]` and uses fixed errors 1e-4 and 1e-7.
The maximum degree is 7999, the largest odd degree no greater than 8000.
Neither of the earlier result directories is modified.

`degree_vs_delta.png` and `.pdf` show only verified minimum-degree points and
reference curves. No lower-bound markers, failure markers, or projections are
plotted. Unmet targets remain in `results.json`; absence of a plotted point
does not imply success at the degree budget. Open markers flag rescaling.

## Results

All 18 requested targets were evaluated: 15 verified minima and three targets
outside the degree budget. The 15 new fits all converged and passed the
independent audits; earlier cached fits support the remaining points.

| delta | Minimum degree, epsilon=1e-4 | Minimum degree, epsilon=1e-7 |
| --- | ---: | ---: |
| 0.1 | 87 | 153 |
| 0.05 | 173 | 305 |
| 0.02 | 429 | 761 |
| 0.01 | 857 | 1521 |
| 0.005 | 1713 | 3041 |
| 0.003 | 2855 | 5069 |
| 0.002 | 4283 | 7601 |
| 0.0015 | 5711 | Not reached through 7999 |
| 0.001 | Not reached through 7999 | Not reached through 7999 |

At degree 7999 the bounded errors were 2.75333397e-6 for delta=0.0015 and
1.81582614e-4 for delta=0.001. Those fits passed the audits; their accuracy
simply missed the relevant requested epsilon. They are recorded but omitted
from the plot.

The largest raw ripple-magnitude spread among new fits was 6.69e-14. Their
maximum magnitude-bound overshoot was 2.22e-16, at floating-point roundoff.
Every plotted minimum has an independently checked failing predecessor d-2.

## Search and verification

The sweep uses delta values 0.1, 0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.0015,
and 0.001. Existing audited fits are reused read-only. For new points, the
nearest completed minimum suggests a starting degree using inverse-delta
scaling. Actual Remez fits establish the passing/failing bracket, and binary
search resolves it to adjacent odd degrees. Predictions never establish a
minimum on their own.

Every reported minimum passes the requested epsilon, while degree d-2 fails,
with more than 5e-13 separation from the threshold. Fits use the original
experiment's independent derivative-root, alternation, full-domain magnitude,
and raw-to-bounded error-identity checks. See the [baseline methodology](../sign_degree_scaling/README.md).
The numerical tolerances and QSPPACK library code are unchanged.

## Runtime and memory

Large fits run sequentially, each in a separate subprocess with a 30-minute
timeout. NumPy BLAS and Accelerate thread counts are set to one. This machine
reports 24 GiB of physical memory. The experiment does not impose a new degree
cutoff below 7999 based on extrapolated runtime.

- `fits.json`: newly computed coefficients, errors, extrema, timings, and
  process peak resident memory. Its `seconds` field includes Remez and the
  independent audit, excluding Python startup and JSON serialization.
- `attempts.json`: elapsed process wall times and complete stdout/stderr for
  every new attempt, including failures and timeouts if encountered.
- `results.json`: minimum-degree brackets or explicit unmet/failed statuses.
- `certificates.json`: the fits supporting every final minimum and unmet
  target, including reused earlier fits.

Peak resident memory includes the whole worker process, not just one array.
Recorded elapsed times are exploratory measurements, not controlled repeated
benchmarks. All coefficients are ascending full Chebyshev coefficients.

Measured examples (GB below means decimal gigabytes):

| Degree | Fit plus audit time | Peak resident memory |
| ---: | ---: | ---: |
| 3041 | 33.4 s | 1.23 GB |
| 4283 | 72.9 s | 2.21 GB |
| 5069 | 120.0 s | 2.93 GB |
| 5711 | 162.6 s | 3.73 GB |
| 7601 | 344.4 s | 6.27 GB |
| 7999 (delta=0.001) | 387.1 s | 6.97 GB |

The 15 new fits used 2610.7 seconds (43.5 minutes) in total, excluding process
startup and serialization. Degree 7999 was reached without a timeout or
numerical audit failure. A cached-cap bookkeeping error interrupted final
result assembly after those fits completed; it was fixed and regression-checked
for both epsilons using cached data, with no numerical reruns necessary.

## Reproduce

From the repository root with its dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qsppack-sign-mpl python optimization/sign_degree_8000.py
```

The script resumes completed rows and cached fits. Add `--plots-only` to
regenerate the figure. Both earlier studies supply reusable fits when present.
