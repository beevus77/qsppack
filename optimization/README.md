# Sign approximation experiments

All studies approximate 1 on `[delta,1]` with an odd polynomial, with the final
polynomial bounded by 1 in magnitude on `[-1,1]`. Original figures are retained
in their own directories.

| Study | Results | Script |
| --- | --- | --- |
| Baseline degree and accuracy scaling | [sign_degree_scaling](sign_degree_scaling/README.md) | `sign_degree_scaling.py` |
| Small gaps through 1e-8, degree capped at 2001 | [sign_degree_stress](sign_degree_stress/README.md) | `sign_degree_stress.py` |
| Gaps 1e-1 through 1e-3, degrees through 7999 | [sign_degree_8000](sign_degree_8000/README.md) | `sign_degree_8000.py` |
| Fixed delta=0.002, precision through 1e-10 | [sign_precision](sign_precision/README.md) | `sign_precision.py` |

Each results directory contains PNG/PDF figures and JSON data. A reported
minimum degree always has an audited passing polynomial and a failing
preceding odd degree. Bounds, projections, and audit failures are distinguished
from measured minima. All scripts use the repository's Remez implementation.
