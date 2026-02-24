#!/usr/bin/env python3
"""
Generate data for Figure 1: maximum constraint violation vs discretization size.

We approximate g(x) = x/a on (0, a) with an odd degree-101 polynomial using
cvx_poly_coef, enforcing |f(x)| <= 1 at npts Chebyshev-like grid points.

For each npts in {2^7, ..., 2^20}, we:
  - solve for the Chebyshev coefficients via cvx_poly_coef
  - compute the true maximum of |f(x)| on [-1, 1] by:
        * differentiating the Chebyshev series
        * finding roots of f'(x) in (-1, 1)
        * evaluating |f(x)| at those roots and at x = ±1
  - record max_violation = max(0, max_{x in [-1,1]} |f(x)| - 1)
  - record max_linf_error_fit_domain = max_{x in [0,a]} |f(x) - (1-epsil)*x/a|

Results are saved to:
  explorations/recovery/data/fig_1_constraint_violation_vs_npts.csv
"""

import argparse
import csv
import os
import time
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as cheb

from qsppack.utils import cvx_poly_coef


def max_abs_on_interval_cheb(coef_full: np.ndarray, a: float = -1.0, b: float = 1.0) -> float:
    """
    Compute max_{x in [a,b]} |f(x)| where f is given in Chebyshev coefficients.

    We:
      - take the Chebyshev derivative
      - find its roots
      - restrict to real roots in (a,b)
      - evaluate f at those roots and at endpoints a, b
    """
    coef_full = np.asarray(coef_full, dtype=float)
    # Derivative in Chebyshev basis
    dcoef = cheb.chebder(coef_full)
    # Roots of derivative (may be complex)
    roots = cheb.chebroots(dcoef)
    # Keep real roots
    real_mask = np.isclose(roots.imag, 0.0, atol=1e-12)
    real_roots = roots[real_mask].real
    # Restrict to open interval (a, b)
    interior = real_roots[(real_roots > a) & (real_roots < b)]

    # Candidates: endpoints and interior critical points
    candidates: Sequence[float] = np.concatenate(([a, b], interior)) if interior.size > 0 else np.array([a, b])

    vals = cheb.chebval(candidates, coef_full)
    max_abs = float(np.max(np.abs(vals)))
    return max_abs


def max_linf_error_on_fit_domain(
    coef_full: np.ndarray, a: float, epsil: float
) -> float:
    """
    Compute max_{x in [0,a]} |f(x) - (1-epsil)*x/a| exactly via critical points.

    h(x) = f(x) - (1-epsil)*x/a; the maximum of |h(x)| occurs at 0, a, or where
    h'(x) = 0. We get h' in Chebyshev as f'(x) - (1-epsil)/a, find its roots
    in (0, a), and evaluate |h| at those points and at the endpoints.
    """
    coef_full = np.asarray(coef_full, dtype=float)
    dcoef = cheb.chebder(coef_full)
    # h'(x) = f'(x) - (1-epsil)/a; in Chebyshev, constant term is index 0 (T_0 = 1)
    dcoef_h = np.copy(dcoef)
    if dcoef_h.size > 0:
        dcoef_h[0] -= (1.0 - epsil) / a
    roots = cheb.chebroots(dcoef_h)
    real_mask = np.isclose(roots.imag, 0.0, atol=1e-12)
    real_roots = roots[real_mask].real
    interior = real_roots[(real_roots > 0.0) & (real_roots < a)]
    candidates = (
        np.concatenate(([0.0, a], interior)) if interior.size > 0 else np.array([0.0, a])
    )
    f_vals = cheb.chebval(candidates, coef_full)
    g_vals = (1.0 - epsil) * candidates / a
    return float(np.max(np.abs(f_vals - g_vals)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data for Fig. 1: max constraint violation vs npts."
    )
    parser.add_argument(
        "--deg",
        type=int,
        default=101,
        help="Degree of polynomial (default: 101, should be odd).",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.2,
        help="Interval endpoint a for target g(x) = x/a on (0,a) (default: 0.2).",
    )
    parser.add_argument(
        "--epsil",
        type=float,
        default=0.0,
        help="Epsilon used in cvx_poly_coef (default: 0). Target is (1-epsil)*x/a.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: explorations/recovery/data/fig_1_constraint_violation_vs_npts.csv).",
    )
    args = parser.parse_args()

    deg = args.deg
    a = args.a
    epsil = args.epsil

    if deg % 2 == 0:
        raise ValueError("Expected an odd degree (for odd polynomial), got deg = {}".format(deg))

    # Target g(x) = x/a on (0,a). With epsil=0 we approximate up to the bound; use (1-epsil)*x/a if slack desired.
    targ = lambda x: (1.0 - epsil) * x / a

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if args.out is None:
        csv_path = os.path.join(data_dir, "fig_1_constraint_violation_vs_npts.csv")
    else:
        csv_path = args.out

    # Powers of two from 2^7 up to 2^20 inclusive (2^21+ too expensive in practice)
    npts_values = [2 ** k for k in range(7, 21)]
    fieldnames = [
        "npts",
        "degree",
        "parity",
        "iteration_time",
        "max_abs",
        "max_violation",
        "max_linf_error_fit_domain",
    ]

    # Resume: if CSV exists, only run for npts not already present.
    # Backward compatibility: if CSV has no max_linf_error_fit_domain column, add it (NaN) and write back.
    existing_npts = set()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "max_linf_error_fit_domain" not in df.columns:
            df["max_linf_error_fit_domain"] = np.nan
            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        if "npts" in df.columns:
            existing_npts = set(df["npts"].astype(int))
        npts_to_run = [n for n in npts_values if n not in existing_npts]
    else:
        npts_to_run = list(npts_values)

    if not npts_to_run:
        print("All npts already computed. Use --force in generate_fig_1.sh to overwrite.")
        return

    resuming = len(existing_npts) > 0
    print("Generating Fig. 1 data")
    print(f"  degree = {deg}")
    print(f"  a      = {a}")
    print(f"  epsil  = {epsil}")
    print(f"  output = {csv_path}")
    if resuming:
        print(f"  resuming: {len(existing_npts)} existing, {len(npts_to_run)} to run")
        print(f"  npts to run = {npts_to_run}")
    else:
        print(f"  npts   = {npts_to_run}")
    print("-" * 60)

    with open(csv_path, "a" if resuming else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not resuming:
            writer.writeheader()

        for npts in npts_to_run:
            print(f"npts = {npts}")
            start = time.time()

            opts = {
                "intervals": [0, a],
                "objnorm": np.inf,
                "epsil": epsil,
                "npts": int(npts),
                "isplot": False,
                "fscale": 1,
                "method": "cvxpy",
            }

            coef_full = cvx_poly_coef(targ, deg, opts)

            # Compute true max |f(x)| on [-1,1] using derivative + roots
            max_abs = max_abs_on_interval_cheb(coef_full, -1.0, 1.0)
            max_violation = max(0.0, max_abs - 1.0)
            max_linf_error_fit_domain = max_linf_error_on_fit_domain(coef_full, a, epsil)

            elapsed = time.time() - start
            parity = deg % 2

            print(f"  max|f(x)| on [-1,1]  = {max_abs:.6e}")
            print(f"  max violation        = {max_violation:.6e}")
            print(f"  max L-inf error [0,a] = {max_linf_error_fit_domain:.6e}")
            print(f"  iteration time (s)   = {elapsed:.3f}")

            writer.writerow(
                {
                    "npts": int(npts),
                    "degree": int(deg),
                    "parity": int(parity),
                    "iteration_time": float(elapsed),
                    "max_abs": float(max_abs),
                    "max_violation": float(max_violation),
                    "max_linf_error_fit_domain": float(max_linf_error_fit_domain),
                }
            )
            f.flush()  # commit to disk so killing early preserves all completed rows

    print("-" * 60)
    print(f"Data written to: {csv_path}")


if __name__ == "__main__":
    main()
