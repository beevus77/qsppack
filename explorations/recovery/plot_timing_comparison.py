#!/usr/bin/env python3
"""Compare Figure 7 and exact-bound SDP approximation runtimes."""

from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator, ScalarFormatter


DEFAULT_FIG7_CSV = "data/degree_scaling_uniform_sv_amp_npts19_solver2_odd.csv"
DEFAULT_SDP_CSV = "data/degree_scaling_uniform_sv_amp_constrainapprox.csv"
DEFAULT_COMPARISON_CSV = "data/degree_scaling_timing_comparison.csv"
DEFAULT_PDF = "figures/degree_scaling_timing_comparison.pdf"
DEFAULT_PNG = "figures/degree_scaling_timing_comparison.png"


def benchmark_sdp(degree: int, a: float, epsilon: float, npts: int) -> float:
    """Time only the new constrained-approximation solve, as in its sweep."""
    from constrainapprox import CERTIFICATE_TOL, solve_odd_bounded_approx
    from degree_scaling_constrainapprox import BOUND_TOL, independent_bound_check

    target = lambda x: (1.0 - epsilon) * x / a
    started = time.perf_counter()
    coefficients, diagnostics = solve_odd_bounded_approx(
        target,
        degree=degree,
        fit_intervals=[0.0, a],
        npts=npts,
        bound=1.0,
        solver="SCS",
    )
    runtime = time.perf_counter() - started
    independent_maximum, _ = independent_bound_check(coefficients, degree)
    valid = bool(
        diagnostics["certificate_passed"]
        and diagnostics["diagonal_sum_residual"] <= CERTIFICATE_TOL
        and diagnostics["minimum_gram_eigenvalue"] >= 0.0
        and diagnostics["global_max_abs"] <= 1.0
        and independent_maximum <= 1.0 + BOUND_TOL
    )
    if not valid:
        raise RuntimeError(f"SDP certificate check failed at degree {degree}")
    return runtime


def build_comparison(
    fig7_csv: str,
    sdp_csv: str,
    output_csv: str,
    count: int,
    benchmark_missing: bool,
) -> pd.DataFrame:
    fig7 = pd.read_csv(fig7_csv).sort_values("degree").head(count)
    sdp = pd.read_csv(sdp_csv).sort_values("degree")
    required_fig7 = {"degree", "time_fit"}
    required_sdp = {"degree", "runtime_seconds", "a", "epsilon", "npts"}
    if missing := required_fig7.difference(fig7.columns):
        raise ValueError(f"Figure 7 CSV is missing columns: {sorted(missing)}")
    if missing := required_sdp.difference(sdp.columns):
        raise ValueError(f"SDP CSV is missing columns: {sorted(missing)}")

    sdp_times = dict(zip(sdp["degree"].astype(int), sdp["runtime_seconds"]))
    rows = []
    for row in fig7.itertuples(index=False):
        degree = int(row.degree)
        runtime = sdp_times.get(degree)
        source = "saved sweep"
        if runtime is None:
            if not benchmark_missing:
                raise RuntimeError(
                    f"No exact SDP timing for degree {degree}; rerun with "
                    "--benchmark-missing"
                )
            reference = sdp.iloc[(sdp["degree"] - degree).abs().argsort()[:1]].iloc[0]
            print(f"Benchmarking missing exact SDP degree {degree}")
            runtime = benchmark_sdp(
                degree, float(reference["a"]), float(reference["epsilon"]), int(reference["npts"])
            )
            source = "exact-degree supplemental benchmark"
            print(f"  degree {degree}: {runtime:.6f} s")
        rows.append(
            {
                "degree": degree,
                "fig7_fit_seconds": float(row.time_fit),
                "exact_bound_sdp_seconds": float(runtime),
                "sdp_timing_source": source,
            }
        )

    comparison = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    comparison.to_csv(output_csv, index=False)
    return comparison


def plot(comparison: pd.DataFrame, pdf_path: str, png_path: str) -> None:
    degrees = comparison["degree"].to_numpy(float)
    plt.rcParams.update({"font.family": "serif", "font.size": 16})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        degrees,
        comparison["fig7_fit_seconds"],
        "o--",
        color="#0072B2",
        linewidth=1.5,
        markersize=7,
        label="Fig. 7 sampled fit",
    )
    ax.plot(
        degrees,
        comparison["exact_bound_sdp_seconds"],
        "s--",
        color="#D55E00",
        linewidth=1.5,
        markersize=7,
        label="Exact-bound SDP",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Approximation time (seconds)")
    ax.set_xticks(degrees)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    for path in (pdf_path, png_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig7-csv", default=DEFAULT_FIG7_CSV)
    parser.add_argument("--sdp-csv", default=DEFAULT_SDP_CSV)
    parser.add_argument("--comparison-csv", default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--png", default=DEFAULT_PNG)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--benchmark-missing", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        comparison = pd.read_csv(args.comparison_csv)
    else:
        comparison = build_comparison(
            args.fig7_csv,
            args.sdp_csv,
            args.comparison_csv,
            args.count,
            args.benchmark_missing,
        )
    plot(comparison, args.pdf, args.png)
    print(comparison.to_string(index=False))
    print(f"Saved {args.pdf} and {args.png}")


if __name__ == "__main__":
    main()
