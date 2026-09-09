"""Resource-bounded stress tests of QSPPACK's dense Remez implementation.

Each fit runs in a separate process with a 90 second timeout and degree <=2001.
Missing minima are reported as verified lower bounds or numerical failures,
never as measured extrapolations. Run from the repository root.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from sign_degree_scaling import (Experiment, ROUND_OFF, audit,
                                 ConstrainedRemezFitter, RemezOptions)

OUTPUT = Path(__file__).resolve().parent / "sign_degree_stress"
MAX_DEGREE = 2001
TIMEOUT = 90
DELTAS = [1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def diagnose(delta):
    degree = MAX_DEGREE
    fitter = ConstrainedRemezFitter(
        lambda x: np.ones_like(x), [(delta, 1)],
        target_derivative=lambda x: np.zeros_like(x),
        options=RemezOptions(exchange_tolerance=2e-14, root_grid_size=12*degree))
    seed = np.sqrt(delta**2+(1-delta**2)*np.sin(
        np.linspace(0, np.pi/2, (degree+3)//2))**2)
    result = fitter.fit(degree, initial_extremals=seed, strict=True)
    extrema, errors, error, spread, magnitude = audit(result.raw_coefficients, delta)
    bounded_error = float(np.max(1-result.evaluate(extrema)))
    return dict(delta=delta, degree=degree, remez_converged=result.converged,
                exchange_iterations=result.exchange_iterations, raw_error=error,
                bounded_error=bounded_error, raw_ripple_spread=spread,
                alternation_count=len(extrema), raw_max_magnitude=magnitude,
                scale_factor=result.scale_factor,
                bounded_max_magnitude=magnitude/result.scale_factor,
                raw_to_bounded_identity_residual=bounded_error-2*error/(1+error),
                identity_tolerance=ROUND_OFF,
                classification="Remez converged; root-count and alternation checks passed; "
                               "strict identity tolerance exceeded.")


class StressExperiment(Experiment):
    def fit(self, delta, degree):
        self.last_attempted_degree = degree
        if degree > MAX_DEGREE:
            raise RuntimeError(f"Degree {degree} exceeds resource cap {MAX_DEGREE}")
        key = f"{delta:.12g}:{degree}"
        if key in self.cache:
            return self.cache[key]
        started = time.monotonic()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   str(delta), str(degree)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT,
                                    env=dict(os.environ, OPENBLAS_NUM_THREADS="1",
                                             VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg"))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Timeout after {TIMEOUT}s at degree {degree}") from exc
        if result.returncode:
            raise RuntimeError(result.stderr[-1800:] or result.stdout[-1800:])
        self.cache = json.loads(self.cache_path.read_text())
        print(result.stdout.strip(), flush=True)
        self.cache[key]["wall_seconds_with_process_startup"] = time.monotonic()-started
        return self.cache[key]

    def minimum_capped(self, delta, epsilon):
        row = dict(delta=delta, epsilon=epsilon, degree_cap=MAX_DEGREE)
        upper = min((MAX_DEGREE-1)//2, max(1, math.ceil(math.log(1/epsilon)/delta/2)))
        lower = 0
        try:
            candidate = self.fit(delta, 2*upper+1)
            if candidate["bounded_error"] > epsilon+ROUND_OFF:
                if 2*upper+1 != MAX_DEGREE:
                    candidate = self.fit(delta, MAX_DEGREE)
                    upper = (MAX_DEGREE-1)//2
                if candidate["bounded_error"] > epsilon+ROUND_OFF:
                    return dict(row, status="verified_lower_bound", degree_lower_bound=MAX_DEGREE+2,
                                tested_degree=MAX_DEGREE, tested_error=candidate["bounded_error"],
                                alternation_count=candidate["alternation_count"],
                                requires_rescaling=candidate["requires_rescaling"])
            while upper-lower > 1:
                middle = (lower+upper)//2
                if self.fit(delta, 2*middle+1)["bounded_error"] <= epsilon:
                    upper = middle
                else:
                    lower = middle
            selected = self.fit(delta, 2*upper+1)
            previous = self.fit(delta, 2*upper-1)
            if not selected["bounded_error"]+ROUND_OFF < epsilon < previous["bounded_error"]-ROUND_OFF:
                raise RuntimeError("Degree bracket is within numerical tolerance")
            return dict(row, status="verified_minimum", degree=selected["degree"],
                        error=selected["bounded_error"], previous_degree=previous["degree"],
                        previous_error=previous["bounded_error"], scale_factor=selected["scale_factor"],
                        requires_rescaling=selected["requires_rescaling"],
                        alternation_count=selected["alternation_count"])
        except RuntimeError as exc:
            return dict(row, status="numerical_or_timeout_failure",
                        attempted_degree=getattr(self, "last_attempted_degree", 2*upper+1),
                        message=str(exc))


def plots(rows):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#176B91", "#C45824"]
    fig, ax = plt.subplots(figsize=(9, 6.3), layout="constrained")
    for epsilon, color in zip([1e-4, 1e-7], colors):
        subset = [r for r in rows if r["sweep"] == "delta" and r["epsilon"] == epsilon]
        measured = [r for r in subset if r["status"] == "verified_minimum"]
        ax.loglog([r["delta"] for r in measured], [r["degree"] for r in measured], "o",
                  mfc="white", mew=1.8, color=color,
                  label=rf"Verified minimum, $\epsilon=10^{{{round(math.log10(epsilon))}}}$")
        anchor = max(measured, key=lambda r: r["delta"])
        x = np.geomspace(1e-8, 1e-2, 400)
        ax.loglog(x, anchor["degree"]*anchor["delta"]/x, "--", color=color, alpha=.65,
                  label=rf"Extrapolation: ${anchor['degree']*anchor['delta']:.2f}/\delta$")
        bounds = [r for r in subset if r["status"] == "verified_lower_bound"]
        ax.scatter([r["delta"] for r in bounds], [r["degree_lower_bound"] for r in bounds],
                   marker="^", facecolors="none", edgecolors=color, s=100 if epsilon == 1e-4 else 45,
                   label=rf"Lower bound only, $\epsilon=10^{{{round(math.log10(epsilon))}}}$")
        failed = [r for r in subset if r["status"] == "numerical_or_timeout_failure"]
        if failed:
            ax.scatter([r["delta"] for r in failed], [MAX_DEGREE]*len(failed), marker="x",
                       color=".25", label="Strict audit / timeout failure" if epsilon == 1e-4 else None)
    x = np.geomspace(1e-8, 1e-2, 400)
    ax.loglog(x, (1-1e-4)/np.arcsin(x), "-.", color=".25", lw=1.3,
              label=r"Analytic lower bound: $(1-10^{-4})/\arcsin\delta$")
    ax.loglog(x, 86/np.sqrt(x), ":", color="#8B3973", label=r"Reference: $86/\sqrt{\delta}$")
    ax.axhline(MAX_DEGREE, color=".5", lw=.8)
    ax.set(xlabel=r"Gap endpoint $\delta$", ylabel="Odd degree", xlim=(7e-9, 1.4e-2),
           title="Small-gap stress test: measurements and extrapolations")
    ax.grid(which="major", alpha=.2)
    ax.legend(fontsize=9, loc="upper right")
    fig.get_layout_engine().set(rect=(0, .085, 1, .915))
    fig.text(.07, .018, "Triangles mean minimum degree > 2001; dashed curves are unverified estimates.\n"
             "Open circles are verified bounded minimax fits requiring rescaling.", fontsize=9)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"degree_vs_delta_stress.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.8), layout="constrained")
    for delta, color in zip([.01, .005, .001], [*colors, "#6E54A3"]):
        subset = sorted([r for r in rows if r["sweep"] == "epsilon" and r["delta"] == delta],
                        key=lambda r: r["epsilon"])
        measured = [r for r in subset if r["status"] == "verified_minimum"]
        if measured:
            ax.loglog([r["epsilon"] for r in measured], [r["degree"] for r in measured], "o-",
                      color=color, mfc="white", label=rf"Verified minima, $\delta={delta:g}$")
        bounds = [r for r in subset if r["status"] == "verified_lower_bound"]
        if bounds:
            ax.scatter([r["epsilon"] for r in bounds], [r["degree_lower_bound"] for r in bounds],
                       marker="^", facecolors="none", edgecolors=color, s=110 if delta == .001 else 45,
                       label=rf"Lower bounds, $\delta={delta:g}$")
    x = np.geomspace(1e-8, 1e-2, 200)
    ax.loglog(x, 95*np.log(1/x), "--", color=".4", label=r"Reference: $95\ln(1/\epsilon)$")
    ax.axhline(MAX_DEGREE, color=".6", lw=.8)
    ax.set(xlabel=r"Requested uniform error $\epsilon$", ylabel="Odd degree",
           title="Accuracy sweep at smaller gaps (degree cap: 2001)")
    ax.grid(which="major", alpha=.2)
    ax.legend(fontsize=9)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"degree_vs_epsilon_stress.{suffix}", dpi=220)
    plt.close(fig)

    cache = json.loads((OUTPUT / "fits.json").read_text())
    capped = sorted([r for r in cache.values() if r["degree"] == MAX_DEGREE],
                    key=lambda r: r["delta"])
    if capped:
        fig, ax = plt.subplots(figsize=(8.5, 5.8), layout="constrained")
        ax.loglog([r["delta"] for r in capped], [r["bounded_error"] for r in capped],
                  "o-", color=colors[0], mfc="white", label="Bounded minimax error (rescaled)")
        ax.loglog([r["delta"] for r in capped], [r["raw_error"] for r in capped],
                  "s--", color=colors[1], mfc="white", label="Raw minimax error")
        diagnostic_path = OUTPUT / "failure_diagnostics.json"
        if diagnostic_path.exists():
            diagnostics = json.loads(diagnostic_path.read_text())
            ax.scatter([r["delta"] for r in diagnostics], [r["bounded_error"] for r in diagnostics],
                       marker="x", color=".2", s=65, zorder=5,
                       label="Bounded error: strict identity check flagged")
        for epsilon in [1e-4, 1e-7]:
            ax.axhline(epsilon, color=".6", ls=":", lw=1,
                       label=rf"Requested $\epsilon=10^{{{round(math.log10(epsilon))}}}$")
        ax.set(xlabel=r"Gap endpoint $\delta$", ylabel="Verified uniform error",
               title="Actual stress-test errors at fixed degree 2001")
        ax.grid(which="major", alpha=.2)
        ax.legend(fontsize=10)
        for suffix in ["png", "pdf"]:
            fig.savefig(OUTPUT / f"fixed_degree_error.{suffix}", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, type=float)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--diagnose", nargs="+", type=float)
    args = parser.parse_args()
    if args.diagnose:
        diagnostics = [diagnose(delta) for delta in args.diagnose]
        (OUTPUT / "failure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
        print(json.dumps(diagnostics, indent=2), flush=True)
        return
    if args.worker:
        delta, degree = args.worker
        if int(degree) > MAX_DEGREE:
            raise ValueError("Worker degree exceeds resource cap")
        Experiment(OUTPUT).fit(delta, int(degree))
        return
    experiment = StressExperiment(OUTPUT)
    path = OUTPUT / "results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if not args.plots_only:
        cases = [("delta", delta, epsilon) for delta in DELTAS for epsilon in [1e-4, 1e-7]]
        cases += [("epsilon", delta, epsilon) for delta in [.01, .005, .001]
                  for epsilon in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
        for sweep, delta, epsilon in cases:
            if any(r["sweep"] == sweep and r["delta"] == delta and r["epsilon"] == epsilon for r in rows):
                continue
            row = dict(sweep=sweep, **experiment.minimum_capped(delta, epsilon))
            rows.append(row)
            path.write_text(json.dumps(rows, indent=2)+"\n")
            print("RESULT", row, flush=True)
    plots(rows)


if __name__ == "__main__":
    main()
