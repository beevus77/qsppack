"""Reproducible odd-sign Remez degree searches and numerical certificates.

Run from the repository root with a single BLAS thread, for example::

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg \
        python optimization/sign_degree_scaling.py

Only the existing QSPPACK Remez implementation constructs polynomials.  This
script independently locates derivative roots to audit errors and alternation.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsppack.remez import ConstrainedRemezFitter, RemezOptions


OUTPUT = Path(__file__).resolve().parent / "sign_degree_scaling"
DELTAS = [0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.035, 0.025, 0.02]
FIXED_EPSILONS = [1e-4, 1e-7]
EPSILONS = np.logspace(-2, -8, 13).tolist()
FIXED_DELTAS = [0.2, 0.1, 0.05]
ROUND_OFF = 5e-13


def audit(coefficients, delta):
    """Account for every derivative root by counting positive roots and parity.

    For d=2m-1, finding m-1 distinct positive roots accounts for all d-1
    derivative roots, since the derivative is even.  Endpoints plus those roots
    therefore suffice for the continuous-domain extrema, up to floating point.
    """
    degree = len(coefficients) - 1
    derivative = cheb.chebder(coefficients)
    grid = np.cos(np.linspace(np.pi / 2, 0, max(4097, 32 * degree)))
    values = cheb.chebval(grid, derivative)
    changes = np.flatnonzero(values[:-1] * values[1:] < 0)
    roots = [brentq(lambda x: cheb.chebval(x, derivative), grid[i], grid[i+1],
                    xtol=5e-15, rtol=1e-14) for i in changes]
    roots = np.unique(np.r_[roots, grid[values == 0]])
    roots = roots[(roots > 0) & (roots < 1)]
    expected_roots = (degree - 1) // 2
    if len(roots) != expected_roots or np.any(roots <= delta):
        raise RuntimeError(f"Incomplete derivative-root certificate: d={degree}, "
                           f"delta={delta}, roots={len(roots)}, expected={expected_roots}")
    extrema = np.r_[delta, roots, 1.0]
    signed_error = cheb.chebval(extrema, coefficients) - 1
    error = float(np.max(np.abs(signed_error)))
    spread = float(np.ptp(np.abs(signed_error)))
    alternating = bool(np.all(signed_error[:-1] * signed_error[1:] < 0))
    if not alternating or spread > max(ROUND_OFF, 2e-5 * error):
        raise RuntimeError(f"Failed alternation: d={degree}, delta={delta}, "
                           f"error={error}, spread={spread}")
    magnitude = float(np.max(np.abs(cheb.chebval(np.r_[0, roots, 1], coefficients))))
    return extrema, signed_error, error, spread, magnitude


class Experiment:
    def __init__(self, output):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache_path = output / "fits.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def fit(self, delta, degree):
        key = f"{delta:.12g}:{degree}"
        if key in self.cache:
            return self.cache[key]
        started = time.monotonic()
        fitter = ConstrainedRemezFitter(
            target=lambda x: np.ones_like(x), fit_intervals=[(delta, 1)],
            target_derivative=lambda x: np.zeros_like(x),
            options=RemezOptions(exchange_tolerance=2e-14,
                                 root_grid_size=max(2001, 12 * degree)))
        # Odd P(x)=x Q(x^2): Chebyshev-distributed points in x^2 provide a
        # stable initial exchange set at small ripples / higher degrees.
        seed = np.sqrt(delta**2 + (1-delta**2) * np.sin(
            np.linspace(0, np.pi/2, (degree+1)//2+1))**2)
        result = fitter.fit(degree, initial_extremals=seed, strict=True)
        extrema, raw_errors, raw_error, spread, raw_max = audit(result.raw_coefficients, delta)
        bounded_errors = 1 - result.evaluate(extrema)
        bounded_error = float(np.max(np.abs(bounded_errors)))
        bounded_max = raw_max / result.scale_factor
        centered = bounded_errors - bounded_error/2
        centered_spread = float(np.ptp(np.abs(centered)))
        if bounded_max > 1 + ROUND_OFF or np.min(bounded_errors) < -ROUND_OFF:
            raise RuntimeError("Scaled polynomial violates the magnitude bound.")
        if centered_spread > max(ROUND_OFF, 2e-5 * bounded_error):
            raise RuntimeError("Bounded error does not oscillate about its midpoint.")
        if abs(bounded_error - 2*raw_error/(1+raw_error)) > ROUND_OFF:
            raise RuntimeError("Raw-to-bounded minimax error identity failed.")
        record = dict(delta=delta, degree=degree, raw_error=raw_error,
                      bounded_error=bounded_error, scale_factor=result.scale_factor,
                      requires_rescaling=bool(result.scale_factor > 1+ROUND_OFF),
                      raw_max_magnitude=raw_max, bounded_max_magnitude=bounded_max,
                      alternation_count=len(extrema), expected_alternation_count=(degree+3)//2,
                      raw_ripple_spread=spread, bounded_centered_ripple_spread=centered_spread,
                      converged=result.converged, exchange_iterations=result.exchange_iterations,
                      seconds=time.monotonic()-started,
                      extrema=extrema.tolist(), raw_signed_errors=raw_errors.tolist(),
                      raw_coefficients=result.raw_coefficients.tolist())
        self.cache[key] = record
        self.cache_path.write_text(json.dumps(self.cache, indent=2) + "\n")
        print(f"delta={delta:g} d={degree:4d} bounded_error={bounded_error:.9g} "
              f"alternation={len(extrema)} scale-1={result.scale_factor-1:.3g} "
              f"time={record['seconds']:.2f}s", flush=True)
        return record

    def minimum(self, delta, epsilon):
        # Binary search index k with d=2k+1.  Nested odd polynomial spaces make
        # optimal errors nonincreasing.  Both sides of the final bracket are audited.
        lower = 0
        if self.fit(delta, 1)["bounded_error"] <= epsilon:
            upper = 0
        else:
            upper = max(1, math.ceil(math.log(1/epsilon)/delta/2))
            while self.fit(delta, 2*upper+1)["bounded_error"] > epsilon:
                lower, upper = upper, math.ceil(1.2 * upper) + 1
            while upper-lower > 1:
                middle = (lower+upper)//2
                if self.fit(delta, 2*middle+1)["bounded_error"] <= epsilon:
                    upper = middle
                else:
                    lower = middle
        selected = self.fit(delta, 2*upper+1)
        predecessor = self.fit(delta, 2*upper-1) if upper else None
        if selected["bounded_error"] + ROUND_OFF >= epsilon:
            raise RuntimeError("Passing degree is too close to the numerical error floor.")
        if predecessor and predecessor["bounded_error"] - ROUND_OFF <= epsilon:
            raise RuntimeError("Failing degree is too close to the numerical error floor.")
        return dict(delta=delta, epsilon=epsilon, degree=selected["degree"],
                    error=selected["bounded_error"], previous_degree=2*upper-1 if upper else None,
                    previous_error=predecessor["bounded_error"] if predecessor else None,
                    scale_factor=selected["scale_factor"],
                    requires_rescaling=selected["requires_rescaling"],
                    raw_ripple_spread=selected["raw_ripple_spread"],
                    alternation_count=selected["alternation_count"])


def plots(rows, experiment):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    colors = ["#176B91", "#C45824", "#6E54A3"]

    def decorate(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Minimum odd degree")
        ax.grid(which="major", color="#D9DFE5", linewidth=.7)
        ax.grid(which="minor", color="#EDF0F3", linewidth=.4)

    def save(fig, name):
        fig.savefig(experiment.output / f"{name}.png", dpi=220)
        fig.savefig(experiment.output / f"{name}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 6), layout="constrained")
    for color, epsilon in zip(colors, FIXED_EPSILONS):
        data = sorted([r for r in rows if r["sweep"] == "delta" and r["epsilon"] == epsilon],
                      key=lambda r: r["delta"])
        ax.plot([r["delta"] for r in data], [r["degree"] for r in data], "o-",
                color=color, mfc="white", mew=1.8, lw=2,
                label=rf"$\epsilon=10^{{{round(math.log10(epsilon))}}}$")
    x = np.geomspace(min(DELTAS), max(DELTAS), 200)
    ax.plot(x, 7.5/x, "--", color="#454B54", lw=1.5, label=r"Reference: $7.5/\delta$")
    ax.plot(x, 16/x, "--", color="#8D939B", lw=1.5, label=r"Reference: $16/\delta$")
    ax.plot(x, 30/np.sqrt(x), ":", color="#8B3973", lw=2,
            label=r"Requested reference: $30/\sqrt{\delta}$")
    decorate(ax)
    ax.xaxis.set_major_locator(FixedLocator([.02, .03, .05, .1, .2, .3]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(FixedLocator([30, 50, 100, 200, 500]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel(r"Gap endpoint $\delta$   (fit on $[\delta,1]$)")
    ax.set_title("Odd sign approximation: degree versus gap", loc="left", weight="bold", pad=28)
    ax.text(0, 1.015, r"Bounded minimax: $|P|\leq 1$ on $[-1,1]$; open markers require rescaling",
            transform=ax.transAxes, fontsize=10, color="#535B65")
    ax.legend(fontsize=10, framealpha=.95)
    save(fig, "degree_vs_delta")

    fig, ax = plt.subplots(figsize=(8.4, 6), layout="constrained")
    x = np.geomspace(min(EPSILONS), max(EPSILONS), 200)
    for color, delta in zip(colors, FIXED_DELTAS):
        data = sorted([r for r in rows if r["sweep"] == "epsilon" and r["delta"] == delta],
                      key=lambda r: r["epsilon"])
        ax.plot([r["epsilon"] for r in data], [r["degree"] for r in data], "o-",
                color=color, mfc="white", mew=1.8, lw=2, label=rf"$\delta={delta:g}$")
        anchor = min(data, key=lambda r: abs(math.log10(r["epsilon"])+5))
        constant = anchor["degree"] / math.log(1/anchor["epsilon"])
        ax.plot(x, constant*np.log(1/x), "--", color=color, alpha=.65, lw=1.4,
                label=rf"${constant:.1f}\,\ln(1/\epsilon)$")
    decorate(ax)
    ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 400]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel(r"Requested uniform error $\epsilon$")
    ax.set_title("Odd sign approximation: degree versus accuracy", loc="left", weight="bold", pad=28)
    ax.text(0, 1.015, "Bounded minimax; open markers require rescaling; degree d − 2 checked",
            transform=ax.transAxes, fontsize=10, color="#535B65")
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=10, ncol=2, framealpha=.95)
    save(fig, "degree_vs_epsilon")

    row = next(r for r in rows if r["delta"] == .1 and r["epsilon"] == 1e-4)
    record = experiment.fit(row["delta"], row["degree"])
    raw = np.asarray(record["raw_coefficients"])
    bounded = raw / record["scale_factor"]
    delta, degree, error = row["delta"], row["degree"], row["error"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained",
                             gridspec_kw={"height_ratios": [1.35, 1, 1]})
    x = np.linspace(-1, 1, 6001)
    axes[0].plot(x, cheb.chebval(x, bounded), color=colors[0], lw=2, label="Bounded polynomial")
    axes[0].plot([-1, -delta], [-1, -1], "k--", lw=1.2, label="Sign target")
    axes[0].plot([delta, 1], [1, 1], "k--", lw=1.2)
    axes[0].axvspan(-delta, delta, color="#EBEEF2", label="Transition gap")
    axes[0].set(xlabel="x", ylabel="P(x)", xlim=(-1,1))
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title(rf"Verified equioscillation: $\delta={delta:g}$, $\epsilon=10^{{-4}}$, degree {degree}",
                      loc="left", weight="bold")
    x = np.linspace(delta, 1, 10001)
    extrema = np.asarray(record["extrema"])
    for ax, coef, denom, label, levels, color in [
        (axes[1], raw, record["raw_error"], r"$(P_{raw}(x)-1)/e_{raw}$", [-1,1], colors[1]),
        (axes[2], bounded, error, r"$(1-P(x))/E$", [0,1], colors[0])]:
        sign = -1 if ax is axes[2] else 1
        ax.plot(x, sign*(cheb.chebval(x, coef)-1)/denom, color=color, lw=1.2)
        ax.plot(extrema, sign*(cheb.chebval(extrema,coef)-1)/denom, "o", color=color,
                ms=3.5, mfc="white", clip_on=False)
        for level in levels:
            ax.axhline(level, color="#686E76", ls="--", lw=.8)
        ax.set(xlabel="x on the positive fitting interval", ylabel=label, xlim=(delta,1))
        ax.grid(alpha=.15)
    axes[1].set_title(f"Raw minimax: {len(extrema)} alternating extrema, error = {record['raw_error']:.6g}",
                      fontsize=11, loc="left")
    axes[2].set_title(f"Bounded minimax: error alternates between 0 and E = {error:.6g}; "
                      f"scale = {record['scale_factor']:.9f}", fontsize=11, loc="left")
    save(fig, "equioscillation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    experiment = Experiment(args.output)
    results_path = args.output / "minimum_degrees.json"
    if args.plots_only:
        rows = json.loads(results_path.read_text())
    else:
        rows = []
        for epsilon in FIXED_EPSILONS:
            for delta in DELTAS:
                rows.append(dict(sweep="delta", **experiment.minimum(delta, epsilon)))
                results_path.write_text(json.dumps(rows, indent=2) + "\n")
        for delta in FIXED_DELTAS:
            for epsilon in EPSILONS:
                rows.append(dict(sweep="epsilon", **experiment.minimum(delta, epsilon)))
                results_path.write_text(json.dumps(rows, indent=2) + "\n")
    plots(rows, experiment)
    print(f"Saved {len(rows)} certified minimum-degree results and three figures to {args.output}")


if __name__ == "__main__":
    main()
