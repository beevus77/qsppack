"""Odd-sign degree sweep through 7999, preserving both earlier experiments.

Run with OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg.
Only verified minima are plotted. Every target and fit attempt is recorded.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from sign_degree_scaling import Experiment, ROUND_OFF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "sign_degree_8000"
MAX_DEGREE = 7999
TIMEOUT = 1800
DELTAS = [.1, .05, .02, .01, .005, .003, .002, .0015, .001]
EPSILONS = [1e-4, 1e-7]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(path)


class Sweep:
    def __init__(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        for directory in ["sign_degree_scaling", "sign_degree_stress", "sign_degree_8000"]:
            path = BASE / directory / "fits.json"
            if path.exists():
                self.cache.update(json.loads(path.read_text()))
        self.attempts_path = OUTPUT / "attempts.json"
        self.attempts = json.loads(self.attempts_path.read_text()) if self.attempts_path.exists() else []

    def fit(self, delta, degree):
        if degree < 1 or degree > MAX_DEGREE or degree % 2 != 1:
            raise ValueError("Degree must be odd and between 1 and 7999")
        key = f"{delta:.12g}:{degree}"
        if key in self.cache:
            return self.cache[key]
        print(f"START delta={delta:g} degree={degree}", flush=True)
        started = time.monotonic()
        attempt = dict(delta=delta, degree=degree, timeout_seconds=TIMEOUT)
        try:
            run = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(delta), str(degree)],
                capture_output=True, text=True, timeout=TIMEOUT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                         MPLBACKEND="Agg"))
            attempt.update(wall_seconds=time.monotonic()-started, returncode=run.returncode,
                           stdout=run.stdout, stderr=run.stderr)
        except subprocess.TimeoutExpired as exc:
            attempt.update(wall_seconds=time.monotonic()-started, status="timeout")
            self.attempts.append(attempt)
            write_json(self.attempts_path, self.attempts)
            raise RuntimeError(f"Fit exceeded {TIMEOUT}s: delta={delta}, degree={degree}") from exc
        attempt["status"] = "passed" if run.returncode == 0 else "failed"
        self.attempts.append(attempt)
        write_json(self.attempts_path, self.attempts)
        if run.returncode:
            raise RuntimeError(f"Fit failed: delta={delta}, degree={degree}: {run.stderr[-1500:]}")
        self.cache.update(json.loads((OUTPUT / "fits.json").read_text()))
        print(run.stdout.strip(), flush=True)
        return self.cache[key]

    def minimum(self, delta, epsilon, previous_rows):
        row = dict(delta=delta, epsilon=epsilon, degree_cap=MAX_DEGREE)
        known = [r for r in self.cache.values() if r["delta"] == delta and r["degree"] <= MAX_DEGREE]
        passing = [r["degree"] for r in known if r["bounded_error"]+ROUND_OFF < epsilon]
        failing = [r["degree"] for r in known if r["bounded_error"]-ROUND_OFF > epsilon]
        low, high = max(failing, default=-1), min(passing, default=MAX_DEGREE+2)
        if low == MAX_DEGREE:
            record = self.fit(delta, MAX_DEGREE)
            return dict(row, status="outside_degree_budget",
                        tested_error=record["bounded_error"], tested_degree=MAX_DEGREE)
        if high-low > 2:
            prior = [r for r in previous_rows if r["epsilon"] == epsilon and r["status"] == "verified_minimum"]
            if prior:
                anchor = min(prior, key=lambda r: abs(math.log(r["delta"]/delta)))
                estimate = anchor["degree"]*anchor["delta"]/delta
            else:
                estimate = math.log(1/epsilon)/delta
            guess = min(MAX_DEGREE, max(1, 2*round((estimate-1)/2)+1))
            guess = min(high-2, max(low+2, guess))
            step = 2
            while high > MAX_DEGREE or low < 1:
                record = self.fit(delta, guess)
                error = record["bounded_error"]
                if abs(error-epsilon) <= ROUND_OFF:
                    raise RuntimeError("Error too close to requested epsilon for a degree decision")
                if error < epsilon:
                    high = min(high, guess)
                    if low >= 1:
                        break
                    guess = max(1, high-step)
                else:
                    low = max(low, guess)
                    if low == MAX_DEGREE:
                        return dict(row, status="outside_degree_budget", tested_error=error,
                                    tested_degree=MAX_DEGREE)
                    if high <= MAX_DEGREE:
                        break
                    guess = min(MAX_DEGREE, low+step)
                step *= 2
            # Tighten a broad cached bracket near the scaling estimate before
            # binary search. Every update uses an audited fit, never an estimate.
            if high-low > 2:
                guess = min(high-2, max(low+2, 2*round((estimate-1)/2)+1))
                record = self.fit(delta, guess)
                if record["bounded_error"] < epsilon:
                    high = guess
                    candidate = max(low, high-2)
                    if candidate > low:
                        probe = self.fit(delta, candidate)
                        if probe["bounded_error"] > epsilon:
                            low = candidate
                else:
                    low = guess
                    candidate = min(high, low+2)
                    if candidate < high:
                        probe = self.fit(delta, candidate)
                        if probe["bounded_error"] < epsilon:
                            high = candidate
            while high-low > 2:
                middle = low + 2*((high-low)//4)
                record = self.fit(delta, middle)
                if record["bounded_error"] < epsilon:
                    high = middle
                else:
                    low = middle
        selected, previous = self.fit(delta, high), self.fit(delta, high-2)
        if not selected["bounded_error"]+ROUND_OFF < epsilon < previous["bounded_error"]-ROUND_OFF:
            raise RuntimeError("Final bracket not separated from epsilon by numerical tolerance")
        return dict(row, status="verified_minimum", degree=high,
                    error=selected["bounded_error"], previous_degree=high-2,
                    previous_error=previous["bounded_error"],
                    scale_factor=selected["scale_factor"],
                    requires_rescaling=selected["requires_rescaling"],
                    alternation_count=selected["alternation_count"],
                    raw_ripple_spread=selected["raw_ripple_spread"])


def plots(rows):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.6, 6), layout="constrained")
    for epsilon, color in zip(EPSILONS, ["#176B91", "#C45824"]):
        data = sorted([r for r in rows if r["epsilon"] == epsilon and r["status"] == "verified_minimum"],
                      key=lambda r: r["delta"])
        ax.loglog([r["delta"] for r in data], [r["degree"] for r in data], "o-",
                  color=color, mfc="white", mew=1.6, lw=2,
                  label=rf"$\epsilon=10^{{{round(math.log10(epsilon))}}}$")
    x = np.geomspace(.001, .1, 200)
    ax.loglog(x, 8/x, "--", color=".5", lw=1.2, label=r"Reference: $8/\delta$")
    ax.loglog(x, 27/np.sqrt(x), ":", color="#8B3973", lw=1.5,
              label=r"Reference: $27/\sqrt{\delta}$")
    ax.set(xlabel=r"Gap endpoint $\delta$", ylabel="Verified minimum odd degree",
           xlim=(.0009, .112), ylim=(60, 9500))
    ax.xaxis.set_major_locator(FixedLocator([.001, .002, .005, .01, .02, .05, .1]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000, 5000, 8000]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.grid(which="major", alpha=.22)
    ax.grid(which="minor", alpha=.07)
    ax.set_title("Odd sign approximation: degree sweep through 7999", loc="left", weight="bold", pad=28)
    ax.text(0, 1.015, "Verified minima only; open markers require rescaling; unmet targets omitted",
            transform=ax.transAxes, fontsize=9.5, color=".35")
    ax.legend(fontsize=10)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"degree_vs_delta.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, type=float)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        delta, degree = args.worker
        if not 1 <= degree <= MAX_DEGREE or int(degree) % 2 != 1:
            raise ValueError("Worker degree outside allowed range")
        experiment = Experiment(OUTPUT)
        record = experiment.fit(delta, int(degree))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak*1024)
        experiment.cache[f"{delta:.12g}:{int(degree)}"] = record
        write_json(experiment.cache_path, experiment.cache)
        print(f"PEAK_RSS_GB={record['peak_rss_bytes']/1e9:.3f}", flush=True)
        return
    sweep = Sweep()
    path = OUTPUT / "results.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if not args.plots_only:
        for delta in DELTAS:
            for epsilon in EPSILONS:
                if any(r["delta"] == delta and r["epsilon"] == epsilon for r in rows):
                    continue
                try:
                    row = sweep.minimum(delta, epsilon, rows)
                except RuntimeError as exc:
                    row = dict(delta=delta, epsilon=epsilon, status="failed", message=str(exc))
                rows.append(row)
                write_json(path, rows)
                print("RESULT", json.dumps(row), flush=True)
                plots(rows)
        used = {}
        for row in rows:
            for field in ["degree", "previous_degree", "tested_degree"]:
                if field in row:
                    key = f"{row['delta']:.12g}:{row[field]}"
                    used[key] = sweep.cache[key]
        write_json(OUTPUT / "certificates.json", used)
    plots(rows)


if __name__ == "__main__":
    main()
