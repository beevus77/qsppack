"""Fixed-delta accuracy study, direct extrapolation, and precision diagnostics.

The polynomial construction is unchanged QSPPACK Remez in float64. Workers
preserve returned polynomials even when an independent audit fails.
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
import warnings

from sign_degree_scaling import (audit, ConstrainedRemezFitter, RemezOptions, ROUND_OFF)
from sign_degree_8000 import write_json
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
from numpy.polynomial import chebyshev as cheb

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "sign_precision"
DELTA = .002
MAX_DEGREE = 12001
TIMEOUT = 5400


def spotcheck(degree):
    """80-digit evaluation of selected extrema; not a full new certificate."""
    import mpmath as mp
    record=json.loads((OUTPUT/f"fit_{degree}.json").read_text())
    raw=np.asarray(record["raw_coefficients"])
    selected=raw/record["scale_factor"]
    extrema=np.asarray(record.get("extrema",record["library_extrema"]))
    if extrema.size==0:
        raise RuntimeError("No extrema available for precision spot checks")
    double_errors=1-cheb.chebval(extrema,selected)
    indices=np.unique(np.r_[np.linspace(0,len(extrema)-1,8,dtype=int),
                            np.argmax(double_errors),np.argmin(double_errors)])
    points=[]
    started=time.monotonic()
    with mp.workdps(80):
        for index in indices:
            x=mp.mpf(float(extrema[index]))
            b1=b2=mp.mpf(0)
            for coefficient in selected[:0:-1]:
                b0=mp.mpf(float(coefficient))+2*x*b1-b2
                b2,b1=b1,b0
            value=mp.mpf(float(selected[0]))+x*b1-b2
            error=1-value
            points.append(dict(x=float(x), float64_error=float(double_errors[index]),
                               error_80_digits=mp.nstr(error,50),
                               evaluation_difference=float(mp.mpf(float(double_errors[index]))-error)))
    result=dict(degree=degree, decimal_precision=80, points=points,
                seconds=time.monotonic()-started,
                max_evaluation_difference=max(abs(p["evaluation_difference"]) for p in points),
                scope="Selected extrema of stored float64 coefficients; no reoptimization and no full-domain certificate")
    write_json(OUTPUT/f"spotcheck_{degree}.json",result)
    print(json.dumps(result),flush=True)


class TimedFitter(ConstrainedRemezFitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_seconds = {"exchange": 0., "metrics": 0., "root_search": 0.}

    def _checkpoint(self, stage):
        systems=getattr(self,"linear_system_diagnostics",[])
        write_json(OUTPUT/"progress.json", dict(degree=self.profile_degree, stage=stage,
                   timestamp=time.time(), stage_seconds=self.stage_seconds,
                   linear_system_count=len(systems), last_linear_system=systems[-1] if systems else None))

    def _exchange(self, *args, **kwargs):
        self._checkpoint("exchange start")
        started = time.monotonic()
        try:
            return super()._exchange(*args, **kwargs)
        finally:
            self.stage_seconds["exchange"] += time.monotonic()-started
            self._checkpoint("exchange complete")

    def _metrics(self, *args, **kwargs):
        self._checkpoint("feasibility metrics start")
        started = time.monotonic()
        try:
            return super()._metrics(*args, **kwargs)
        finally:
            self.stage_seconds["metrics"] += time.monotonic()-started
            self._checkpoint("feasibility metrics complete")

    def _roots(self, *args, **kwargs):
        self._checkpoint("root search start")
        started = time.monotonic()
        try:
            return super()._roots(*args, **kwargs)
        finally:
            self.stage_seconds["root_search"] += time.monotonic()-started
            self._checkpoint("root search complete")


def worker(degree):
    started = time.monotonic()
    fitter = TimedFitter(
        lambda x: np.ones_like(x), [(DELTA, 1)],
        target_derivative=lambda x: np.zeros_like(x),
        options=RemezOptions(exchange_tolerance=2e-14, root_grid_size=max(2001, 12*degree)))
    fitter.profile_degree=degree
    seed = np.sqrt(DELTA**2+(1-DELTA**2)*np.sin(
        np.linspace(0, np.pi/2, (degree+3)//2))**2)
    systems=[]
    fitter.linear_system_diagnostics=systems
    original_lstsq=np.linalg.lstsq
    def observed_lstsq(matrix, rhs, *args, **kwargs):
        # Preserve NumPy's exact inputs/outputs; retain diagnostics that the
        # library normally discards. This patch is local to the worker process.
        answer=original_lstsq(matrix,rhs,*args,**kwargs)
        singular=answer[3]
        systems.append(dict(shape=list(matrix.shape), rank=int(answer[2]),
                            largest_singular_value=float(singular[0]),
                            smallest_singular_value=float(singular[-1]),
                            condition_number=float(singular[0]/singular[-1]) if singular[-1] else None,
                            default_relative_rank_cutoff=float(np.finfo(float).eps*max(matrix.shape))))
        return answer
    np.linalg.lstsq=observed_lstsq
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fitter.fit(degree, initial_extremals=seed, strict=False)
    finally:
        np.linalg.lstsq=original_lstsq
    record = dict(delta=DELTA, degree=degree, converged=result.converged,
                  message=result.message, warnings=[str(w.message) for w in caught],
                  exchange_iterations=result.exchange_iterations,
                  scale_factor=result.scale_factor,
                  requires_rescaling=result.scale_factor > 1+ROUND_OFF,
                  raw_coefficients=result.raw_coefficients.tolist(),
                  library_extrema=result.extremal_points.tolist(),
                  library_raw_error=result.raw_metrics.max_error,
                  library_bounded_error=result.metrics.max_error,
                  library_bounded_max=result.metrics.max_magnitude,
                  fitter_stage_seconds=fitter.stage_seconds, linear_systems=systems)
    # Save the expensive result before independent auditing can fail.
    path = OUTPUT / f"fit_{degree}.json"
    write_json(path, record)
    audit_started = time.monotonic()
    fitter._checkpoint("independent audit start")
    try:
        extrema, errors, raw_error, spread, raw_max = audit(result.raw_coefficients, DELTA)
        bounded_errors = 1-result.evaluate(extrema)
        bounded_error = float(np.max(np.abs(bounded_errors)))
        centered_spread = float(np.ptp(np.abs(bounded_errors-bounded_error/2)))
        identity_residual = bounded_error-2*raw_error/(1+raw_error)
        checks = dict(convergence=result.converged,
                      magnitude=raw_max/result.scale_factor <= 1+ROUND_OFF,
                      error_nonnegative=float(np.min(bounded_errors)) >= -ROUND_OFF,
                      centered_ripple=centered_spread <= max(ROUND_OFF, 2e-5*bounded_error),
                      error_identity=abs(identity_residual) <= ROUND_OFF)
        record.update(raw_error=raw_error, bounded_error=bounded_error,
                      raw_ripple_spread=spread, bounded_centered_ripple_spread=centered_spread,
                      identity_residual=identity_residual,
                      raw_max_magnitude=raw_max, bounded_max_magnitude=raw_max/result.scale_factor,
                      alternation_count=len(extrema), expected_alternation_count=(degree+3)//2,
                      extrema=extrema.tolist(), raw_signed_errors=errors.tolist(),
                      raw_alternation_lower_bound=float(np.min(np.abs(errors))),
                      checks=checks, audit_passed=all(checks.values()))
    except RuntimeError as exc:
        record.update(audit_passed=False, audit_failure=str(exc))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(audit_seconds=time.monotonic()-audit_started,
                  seconds=time.monotonic()-started,
                  peak_rss_bytes=int(peak if sys.platform == "darwin" else peak*1024))
    write_json(path, record)
    fitter._checkpoint("fit and audit complete")
    print(json.dumps({k:record.get(k) for k in ["degree", "converged", "audit_passed", "audit_failure",
                     "bounded_error", "raw_ripple_spread", "seconds", "peak_rss_bytes"]}), flush=True)


class Study:
    def __init__(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        for directory in ["sign_degree_scaling", "sign_degree_stress", "sign_degree_8000"]:
            path = BASE / directory / "fits.json"
            if path.exists():
                for r in json.loads(path.read_text()).values():
                    if r["delta"] == DELTA:
                        self.cache[r["degree"]] = dict(r, audit_passed=True)
        for path in OUTPUT.glob("fit_*.json"):
            r = json.loads(path.read_text())
            if "seconds" in r:
                self.cache[r["degree"]] = r
        path = OUTPUT / "attempts.json"
        self.attempts = json.loads(path.read_text()) if path.exists() else []

    def fit(self, degree):
        if not 1 <= degree <= MAX_DEGREE or degree % 2 != 1:
            raise ValueError("Odd degree outside experiment range")
        if degree in self.cache:
            return self.cache[degree]
        print(f"START degree={degree}, delta={DELTA}", flush=True)
        started = time.monotonic()
        try:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(degree)],
                                 capture_output=True, text=True, timeout=TIMEOUT,
                                 env=dict(os.environ, OPENBLAS_NUM_THREADS="1",
                                          VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg"))
        except subprocess.TimeoutExpired:
            self.attempts.append(dict(degree=degree, status="timeout", seconds=time.monotonic()-started))
            write_json(OUTPUT / "attempts.json", self.attempts)
            raise RuntimeError(f"Degree {degree} exceeded {TIMEOUT} seconds")
        self.attempts.append(dict(degree=degree, returncode=run.returncode,
                                 seconds=time.monotonic()-started, stdout=run.stdout, stderr=run.stderr))
        write_json(OUTPUT / "attempts.json", self.attempts)
        print(run.stdout.strip(), flush=True)
        if run.returncode:
            raise RuntimeError(run.stderr[-2000:])
        record = json.loads((OUTPUT / f"fit_{degree}.json").read_text())
        self.cache[degree] = record
        return record

    def minimum(self, epsilon, guess):
        low, high = None, None
        degree, step = guess, 2
        while low is None or high is None:
            record = self.fit(degree)
            if not record["audit_passed"]:
                raise RuntimeError(f"Degree {degree} failed independent audit")
            error = record["bounded_error"]
            if abs(error-epsilon) <= ROUND_OFF:
                raise RuntimeError("Degree decision below existing tolerance")
            if error < epsilon:
                high = degree
                degree = high-step
            else:
                low = degree
                degree = low+step
            step *= 2
        while high-low > 2:
            degree = low+2*((high-low)//4)
            record = self.fit(degree)
            if not record["audit_passed"]:
                raise RuntimeError("Independent audit failed")
            if record["bounded_error"] < epsilon:
                high = degree
            else:
                low = degree
        upper, lower = self.fit(high), self.fit(low)
        if not upper["bounded_error"]+ROUND_OFF < epsilon < lower["bounded_error"]-ROUND_OFF:
            raise RuntimeError("Final bracket below decision tolerance")
        return dict(epsilon=epsilon, degree=high, error=upper["bounded_error"],
                    previous_degree=low, previous_error=lower["bounded_error"],
                    status="verified_minimum")


def plots(rows, model, attempts):
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(8.6,6), layout="constrained")
    ax.loglog([r["epsilon"] for r in rows], [r["degree"] for r in rows], "o-",
              color="#176B91", mfc="white", lw=2, label="Verified minimum odd degree")
    x=np.geomspace(1e-10,1e-4,300)
    ax.loglog(x, model["slope"]*np.log(1/x)+model["intercept"], "--", color=".45",
              label=rf"Fit: ${model['slope']:.1f}\ln(1/\epsilon){model['intercept']:+.1f}$")
    successful=[a for a in attempts if a.get("audit_passed")
                and a.get("bounded_error",float("inf"))+ROUND_OFF < 1e-10]
    if successful:
        a=min(successful,key=lambda r:r["degree"])
        ax.scatter([1e-10], [a["degree"]], marker="D", s=65, facecolors="white",
                   edgecolors="#C45824", zorder=5,
                   label="Verified feasible degree; minimum unresolved")
    ax.set(xlabel=r"Requested uniform error $\epsilon$", ylabel="Odd polynomial degree",
           xlim=(7e-11,1.4e-4))
    ax.yaxis.set_major_locator(FixedLocator([4000,5000,6000,8000,10000,12000]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_title(r"Sign approximation at fixed $\delta=0.002$", loc="left", weight="bold", pad=28)
    ax.text(0,1.015,"Odd bounded polynomials; rescaling applied to every fit",
            transform=ax.transAxes,fontsize=9.5,color=".35")
    ax.grid(which="major", alpha=.2)
    ax.legend(fontsize=9)
    for suffix in ["png","pdf"]:
        fig.savefig(OUTPUT/f"degree_vs_epsilon.{suffix}", dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--spotcheck",type=int)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.spotcheck is not None:
        spotcheck(args.spotcheck)
        return
    if args.worker:
        if not 1 <= args.worker <= MAX_DEGREE or args.worker%2!=1:
            raise ValueError("Worker degree outside budget")
        worker(args.worker)
        return
    if args.plots_only:
        rows=json.loads((OUTPUT/"results.json").read_text())
        model=json.loads((OUTPUT/"log_fit.json").read_text())
        probe_path=OUTPUT/"precision_probes.json"
        probes=json.loads(probe_path.read_text()) if probe_path.exists() else []
        plots(rows,model,probes)
        return
    study=Study()
    path=OUTPUT/"results.json"
    rows=json.loads(path.read_text()) if path.exists() else []
    for epsilon, guess in [(1e-4,4283),(1e-5,5385),(1e-6,6495),(1e-7,7601)]:
        if not any(r["epsilon"]==epsilon for r in rows):
            row=study.minimum(epsilon,guess)
            rows.append(row)
            write_json(path,rows)
            print("RESULT",json.dumps(row),flush=True)
    rows.sort(key=lambda r:r["epsilon"],reverse=True)
    xs=np.log([1/r["epsilon"] for r in rows])
    ys=np.array([r["degree"] for r in rows])
    slope, intercept=np.polyfit(xs,ys,1)
    prediction=slope*np.log(1e10)+intercept
    model=dict(slope=float(slope),intercept=float(intercept),
               residuals=(ys-(slope*xs+intercept)).tolist(), prediction_1e10=float(prediction),
               initial_degree=2*round((prediction-1)/2)+1)
    write_json(OUTPUT/"log_fit.json",model)
    print("MODEL",json.dumps(model),flush=True)
    plots(rows,model,[])
    probes=[]
    degree=model["initial_degree"]
    first=study.fit(degree)
    probes.append({k:v for k,v in first.items() if k not in ["raw_coefficients","extrema",
                                                          "raw_signed_errors","library_extrema"]})
    write_json(OUTPUT/"precision_probes.json",probes)
    # One observed-error correction to aim safely below epsilon. This is a
    # feasible approximation probe, not an unsupported exact-degree claim.
    if first.get("audit_passed") and first["bounded_error"] >= 1e-10-ROUND_OFF:
        corrected=degree+model["slope"]*math.log(first["bounded_error"]/(.99e-10))
        degree=2*math.ceil((corrected-1)/2)+1
        second=study.fit(degree)
        probes.append({k:v for k,v in second.items() if k not in ["raw_coefficients","extrema",
                                                              "raw_signed_errors","library_extrema"]})
        write_json(OUTPUT/"precision_probes.json",probes)
    plots(rows,model,probes)


if __name__=="__main__":
    main()
