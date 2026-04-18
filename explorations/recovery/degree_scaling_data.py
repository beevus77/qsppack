#!/usr/bin/env python3
"""
Generate degree-scaling data for matrix inversion and uniform singular value
amplification target functions.

When run with --npts N, sweeps over polynomial degrees (same grid as
degree_scaling_thresh_proj.py), fits the target on the problem-specific interval,
runs QSP retraction, and writes max error vs target (and constraint checks) to CSV.

Data are written to:
  explorations/recovery/data/degree_scaling_{problem_type}_npts{k}.csv

where k = int(log2(N)). With ``--solver2``, the basename gets ``_solver2`` and cvx_poly_coef uses OSQP
instead of CLARABEL. Supports --problem-type mat_inv and uniform_sv_amp.
"""

import argparse
import csv
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import dct

from qsppack.solver import solve
from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry


# Number of Chebyshev nodes for constraint check (|p(x)| <= 1 on [-1, 1])
M_CHEB_CONSTRAINT = 1_000_000


def chebval_dct(c: np.ndarray, M: int) -> np.ndarray:
    """
    Evaluate Chebyshev series with full coefficients c at M Chebyshev nodes.

    Nodes: x_j = cos(j*pi/(M-1)), j=0..M-1.
    Implementation: zero-pad coefficients to length M and apply DCT-I.
    Returns array of length M with values at the Chebyshev nodes.
    """
    if M < 2:
        raise ValueError("M must be >= 2 for DCT-I evaluation")
    c = np.asarray(c, dtype=float)
    if len(c) > M:
        raise ValueError("Number of coefficients exceeds desired number of samples M")
    c_pad = np.zeros(M, dtype=float)
    c_pad[: len(c)] = c
    if M > 2:
        c_pad[1 : M - 1] *= 0.5
    return dct(c_pad, type=1, norm=None)


def degree_grid() -> Tuple[np.ndarray, List[int]]:
    """
    Degrees from 2^5 to 2^9, including ceilings of half-integer exponents.

    Exponents: 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9  (9 total)
    degree_k = smallest even integer >= 2**exponent_k
    """
    exponents = np.arange(5.0, 9.0 + 0.5, 0.5)
    raw_degrees = np.ceil(2.0**exponents).astype(int)
    degrees = [int(d if d % 2 == 0 else d + 1) for d in raw_degrees]
    return exponents, degrees


def run_single_degree(
    deg: int,
    target: Callable[[np.ndarray], np.ndarray],
    intervals: List[float],
    x_lo: float,
    x_hi: float,
    npts: int,
    epsil: float,
    N_weiss: int = 2**14,
    n_xplot: int = 1000,
    cvx_solver: Optional[str] = None,
    cvx_verbose: bool = False,
) -> Dict[str, object]:
    """
    Run the fitting + QSP retraction for a single degree.

    Returns a dictionary with:
      - degree, parity, npts, N_weiss
      - time_fit, time_qsp
      - max_error_poly, max_error_qsp
      - constraint_violated (True if |p(x)| > 1 at Chebyshev nodes)
      - coef_full, coef, phi_proc
    """
    parity = deg % 2

    opts_fit = {
        "intervals": intervals,
        "objnorm": np.inf,
        "epsil": epsil,
        "npts": npts,
        "fscale": 1,
        "maxiter": 100,
        "isplot": False,
        "method": "cvxpy",
    }
    if cvx_solver is not None:
        opts_fit["solver"] = cvx_solver
    if cvx_verbose:
        opts_fit["verbose"] = True

    t0 = time.time()
    coef_full = cvx_poly_coef(target, deg, opts_fit)
    time_fit = time.time() - t0
    coef = coef_full[parity::2]

    opts_qsp = dict(opts_fit)
    opts_qsp.update(
        {
            "N": int(N_weiss),
            "method": "NLFT",
            "targetPre": False,
            "typePhi": "reduced",
        }
    )

    t1 = time.time()
    phi_proc, out = solve(coef, parity, opts_qsp)
    time_qsp = time.time() - t1
    out["typePhi"] = "full"
    # NLFT returns phases for the real (Pre) QSP response; match get_entry to that channel.
    out["targetPre"] = True

    xlist = np.linspace(x_lo, x_hi, n_xplot)
    func = lambda x: chebyshev_to_func(x, coef, parity, True)
    targ_value = target(xlist)
    func_value = func(xlist)
    qsp_value = get_entry(xlist, phi_proc, out)

    abs_err_poly = np.abs(func_value - targ_value)
    abs_err_qsp = np.abs(qsp_value - targ_value)

    max_error_poly = float(np.max(abs_err_poly))
    max_error_qsp = float(np.max(abs_err_qsp))

    # Constraint check: non-retracted polynomial must satisfy |p(x)| <= 1 on [-1, 1]
    M_cheb = M_CHEB_CONSTRAINT
    try:
        vals_coef = chebval_dct(coef_full, M_cheb)
        max_abs_coef = float(np.max(np.abs(vals_coef))) - 1.0
    except MemoryError:
        M_cheb = 100_000
        vals_coef = chebval_dct(coef_full, M_cheb)
        max_abs_coef = float(np.max(np.abs(vals_coef))) - 1.0
    constraint_violated = max_abs_coef > 0.0

    return {
        "degree": int(deg),
        "parity": int(parity),
        "npts": int(npts),
        "N_weiss": int(N_weiss),
        "time_fit": float(time_fit),
        "time_qsp": float(time_qsp),
        "max_error_poly": max_error_poly,
        "max_error_qsp": max_error_qsp,
        "constraint_violated": constraint_violated,
        "coef_full": coef_full,
        "coef": coef,
        "phi_proc": np.asarray(phi_proc),
    }


def build_target_and_domain(
    problem_type: str,
    a: float,
    epsil: float,
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[float], float, float]:
    """Return (target callable, intervals list, x_lo, x_hi) for evaluation."""
    if problem_type == "uniform_sv_amp":
        target = lambda x: (1.0 - epsil) * x / a
        intervals = [0.0, a]
        x_lo, x_hi = 0.0, a
    elif problem_type == "mat_inv":
        target = lambda x: a / x
        intervals = [a, 1.0]
        x_lo, x_hi = a, 1.0
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    return target, intervals, x_lo, x_hi


def generate_data(
    problem_type: str,
    npts: int,
    csv_path: str,
    a: float = 0.2,
    epsil: float = 0.0,
    N_weiss: int = 2**14,
    force: bool = False,
    cvx_solver: Optional[str] = None,
    cvx_verbose: bool = False,
) -> None:
    """
    Generate / append data for the degree sweep and write to CSV.

    If csv_path exists and force is False, only missing degrees are computed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(csv_path)
    if not data_dir:
        data_dir = os.path.join(script_dir, "data")
        csv_path = os.path.join(data_dir, csv_path)
    os.makedirs(data_dir, exist_ok=True)

    target, intervals, x_lo, x_hi = build_target_and_domain(problem_type, a, epsil)
    exponents, degrees_all = degree_grid()

    if force and os.path.exists(csv_path):
        print(f"Forcing full regeneration: removing existing data at {csv_path}")
        os.remove(csv_path)

    existing_degrees = set()
    if os.path.exists(csv_path):
        import pandas as pd

        df_existing = pd.read_csv(csv_path)
        if "degree" in df_existing.columns:
            existing_degrees = set(df_existing["degree"].astype(int))

    degrees_to_run = [d for d in degrees_all if (force or d not in existing_degrees)]

    if not degrees_to_run:
        print("All requested degrees already present in CSV; skipping data generation.")
        return

    fieldnames = [
        "degree",
        "exp2",
        "parity",
        "npts",
        "N_weiss",
        "time_fit",
        "time_qsp",
        "max_error_poly",
        "max_error_qsp",
        "constraint_violated",
        "coef",
        "coef_full",
        "phi_proc",
    ]

    resuming = os.path.exists(csv_path) and not force and bool(existing_degrees)
    mode = "a" if resuming else "w"

    print(f"Generating degree-scaling data for {problem_type} (a={a}, epsil={epsil})")
    print(f"  CSV path = {csv_path}")
    if cvx_solver is not None:
        print(f"  cvx_poly_coef solver = {cvx_solver}")
    print(f"  npts = {npts}")
    print(f"  degrees = {degrees_all}")
    if resuming:
        print(f"  resuming: {len(existing_degrees)} existing, {len(degrees_to_run)} to run")
        print(f"  degrees to run = {degrees_to_run}")
    print("-" * 60)

    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if not resuming:
            writer.writeheader()

        for deg, exp2 in zip(degrees_all, exponents):
            if deg not in degrees_to_run:
                continue

            print(f"degree = {deg} (2^{exp2:.1f}), npts = {npts}")
            result = run_single_degree(
                deg,
                target=target,
                intervals=intervals,
                x_lo=x_lo,
                x_hi=x_hi,
                npts=npts,
                epsil=epsil,
                N_weiss=N_weiss,
                cvx_solver=cvx_solver,
                cvx_verbose=cvx_verbose,
            )

            phi_arr = np.asarray(result["phi_proc"])
            imag_max = float(np.max(np.abs(phi_arr.imag)))
            tol_imag = 1e-10
            if imag_max > tol_imag:
                raise ValueError(
                    f"phi_proc has unexpectedly large imaginary part: "
                    f"max |Im(phi_proc)| = {imag_max:.3e} > {tol_imag:.1e}"
                )
            phi_real = phi_arr.real

            row = {
                "degree": int(result["degree"]),
                "exp2": float(exp2),
                "parity": int(result["parity"]),
                "npts": int(result["npts"]),
                "N_weiss": int(result["N_weiss"]),
                "time_fit": float(result["time_fit"]),
                "time_qsp": float(result["time_qsp"]),
                "max_error_poly": float(result["max_error_poly"]),
                "max_error_qsp": float(result["max_error_qsp"]),
                "constraint_violated": bool(result["constraint_violated"]),
                "coef": json.dumps(np.asarray(result["coef"]).tolist()),
                "coef_full": json.dumps(np.asarray(result["coef_full"]).tolist()),
                "phi_proc": json.dumps(phi_real.tolist()),
            }
            writer.writerow(row)
            f.flush()

    print("-" * 60)
    print(f"Data written to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Degree-scaling data generation for matrix inversion and uniform "
            "singular value amplification: sweep degrees with fixed npts, write CSV."
        )
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=("mat_inv", "uniform_sv_amp"),
        default="mat_inv",
        help=(
            "Target: 'mat_inv' = a/x on [a,1]; "
            "'uniform_sv_amp' = (1-eps)*x/a on [0,a]. Default: mat_inv."
        ),
    )
    parser.add_argument(
        "--npts",
        type=int,
        required=True,
        metavar="N",
        help="Exact number of convex-optimization points for every degree (e.g. 256).",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.2,
        help="Parameter a: interval [0,a] or [a,1] and target scale. Must be in (0,1). Default: 0.2.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon for uniform_sv_amp target (1-eps)*x/a; also used in opts for mat_inv. Default: 0.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=2**14,
        help="Weiss N for QSP solver. Default: 2^14.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Output CSV path (default: data/degree_scaling_{problem_type}_npts{k}.csv). "
            "With --solver2, _solver2 is inserted before the extension."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full regeneration of data, even if CSV already exists.",
    )
    parser.add_argument(
        "--solver2",
        action="store_true",
        help=(
            "Use OSQP for cvx_poly_coef (instead of default CLARABEL). "
            "Appends _solver2 to the CSV basename so default runs are unchanged."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose=True to cvx_poly_coef's CVXPY solve (does not change output paths).",
    )
    args = parser.parse_args()

    if not (0.0 < args.a < 1.0):
        raise ValueError("--a must lie in (0, 1).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    npts_exp = int(np.log2(args.npts))
    if args.csv is None:
        base = f"degree_scaling_{args.problem_type}_npts{npts_exp}"
        if args.solver2:
            base += "_solver2"
        csv_path = os.path.join(data_dir, f"{base}.csv")
    else:
        if args.solver2:
            root, ext = os.path.splitext(args.csv)
            csv_path = root + "_solver2" + (ext if ext else ".csv")
        else:
            csv_path = args.csv

    generate_data(
        problem_type=args.problem_type,
        npts=args.npts,
        csv_path=csv_path,
        a=args.a,
        epsil=args.epsilon,
        N_weiss=args.N,
        force=args.force,
        cvx_solver="OSQP" if args.solver2 else None,
        cvx_verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
