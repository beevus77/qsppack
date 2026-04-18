#!/usr/bin/env python3
"""
Degree-scaling experiment for the threshold projection problem.

This script now serves two roles:

- Generate data for a figure with:
    x-axis: polynomial degree (log scale)
    y-axis: max error vs the target function (log scale)
  for both the fitting polynomial and the retracted (QSP) polynomial.

- Optionally, reproduce the original per-degree plots of:
    - the target, polynomial approximation, and QSP curve
    - the corresponding pointwise errors

Data are written to:
  explorations/recovery/data/degree_scaling_thresh_proj_1x.csv  (single: npts = degree)
  explorations/recovery/data/degree_scaling_thresh_proj_2x.csv (double: npts = 2*degree)
  explorations/recovery/data/degree_scaling_thresh_proj_npts{k}.csv (when --npts N is set; k = int(log2(N)))

With ``--npts N``, optional ``--ploterror --errordegree D`` writes a two-panel figure:
degree scaling (same as the usual summary) plus pointwise $|q-f|$ and $|p/s-f|$
on $[-1,1]$ (with $s=\max_{[-1,1]}|p|$ from Chebyshev samples), excluding narrow bands
around $x=\pm\frac12$.

The main execution path:
  - checks whether the CSV already exists and contains all requested degrees
  - only runs the expensive optimization for missing degrees (unless --force)
  - produces a summary log-log plot of max error vs degree
"""

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.fft import dct

from qsppack.solver import solve
from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry


BLUE = "#0072B2"
MAIZE = "#E69F00"
GREEN = "#009E73"

# Marker sizes for 2-norm polynomial-space plots (BLUE = fit, MAIZE = retraction)
MARKERSIZE_BLUE_X = 10  # 'x' when constraints violated
MARKERSIZE_BLUE_O = 8  # 'o' when constraints satisfied
MARKERSIZE_MAIZE_O = 12  # retraction (open circles)
DELTA = 0.05

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
    # DCT-I synthesis expects half-scaled interior coefficients
    if M > 2:
        c_pad[1 : M - 1] *= 0.5
    return dct(c_pad, type=1, norm=None)


def target(x: np.ndarray) -> np.ndarray:
    """Indicator of |x| < 0.5, vectorized."""
    return np.where(np.abs(x) < 0.5, 1.0, 0.0)


def degree_grid() -> Tuple[np.ndarray, List[int]]:
    """
    Degrees from 2^5 to 2^9, including ceilings of half-integer exponents.

    Exponents: 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9  (9 total)
    degree_k = smallest even integer >= 2**exponent_k
    """
    # 5, 5.5, ..., 9.0
    exponents = np.arange(5.0, 9.0 + 0.5, 0.5)
    # First take the ceiling to an integer, then round up to the nearest even integer.
    raw_degrees = np.ceil(2.0**exponents).astype(int)
    degrees = [int(d if d % 2 == 0 else d + 1) for d in raw_degrees]
    return exponents, degrees


def plot_polynomials_and_errors(
    xlist: np.ndarray,
    targ_value: np.ndarray,
    func_value: np.ndarray,
    qsp_value: np.ndarray,
    show_polynomial_plots: bool,
    show_error_plots: bool,
) -> None:
    """Reproduce the original polynomial / error plots (optional)."""
    if show_polynomial_plots:
        plt.figure()
        plt.plot(xlist, targ_value, label="True")
        plt.plot(xlist, func_value, label="Polynomial Approximation")
        plt.plot(xlist, qsp_value, label="QSP")
        plt.plot(xlist, np.ones(len(xlist)), "k--", label="Constraint")
        plt.plot(xlist, -np.ones(len(xlist)), "k--")
        plt.xlabel("$x$", fontsize=12)
        plt.grid()
        plt.legend(loc="best")
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.show()

    if show_error_plots:
        plt.figure()
        mask1 = xlist < 0.5 - DELTA
        mask2 = xlist > 0.5 + DELTA
        abs_err_poly = np.abs(func_value - targ_value)
        abs_err_qsp = np.abs(qsp_value - targ_value)
        plt.plot(xlist[mask1], abs_err_poly[mask1], color=BLUE)
        plt.plot(xlist[mask2], abs_err_poly[mask2], color=BLUE)
        plt.plot(xlist[mask1], abs_err_qsp[mask1], color=MAIZE)
        plt.plot(xlist[mask2], abs_err_qsp[mask2], color=MAIZE)
        plt.yscale("log")
        plt.xlim([0, 1])
        plt.ylim([1e-7, 1e-2])
        plt.tight_layout()
        plt.show()


def run_single_degree(
    deg: int,
    N_weiss: int = 2**14,
    n_xplot: int = 1000,
    show_polynomial_plots: bool = False,
    show_error_plots: bool = False,
    npts_factor: float = 1.0,
    npts_exact: Optional[int] = None,
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
    npts = int(npts_exact) if npts_exact is not None else int(npts_factor * deg)

    opts_fit = {
        "intervals": [0, 0.5 - DELTA, 0.5 + DELTA, 1],
        "objnorm": np.inf,
        "epsil": 0,
        "npts": npts,
        "fscale": 1,
        "maxiter": 100,
        "isplot": False,
        "method": "cvxpy",
    }

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

    xlist = np.linspace(0.0, 1.0, n_xplot)
    func = lambda x: chebyshev_to_func(x, coef, parity, True)
    targ_value = target(xlist)
    func_value = func(xlist)
    qsp_value = get_entry(xlist, phi_proc, out)

    abs_err_poly = np.abs(func_value - targ_value)
    abs_err_qsp = np.abs(qsp_value - targ_value)

    # Only measure error off the transition region:
    #   [0, 0.5 - DELTA] and [0.5 + DELTA, 1].
    mask_left = xlist <= 0.5 - DELTA
    mask_right = xlist >= 0.5 + DELTA
    mask = mask_left | mask_right
    if not np.any(mask):
        raise RuntimeError("No evaluation points in the off-transition regions.")

    max_error_poly = float(np.max(abs_err_poly[mask]))
    max_error_qsp = float(np.max(abs_err_qsp[mask]))

    # Constraint check: non-retracted polynomial must satisfy |p(x)| <= 1 on [-1, 1]
    # Evaluate via DCT-I at Chebyshev nodes (same as plot_recovery_conv_polynomial_space)
    M_cheb = M_CHEB_CONSTRAINT
    try:
        vals_coef = chebval_dct(coef_full, M_cheb)
        max_abs_coef = float(np.max(np.abs(vals_coef))) - 1.0
    except MemoryError:
        M_cheb = 100_000
        vals_coef = chebval_dct(coef_full, M_cheb)
        max_abs_coef = float(np.max(np.abs(vals_coef))) - 1.0
    constraint_violated = max_abs_coef > 0.0

    if show_polynomial_plots or show_error_plots:
        plot_polynomials_and_errors(
            xlist,
            targ_value,
            func_value,
            qsp_value,
            show_polynomial_plots=show_polynomial_plots,
            show_error_plots=show_error_plots,
        )

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


def generate_data(
    csv_path: str,
    force: bool = False,
    show_polynomial_plots: bool = False,
    show_error_plots: bool = False,
    npts_factor: float = 1.0,
    npts_exact: Optional[int] = None,
) -> None:
    """
    Generate / append data for the degree sweep and write to CSV.

    If csv_path exists and force is False, we only compute missing degrees.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(csv_path)
    if not data_dir:
        data_dir = os.path.join(script_dir, "data")
        csv_path = os.path.join(data_dir, csv_path)
    os.makedirs(data_dir, exist_ok=True)

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

    print("Generating degree-scaling data for threshold projection")
    print(f"  CSV path = {csv_path}")
    print(f"  degrees  = {degrees_all}")
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

            npts_used = int(npts_exact) if npts_exact is not None else int(npts_factor * deg)
            print(f"degree = {deg} (2^{exp2:.1f}), npts = {npts_used}")
            result = run_single_degree(
                deg,
                N_weiss=2**14,
                n_xplot=1000,
                show_polynomial_plots=show_polynomial_plots,
                show_error_plots=show_error_plots,
                npts_factor=npts_factor,
                npts_exact=npts_exact,
            )

            phi_arr = np.asarray(result["phi_proc"])
            # Imaginary part should be purely numerical noise; enforce this.
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


def _parse_constraint_violated(val) -> Optional[bool]:
    """Parse constraint_violated from CSV (bool or string 'True'/'False'). Returns None if missing/NaN (need to compute)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


def _constraint_violated_from_row(row, M_cheb: int = 1_000_000) -> bool:
    """Compute constraint_violated from coef_full in a CSV row (no CSV write)."""
    coef_full = np.array(json.loads(row["coef_full"]), dtype=float)
    try:
        vals = chebval_dct(coef_full, M_cheb)
    except MemoryError:
        vals = chebval_dct(coef_full, 100_000)
    return (np.max(np.abs(vals)) - 1.0) > 0.0


def _load_threshold_summary_dataframe(
    csv_path: str,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
):
    """
    Read threshold-projection CSV and return a processed DataFrame for the
    degree-vs-max-error summary plot (same filtering as plot_summary).
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if "degree" not in data.columns:
        raise ValueError("CSV must contain 'degree' column.")
    if "max_error_poly" not in data.columns or "max_error_qsp" not in data.columns:
        raise ValueError("CSV must contain 'max_error_poly' and 'max_error_qsp' columns.")

    if "constraint_violated" not in data.columns:
        data["constraint_violated"] = None
    constraint_violated_list = []
    for idx, row in data.iterrows():
        cached = _parse_constraint_violated(row.get("constraint_violated"))
        if cached is not None:
            constraint_violated_list.append(cached)
        elif "coef_full" in row and pd.notna(row.get("coef_full")):
            constraint_violated_list.append(_constraint_violated_from_row(row))
        else:
            constraint_violated_list.append(False)
    data["constraint_violated"] = constraint_violated_list

    data = data.sort_values("degree").reset_index(drop=True)
    if leave_out_last and len(data) > 0:
        data = data.iloc[:-1, :].reset_index(drop=True)
    if max_exp is not None:
        if "exp2" not in data.columns:
            data = data[data["degree"] <= 2**max_exp].reset_index(drop=True)
        else:
            data = data[data["exp2"] <= max_exp].reset_index(drop=True)
    return data


def _plot_degree_scaling_on_ax(ax: Axes, data) -> None:
    """Draw the log-log degree vs max error summary on an existing Axes."""
    from matplotlib.lines import Line2D

    degrees = data["degree"].to_numpy(dtype=float)
    exponents = (
        data["exp2"].to_numpy(dtype=float)
        if "exp2" in data.columns
        else np.log2(degrees)
    )
    err_poly = data["max_error_poly"].to_numpy(dtype=float)
    err_qsp = data["max_error_qsp"].to_numpy(dtype=float)
    constraint_violated = data["constraint_violated"].to_numpy(dtype=bool)

    degrees_clamped = np.where(degrees > 0.0, degrees, 1e-16)
    err_poly_clamped = np.where(err_poly > 0.0, err_poly, 1e-16)
    err_qsp_clamped = np.where(err_qsp > 0.0, err_qsp, 1e-16)

    for i in range(len(degrees_clamped)):
        if constraint_violated[i]:
            ax.plot(
                degrees_clamped[i],
                err_poly_clamped[i],
                "x",
                color=BLUE,
                markersize=MARKERSIZE_BLUE_X,
                markeredgewidth=2,
            )
        else:
            ax.plot(
                degrees_clamped[i],
                err_poly_clamped[i],
                "o",
                color=BLUE,
                markersize=MARKERSIZE_BLUE_O,
                markeredgewidth=1,
                fillstyle="none",
            )
        ax.plot(
            degrees_clamped[i],
            err_qsp_clamped[i],
            "o",
            color=MAIZE,
            markersize=MARKERSIZE_MAIZE_O,
            markeredgewidth=1,
            fillstyle="none",
        )
    ax.plot(degrees_clamped, err_poly_clamped, "k--", alpha=0.7, linewidth=1)
    ax.plot(degrees_clamped, err_qsp_clamped, "k--", alpha=0.7, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Maximum error vs target")
    ax.xaxis.set_major_locator(FixedLocator(degrees_clamped))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(degrees_clamped)
    ax.set_xticklabels([rf"$2^{{{e:g}}}$" for e in exponents])
    ax.grid(True, which="both", alpha=0.3)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color=BLUE,
            linestyle="None",
            markersize=8,
            markeredgewidth=2,
            label="Polynomial max error (constraints violated)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            linestyle="None",
            markersize=7,
            fillstyle="none",
            label="Polynomial max error (constraints satisfied)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=MAIZE,
            linestyle="None",
            markersize=5,
            fillstyle="none",
            label="Retracted polynomial max error",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")


def _max_abs_poly_on_interval_minus1_1(coef_full: np.ndarray) -> float:
    """Max |p(x)| on [-1, 1] via Chebyshev nodes (same spirit as constraint check)."""
    M_cheb = M_CHEB_CONSTRAINT
    try:
        vals = chebval_dct(coef_full, M_cheb)
    except MemoryError:
        vals = chebval_dct(coef_full, 100_000)
    return float(np.max(np.abs(vals)))


def _off_jump_masks_unit_interval(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Masks for x in [0, 1] where the threshold target is constant, excluding
    the transition band around x = 0.5 (same convention as run_single_degree).
    """
    m_left = x <= 0.5 - DELTA
    m_right = x >= 0.5 + DELTA
    return m_left, m_right


def plot_pointwise_retraction_and_scaled_poly_errors_on_ax(
    ax: Axes,
    row: Any,
    n_x: int = 4000,
) -> None:
    """
    Log-linear plot of |q-f|, |p/s-f|, and |p-f| on [0, 1], excluding the transition
    band around x = 0.5. Here ``p`` comes from the CSV ``coef`` column (unretracted
    parity-stripped Chebyshev coefficients); ``coef_full`` is only used for
    ``s = max_{[-1,1]} |p|``. When ``s`` is essentially 1 and ``q`` is close to ``p``,
    the scaled and retracted error curves can coincide on a log scale.
    """
    coef = np.array(json.loads(row["coef"]), dtype=float)
    coef_full = np.array(json.loads(row["coef_full"]), dtype=float)
    phi_proc = np.array(json.loads(row["phi_proc"]), dtype=float)
    parity = int(row["parity"])

    # CSV phi_proc is a full phase list for the Pre-channel polynomial values.
    # This matches the max_error_qsp values already stored in the CSV summary plot.
    out_qsp = {"targetPre": True, "parity": parity, "typePhi": "full"}

    x = np.linspace(0.0, 1.0, n_x)
    targ = target(x)
    p = chebyshev_to_func(x, coef, parity, True)
    q = get_entry(x, phi_proc, out_qsp)

    s = _max_abs_poly_on_interval_minus1_1(coef_full)
    if s <= 0.0:
        raise ValueError("max |p| on [-1, 1] is zero; cannot scale unretracted polynomial.")

    floor = 1e-20
    abs_err_qsp = np.maximum(np.abs(q - targ), floor)
    abs_err_scaled_poly = np.maximum(np.abs(p / s - targ), floor)
    abs_err_fit_poly = np.maximum(np.abs(p - targ), floor)

    m_left, m_right = _off_jump_masks_unit_interval(x)
    first = True
    for mask in (m_left, m_right):
        if not np.any(mask):
            continue
        lbl_q = "Retracted $|q-f|$" if first else None
        lbl_ps = "Scaled fit $|p/s-f|$" if first else None
        lbl_p = "Unscaled fit $|p-f|$" if first else None
        ax.plot(x[mask], abs_err_qsp[mask], color=MAIZE, linewidth=1.2, label=lbl_q)
        ax.plot(x[mask], abs_err_scaled_poly[mask], color=BLUE, linewidth=1.2, label=lbl_ps)
        ax.plot(
            x[mask],
            abs_err_fit_poly[mask],
            color=GREEN,
            linewidth=1.2,
            linestyle="--",
            label=lbl_p,
        )
        first = False

    # Directly report the same "max off-transition error" definition as the left plot data.
    off_transition_mask = m_left | m_right
    max_err_qsp = float(np.max(abs_err_qsp[off_transition_mask]))
    max_err_scaled_poly = float(np.max(abs_err_scaled_poly[off_transition_mask]))
    max_err_fit_poly = float(np.max(abs_err_fit_poly[off_transition_mask]))
    max_qp = float(np.max(np.abs((q - p)[off_transition_mask])))
    max_p_ps = float(np.max(np.abs((p / s - p)[off_transition_mask])))

    ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel("Absolute error")
    ax.set_xlim([0.0, 1.0])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    deg = int(row["degree"])
    ax.set_title(
        f"Pointwise errors (degree {deg})\n"
        f"max |q-f|={max_err_qsp:.2e}, max |p/s-f|={max_err_scaled_poly:.2e}, "
        f"max |p-f|={max_err_fit_poly:.2e}\n"
        f"s={s:.10f}, max|q-p|={max_qp:.2e}, max|p/s-p|={max_p_ps:.2e}"
    )


def plot_summary_and_pointwise_errors(
    csv_path: str,
    fig_path: str,
    errordegree: int,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
) -> None:
    """Left: degree scaling summary; right: pointwise errors for one degree."""
    import pandas as pd

    raw = pd.read_csv(csv_path)
    if "degree" not in raw.columns:
        raise ValueError("CSV must contain 'degree' column.")
    degrees_in_file = set(int(d) for d in raw["degree"].tolist())
    if int(errordegree) not in degrees_in_file:
        raise ValueError(
            f"errordegree={errordegree} is not among degrees in {csv_path}: "
            f"{sorted(degrees_in_file)}"
        )

    data = _load_threshold_summary_dataframe(csv_path, leave_out_last, max_exp)
    matches = raw.loc[raw["degree"].astype(int) == int(errordegree)]
    if matches.empty:
        raise ValueError(f"No CSV row found for degree {errordegree}.")
    row = matches.iloc[0]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_degree_scaling_on_ax(ax0, data)
    ax0.set_title("Max error vs degree")
    plot_pointwise_retraction_and_scaled_poly_errors_on_ax(ax1, row)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


def plot_summary(
    csv_path: str,
    fig_path: str,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
) -> None:
    """Plot degree (log) vs max error (log) for fit and QSP.
    Polynomial: x if constraint violated, o (larger) if satisfied. Retraction: o (larger).
    Does not write to CSV (does not overwrite existing data).
    If max_exp is set (e.g. 8), only plot degrees with exp2 <= max_exp (degree <= 2^max_exp).
    """
    data = _load_threshold_summary_dataframe(csv_path, leave_out_last, max_exp)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_degree_scaling_on_ax(ax, data)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


def plot_summary_both(
    csv_single: str,
    csv_double: str,
    fig_path: str,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
) -> None:
    """Plot degree vs max error for both npts=deg and npts=2*deg on one figure.
    Polynomial: x if constraint violated, o (larger) if satisfied. Retraction: o (larger).
    Does not write to CSV (does not overwrite existing data).
    If max_exp is set (e.g. 8), only plot degrees with exp2 <= max_exp (degree <= 2^max_exp).
    """
    import pandas as pd
    from matplotlib.lines import Line2D

    if not os.path.exists(csv_single):
        raise FileNotFoundError(f"CSV file not found for single-points data: {csv_single}")
    if not os.path.exists(csv_double):
        raise FileNotFoundError(f"CSV file not found for double-points data: {csv_double}")

    data1 = pd.read_csv(csv_single)
    data2 = pd.read_csv(csv_double)

    required_cols = {"degree", "max_error_poly", "max_error_qsp"}
    for name, df in (("single", data1), ("double", data2)):
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                f"CSV for {name}-points data is missing required columns: {missing}"
            )

    # Build constraint_violated for each dataset (no CSV write); avoid bool-in-float column
    for data in (data1, data2):
        if "constraint_violated" not in data.columns:
            data["constraint_violated"] = None
        lst = []
        for idx, row in data.iterrows():
            cached = _parse_constraint_violated(row.get("constraint_violated"))
            if cached is not None:
                lst.append(cached)
            elif "coef_full" in row and pd.notna(row.get("coef_full")):
                lst.append(_constraint_violated_from_row(row))
            else:
                lst.append(False)
        data["constraint_violated"] = lst

    data1 = data1.sort_values("degree").reset_index(drop=True)
    data2 = data2.sort_values("degree").reset_index(drop=True)
    if leave_out_last and len(data1) > 0 and len(data2) > 0:
        data1 = data1.iloc[:-1, :].reset_index(drop=True)
        data2 = data2.iloc[:-1, :].reset_index(drop=True)
    if max_exp is not None:
        if "exp2" in data1.columns:
            data1 = data1[data1["exp2"] <= max_exp].reset_index(drop=True)
            data2 = data2[data2["exp2"] <= max_exp].reset_index(drop=True)
        else:
            data1 = data1[data1["degree"] <= 2**max_exp].reset_index(drop=True)
            data2 = data2[data2["degree"] <= 2**max_exp].reset_index(drop=True)

    degrees1 = data1["degree"].to_numpy(dtype=float)
    degrees2 = data2["degree"].to_numpy(dtype=float)
    if not np.array_equal(degrees1, degrees2):
        raise ValueError(
            "Degree grids for single and double npts data do not match; "
            "cannot overlay them cleanly."
        )

    degrees = degrees1
    err_poly_single = data1["max_error_poly"].to_numpy(dtype=float)
    err_qsp_single = data1["max_error_qsp"].to_numpy(dtype=float)
    err_poly_double = data2["max_error_poly"].to_numpy(dtype=float)
    err_qsp_double = data2["max_error_qsp"].to_numpy(dtype=float)
    violated_single = data1["constraint_violated"].to_numpy(dtype=bool)
    violated_double = data2["constraint_violated"].to_numpy(dtype=bool)

    # Clamp to positive for log scale
    degrees_clamped = np.where(degrees > 0.0, degrees, 1e-16)
    err_poly_single_clamped = np.where(err_poly_single > 0.0, err_poly_single, 1e-16)
    err_qsp_single_clamped = np.where(err_qsp_single > 0.0, err_qsp_single, 1e-16)
    err_poly_double_clamped = np.where(err_poly_double > 0.0, err_poly_double, 1e-16)
    err_qsp_double_clamped = np.where(err_qsp_double > 0.0, err_qsp_double, 1e-16)

    # Exponents for x-tick labels
    if "exp2" in data1.columns:
        exponents = data1["exp2"].to_numpy(dtype=float)
    else:
        exponents = np.log2(degrees_clamped)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(8, 6))

    # Polynomial max error: blue — x if violated, o if satisfied
    for i in range(len(degrees_clamped)):
        if violated_single[i]:
            ax.plot(
                degrees_clamped[i],
                err_poly_single_clamped[i],
                "x",
                color=BLUE,
                markersize=8,
                markeredgewidth=2,
            )
        else:
            ax.plot(
                degrees_clamped[i],
                err_poly_single_clamped[i],
                "o",
                color=BLUE,
                markersize=7,
                markeredgewidth=1,
                fillstyle="none",
            )
        if violated_double[i]:
            ax.plot(
                degrees_clamped[i],
                err_poly_double_clamped[i],
                "x",
                color=BLUE,
                markersize=8,
                markeredgewidth=2,
            )
        else:
            ax.plot(
                degrees_clamped[i],
                err_poly_double_clamped[i],
                "o",
                color=BLUE,
                markersize=7,
                markeredgewidth=1,
                fillstyle="none",
            )
        ax.plot(
            degrees_clamped[i],
            err_qsp_single_clamped[i],
            "o",
            color=MAIZE,
            markersize=5,
            markeredgewidth=1,
            fillstyle="none",
        )
        ax.plot(
            degrees_clamped[i],
            err_qsp_double_clamped[i],
            "o",
            color=MAIZE,
            markersize=5,
            markeredgewidth=1,
            fillstyle="none",
        )
    ax.plot(degrees_clamped, err_poly_single_clamped, "k--", alpha=0.7, linewidth=1)
    ax.plot(degrees_clamped, err_poly_double_clamped, "k--", alpha=0.7, linewidth=1)
    ax.plot(degrees_clamped, err_qsp_single_clamped, "k-.", alpha=0.7, linewidth=1)
    ax.plot(degrees_clamped, err_qsp_double_clamped, "k-.", alpha=0.7, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Maximum error vs target")
    # Only show ticks/labels at our data points (integer and half-integer powers of 2)
    ax.xaxis.set_major_locator(FixedLocator(degrees_clamped))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(degrees_clamped)
    ax.set_xticklabels([rf"$2^{{{e:g}}}$" for e in exponents])
    ax.grid(True, which="both", alpha=0.3)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color=BLUE,
            linestyle="None",
            markersize=MARKERSIZE_BLUE_X,
            markeredgewidth=2,
            label="Polynomial (constraints violated)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            linestyle="None",
            markersize=MARKERSIZE_BLUE_O,
            fillstyle="none",
            label="Polynomial (constraints satisfied)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=MAIZE,
            linestyle="None",
            markersize=MARKERSIZE_MAIZE_O,
            fillstyle="none",
            label="Retracted polynomial",
        ),
        Line2D([0], [0], linestyle="--", color="k", label="npts = degree"),
        Line2D([0], [0], linestyle="-.", color="k", label="npts = 2·degree"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Degree-scaling threshold projection experiment: "
            "generate data and plot max error vs degree."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Output CSV path "
            "(default: degree_scaling_thresh_proj_1x.csv, _2x.csv, or _nptsN.csv)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output figure path "
            "(default: degree_scaling_thresh_proj_1x.pdf, _2x.pdf, or _nptsN.pdf)."
        ),
    )
    parser.add_argument(
        "--plotting-mode",
        type=str,
        choices=["single", "double", "both"],
        default="single",
        help=(
            "Which npts configuration to use for plotting: "
            "'single' (npts = degree), 'double' (npts = 2 * degree), "
            "or 'both' (overlay both on one plot; requires existing CSVs)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full regeneration of data, even if CSV already exists.",
    )
    parser.add_argument(
        "--double-points",
        action="store_true",
        help=(
            "[Deprecated] Shortcut for --plotting-mode=double. "
            "Use twice as many convex optimization points: npts = 2 * degree. "
            "Results are written to *_2x.csv / *_2x.pdf by default."
        ),
    )
    parser.add_argument(
        "--per-degree-plots",
        action="store_true",
        help=(
            "Show per-degree polynomial and error plots as each degree is computed "
            "(off by default)."
        ),
    )
    parser.add_argument(
        "--no-summary-plot",
        action="store_true",
        help="Skip the summary degree-vs-error plot (still generates data).",
    )
    parser.add_argument(
        "--leave-out-last",
        action="store_true",
        help="Omit the largest degree from the summary plot (keeps data intact).",
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use this exact number of convex-optimization points for every degree "
            "(overrides plotting-mode: single/double). "
            "Filename uses k = int(log2(N)), e.g. --npts 256 -> degree_scaling_thresh_proj_npts8.csv."
        ),
    )
    parser.add_argument(
        "--max-exp",
        type=int,
        default=None,
        metavar="K",
        help=(
            "When plotting, only include degrees with exp2 <= K (degree <= 2^K). "
            "E.g. --max-exp 8 plots up to 2^8 (256) instead of 2^9."
        ),
    )
    parser.add_argument(
        "--ploterror",
        action="store_true",
        help=(
            "Only with --npts: write a two-panel PDF (degree summary + pointwise errors). "
            "Requires --errordegree."
        ),
    )
    parser.add_argument(
        "--errordegree",
        type=int,
        default=None,
        metavar="D",
        help=(
            "Degree row to use for the pointwise error subplot; must appear in the CSV "
            "(see --ploterror)."
        ),
    )
    args = parser.parse_args()

    if args.ploterror and args.npts is None:
        raise ValueError("--ploterror is only supported when --npts is set.")
    if args.ploterror and args.errordegree is None:
        raise ValueError("--ploterror requires --errordegree.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    figures_dir = os.path.join(script_dir, "figures")

    # Resolve plotting mode, allowing --double-points as a shorthand for "double".
    mode = args.plotting_mode
    if args.double_points and mode == "single":
        mode = "double"

    if mode == "both" and args.csv is not None:
        raise ValueError(
            "--csv cannot be used with --plotting-mode=both. "
            "Use the default CSV locations for single and double runs instead."
        )
    if args.npts is not None and mode == "both":
        raise ValueError(
            "--npts cannot be used with --plotting-mode=both."
        )

    # Paths for single- and double-points CSVs/figures (used depending on mode).
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    base_csv_single = os.path.join(data_dir, "degree_scaling_thresh_proj_1x.csv")
    base_csv_double = os.path.join(data_dir, "degree_scaling_thresh_proj_2x.csv")
    base_fig_single = os.path.join(figures_dir, "degree_scaling_thresh_proj_1x.pdf")
    base_fig_double = os.path.join(figures_dir, "degree_scaling_thresh_proj_2x.pdf")
    base_fig_both = os.path.join(figures_dir, "degree_scaling_thresh_proj_both.pdf")

    # Exact npts override: one CSV/fig per npts value (name uses log2(npts) as in fgt_polynomial_space).
    if args.npts is not None:
        npts_exp = int(np.log2(args.npts))
        if args.csv is None:
            csv_path = os.path.join(data_dir, f"degree_scaling_thresh_proj_npts{npts_exp}.csv")
        else:
            csv_path = args.csv
        if args.output is None:
            fig_path = os.path.join(figures_dir, f"degree_scaling_thresh_proj_npts{npts_exp}.pdf")
        else:
            fig_path = args.output
            out_dir = os.path.dirname(os.path.abspath(fig_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        generate_data(
            csv_path,
            force=args.force,
            show_polynomial_plots=args.per_degree_plots,
            show_error_plots=args.per_degree_plots,
            npts_exact=args.npts,
        )
        if args.ploterror:
            plot_summary_and_pointwise_errors(
                csv_path,
                fig_path,
                args.errordegree,
                leave_out_last=args.leave_out_last,
                max_exp=args.max_exp,
            )
        elif not args.no_summary_plot:
            plot_summary(
                csv_path,
                fig_path,
                leave_out_last=args.leave_out_last,
                max_exp=args.max_exp,
            )

    # Mode-based paths (single = 1x, double = 2x).
    elif mode in ("single", "double"):
        if args.csv is None:
            csv_path = base_csv_single if mode == "single" else base_csv_double
        else:
            csv_path = args.csv

        if args.output is None:
            fig_path = base_fig_single if mode == "single" else base_fig_double
        else:
            fig_path = args.output
            out_dir = os.path.dirname(os.path.abspath(fig_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        # Generate data (with caching / resumption) and optional per-degree plots.
        npts_factor = 1.0 if mode == "single" else 2.0
        generate_data(
            csv_path,
            force=args.force,
            show_polynomial_plots=args.per_degree_plots,
            show_error_plots=args.per_degree_plots,
            npts_factor=npts_factor,
        )

        # If data already existed and user only wants the figure, generate_data()
        # will be cheap and will not rerun expensive optimization.
        if not args.no_summary_plot:
            plot_summary(
                csv_path,
                fig_path,
                leave_out_last=args.leave_out_last,
                max_exp=args.max_exp,
            )

    elif mode == "both":
        # No data generation in 'both' mode: we reuse existing single/double CSVs.
        fig_path = args.output if args.output is not None else base_fig_both
        out_dir = os.path.dirname(os.path.abspath(fig_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if not args.no_summary_plot:
            plot_summary_both(
                base_csv_single,
                base_csv_double,
                fig_path,
                leave_out_last=args.leave_out_last,
                max_exp=args.max_exp,
            )


if __name__ == "__main__":
    main()
