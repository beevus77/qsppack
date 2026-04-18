#!/usr/bin/env python3
"""
Plot degree-scaling summary for matrix inversion and uniform singular value
amplification (data produced by degree_scaling_data.py).

Reads CSV with degree, max_error_poly, max_error_qsp, constraint_violated, etc.,
and produces a log-log plot of polynomial degree vs maximum error vs target,
with the same style as degree_scaling_thresh_proj (polynomial: x if constraints
violated, o if satisfied; retraction: o).

With ``--ploterror --errordegree D --problem-type ...``, also writes a two-panel
figure: the usual summary plus pointwise $|q-f|$ and $|p/s-f|$ on the problem domain
($s=\max_{[-1,1]}|p|$ from Chebyshev samples). Pass ``--a`` / ``--epsilon`` to match
the values used when generating the CSV (defaults 0.2 and 0.0).
"""

import argparse
import json
import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.fft import dct

BLUE = "#0072B2"
MAIZE = "#E69F00"
GREEN = "#009E73"


def _mathtext_log_tick_label(x: float, pos=None) -> str:
    """Format positive x as m×10^e for log-scale axis ticks (FixedLocator positions)."""
    if x <= 0.0 or not np.isfinite(x):
        return ""
    exp = int(np.floor(np.log10(x) + 1e-12))
    m = x * 10 ** (-exp)
    if abs(m - 1.0) < 1e-9:
        return rf"$10^{{{exp}}}$"
    m_int = int(round(m))
    if abs(m - m_int) < 1e-5 and m_int != 0:
        return rf"${m_int}\times 10^{{{exp}}}$"
    return rf"${m:g}\times 10^{{{exp}}}$"

# Marker sizes for 2-norm polynomial-space plots (BLUE = fit, MAIZE = retraction)
MARKERSIZE_BLUE_X = 10  # 'x' when constraints violated
MARKERSIZE_BLUE_O = 8  # 'o' when constraints satisfied
MARKERSIZE_MAIZE_O = 12  # retraction (open circles)

M_CHEB_CONSTRAINT = 1_000_000


def chebval_dct(c: np.ndarray, M: int) -> np.ndarray:
    """
    Evaluate Chebyshev series with full coefficients c at M Chebyshev nodes.

    Nodes: x_j = cos(j*pi/(M-1)), j=0..M-1.
    Implementation: zero-pad coefficients to length M and apply DCT-I.
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


def _parse_constraint_violated(val) -> Optional[bool]:
    """Parse constraint_violated from CSV (bool or string 'True'/'False'). Returns None if missing/NaN."""
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


def _max_abs_poly_on_interval_minus1_1(coef_full: np.ndarray) -> float:
    """Max |p(x)| on [-1, 1] via Chebyshev nodes (same spirit as constraint check)."""
    M_cheb = M_CHEB_CONSTRAINT
    try:
        vals = chebval_dct(coef_full, M_cheb)
    except MemoryError:
        vals = chebval_dct(coef_full, 100_000)
    return float(np.max(np.abs(vals)))


def _load_degree_scaling_dataframe(
    csv_path: str,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
) -> pd.DataFrame:
    """Read CSV and return processed rows for the degree-vs-error summary plot."""
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


def _plot_degree_scaling_on_ax(
    ax: Axes,
    data: pd.DataFrame,
    refinescale: bool = False,
) -> None:
    """Draw the log-log degree vs max error summary on an existing Axes."""
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
    if refinescale:
        y_all = np.concatenate([err_poly_clamped, err_qsp_clamped])
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        span_log = max(np.log10(y_max / y_min), 1e-15)
        pad = 0.03 + 0.06 * min(span_log, 1.0)
        y_lo = 10.0 ** (np.log10(y_min) - pad)
        y_hi = 10.0 ** (np.log10(y_max) + pad)
        ax.set_ylim(y_lo, y_hi)

        exp_lo = int(np.floor(np.log10(y_lo) - 1e-12))
        exp_hi = int(np.ceil(np.log10(y_hi) + 1e-12))
        tickvals: list[float] = []
        for e in range(exp_lo, exp_hi + 1):
            base = 10.0**e
            for m in range(1, 10):
                v = m * base
                if y_lo <= v <= y_hi:
                    tickvals.append(v)
        if len(tickvals) < 3:
            tickvals = list(
                10.0 ** np.linspace(np.log10(y_lo), np.log10(y_hi), max(5, len(tickvals) + 3))
            )
        ax.yaxis.set_major_locator(FixedLocator(np.asarray(tickvals)))
        ax.yaxis.set_major_formatter(FuncFormatter(_mathtext_log_tick_label))
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
            markersize=MARKERSIZE_BLUE_X,
            markeredgewidth=2,
            label="Polynomial max error (constraints violated)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=BLUE,
            linestyle="None",
            markersize=MARKERSIZE_BLUE_O,
            fillstyle="none",
            label="Polynomial max error (constraints satisfied)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=MAIZE,
            linestyle="None",
            markersize=MARKERSIZE_MAIZE_O,
            fillstyle="none",
            label="Retracted polynomial max error",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")


def plot_pointwise_retraction_and_scaled_poly_errors_on_ax(
    ax: Axes,
    row: Any,
    target_fn: Callable[[np.ndarray], np.ndarray],
    x_lo: float,
    x_hi: float,
    n_x: int = 4000,
) -> None:
    """
    Log-linear plot of |q-f|, |p/s-f|, and |p-f| on [x_lo, x_hi] (problem domain).

    Here ``p`` is the unretracted fit from the CSV ``coef`` column (parity-stripped
    Chebyshev coefficients); ``coef_full`` is used only to form
    ``s = max_{[-1,1]} |p|`` on Chebyshev nodes. When ``s`` is essentially 1 and the
    retraction satisfies ``q`` approximately ``p``, the scaled curve ``p/s`` is almost the same
    as ``p``, so its error track can sit on top of the retracted error track.
    """
    from qsppack.utils import chebyshev_to_func, get_entry

    coef = np.array(json.loads(row["coef"]), dtype=float)
    coef_full = np.array(json.loads(row["coef_full"]), dtype=float)
    phi_proc = np.array(json.loads(row["phi_proc"]), dtype=float)
    parity = int(row["parity"])

    out_qsp = {"targetPre": True, "parity": parity, "typePhi": "full"}

    x = np.linspace(float(x_lo), float(x_hi), n_x)
    targ = target_fn(x)
    p = chebyshev_to_func(x, coef, parity, True)
    q = get_entry(x, phi_proc, out_qsp)

    s = _max_abs_poly_on_interval_minus1_1(coef_full)
    if s <= 0.0:
        raise ValueError("max |p| on [-1, 1] is zero; cannot scale unretracted polynomial.")

    floor = 1e-20
    abs_err_qsp = np.maximum(np.abs(q - targ), floor)
    abs_err_scaled_poly = np.maximum(np.abs(p / s - targ), floor)
    abs_err_fit_poly = np.maximum(np.abs(p - targ), floor)

    ax.plot(x, abs_err_qsp, color=MAIZE, linewidth=1.2, label="Retracted $|q-f|$")
    ax.plot(x, abs_err_scaled_poly, color=BLUE, linewidth=1.2, label="Scaled fit $|p/s-f|$")
    ax.plot(
        x,
        abs_err_fit_poly,
        color=GREEN,
        linewidth=1.2,
        linestyle="--",
        label="Unscaled fit $|p-f|$",
    )

    max_err_qsp = float(np.max(np.abs(q - targ)))
    max_err_scaled_poly = float(np.max(np.abs(p / s - targ)))
    max_err_fit_poly = float(np.max(np.abs(p - targ)))
    max_qp = float(np.max(np.abs(q - p)))
    max_p_ps = float(np.max(np.abs(p / s - p)))

    ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel("Absolute error")
    ax.set_xlim([float(x_lo), float(x_hi)])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    deg = int(row["degree"])
    csv_q = float(row["max_error_qsp"])
    ax.set_title(
        f"Pointwise errors (degree {deg})\n"
        f"max |q-f|={max_err_qsp:.2e} (CSV {csv_q:.2e}), "
        f"max |p/s-f|={max_err_scaled_poly:.2e}, max |p-f|={max_err_fit_poly:.2e}\n"
        f"s={s:.10f}, max|q-p|={max_qp:.2e}, max|p/s-p|={max_p_ps:.2e}"
    )


def plot_summary_and_pointwise_errors(
    csv_path: str,
    fig_path: str,
    errordegree: int,
    problem_type: str,
    a: float,
    epsil: float,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
    refinescale: bool = False,
) -> None:
    """Left: degree scaling summary; right: pointwise errors for one degree."""
    from degree_scaling_data import build_target_and_domain

    raw = pd.read_csv(csv_path)
    if "degree" not in raw.columns:
        raise ValueError("CSV must contain 'degree' column.")
    degrees_in_file = set(int(d) for d in raw["degree"].tolist())
    if int(errordegree) not in degrees_in_file:
        raise ValueError(
            f"errordegree={errordegree} is not among degrees in {csv_path}: "
            f"{sorted(degrees_in_file)}"
        )

    target_fn, _intervals, x_lo, x_hi = build_target_and_domain(problem_type, a, epsil)
    data = _load_degree_scaling_dataframe(csv_path, leave_out_last, max_exp)
    matches = raw.loc[raw["degree"].astype(int) == int(errordegree)]
    if matches.empty:
        raise ValueError(f"No CSV row found for degree {errordegree}.")
    row = matches.iloc[0]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_degree_scaling_on_ax(ax0, data, refinescale=refinescale)
    ax0.set_title("Max error vs degree")
    plot_pointwise_retraction_and_scaled_poly_errors_on_ax(
        ax1, row, target_fn, x_lo, x_hi
    )

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


def plot_summary(
    csv_path: str,
    fig_path: str,
    leave_out_last: bool = False,
    max_exp: Optional[int] = None,
    refinescale: bool = False,
) -> None:
    """Plot degree (log) vs max error (log) for fit and QSP.

    Polynomial: x if constraint violated, o (larger) if satisfied. Retraction: o (larger).
    Does not write to CSV. If max_exp is set (e.g. 8), only plot degrees with exp2 <= max_exp.
    If refinescale is True, tighten y-limits and place explicit log-spaced y ticks (1–9×10^k
    in view) so narrow error ranges get multiple labeled ticks.
    """
    data = _load_degree_scaling_dataframe(csv_path, leave_out_last, max_exp)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_degree_scaling_on_ax(ax, data, refinescale=refinescale)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot degree-scaling summary (degree vs max error) from CSV "
            "produced by degree_scaling_data.py."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (e.g. data/degree_scaling_mat_inv_npts8.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: figures/degree_scaling_<base>.pdf from CSV dir).",
    )
    parser.add_argument(
        "--leave-out-last",
        action="store_true",
        help="Omit the largest degree from the plot.",
    )
    parser.add_argument(
        "--max-exp",
        type=int,
        default=None,
        metavar="K",
        help="Only include degrees with exp2 <= K (e.g. --max-exp 8 for degree <= 256).",
    )
    parser.add_argument(
        "--refinescale",
        action="store_true",
        help=(
            "Explicit log-spaced y tick positions (1–9×10^k in view) with readable "
            "m×10^e labels and padded y-limits; default plot behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--ploterror",
        action="store_true",
        help=(
            "Write a two-panel PDF (summary + pointwise $|q-f|$ and $|p/s-f|$ on the "
            "problem domain). Requires --errordegree and --problem-type; set --a and "
            "--epsilon to match degree_scaling_data.py."
        ),
    )
    parser.add_argument(
        "--errordegree",
        type=int,
        default=None,
        metavar="D",
        help="Degree for the pointwise error panel (must appear in the CSV).",
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["mat_inv", "uniform_sv_amp"],
        default=None,
        help="Which target was used to build the CSV (required with --ploterror).",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.2,
        help="Parameter a; must match degree_scaling_data.py (default 0.2).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon; must match degree_scaling_data.py --epsilon (default 0).",
    )
    args = parser.parse_args()

    if args.ploterror:
        if args.errordegree is None:
            raise ValueError("--ploterror requires --errordegree.")
        if args.problem_type is None:
            raise ValueError("--ploterror requires --problem-type (mat_inv or uniform_sv_amp).")
        if not (0.0 < args.a < 1.0):
            raise ValueError("--a must lie in (0, 1).")

    csv_path = args.csv
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, "data", os.path.basename(csv_path))
        if os.path.exists(alt):
            csv_path = alt

    if args.output is not None:
        fig_path = args.output
        out_dir = os.path.dirname(os.path.abspath(fig_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        figures_dir = os.path.join(script_dir, "figures")
        base = os.path.splitext(os.path.basename(csv_path))[0]
        fig_path = os.path.join(figures_dir, f"{base}.pdf")
        os.makedirs(figures_dir, exist_ok=True)

    if args.ploterror:
        plot_summary_and_pointwise_errors(
            csv_path,
            fig_path,
            args.errordegree,
            problem_type=args.problem_type,
            a=args.a,
            epsil=args.epsilon,
            leave_out_last=args.leave_out_last,
            max_exp=args.max_exp,
            refinescale=args.refinescale,
        )
    else:
        plot_summary(
            csv_path,
            fig_path,
            leave_out_last=args.leave_out_last,
            max_exp=args.max_exp,
            refinescale=args.refinescale,
        )


if __name__ == "__main__":
    main()
