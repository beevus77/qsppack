#!/usr/bin/env python3
"""
Single degree/npts plot for the threshold projection problem.

Generates one figure: target function, fitting polynomial, and retracted (QSP) polynomial.
No error analysis, no degree sweep. Usage:

  python plot_thresh_proj_ex.py --degree 32 --npts 32 [--output fig.pdf]
  python plot_thresh_proj_ex.py --degree 128 --npts 256 --zoom-inset [--output fig.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from qsppack.solver import solve
from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry

BLUE = "#0072B2"
MAIZE = "#E69F00"
DELTA = 0.05
N_WEISS = 2**14
N_XPLOT = 3000

# Match Fig 8 / mat_inv_ex.py for axis label; legend slightly smaller for this figure.
AXIS_LABEL_FONTSIZE = 26
LEGEND_FONTSIZE = 11

# Inset geometry in parent axes coordinates (fractions of main axes width / height).
INSET_WIDTH_FRAC = 0.35   # with anchor 0.08: ~0.08 to 0.43 horizontally
INSET_HEIGHT_FRAC = 0.65  # ~0.10 to 0.75 vertically (tall inset so 1±1e-5 is readable)
INSET_ANCHOR_X = 0.08
INSET_ANCHOR_Y = 0.10
INSET_TICK_FONTSIZE = 11


def _apply_fig8_style() -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 24
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["legend.fontsize"] = LEGEND_FONTSIZE
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24


def target(x: np.ndarray) -> np.ndarray:
    """Indicator of |x| < 0.5, vectorized."""
    return np.where(np.abs(x) < 0.5, 1.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot target, fitting polynomial, and retracted polynomial for one (degree, npts)."
    )
    parser.add_argument(
        "--degree",
        type=int,
        required=True,
        metavar="D",
        help="Polynomial degree (even integer).",
    )
    parser.add_argument(
        "--npts",
        type=int,
        required=True,
        metavar="N",
        help="Number of convex-optimization points.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save figure to PATH (default: show interactively).",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Do not show the legend.",
    )
    parser.add_argument(
        "--zoom-inset",
        action="store_true",
        help=(
            "Draw a zoomed inset (inset_axes + mark_inset) on the left with a tall "
            "aspect ratio so the 1±1e-5 band is readable; default x/y limits match the "
            "former manual PowerPoint crop."
        ),
    )
    parser.add_argument(
        "--inset-xmin",
        type=float,
        default=0.2,
        help="Inset x-axis lower limit (default: 0.2).",
    )
    parser.add_argument(
        "--inset-xmax",
        type=float,
        default=0.45,
        help="Inset x-axis upper limit (default: 0.45).",
    )
    parser.add_argument(
        "--inset-ymin",
        type=float,
        default=1.0 - 1e-5,
        help="Inset y-axis lower limit (default: 1 - 1e-5).",
    )
    parser.add_argument(
        "--inset-ymax",
        type=float,
        default=1.0 + 1.1e-5,
        help="Inset y-axis upper limit (default: 1 + 1.1e-5, slightly above 1 + 1e-5).",
    )
    args = parser.parse_args()

    deg = args.degree
    npts = args.npts
    parity = deg % 2

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

    coef_full = cvx_poly_coef(target, deg, opts_fit)
    coef = coef_full[parity::2]

    opts_qsp = dict(opts_fit)
    opts_qsp.update(
        {
            "N": int(N_WEISS),
            "method": "NLFT",
            "targetPre": False,
            "typePhi": "reduced",
        }
    )
    phi_proc, out = solve(coef, parity, opts_qsp)
    out["typePhi"] = "full"

    xlist = np.linspace(0.0, 1.0, N_XPLOT)
    func = lambda x: chebyshev_to_func(x, coef, parity, True)
    targ_value = target(xlist)
    func_value = func(xlist)
    qsp_value = get_entry(xlist, phi_proc, out)

    _apply_fig8_style()
    fig, ax = plt.subplots()

    ax.plot(xlist, targ_value, "k", label="Target")
    ax.plot(xlist, func_value, color=BLUE, label="Polynomial approximation")
    ax.plot(xlist, qsp_value, "--", color=MAIZE, label="Retraction")
    ax.set_xlabel(r"$x$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid()
    if not args.hide_legend:
        ax.legend(loc="upper right", framealpha=1, fontsize=LEGEND_FONTSIZE)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([0, 1])

    if args.zoom_inset:
        # Tall, narrow inset in parent axes coordinates: lower-left (0.08, 0.10),
        # width 0.35 (~0.08–0.43 horizontally), height 0.65 (~0.10–0.75 vertically)
        # so the 1±1e-5 band uses most of the inset height.
        bbox = (INSET_ANCHOR_X, INSET_ANCHOR_Y, INSET_WIDTH_FRAC, INSET_HEIGHT_FRAC)
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=bbox,
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        axins.plot(xlist, targ_value, "k", linewidth=0.9)
        axins.plot(xlist, func_value, color=BLUE, linewidth=0.9)
        axins.plot(xlist, qsp_value, "--", color=MAIZE, linewidth=0.9)
        axins.set_xlim(args.inset_xmin, args.inset_xmax)
        axins.set_ylim(args.inset_ymin, args.inset_ymax)
        axins.set_aspect("auto")
        axins.grid(True, alpha=0.6)
        axins.tick_params(axis="both", labelsize=INSET_TICK_FONTSIZE)
        # Offset / multiplier text (e.g. ×10⁻⁵) otherwise inherits rcParams font.size (main ticks).
        axins.xaxis.offsetText.set_fontsize(INSET_TICK_FONTSIZE)
        axins.yaxis.offsetText.set_fontsize(INSET_TICK_FONTSIZE)
        axins.set_facecolor("white")
        axins.patch.set_edgecolor("0.35")
        axins.patch.set_linewidth(0.8)
        # Connect inset to the zoomed rectangle on the main axes.
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", linewidth=0.8)

    if args.zoom_inset:
        # tight_layout clashes with inset_axes; savefig(..., bbox_inches="tight") still trims.
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)
    else:
        fig.tight_layout()

    if args.output:
        out_path = os.path.abspath(args.output)
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Figure saved to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
