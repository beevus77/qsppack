#!/usr/bin/env python3
"""
Single degree/npts plot for the threshold projection problem.

Generates one figure: target function, fitting polynomial, and retracted (QSP) polynomial.
No error analysis, no degree sweep. Usage:

  python plot_thresh_proj_ex.py --degree 32 --npts 32 [--output fig.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from qsppack.solver import solve
from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry

BLUE = "#0072B2"
MAIZE = "#E69F00"
DELTA = 0.05
N_WEISS = 2**14
N_XPLOT = 3000


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

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.figure()
    plt.plot(xlist, targ_value, "k", label="Target")
    plt.plot(xlist, func_value, color=BLUE, label="Polynomial approximation")
    plt.plot(xlist, qsp_value, "--", color=MAIZE, label="Retraction")
    plt.xlabel(r"$x$", fontsize=12)
    plt.grid()
    if not args.hide_legend:
        plt.legend(loc="best", fontsize=18)
    plt.ylim([-0.1, 1.1])
    plt.xlim([0, 1])
    plt.tight_layout()

    if args.output:
        out_path = os.path.abspath(args.output)
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Figure saved to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
