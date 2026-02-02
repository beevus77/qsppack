"""
Convex polynomial approximation examples for QSP: linear system and gaussian state.
Run with: python cvx_poly_ex.py linear_system | gaussian_state \
    [--xlim XMIN XMAX] [--ylim YMIN YMAX] \
    [--xlim-err XMIN XMAX] [--ylim-err YMIN YMAX]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from qsppack.utils import cvx_poly_coef, chebyshev_to_func


def run_example(
    targ,
    deg,
    opts,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
    xlim_err=(-1.1, 1.1),
    ylim_err=(-1.1, 1.1),
):
    """Run the common workflow: solve, evaluate, and plot for a given target and options."""
    parity = deg % 2

    # coef_full = cvx_poly_coef(targ, deg, opts)
    coef_full = cvx_poly_coef(targ, deg, opts)
    coef = coef_full[parity::2]

    xlist = np.linspace(-1, 1, 10000)
    func = lambda x: chebyshev_to_func(x, coef, parity, True)
    targ_value = targ(xlist)
    func_value = func(xlist)

    # Plot the difference
    diff_value = targ_value - func_value
    plt.plot(xlist, diff_value, label="$g - f$")
    plt.xlabel("x")
    plt.xlim(xlim_err)
    plt.ylim(ylim_err)
    plt.grid(True, alpha=0.3)
    # plt.yscale("log")
    plt.show()

    # Plot the target and approximation
    plt.plot(xlist, targ_value, label="$g$")
    plt.plot(xlist, func_value, label="$f$")
    plt.xlabel("x")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.show()


def linear_system(xlim, ylim, xlim_err, ylim_err):
    """Linear system example: targ(x) = 1/(kappa*x)."""
    kappa = 10
    targ = lambda x: 1 / (kappa * x)
    deg = 71

    opts = {
        "intervals": [1 / kappa, 1],
        "objnorm": np.inf,
        "epsil": 0.01,
        "npts": 200,
        "fscale": 1,
        "isplot": False,
        "method": "cvxpy",
        "constraint_intervals": [-1, 0],
    }
    run_example(
        targ,
        deg,
        opts,
        xlim=xlim,
        ylim=ylim,
        xlim_err=xlim_err,
        ylim_err=ylim_err,
    )


def gaussian_state(xlim, ylim, xlim_err, ylim_err):
    """Gaussian state example: targ(x) = exp(-beta/2 * arcsin(x)^2)."""
    beta = 100
    targ = lambda x: np.exp(-beta / 2 * np.arcsin(x) ** 2)
    deg = 100

    opts = {
        "intervals": [0, np.sin(1)],
        "objnorm": np.inf,
        "epsil": 0,
        "npts": 100,
        "fscale": 1,
        "isplot": False,
        "method": "cvxpy",
    }
    run_example(
        targ,
        deg,
        opts,
        xlim=xlim,
        ylim=ylim,
        xlim_err=xlim_err,
        ylim_err=ylim_err,
    )


EXAMPLES = {
    "linear_system": linear_system,
    "gaussian_state": gaussian_state,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run convex polynomial approximation examples for QSP."
    )
    parser.add_argument(
        "example",
        choices=list(EXAMPLES.keys()),
        help="Which example to run: linear_system or gaussian_state",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        help="x-axis limits for target/approximation plot",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        help="y-axis limits for target/approximation plot",
    )
    parser.add_argument(
        "--xlim-err",
        nargs=2,
        type=float,
        default=None,
        help="x-axis limits for error plot",
    )
    parser.add_argument(
        "--ylim-err",
        nargs=2,
        type=float,
        default=None,
        help="y-axis limits for error plot",
    )
    args = parser.parse_args()
    if args.example == "linear_system":
        xlim = args.xlim if args.xlim is not None else [-1, 0]
        ylim = args.ylim if args.ylim is not None else [-1.1, 1.1]
        xlim_err = args.xlim_err if args.xlim_err is not None else [0.1, 1]
        ylim_err = args.ylim_err if args.ylim_err is not None else [-0.02, 0.02]
    elif args.example == "gaussian_state":
        xlim = args.xlim if args.xlim is not None else [-1, 0]
        ylim = args.ylim if args.ylim is not None else [-0.2, 1.1]
        xlim_err = args.xlim_err if args.xlim_err is not None else [0, np.sin(1)]
        ylim_err = args.ylim_err if args.ylim_err is not None else [-1.1e-8, 1.1e-8]
    else:
        xlim = args.xlim if args.xlim is not None else [-1.1, 1.1]
        ylim = args.ylim if args.ylim is not None else [-1.1, 1.1]
        xlim_err = args.xlim_err if args.xlim_err is not None else [-1.1, 1.1]
        ylim_err = args.ylim_err if args.ylim_err is not None else [-1.1, 1.1]

    EXAMPLES[args.example](xlim, ylim, xlim_err, ylim_err)


if __name__ == "__main__":
    main()
