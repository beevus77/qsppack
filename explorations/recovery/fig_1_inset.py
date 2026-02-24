#!/usr/bin/env python3
"""
Generate and plot Fig. 1 inset: L-infinity error on [0,a] vs npts for 10 points
from 2^7 to 2^8 (log-uniform). Data generation and plotting in one script;
no CSV. Output: explorations/recovery/figures/fig_1_inset.pdf
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.polynomial import chebyshev as cheb

from qsppack.utils import cvx_poly_coef


def max_linf_error_on_fit_domain(
    coef_full: np.ndarray, a: float, epsil: float
) -> float:
    """Max_{x in [0,a]} |f(x) - (1-epsil)*x/a| via critical points."""
    coef_full = np.asarray(coef_full, dtype=float)
    dcoef = cheb.chebder(coef_full)
    dcoef_h = np.copy(dcoef)
    if dcoef_h.size > 0:
        dcoef_h[0] -= (1.0 - epsil) / a
    roots = cheb.chebroots(dcoef_h)
    real_mask = np.isclose(roots.imag, 0.0, atol=1e-12)
    real_roots = roots[real_mask].real
    interior = real_roots[(real_roots > 0.0) & (real_roots < a)]
    candidates = (
        np.concatenate(([0.0, a], interior)) if interior.size > 0 else np.array([0.0, a])
    )
    f_vals = cheb.chebval(candidates, coef_full)
    g_vals = (1.0 - epsil) * candidates / a
    return float(np.max(np.abs(f_vals - g_vals)))


def main() -> None:
    deg = 101
    a = 0.2
    epsil = 0.0
    targ = lambda x: (1.0 - epsil) * x / a

    # 10 points from 2^7 to 2^8, approximately log-uniform, rounded to integer
    npts_arr = np.round(2 ** np.linspace(7, 8, 10)).astype(int)
    npts_arr = np.unique(npts_arr)  # avoid duplicate npts after rounding

    linf_errors = []
    for npts in npts_arr:
        opts = {
            "intervals": [0, a],
            "objnorm": np.inf,
            "epsil": epsil,
            "npts": int(npts),
            "isplot": False,
            "fscale": 1,
            "method": "cvxpy",
        }
        coef_full = cvx_poly_coef(targ, deg, opts)
        linf_errors.append(max_linf_error_on_fit_domain(coef_full, a, epsil))
    linf_errors = np.array(linf_errors)

    # Clamp for log scale
    linf_clamped = np.where(linf_errors > 0.0, linf_errors, 1e-16)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(
        npts_arr, linf_clamped, marker="o", s=50, facecolors="none",
        edgecolors="#0072B2", linewidths=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Two major x-ticks at 1.5e2 and 2.5e2, exponential style; no labels on minor ticks
    ax.xaxis.set_major_locator(mticker.FixedLocator([150.0, 250.0]))
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(mticker.LogLocator(subs=[2.0, 5.0]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.grid(True, which="both", alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "fig_1_inset.pdf")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")


if __name__ == "__main__":
    main()
