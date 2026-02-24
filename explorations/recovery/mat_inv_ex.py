"""
Matrix inversion polynomials and plotting: optimal polynomial (Sünderhauf et al.,
arXiv:2507.15537), retraction via Weiss + nonlinear FFT, and windowed construction
using windowing functions from PRX Quantum 2, 040203 (2021) / arXiv:2105.02859v5,
Appendix C (Construction of the Matrix-Inversion Polynomial).
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.special import erf
from numpy.polynomial import chebyshev as cheb

from qsppack.nlfa import b_from_cheb, weiss, inverse_nonlinear_FFT, forward_nonlinear_FFT
from qsppack.utils import cvx_poly_coef


# Match LaTeX/plotting style used elsewhere in this directory
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


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


def alpha(a: float) -> float:
    """α(a) from Eq. (9) in Sünderhauf et al."""
    if not (0.0 < a < 1.0):
        raise ValueError("Parameter 'a' must lie in (0, 1).")
    return (1.0 + a) / (2.0 * (1.0 - a))


def L_n_recurrence(x: np.ndarray, a: float, n: int) -> np.ndarray:
    """
    Compute 𝓛_n(x; a) via the stable recurrence (Eqs. (9), (11), (12)).

    This is the scaled polynomial 𝓛_n defined in Appendix A:
        𝓛_n(x; a) = L_n(x; a) / α(a)^n
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    x = np.asarray(x, dtype=float)
    alpha_val = alpha(a)

    # Initial values (Eq. (12))
    if n == 1:
        return (x + (1.0 - a) / (1.0 + a)) / alpha_val

    L1 = (x + (1.0 - a) / (1.0 + a)) / alpha_val

    if n == 2:
        return (x**2 + (1.0 - a) / (2.0 * (1.0 + a)) * x - 0.5) / (alpha_val**2)

    L2 = (x**2 + (1.0 - a) / (2.0 * (1.0 + a)) * x - 0.5) / (alpha_val**2)

    # Recurrence (Eq. (11))
    L_nm2 = L1  # 𝓛_{n-1} placeholder during iteration
    L_nm1 = L2  # 𝓛_n placeholder during iteration
    for k in range(3, n + 1):
        L_k = (x * L_nm1) / alpha_val - L_nm2 / (4.0 * alpha_val**2)
        L_nm2, L_nm1 = L_nm1, L_k
    return L_nm1


def optimal_poly_eval(x: np.ndarray, a: float, n: int) -> np.ndarray:
    """
    Evaluate the optimal matrix inversion polynomial P_{2n-1}(x; a).

    Uses the numerically stable expression (Eq. (10)) in the paper:
        P_{2n-1}(x; a) = (1 - (-1)^n (1+a)^2/(4a) * 𝓛_n(z; a)) / x
    where
        z = (2 x^2 - (1+a^2)) / (1-a^2).
    We define P(0) = 0 by continuity/symmetry (odd polynomial).
    """
    x = np.asarray(x, dtype=float)
    d = 2 * n - 1
    if d <= 0 or d % 2 != 1:
        raise ValueError("Degree must be an odd positive integer (2n-1 with n>=1).")

    # Map to z as in Eq. (10)
    z = (2.0 * x**2 - (1.0 + a**2)) / (1.0 - a**2)

    L_n_vals = L_n_recurrence(z, a, n)
    prefactor = (1.0 + a) ** 2 / (4.0 * a)

    numerator = 1.0 - ((-1) ** n) * prefactor * L_n_vals

    # Handle x=0 explicitly (odd polynomial => P(0) = 0)
    P = np.empty_like(x, dtype=float)
    nonzero_mask = x != 0.0
    P[nonzero_mask] = numerator[nonzero_mask] / x[nonzero_mask]
    P[~nonzero_mask] = 0.0
    return P


def optimal_poly_cheb_coeffs(a: float, n: int) -> np.ndarray:
    """
    Compute Chebyshev coefficients of P_{2n-1}(x; a) on [-1, 1].

    Uses exact Chebyshev interpolation at d+1 Chebyshev-Lobatto nodes:
        x_j = cos(j*pi/d), j=0..d, d = 2n-1.
    """
    d = 2 * n - 1
    # Chebyshev-Lobatto nodes
    j = np.arange(d + 1)
    x_nodes = np.cos(np.pi * j / d)
    y_vals = optimal_poly_eval(x_nodes, a, n)

    # Fit polynomial in Chebyshev basis (ascending order c_0..c_d)
    coefs = cheb.chebfit(x_nodes, y_vals, deg=d)
    # Scale polynomial by a
    coefs *= a
    return coefs


# ---------------------------------------------------------------------------
# P_Θ: step/sign function approximation (PRX Quantum Appendix C, ref [21]).
# [21] G. H. Low, Quantum signal processing by single-qubit dynamics, Ph.D. thesis, MIT (2017).
# P_Θ_{ε,Δ}(x) is an odd polynomial that ε-approximates erf(k*x) on [-1,1], where
#   k = sqrt(2)/Δ * sqrt(log(2/(π ε²))).
# So it approximates the sign function with transition width ~ Δ. We construct it
# by Chebyshev-fitting erf(k*x) and zeroing even coefficients (enforcing odd parity).
# ---------------------------------------------------------------------------


def p_theta_k(eps: float, Delta: float) -> float:
    """Scaling factor k such that P_Θ approximates erf(k*x). PRX Quantum Eq. after (C4)."""
    if eps <= 0 or eps >= 1 or Delta <= 0:
        raise ValueError("eps must be in (0,1), Delta > 0")
    return np.sqrt(2.0) / Delta * np.sqrt(np.log(2.0 / (np.pi * eps**2)))


def p_theta_cheb_coeffs(
    eps: float,
    Delta: float,
    degree: int,
) -> np.ndarray:
    """
    Chebyshev coefficients (ascending) of P_Θ_{ε,Δ}(x), odd polynomial approximating
    erf(k*x) on [-1, 1] with k = p_theta_k(eps, Delta). Degree should be odd.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    degree = degree | 1  # ensure odd
    k = p_theta_k(eps, Delta)
    j = np.arange(degree + 1)
    x_nodes = np.cos(np.pi * j / degree) if degree > 0 else np.array([1.0])
    # Target: erf(k*x) (odd function)
    y_vals = erf(k * x_nodes)
    coefs = cheb.chebfit(x_nodes, y_vals, deg=degree)
    # Enforce odd: P_Θ has odd parity => only T_1, T_3, ...
    coefs[0::2] = 0.0
    return coefs


def p_theta_eval(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Evaluate P_Θ(x) = sum_k c_k T_k(x) at x."""
    return cheb.chebval(x, coefs)


# ---------------------------------------------------------------------------
# Windowing functions (PRX Quantum 2, 040203 (2021) / arXiv:2105.02859v5, Appendix C).
# Option A (smoothstep): W(x) = 0 for |x|<=a-delta, smoothstep on [a-delta,a], 1 for |x|>=a.
# Option B (P_Θ): W(x) = (1 + P_Θ(|x| - (a - delta/2))) / 2 with Delta = delta, so step
#   centered at a - delta/2, transition width ~ delta; P_Θ from ref [21] above.
# We approximate W by a Chebyshev polynomial of degree window_deg (default 2n),
# then zero odd coefficients so the window polynomial is even.
# The windowed curve is P_opt(x) * W(x), full degree 4n-1.
# ---------------------------------------------------------------------------


def window_eval(
    x: np.ndarray,
    a: float,
    delta: float = 0.01,
    use_p_theta: bool = False,
    p_theta_coefs: np.ndarray | None = None,
    p_theta_eps: float = 1e-6,
) -> np.ndarray:
    """
    Evaluate the window function W(x) on [-1, 1].

    Even function: transition from 0 to 1 on [a-delta, a].
    If use_p_theta is False (default): smoothstep
      W(x) = 0 for |x| <= a - delta, cubic smoothstep on [a-delta, a], 1 for |x| >= a.
    If use_p_theta is True: W(x) = (1 + P_Θ(z_norm)) / 2 with z_norm mapping
      [a-delta, a] to [-1,1]; P_Θ from ref [21] (ε-approx to erf). Requires
      p_theta_coefs or we build with p_theta_eps and Delta=delta.
    """
    x = np.asarray(x, dtype=float)
    u = np.abs(x)
    if not use_p_theta:
        out = np.zeros_like(x)
        x_low = a - delta
        if delta <= 0:
            out[u >= a] = 1.0
            return out
        mask_rise = (u > x_low) & (u < a)
        mask_one = u >= a
        t = (u[mask_rise] - x_low) / delta
        t = np.clip(t, 0.0, 1.0)
        out[mask_rise] = 3.0 * t * t - 2.0 * t * t * t
        out[mask_one] = 1.0
        return out
    # P_Θ-based window: center at a - delta/2, map [a-delta, a] -> [-1,1] for P_Θ
    z = u - (a - delta / 2.0)
    z_norm = (2.0 / delta) * z
    z_norm = np.clip(z_norm, -1.0, 1.0)
    if p_theta_coefs is None:
        deg_pt = max(3, int(round(2.0 * (1.0 / max(delta, 1e-10)) * max(1.0, np.log(1.0 / max(p_theta_eps, 1e-12))))))
        deg_pt = deg_pt | 1
        p_theta_coefs = p_theta_cheb_coeffs(p_theta_eps, delta, deg_pt)
    w_vals = (1.0 + p_theta_eval(z_norm, p_theta_coefs)) / 2.0
    return w_vals


def window_cheb_coeffs(
    a: float,
    degree_w: int,
    delta: float = 0.01,
    use_p_theta: bool = False,
    p_theta_eps: float = 1e-6,
) -> np.ndarray:
    """
    Chebyshev coefficients (ascending order) of the window W(x) on [-1, 1].
    W is even, so we fit at Chebyshev nodes then zero odd-index coefficients.
    degree_w: approximant degree (e.g. 2n). delta: transition on [a-delta, a].
    If use_p_theta is True, W is built from P_Θ (ref [21]) instead of smoothstep.
    """
    if degree_w < 0:
        raise ValueError("degree_w must be non-negative")
    if use_p_theta:
        deg_pt = max(3, int(round(2.0 * (1.0 / max(delta, 1e-10)) * max(1.0, np.log(1.0 / max(p_theta_eps, 1e-12))))))
        deg_pt = deg_pt | 1
        p_theta_coefs = p_theta_cheb_coeffs(p_theta_eps, delta, deg_pt)
        j = np.arange(degree_w + 1)
        x_nodes = np.cos(np.pi * j / degree_w) if degree_w > 0 else np.array([1.0])
        y_vals = window_eval(x_nodes, a, delta, use_p_theta=True, p_theta_coefs=p_theta_coefs)
    else:
        j = np.arange(degree_w + 1)
        x_nodes = np.cos(np.pi * j / degree_w) if degree_w > 0 else np.array([1.0])
        y_vals = window_eval(x_nodes, a, delta)
    coefs = cheb.chebfit(x_nodes, y_vals, deg=degree_w)
    coefs[1::2] = 0.0
    return coefs


def windowed_poly_cheb_coeffs(
    a: float,
    n: int,
    window_deg: int | None = None,
    delta: float = 0.01,
    use_p_theta: bool = False,
    p_theta_eps: float = 1e-6,
) -> np.ndarray:
    """
    Chebyshev coefficients of the windowed optimal polynomial: P_opt(x) * W(x).

    We use the optimal polynomial P_opt from Sünderhauf et al. (arxiv:2507.15537),
    scaled by a, and multiply by the window W. The product is returned at full
    degree 4n - 1 (no truncation). W can be smoothstep (default) or P_Θ-based (ref [21]).
    """
    if window_deg is None:
        window_deg = 2 * n
    coef_opt = optimal_poly_cheb_coeffs(a, n)
    coef_w = window_cheb_coeffs(
        a, window_deg, delta, use_p_theta=use_p_theta, p_theta_eps=p_theta_eps
    )
    product = cheb.chebmul(coef_opt, coef_w)
    return product


def get_coef_full(a: float, n: int, method: str) -> np.ndarray:
    """
    Return full Chebyshev coefficient vector (length degree+1) for the polynomial.

    method == 'sunderhof': optimal polynomial from Sünderhauf et al. (scaled by a).
    method == 'cvxpy': convex optimization via cvx_poly_coef, target a/x on S(a),
        with npts = nearest power of 2 to 2*degree for constraint discretization.
    method == 'window': same optimal polynomial as sunderhof (windowed version is
        computed separately via windowed_poly_cheb_coeffs for the second curve).
    """
    degree = 2 * n - 1
    if method == "sunderhof":
        return optimal_poly_cheb_coeffs(a, n)
    if method == "cvxpy":
        npts_cvx = degree
        print(f"Using cvxpy discretization npts_cvx = {npts_cvx}")
        targ = lambda x: a / x
        opts = {
            "intervals": [a, 1],
            "objnorm": np.inf,
            "epsil": 0.0,
            "npts": npts_cvx,
            "isplot": False,
            "fscale": 1,
            "method": "cvxpy",
        }
        return cvx_poly_coef(targ, degree, opts)
    if method == "window":
        return optimal_poly_cheb_coeffs(a, n)
    raise ValueError("method must be 'sunderhof', 'cvxpy', or 'window'")


def _draw_one_plot(
    ax: plt.Axes,
    xlist: np.ndarray,
    targ_value: np.ndarray,
    coef_full: np.ndarray,
    coef_recovered_full: np.ndarray,
    M: int,
    title: str,
    legend_loc: str = "upper right",
    use_windowed_curve: bool = False,
) -> None:
    """Draw target, polynomial fit, and either retraction or windowed curve on a single axes."""
    func_value = chebval_dct(coef_full, M)
    recovered_value = chebval_dct(coef_recovered_full, M)
    ax.plot(xlist, targ_value, label="Target", color="black", linewidth=3)
    ax.plot(
        xlist,
        func_value,
        label="Polynomial Fit",
        color="#0072B2",
        linewidth=2,
    )
    if use_windowed_curve:
        ax.plot(
            xlist,
            recovered_value,
            label="Windowed",
            color="#E69F00",
            linewidth=2,
            linestyle="--",
        )
    else:
        ax.plot(
            xlist,
            recovered_value,
            label="Retraction",
            color="#E69F00",
            linewidth=2,
            linestyle="--",
        )
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.1, 1.1])
    ax.legend(loc=legend_loc, framealpha=1)
    ax.set_xlabel(r"$x$")
    ax.set_title(title)


def plot_optimal_and_retraction(
    a: float,
    n: int,
    N_weiss: int,
    method: str = "sunderhof",
    error_plot: bool = False,
    delta: float = 0.01,
    use_p_theta: bool = False,
    p_theta_eps: float = 1e-6,
) -> None:
    """
    Plot target a/x (on S(a)), polynomial approximation, and its retraction.
    Repeats the routine from plot_fit_polynomial_space.py (lines 68-92) using
    degree and N_weiss from mat_inv_ex. coef_full from get_coef_full(a, n, method).
    If method == 'both', draw two subplots (left: sunderhof, right: cvxpy).
    If error_plot is True, add aligned error plots |retraction - target| below.
    delta: for window/all methods, window transitions 0->1 on [a - delta, a].
    use_p_theta: if True, build window from P_Θ (ref [21]) instead of smoothstep.
    """
    degree = 2 * n - 1
    parity = degree % 2  # 1 for odd polynomial

    # Shared evaluation grid and target
    M = 10_000
    print("Computing Chebyshev nodes...")
    xlist = np.cos(np.pi * np.arange(M) / (M - 1))
    targ_value = np.full_like(xlist, np.nan, dtype=float)
    # We only care about the positive side [a, 1] for these odd functions
    domain_mask = xlist >= a
    targ_value[domain_mask] = a / xlist[domain_mask]

    if method == "both":
        print("Creating plot (both methods)...")
        if error_plot:
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
            ax_left = fig.add_subplot(gs[0, 0])
            ax_right = fig.add_subplot(gs[0, 1])
            ax_err_left = fig.add_subplot(gs[1, 0], sharex=ax_left)
            # Share y-axis between error plots so scales line up
            ax_err_right = fig.add_subplot(gs[1, 1], sharex=ax_right, sharey=ax_err_left)
            axes = [
                (ax_left, ax_err_left, "sunderhof", "Optimal + Retraction (deg 101)"),
                (ax_right, ax_err_right, "cvxpy", "Optimal Constrained + Retraction (deg 101)"),
            ]
            for ax_main, ax_err, m, title in axes:
                print(f"  Computing coefficients and retraction for {m}...")
                coef_full = get_coef_full(a, n, m)
                b_coeffs = b_from_cheb(coef_full[parity::2], parity)
                a_coeffs = weiss(b_coeffs, N_weiss)
                gammas, _, _ = inverse_nonlinear_FFT(a_coeffs, b_coeffs)
                _, new_b = forward_nonlinear_FFT(gammas)
                coef_recovered_full = np.zeros(degree + 1)
                coef_recovered_full[1::2] = new_b[int(len(new_b) / 2 - 1) :: -1] + new_b[int(len(new_b) / 2) : :]
                _draw_one_plot(
                    ax_main,
                    xlist,
                    targ_value,
                    coef_full,
                    coef_recovered_full,
                    M,
                    title,
                    legend_loc="upper right",
                )
                # Top plots share x-axis with error plots: suppress x-labels on top row
                ax_main.set_xlabel("")
                # Error plots: |fit - target| (blue) and |retraction - target| (orange), only on |x| >= a
                recovered_value = chebval_dct(coef_recovered_full, M)
                fit_value = chebval_dct(coef_full, M)
                err_fit = np.full_like(xlist, np.nan, dtype=float)
                err_ret = np.full_like(xlist, np.nan, dtype=float)
                err_fit[domain_mask] = np.abs(fit_value[domain_mask] - targ_value[domain_mask])
                err_ret[domain_mask] = np.abs(recovered_value[domain_mask] - targ_value[domain_mask])
                ax_err.plot(xlist, err_fit, color="#0072B2", linewidth=1.0)
                ax_err.plot(xlist, err_ret, color="#E69F00", linewidth=1.5)
                ax_err.set_yscale("log")
                ax_err.grid(True, alpha=0.3)
                ax_err.set_xlim([0, 1])
                ax_err.set_xlabel(r"$x$")
                # Only label error y-axis on the far-left subplot
                if ax_err is ax_err_left:
                    ax_err.set_ylabel("Error")
            # Hide y-axis ticks/labels on the right column (main and error)
            ax_right.tick_params(labelleft=False)
            ax_err_right.tick_params(labelleft=False)
            # Hide x-axis tick labels on the top row (shared x with error plots)
            ax_left.tick_params(labelbottom=False)
            ax_right.tick_params(labelbottom=False)
            plt.tight_layout()
            plt.show()
            return

        # both, no error plot
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        for ax, m, title in [
            (ax_left, "sunderhof", "Optimal + Retraction (deg 101)"),
            (ax_right, "cvxpy", "Optimal Constrained + Retraction (deg 101)"),
        ]:
            print(f"  Computing coefficients and retraction for {m}...")
            coef_full = get_coef_full(a, n, m)
            b_coeffs = b_from_cheb(coef_full[parity::2], parity)
            a_coeffs = weiss(b_coeffs, N_weiss)
            gammas, _, _ = inverse_nonlinear_FFT(a_coeffs, b_coeffs)
            _, new_b = forward_nonlinear_FFT(gammas)
            coef_recovered_full = np.zeros(degree + 1)
            coef_recovered_full[1::2] = new_b[int(len(new_b) / 2 - 1) :: -1] + new_b[int(len(new_b) / 2) : :]
            _draw_one_plot(ax, xlist, targ_value, coef_full, coef_recovered_full, M, title, legend_loc="upper right")
        plt.tight_layout()
        plt.show()
        return

    if method == "all":
        print("Creating plot (all three methods: window, sunderhof, cvxpy)...")
        deg_label = 2 * n - 1
        if error_plot:
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
            ax_w = fig.add_subplot(gs[0, 0])
            ax_s = fig.add_subplot(gs[0, 1])
            ax_c = fig.add_subplot(gs[0, 2])
            ax_err_w = fig.add_subplot(gs[1, 0], sharex=ax_w)
            ax_err_s = fig.add_subplot(gs[1, 1], sharex=ax_s, sharey=ax_err_w)
            ax_err_c = fig.add_subplot(gs[1, 2], sharex=ax_c, sharey=ax_err_w)
            # Window (left)
            coef_w = get_coef_full(a, n, "window")
            coef_w_rec = windowed_poly_cheb_coeffs(
                a, n, delta=delta, use_p_theta=use_p_theta, p_theta_eps=p_theta_eps
            )
            _draw_one_plot(
                ax_w, xlist, targ_value, coef_w, coef_w_rec, M,
                f"Optimal + Window (deg {2*n-1}+{2*n})", use_windowed_curve=True,
            )
            ax_w.set_xlabel("")
            fit_w = chebval_dct(coef_w, M)
            rec_w = chebval_dct(coef_w_rec, M)
            err_fit_w = np.full_like(xlist, np.nan, dtype=float)
            err_win_w = np.full_like(xlist, np.nan, dtype=float)
            err_fit_w[domain_mask] = np.abs(fit_w[domain_mask] - targ_value[domain_mask])
            err_win_w[domain_mask] = np.abs(rec_w[domain_mask] - targ_value[domain_mask])
            ax_err_w.plot(xlist, err_fit_w, color="#0072B2", linewidth=1.0)
            ax_err_w.plot(xlist, err_win_w, color="#E69F00", linewidth=1.5)
            ax_err_w.set_yscale("log")
            ax_err_w.grid(True, alpha=0.3)
            ax_err_w.set_xlim([0, 1])
            ax_err_w.set_xlabel(r"$x$")
            ax_err_w.set_ylabel("Error")
            # Sunderhof (center)
            coef_s = get_coef_full(a, n, "sunderhof")
            b_s = b_from_cheb(coef_s[parity::2], parity)
            a_s = weiss(b_s, N_weiss)
            gammas_s, _, _ = inverse_nonlinear_FFT(a_s, b_s)
            _, new_b_s = forward_nonlinear_FFT(gammas_s)
            coef_s_rec = np.zeros(degree + 1)
            coef_s_rec[1::2] = new_b_s[int(len(new_b_s) / 2 - 1) :: -1] + new_b_s[int(len(new_b_s) / 2) : :]
            _draw_one_plot(
                ax_s, xlist, targ_value, coef_s, coef_s_rec, M,
                f"Optimal + Retraction (deg {deg_label})",
            )
            ax_s.set_xlabel("")
            ax_s.tick_params(labelleft=False)
            fit_s = chebval_dct(coef_s, M)
            rec_s = chebval_dct(coef_s_rec, M)
            err_fit_s = np.full_like(xlist, np.nan, dtype=float)
            err_ret_s = np.full_like(xlist, np.nan, dtype=float)
            err_fit_s[domain_mask] = np.abs(fit_s[domain_mask] - targ_value[domain_mask])
            err_ret_s[domain_mask] = np.abs(rec_s[domain_mask] - targ_value[domain_mask])
            ax_err_s.plot(xlist, err_fit_s, color="#0072B2", linewidth=1.0)
            ax_err_s.plot(xlist, err_ret_s, color="#E69F00", linewidth=1.5)
            ax_err_s.set_yscale("log")
            ax_err_s.grid(True, alpha=0.3)
            ax_err_s.set_xlim([0, 1])
            ax_err_s.set_xlabel(r"$x$")
            ax_err_s.tick_params(labelleft=False)
            # CVXPY (right)
            coef_c = get_coef_full(a, n, "cvxpy")
            b_c = b_from_cheb(coef_c[parity::2], parity)
            a_c = weiss(b_c, N_weiss)
            gammas_c, _, _ = inverse_nonlinear_FFT(a_c, b_c)
            _, new_b_c = forward_nonlinear_FFT(gammas_c)
            coef_c_rec = np.zeros(degree + 1)
            coef_c_rec[1::2] = new_b_c[int(len(new_b_c) / 2 - 1) :: -1] + new_b_c[int(len(new_b_c) / 2) : :]
            _draw_one_plot(
                ax_c, xlist, targ_value, coef_c, coef_c_rec, M,
                f"Optimal Constrained + Retraction (deg {deg_label})",
            )
            ax_c.set_xlabel("")
            ax_c.tick_params(labelleft=False)
            fit_c = chebval_dct(coef_c, M)
            rec_c = chebval_dct(coef_c_rec, M)
            err_fit_c = np.full_like(xlist, np.nan, dtype=float)
            err_ret_c = np.full_like(xlist, np.nan, dtype=float)
            err_fit_c[domain_mask] = np.abs(fit_c[domain_mask] - targ_value[domain_mask])
            err_ret_c[domain_mask] = np.abs(rec_c[domain_mask] - targ_value[domain_mask])
            ax_err_c.plot(xlist, err_fit_c, color="#0072B2", linewidth=1.0)
            ax_err_c.plot(xlist, err_ret_c, color="#E69F00", linewidth=1.5)
            ax_err_c.set_yscale("log")
            ax_err_c.grid(True, alpha=0.3)
            ax_err_c.set_xlim([0, 1])
            ax_err_c.set_xlabel(r"$x$")
            ax_err_c.tick_params(labelleft=False)
            ax_w.tick_params(labelbottom=False)
            ax_s.tick_params(labelbottom=False)
            ax_c.tick_params(labelbottom=False)
            plt.tight_layout()
            plt.show()
        else:
            fig, (ax_w, ax_s, ax_c) = plt.subplots(1, 3, figsize=(18, 6))
            coef_w = get_coef_full(a, n, "window")
            coef_w_rec = windowed_poly_cheb_coeffs(
                a, n, delta=delta, use_p_theta=use_p_theta, p_theta_eps=p_theta_eps
            )
            _draw_one_plot(
                ax_w, xlist, targ_value, coef_w, coef_w_rec, M,
                f"Optimal + Window (deg {2*n-1}+{2*n})", use_windowed_curve=True,
            )
            coef_s = get_coef_full(a, n, "sunderhof")
            b_s = b_from_cheb(coef_s[parity::2], parity)
            a_s = weiss(b_s, N_weiss)
            gammas_s, _, _ = inverse_nonlinear_FFT(a_s, b_s)
            _, new_b_s = forward_nonlinear_FFT(gammas_s)
            coef_s_rec = np.zeros(degree + 1)
            coef_s_rec[1::2] = new_b_s[int(len(new_b_s) / 2 - 1) :: -1] + new_b_s[int(len(new_b_s) / 2) : :]
            _draw_one_plot(
                ax_s, xlist, targ_value, coef_s, coef_s_rec, M,
                f"Optimal + Retraction (deg {deg_label})",
            )
            ax_s.tick_params(labelleft=False)
            coef_c = get_coef_full(a, n, "cvxpy")
            b_c = b_from_cheb(coef_c[parity::2], parity)
            a_c = weiss(b_c, N_weiss)
            gammas_c, _, _ = inverse_nonlinear_FFT(a_c, b_c)
            _, new_b_c = forward_nonlinear_FFT(gammas_c)
            coef_c_rec = np.zeros(degree + 1)
            coef_c_rec[1::2] = new_b_c[int(len(new_b_c) / 2 - 1) :: -1] + new_b_c[int(len(new_b_c) / 2) : :]
            _draw_one_plot(
                ax_c, xlist, targ_value, coef_c, coef_c_rec, M,
                f"Optimal Constrained + Retraction (deg {deg_label})",
            )
            ax_c.tick_params(labelleft=False)
            plt.tight_layout()
            plt.show()
        return

    if method == "window":
        print("Creating plot (window method)...")
        coef_full = get_coef_full(a, n, "window")
        coef_recovered_full = windowed_poly_cheb_coeffs(
            a, n, delta=delta, use_p_theta=use_p_theta, p_theta_eps=p_theta_eps
        )
        if error_plot:
            fig, (ax_main, ax_err) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1]
            )
            _draw_one_plot(
                ax_main,
                xlist,
                targ_value,
                coef_full,
                coef_recovered_full,
                M,
                "",
                use_windowed_curve=True,
            )
            ax_main.set_xlabel("")
            ax_main.tick_params(labelbottom=False)
            recovered_value = chebval_dct(coef_recovered_full, M)
            fit_value = chebval_dct(coef_full, M)
            err_fit = np.full_like(xlist, np.nan, dtype=float)
            err_win = np.full_like(xlist, np.nan, dtype=float)
            err_fit[domain_mask] = np.abs(fit_value[domain_mask] - targ_value[domain_mask])
            err_win[domain_mask] = np.abs(recovered_value[domain_mask] - targ_value[domain_mask])
            ax_err.plot(xlist, err_fit, color="#0072B2", linewidth=1.0)
            ax_err.plot(xlist, err_win, color="#E69F00", linewidth=1.5)
            ax_err.set_yscale("log")
            ax_err.grid(True, alpha=0.3)
            ax_err.set_xlim([0, 1])
            ax_err.set_xlabel(r"$x$")
            ax_err.set_ylabel("Error")
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            _draw_one_plot(
                ax,
                xlist,
                targ_value,
                coef_full,
                coef_recovered_full,
                M,
                "",
                use_windowed_curve=True,
            )
            plt.tight_layout()
            plt.show()
        return

    # Single method (sunderhof or cvxpy)
    coef_full = get_coef_full(a, n, method)
    b_coeffs = b_from_cheb(coef_full[parity::2], parity)
    a_coeffs = weiss(b_coeffs, N_weiss)
    gammas, _, _ = inverse_nonlinear_FFT(a_coeffs, b_coeffs)
    _, new_b = forward_nonlinear_FFT(gammas)
    coef_recovered_full = np.zeros(degree + 1)
    coef_recovered_full[1::2] = new_b[int(len(new_b) / 2 - 1) :: -1] + new_b[int(len(new_b) / 2) : :]

    if error_plot:
        print("Creating plot with error subplot...")
        fig, (ax_main, ax_err) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1]
        )
        _draw_one_plot(ax_main, xlist, targ_value, coef_full, coef_recovered_full, M, "")
        # Suppress x-label on top plot when error subplot is present
        ax_main.set_xlabel("")
        ax_main.tick_params(labelbottom=False)
        recovered_value = chebval_dct(coef_recovered_full, M)
        fit_value = chebval_dct(coef_full, M)
        err_fit = np.full_like(xlist, np.nan, dtype=float)
        err_ret = np.full_like(xlist, np.nan, dtype=float)
        err_fit[domain_mask] = np.abs(fit_value[domain_mask] - targ_value[domain_mask])
        err_ret[domain_mask] = np.abs(recovered_value[domain_mask] - targ_value[domain_mask])
        ax_err.plot(xlist, err_fit, color="#0072B2", linewidth=1.0)
        ax_err.plot(xlist, err_ret, color="#E69F00", linewidth=1.5)
        ax_err.set_yscale("log")
        ax_err.grid(True, alpha=0.3)
        ax_err.set_xlim([0, 1])
        ax_err.set_xlabel(r"$x$")
        ax_err.set_ylabel("Error")
        plt.tight_layout()
        plt.show()
        return

    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_one_plot(ax, xlist, targ_value, coef_full, coef_recovered_full, M, "")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Matrix inversion example: plot 1/x, the optimal polynomial from "
            "Sünderhauf et al. (arxiv:2507.15537), and its polynomial-space retraction "
            "or windowed construction (PRX Quantum 2, 040203 (2021) / arXiv:2105.02859v5, Appendix C)."
        )
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.1,
        help="a parameter (domain S(a) = [-1,-a]∪[a,1]), must be in (0,1). "
        "Default: 0.1.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=51,
        help="n in degree d = 2n-1 of the optimal polynomial. Default: 51 (d=101).",
    )
    parser.add_argument(
        "--N_weiss",
        type=int,
        default=2**12,
        help="N for Weiss transform in recovery (default: 2^12).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("sunderhof", "cvxpy", "both", "window", "all"),
        default="cvxpy",
        help="Method: 'sunderhof', 'cvxpy', 'both' (two subplots), 'window' (optimal + "
        "windowing from PRX Quantum Appendix C instead of retraction), or 'all' "
        "(three columns: window left, Sunderhof center, cvxpy right). Default: cvxpy.",
    )
    parser.add_argument(
        "--error_plot",
        action="store_true",
        help="If set, add an error subplot |retraction - target| below the main plot(s).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="For window/all: window transitions 0->1 on [a - delta, a] (default: 0.01).",
    )
    parser.add_argument(
        "--use-p-theta",
        action="store_true",
        help="Build window from P_Θ (ref [21], erf-based step) instead of smoothstep.",
    )
    parser.add_argument(
        "--p-theta-eps",
        type=float,
        default=1e-6,
        help="Approximation error ε for P_Θ when --use-p-theta (default: 1e-6).",
    )

    args = parser.parse_args()

    # Basic validations
    if args.n < 1:
        raise SystemExit("Error: n must be >= 1.")
    if not (0.0 < args.a < 1.0):
        raise SystemExit("Error: a must lie in (0,1).")
    if not (0.0 < args.delta <= 1.0):
        raise SystemExit("Error: delta must lie in (0, 1].")

    print(
        f"Using a={args.a}, n={args.n} (degree d={2*args.n-1}), N_weiss={args.N_weiss}, "
        f"method={args.method}, error_plot={args.error_plot}, delta={args.delta}, "
        f"use_p_theta={args.use_p_theta}, p_theta_eps={args.p_theta_eps}."
    )
    plot_optimal_and_retraction(
        args.a,
        args.n,
        args.N_weiss,
        args.method,
        args.error_plot,
        delta=args.delta,
        use_p_theta=args.use_p_theta,
        p_theta_eps=args.p_theta_eps,
    )


if __name__ == "__main__":
    main()

