import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from numpy.polynomial import chebyshev as cheb

from fgt_polynomial_space import recovered_coeffs


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
    return coefs


def plot_optimal_and_retraction(a: float, n: int, N_weiss: int) -> None:
    """
    Plot target 1/x (on S(a)), optimal polynomial, and its retraction.
    """
    degree = 2 * n - 1
    parity = degree % 2  # should be 1 (odd)

    # Optimal polynomial Chebyshev coefficients (full, length degree+1)
    coef_full = optimal_poly_cheb_coeffs(a, n)

    # Retraction using the same NLFA pipeline as in fgt_polynomial_space.py
    coef_recovered_full = recovered_coeffs(coef_full, parity, N_weiss)

    # Chebyshev nodes for evaluation (same convention as chebval_dct)
    M = 10_000
    xlist = np.cos(np.pi * np.arange(M) / (M - 1))

    # Target function 1/x on S(a) = [-1, -a] ∪ [a, 1]; mask outside domain
    targ_value = np.full_like(xlist, np.nan, dtype=float)
    domain_mask = np.abs(xlist) >= a
    targ_value[domain_mask] = 1.0 / xlist[domain_mask]

    # Optimal polynomial and its retraction from Chebyshev coefficients
    opt_value = chebval_dct(coef_full, M)
    retr_value = chebval_dct(coef_recovered_full, M)

    plt.figure(figsize=(12, 8))

    # Colors: target black, optimal blue, retraction orange (dashed)
    plt.plot(xlist, targ_value, label=r"Target $1/x$", color="black", linewidth=2)
    plt.plot(
        xlist,
        opt_value,
        label="Optimal polynomial",
        color="#0072B2",
        linewidth=2,
    )
    plt.plot(
        xlist,
        retr_value,
        label="Retraction",
        color="#E69F00",
        linewidth=2,
        linestyle="--",
    )

    plt.grid(True, alpha=0.3)
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.1, 1.1])
    plt.xlabel(r"$x$")
    plt.legend(loc="best", framealpha=1.0)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Matrix inversion example: plot 1/x, the optimal polynomial from "
            "Sünderhauf et al. (arxiv:2507.15537), and its polynomial-space retraction."
        )
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.2,
        help="a parameter (domain S(a) = [-1,-a]∪[a,1]), must be in (0,1). "
        "Default: 0.2.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=26,
        help="n in degree d = 2n-1 of the optimal polynomial. Default: 26 (d=51).",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=2**15,
        help="N parameter for Weiss transform in recovered_coeffs (default: 2^15).",
    )

    args = parser.parse_args()

    # Basic validations
    if args.n < 1:
        raise SystemExit("Error: n must be >= 1.")
    if not (0.0 < args.a < 1.0):
        raise SystemExit("Error: a must lie in (0,1).")

    print(
        f"Using a={args.a}, n={args.n} (degree d={2*args.n-1}), N={args.N} "
        "for Weiss / NLFA retraction."
    )
    plot_optimal_and_retraction(args.a, args.n, args.N)


if __name__ == "__main__":
    main()

