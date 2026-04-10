import numpy as np


def evaluate_from_result(result, omega, which="scaled"):
    """
    Evaluate the trigonometric polynomial stored in a fitter result.

    Parameters
    ----------
    result : dict
        Result returned by ConstrainedRemezQSPFitter.fit(...).
    omega : array_like
        Evaluation points.
    which : {"raw", "scaled"}
        Which coefficient block to use.

    Returns
    -------
    values : np.ndarray
        H(omega).
    """
    omega = np.asarray(omega, dtype=float)
    ks = np.asarray(result["ks"], dtype=float)
    coeffs = np.asarray(result[which]["coeffs"], dtype=float)
    return np.cos(omega[:, None] * ks[None, :]) @ coeffs


def derivative_from_result(result, omega, which="scaled"):
    """
    Evaluate H'(omega) for the trigonometric polynomial stored in result.
    """
    omega = np.asarray(omega, dtype=float)
    ks = np.asarray(result["ks"], dtype=float)
    coeffs = np.asarray(result[which]["coeffs"], dtype=float)
    return -np.sum(
        (ks * coeffs)[None, :] * np.sin(omega[:, None] * ks[None, :]),
        axis=1,
    )


def second_derivative_from_result(result, omega, which="scaled"):
    """
    Evaluate H''(omega).
    """
    omega = np.asarray(omega, dtype=float)
    ks = np.asarray(result["ks"], dtype=float)
    coeffs = np.asarray(result[which]["coeffs"], dtype=float)
    return -np.sum(
        ((ks ** 2) * coeffs)[None, :] * np.cos(omega[:, None] * ks[None, :]),
        axis=1,
    )


def _roots_of_derivative_on_interval(
    result,
    which="scaled",
    domain=(0.0, 0.5 * np.pi),
    grid_size=20001,
    bisection_iters=80,
    zero_tol=1e-13,
):
    """
    Find roots of H'(omega)=0 on a single interval by sign bracketing + bisection.
    Endpoints are not included automatically.
    """
    a, b = domain
    grid = np.linspace(a, b, max(3, int(grid_size)), endpoint=True)
    vals = derivative_from_result(result, grid, which=which)

    roots = []

    # near-zero hits on the grid
    mask_zero = np.abs(vals) < zero_tol
    if np.any(mask_zero):
        roots.extend(grid[mask_zero].tolist())

    # sign changes
    signs = np.sign(vals)
    brk = np.where(signs[:-1] * signs[1:] < 0)[0]

    for i in brk:
        lo, hi = grid[i], grid[i + 1]
        f_lo = derivative_from_result(result, np.array([lo]), which=which)[0]

        for _ in range(bisection_iters):
            mid = 0.5 * (lo + hi)
            f_mid = derivative_from_result(result, np.array([mid]), which=which)[0]
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo = mid
                f_lo = f_mid

        roots.append(0.5 * (lo + hi))

    if not roots:
        return np.array([], dtype=float)

    return np.unique(np.asarray(roots, dtype=float))


def extremal_points_from_result(
    result,
    which="scaled",
    domain=(0.0, 0.5 * np.pi),
    grid_size=20001,
    bisection_iters=80,
):
    """
    Return all candidate extrema points for H on the domain:
    endpoints + numerical roots of H'(omega)=0.
    """
    a, b = domain
    roots = _roots_of_derivative_on_interval(
        result=result,
        which=which,
        domain=domain,
        grid_size=grid_size,
        bisection_iters=bisection_iters,
    )
    pts = np.unique(np.concatenate([np.array([a, b], dtype=float), roots]))
    return pts


def max_magnitude_from_result(
    result,
    which="scaled",
    domain=(0.0, 0.5 * np.pi),
    grid_size=20001,
    bisection_iters=80,
):
    """
    Compute the maximum magnitude of the trigonometric polynomial on the domain
    by evaluating H on all numerical extrema and the endpoints.

    This is more accurate than a plain dense grid search because every interior
    extremum is found from H'(omega)=0.

    Returns
    -------
    info : dict
        A dictionary with max magnitude, argmax location, value, and all extrema.
    """
    pts = extremal_points_from_result(
        result=result,
        which=which,
        domain=domain,
        grid_size=grid_size,
        bisection_iters=bisection_iters,
    )
    vals = evaluate_from_result(result, pts, which=which)
    abs_vals = np.abs(vals)

    idx = int(np.argmax(abs_vals))
    omega_star = float(pts[idx])
    value_star = float(vals[idx])
    max_abs = float(abs_vals[idx])

    # optional classification of ordinary extrema
    h2 = second_derivative_from_result(result, pts, which=which)
    kinds = []
    a, b = domain
    for x, v2 in zip(pts, h2):
        if np.isclose(x, a) or np.isclose(x, b):
            kinds.append("endpoint")
        elif v2 < 0:
            kinds.append("local_max")
        elif v2 > 0:
            kinds.append("local_min")
        else:
            kinds.append("flat_or_undetermined")

    return {
        "which": which,
        "domain": tuple(map(float, domain)),
        "max_abs": max_abs,
        "argmax_abs": omega_star,
        "value_at_argmax_abs": value_star,
        "num_extrema_candidates": int(len(pts)),
        "extremal_points": pts,
        "extremal_values": vals,
        "extremal_types": kinds,
    }
