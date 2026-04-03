
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, List


Interval = Tuple[float, float]


def merge_intervals(
    intervals: Sequence[Interval],
    tol: float = 1e-14,
    domain: Interval = (0.0, 0.5 * np.pi),
) -> List[Interval]:
    if not intervals:
        return []
    a0, b0 = domain
    arr = []
    for a, b in intervals:
        a = max(a0, float(a))
        b = min(b0, float(b))
        if b >= a - tol:
            arr.append((a, b))
    if not arr:
        return []

    arr.sort()
    out = [list(arr[0])]
    for a, b in arr[1:]:
        if a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def enlarge_intervals(
    intervals: Sequence[Interval],
    extension: float,
    domain: Interval = (0.0, 0.5 * np.pi),
) -> List[Interval]:
    if extension <= 0:
        return merge_intervals(intervals, domain=domain)
    enlarged = [(a - extension, b + extension) for a, b in intervals]
    return merge_intervals(enlarged, domain=domain)


def complement_intervals(
    intervals: Sequence[Interval],
    tol: float = 1e-14,
    domain: Interval = (0.0, 0.5 * np.pi),
) -> List[Interval]:
    intervals = merge_intervals(intervals, tol=tol, domain=domain)
    a0, b0 = domain
    out = []
    cur = a0
    for a, b in intervals:
        if a > cur + tol:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < b0 - tol:
        out.append((cur, b0))
    return out


def interval_edges(intervals: Sequence[Interval]) -> np.ndarray:
    if not intervals:
        return np.array([], dtype=float)
    edges = []
    for a, b in intervals:
        edges.extend([a, b])
    return np.unique(np.asarray(edges, dtype=float))


def union_grid(intervals: Sequence[Interval], n_per_interval: int) -> np.ndarray:
    if not intervals:
        return np.array([], dtype=float)
    chunks = []
    for a, b in intervals:
        chunks.append(np.linspace(a, b, max(2, int(n_per_interval)), endpoint=True))
    return np.unique(np.concatenate(chunks))


class ConstrainedRemezQSPFitter:
    """
    Constrained Remez fitter for QSP-style cosine approximation on [0, pi/2].

    The fitted model is
        H(w) = sum_{j in K_d} a_j cos(j w),  K_d = {d, d-2, ...}.

    The method first computes a raw constrained fit, then rescales it by
        scale = max(1, M),
    where M is the raw maximum magnitude on the full domain.

    The fit() method returns two result blocks:
        result["raw"]
        result["scaled"]
    Each block contains coefficients and its own metrics.
    """

    def __init__(
        self,
        omega_fit: Sequence[Interval],
        target: Callable[[np.ndarray], np.ndarray],
        weight: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lower_bound: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        upper_bound: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        domain: Interval = (0.0, 0.5 * np.pi),
        omega_work: Optional[Sequence[Interval]] = None,
        interval_extension: float = 0.0,
        target_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        weight_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        alpha0: float = 0.1,
        rel_tol: float = 1e-3,
        max_outer: int = 40,
        passband_grid_per_interval: int = 4001,
        stopband_grid_per_interval: int = 4001,
        root_grid_per_interval: int = 2001,
        root_bisection_iters: int = 60,
        inner_max_swaps: int = 100,
        inner_tol: float = 1e-9,
        metrics_grid_size: int = 20000,
        metrics_interval_grid_size: int = 4000,
    ) -> None:
        self.domain = domain
        self.omega_fit = merge_intervals(omega_fit, domain=domain)

        if omega_work is None:
            self.omega_work = enlarge_intervals(
                self.omega_fit, interval_extension, domain=domain
            )
        else:
            self.omega_work = merge_intervals(omega_work, domain=domain)

        self.omega_constraint = complement_intervals(self.omega_work, domain=domain)

        self.target = target
        self.target_prime = target_prime
        self.weight = weight or (lambda w: np.ones_like(np.asarray(w, dtype=float)))
        self.weight_prime = weight_prime
        self.lower_bound = lower_bound or (
            lambda w: -np.ones_like(np.asarray(w, dtype=float))
        )
        self.upper_bound = upper_bound or (
            lambda w: np.ones_like(np.asarray(w, dtype=float))
        )

        self.alpha0 = float(alpha0)
        self.rel_tol = float(rel_tol)
        self.max_outer = int(max_outer)
        self.passband_grid_per_interval = int(passband_grid_per_interval)
        self.stopband_grid_per_interval = int(stopband_grid_per_interval)
        self.root_grid_per_interval = int(root_grid_per_interval)
        self.root_bisection_iters = int(root_bisection_iters)
        self.inner_max_swaps = int(inner_max_swaps)
        self.inner_tol = float(inner_tol)
        self.metrics_grid_size = int(metrics_grid_size)
        self.metrics_interval_grid_size = int(metrics_interval_grid_size)

        self._last_result: Optional[Dict[str, Any]] = None

    @staticmethod
    def parity_indices(d: int) -> np.ndarray:
        return np.arange(d, -1, -2, dtype=int)

    @staticmethod
    def _cos_design_matrix_idx(omega: np.ndarray, ks: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        ks = np.asarray(ks, dtype=float)
        return np.cos(omega[:, None] * ks[None, :])

    @classmethod
    def _eval_cos_idx(cls, coeffs: np.ndarray, omega: np.ndarray, ks: np.ndarray) -> np.ndarray:
        return cls._cos_design_matrix_idx(np.asarray(omega), ks) @ coeffs

    @staticmethod
    def _hp_idx(coeffs: np.ndarray, omega: np.ndarray, ks: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        ks = np.asarray(ks, dtype=float)
        return -np.sum(
            (ks * coeffs)[None, :] * np.sin(omega[:, None] * ks[None, :]),
            axis=1,
        )

    @staticmethod
    def _num_deriv(
        fun: Callable[[np.ndarray], np.ndarray],
        omega: np.ndarray,
        h: float = 1e-6,
    ) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        return (fun(omega + h) - fun(omega - h)) / (2 * h)

    def _ewprime_idx(
        self,
        coeffs: np.ndarray,
        omega: np.ndarray,
        ks: np.ndarray,
    ) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        H = self._eval_cos_idx(coeffs, omega, ks)
        Hp = self._hp_idx(coeffs, omega, ks)

        Wp = (
            self._num_deriv(self.weight, omega)
            if self.weight_prime is None
            else self.weight_prime(omega)
        )
        Dp = (
            self._num_deriv(self.target, omega)
            if self.target_prime is None
            else self.target_prime(omega)
        )
        return Wp * (self.target(omega) - H) + self.weight(omega) * (Dp - Hp)

    def _roots_of_f_on_intervals(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        intervals: Sequence[Interval],
    ) -> np.ndarray:
        roots = []

        for a, b in intervals:
            g = np.linspace(a, b, max(3, self.root_grid_per_interval), endpoint=True)
            vals = np.asarray(f(g), dtype=float)

            mask_zero = np.abs(vals) < 1e-12
            if np.any(mask_zero):
                roots.extend(g[mask_zero].tolist())

            s = np.sign(vals)
            brk = np.where(s[:-1] * s[1:] < 0)[0]

            for i in brk:
                lo, hi = g[i], g[i + 1]
                f_lo = f(np.array([lo]))[0]

                for _ in range(self.root_bisection_iters):
                    mid = 0.5 * (lo + hi)
                    f_mid = f(np.array([mid]))[0]
                    if f_lo * f_mid <= 0:
                        hi = mid
                    else:
                        lo = mid
                        f_lo = f_mid

                roots.append(0.5 * (lo + hi))

        if not roots:
            return np.array([], dtype=float)

        return np.unique(np.asarray(roots, dtype=float))

    def _mremez_equation_constrained_union(
        self,
        ks: np.ndarray,
        X_init: np.ndarray,
        eq_freqs: np.ndarray,
        eq_values: np.ndarray,
        omega_work_grid: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        X = np.unique(
            np.concatenate([np.asarray(X_init, float), np.asarray(eq_freqs, float)])
        )
        X.sort()

        ncoef = len(ks)

        for _ in range(self.inner_max_swaps):
            signs = np.array([1 if i % 2 == 0 else -1 for i in range(len(X))], dtype=float)

            is_eq = np.isin(X, eq_freqs)
            X_alt = X[~is_eq]
            S_alt = signs[~is_eq]

            A_alt = np.hstack([
                self.weight(X_alt)[:, None] * self._cos_design_matrix_idx(X_alt, ks),
                S_alt[:, None],
            ])
            b_alt = self.weight(X_alt) * self.target(X_alt)

            if len(eq_freqs) > 0:
                A_eq = np.hstack([
                    self._cos_design_matrix_idx(eq_freqs, ks),
                    np.zeros((len(eq_freqs), 1)),
                ])
                b_eq = np.asarray(eq_values, dtype=float)
                A = np.vstack([A_alt, A_eq])
                b = np.concatenate([b_alt, b_eq])
            else:
                A = A_alt
                b = b_alt

            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            coeffs = sol[:-1]
            Delta = float(sol[-1])

            Ew_grid = self.weight(omega_work_grid) * (
                self.target(omega_work_grid) - self._eval_cos_idx(coeffs, omega_work_grid, ks)
            )

            f = lambda w: self._ewprime_idx(coeffs, w, ks)
            ext = self._roots_of_f_on_intervals(f, self.omega_work)
            cand = np.unique(np.concatenate([ext, interval_edges(self.omega_work)]))
            Ew_cand = self.weight(cand) * (
                self.target(cand) - self._eval_cos_idx(coeffs, cand, ks)
            )

            m_needed = ncoef + 1 - len(eq_freqs)
            if m_needed < 0:
                raise RuntimeError("Too many equality constraints for the available basis size.")

            take = cand[np.argsort(-np.abs(Ew_cand))][:m_needed] if m_needed > 0 else np.array([], dtype=float)
            X_new = np.unique(np.concatenate([take, np.asarray(eq_freqs, dtype=float)]))
            X_new.sort()

            if (
                len(X_new) == len(X)
                and np.allclose(X_new, X)
                and np.max(np.abs(Ew_grid)) <= abs(Delta) + self.inner_tol
            ):
                return coeffs, Delta, X_new

            X = X_new

        return coeffs, Delta, X

    def _compute_max_abs_on_domain(
        self,
        coeffs: np.ndarray,
        ks: np.ndarray,
    ) -> float:
        w_all = np.linspace(self.domain[0], self.domain[1], self.metrics_grid_size, endpoint=True)
        H_all = self._eval_cos_idx(coeffs, w_all, ks)
        return float(np.max(np.abs(H_all)))

    def _max_fit_error_on_intervals(
        self,
        coeffs: np.ndarray,
        ks: np.ndarray,
        intervals: Sequence[Interval],
    ) -> float:
        w = union_grid(intervals, self.metrics_interval_grid_size)
        if len(w) == 0:
            return 0.0
        H = self._eval_cos_idx(coeffs, w, ks)
        return float(np.max(np.abs(H - self.target(w))))

    def _per_interval_fit_errors(
        self,
        coeffs: np.ndarray,
        ks: np.ndarray,
        intervals: Sequence[Interval],
        prefix: str,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, interval in enumerate(intervals):
            out[f"{prefix}_{i}"] = self._max_fit_error_on_intervals(coeffs, ks, [interval])
        return out

    def _constraint_metrics(
        self,
        coeffs: np.ndarray,
        ks: np.ndarray,
    ) -> float:
        w_constraint = union_grid(self.omega_constraint, self.metrics_interval_grid_size)
        if len(w_constraint) == 0:
            return 0.0
        H_constraint = self._eval_cos_idx(coeffs, w_constraint, ks)
        viol_lo = np.maximum(self.lower_bound(w_constraint) - H_constraint, 0.0)
        viol_hi = np.maximum(H_constraint - self.upper_bound(w_constraint), 0.0)
        return float(np.max(np.maximum(viol_lo, viol_hi)))

    def _build_metrics(
        self,
        coeffs: np.ndarray,
        ks: np.ndarray,
        d: int,
        Delta: float,
        raw_max_magnitude: float,
        scaling_factor: float,
        kind: str,
    ) -> Dict[str, float]:
        max_abs_on_domain = self._compute_max_abs_on_domain(coeffs, ks)

        metrics: Dict[str, float] = {
            "kind": kind,
            "degree": int(d),
            "num_coefficients": int(len(coeffs)),
            "ripple_amplitude": float(Delta),
            "raw_max_magnitude": float(raw_max_magnitude),
            "scaling_factor": float(scaling_factor),
            "max_abs_on_domain": float(max_abs_on_domain),
            "max_fit_error_on_fit": self._max_fit_error_on_intervals(coeffs, ks, self.omega_fit),
            "max_fit_error_on_work": self._max_fit_error_on_intervals(coeffs, ks, self.omega_work),
            "max_constraint_violation": self._constraint_metrics(coeffs, ks),
        }
        metrics.update(self._per_interval_fit_errors(coeffs, ks, self.omega_fit, "fit_interval_error"))
        metrics.update(self._per_interval_fit_errors(coeffs, ks, self.omega_work, "work_interval_error"))
        return metrics

    def fit(
        self,
        d: int,
        X0: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        ks = self.parity_indices(d)
        omega_work_grid = union_grid(self.omega_work, self.passband_grid_per_interval)
        omega_constraint_grid = union_grid(self.omega_constraint, self.stopband_grid_per_interval)

        alpha = float(self.alpha0)
        X_prev = np.array([], dtype=float) if X0 is None else np.asarray(X0, dtype=float)
        coeffs = np.zeros(len(ks), dtype=float)
        Delta = 0.0

        for _ in range(self.max_outer):
            if len(omega_constraint_grid) > 0:
                ext_c = self._roots_of_f_on_intervals(
                    lambda w: self._hp_idx(coeffs, w, ks),
                    self.omega_constraint,
                )
                cand_c = np.unique(np.concatenate([ext_c, interval_edges(self.omega_constraint)]))
                Hcand = self._eval_cos_idx(coeffs, cand_c, ks)
                Lc = self.lower_bound(cand_c)
                Uc = self.upper_bound(cand_c)

                mask_lo = Hcand < Lc
                mask_hi = Hcand > Uc
                eq_freqs = np.concatenate([cand_c[mask_lo], cand_c[mask_hi]])
                eq_vals = np.concatenate([Lc[mask_lo], Uc[mask_hi]])
            else:
                eq_freqs = np.array([], dtype=float)
                eq_vals = np.array([], dtype=float)

            ncoef = len(ks)
            m_needed = max(0, ncoef + 1 - len(eq_freqs))

            if X_prev.size == 0:
                if m_needed > 0:
                    idx = np.linspace(0, len(omega_work_grid) - 1, m_needed, dtype=int)
                    X_init = omega_work_grid[idx]
                else:
                    X_init = np.array([], dtype=float)
            else:
                X_init = X_prev

            coeffs, Delta, X_final = self._mremez_equation_constrained_union(
                ks=ks,
                X_init=X_init,
                eq_freqs=eq_freqs,
                eq_values=eq_vals,
                omega_work_grid=omega_work_grid,
            )

            if len(omega_constraint_grid) > 0:
                Hc = self._eval_cos_idx(coeffs, omega_constraint_grid, ks)
                viol_lo = np.maximum(self.lower_bound(omega_constraint_grid) - Hc, 0.0)
                viol_hi = np.maximum(Hc - self.upper_bound(omega_constraint_grid), 0.0)
                max_viol = float(np.max(np.maximum(viol_lo, viol_hi)))
            else:
                max_viol = 0.0

            alpha_new = max_viol

            if abs(alpha_new - alpha) <= self.rel_tol * max(1.0, abs(alpha)):
                X_prev = X_final
                break

            alpha = alpha_new
            X_prev = X_final

        raw_coeffs = coeffs.copy()
        raw_Delta = float(Delta)
        raw_max_magnitude = self._compute_max_abs_on_domain(raw_coeffs, ks)
        scaling_factor = max(1.0, raw_max_magnitude)

        scaled_coeffs = raw_coeffs / scaling_factor
        scaled_Delta = raw_Delta / scaling_factor

        raw_metrics = self._build_metrics(
            raw_coeffs,
            ks,
            d,
            raw_Delta,
            raw_max_magnitude=raw_max_magnitude,
            scaling_factor=1.0,
            kind="raw",
        )
        scaled_metrics = self._build_metrics(
            scaled_coeffs,
            ks,
            d,
            scaled_Delta,
            raw_max_magnitude=raw_max_magnitude,
            scaling_factor=scaling_factor,
            kind="scaled",
        )

        result = {
            "degree": int(d),
            "ks": ks.copy(),
            "extremals": X_prev.copy(),
            "raw": {
                "coeffs": raw_coeffs.copy(),
                "Delta": raw_Delta,
                "metrics": raw_metrics,
            },
            "scaled": {
                "coeffs": scaled_coeffs.copy(),
                "Delta": scaled_Delta,
                "metrics": scaled_metrics,
            },
        }

        self._last_result = result
        return result

    def evaluate(
        self,
        coeffs: np.ndarray,
        omega: np.ndarray,
        d: Optional[int] = None,
        ks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if ks is None:
            if d is None:
                if self._last_result is None:
                    raise ValueError("Provide either d or ks when no fitted result is stored.")
                ks = self._last_result["ks"]
            else:
                ks = self.parity_indices(d)
        return self._eval_cos_idx(np.asarray(coeffs, dtype=float), omega, ks)

    def make_evaluator(
        self,
        coeffs: np.ndarray,
        d: Optional[int] = None,
        ks: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        coeffs = np.asarray(coeffs, dtype=float).copy()
        if ks is None:
            if d is None:
                if self._last_result is None:
                    raise ValueError("Provide either d or ks when no fitted result is stored.")
                ks = self._last_result["ks"]
            else:
                ks = self.parity_indices(d)
        ks = np.asarray(ks, dtype=float).copy()
        return lambda omega: self._eval_cos_idx(coeffs, np.asarray(omega, dtype=float), ks)

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result


def evaluate_from_coeffs(
    fitter: ConstrainedRemezQSPFitter,
    coeffs: np.ndarray,
    omega: np.ndarray,
    d: Optional[int] = None,
    ks: Optional[np.ndarray] = None,
) -> np.ndarray:
    return fitter.evaluate(coeffs=coeffs, omega=omega, d=d, ks=ks)


def make_evaluator(
    fitter: ConstrainedRemezQSPFitter,
    coeffs: np.ndarray,
    d: Optional[int] = None,
    ks: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    return fitter.make_evaluator(coeffs=coeffs, d=d, ks=ks)


def _resolve_plot_result(
    result: Dict[str, Any],
    which: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    which = which.lower()
    if which == "raw":
        return [("raw", result["raw"])]
    if which == "scaled":
        return [("scaled", result["scaled"])]
    if which == "both":
        return [("raw", result["raw"]), ("scaled", result["scaled"])]
    raise ValueError("which must be one of {'raw', 'scaled', 'both'}.")


def visualize_fit(
    fitter: ConstrainedRemezQSPFitter,
    result: Dict[str, Any],
    which: str = "scaled",
    n_plot: int = 5000,
    x_space: bool = False
) -> None:
    ks = result["ks"]
    blocks = _resolve_plot_result(result, which)

    w = np.linspace(fitter.domain[0], fitter.domain[1], n_plot, endpoint=True)
    g = fitter.target(w)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    g_masked = np.full_like(g, np.nan)
    for a0, b0 in fitter.omega_fit:
        mask = (w >= a0) & (w <= b0)
        g_masked[mask] = g[mask]

    if x_space:
        axes[0].plot(np.cos(w), g_masked, "--", label="target on x_fit")
    else:
        axes[0].plot(w / np.pi, g_masked, "--", label="target on omega_fit")
    axes[0].axhline(1.0, linestyle="--")
    axes[0].axhline(-1.0, linestyle="--")

    for label, block in blocks:
        H = fitter.evaluate(block["coeffs"], w, ks=ks)
        if x_space:
            axes[0].plot(np.cos(w), H, label=f"h(x) [{label}]")
        else:
            axes[0].plot(w / np.pi, H, label=f"H(w) [{label}]")

        err = np.abs(H - g)
        err_masked = np.full_like(err, np.nan)
        for a0, b0 in fitter.omega_fit:
            mask = (w >= a0) & (w <= b0)
            err_masked[mask] = err[mask]
        if x_space:
            axes[1].plot(np.cos(w), err_masked, label=f"|h-g| on x_fit [{label}]")
        else:
            axes[1].plot(w / np.pi, err_masked, label=f"|H-g| on omega_fit [{label}]")

    for k, (a0, b0) in enumerate(fitter.omega_fit):
        if x_space:
            axes[0].axvspan(np.cos(b0), np.cos(a0), alpha=0.18, label="x_fit" if k == 0 else None)
        else:
            axes[0].axvspan(a0 / np.pi, b0 / np.pi, alpha=0.18, label="omega_fit" if k == 0 else None)
    for k, (a0, b0) in enumerate(fitter.omega_work):
        if x_space:
            axes[0].axvspan(np.cos(b0), np.cos(a0), alpha=0.08, color="gray", label="x_work" if k == 0 else None)
        else:
            axes[0].axvspan(a0 / np.pi, b0 / np.pi, alpha=0.08, color="gray", label="omega_work" if k == 0 else None)

    for a0, b0 in fitter.omega_fit:
        if x_space:
            axes[1].axvspan(np.cos(b0), np.cos(a0), alpha=0.18)
        else:
            axes[1].axvspan(a0 / np.pi, b0 / np.pi, alpha=0.18)
    for a0, b0 in fitter.omega_work:
        if x_space:
            axes[1].axvspan(np.cos(b0), np.cos(a0), alpha=0.08, color="gray")

        else:
            axes[1].axvspan(a0 / np.pi, b0 / np.pi, alpha=0.08, color="gray")

    axes[0].set_ylabel("value")
    if x_space:
        axes[1].set_xlabel(r"$x$")
    else:
        axes[1].set_xlabel(r"$\omega/\pi$")
    axes[1].set_ylabel("error")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def visualize_fit_separate(
    fitter: ConstrainedRemezQSPFitter,
    result: Dict[str, Any],
    n_plot: int = 5000,
) -> None:
    for which in ("raw", "scaled"):
        visualize_fit(fitter, result, which=which, n_plot=n_plot)
