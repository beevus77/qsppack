import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from qsppack.remez import (
    ConstrainedRemezFitter,
    RemezConvergenceError,
    RemezConvergenceWarning,
    RemezOptions,
    plot_remez_result,
    remez,
)


def threshold_target(x):
    x = np.asarray(x)
    return (x <= 0.45).astype(float)


def test_exact_odd_polynomial_and_coefficient_order():
    result = remez(
        lambda x: 0.5 * x,
        degree=1,
        fit_intervals=[(0.0, 1.0)],
        target_derivative=lambda x: np.full_like(x, 0.5),
        strict=True,
    )

    np.testing.assert_allclose(result.coefficients, [0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.parity_coefficients, [0.5], atol=1e-12)
    np.testing.assert_allclose(result.evaluate([0.0, 0.5, 1.0]), [0.0, 0.25, 0.5])
    assert result.parity == 1
    assert result.converged


def test_target_scale_metrics_use_original_target():
    result = remez(
        lambda x: 0.5 * x,
        degree=1,
        fit_intervals=[(0.0, 1.0)],
        target_scale=0.9,
        target_derivative=lambda x: np.full_like(x, 0.5),
        rescale=False,
        strict=True,
    )

    np.testing.assert_allclose(result.coefficients, [0.0, 0.45], atol=1e-12)
    assert result.metrics.max_error == pytest.approx(0.05, abs=1e-12)


def test_rescaling_is_configurable_and_enforces_bound():
    scaled = remez(
        threshold_target,
        degree=16,
        fit_intervals=[(0.0, 0.45), (0.55, 1.0)],
        target_scale=1.0 - 1e-6,
    )
    raw = remez(
        threshold_target,
        degree=16,
        fit_intervals=[(0.0, 0.45), (0.55, 1.0)],
        target_scale=1.0 - 1e-6,
        rescale=False,
    )

    assert scaled.raw_metrics.max_magnitude > 1.0
    assert scaled.metrics.max_magnitude <= 1.0 + 1e-10
    assert scaled.metrics.max_constraint_violation <= 1e-10
    assert scaled.scale_factor > 1.0
    assert scaled.ripple_amplitude == pytest.approx(
        scaled.raw_ripple_amplitude / scaled.scale_factor
    )
    assert raw.ripple_amplitude == pytest.approx(raw.raw_ripple_amplitude)
    np.testing.assert_allclose(scaled.coefficients, scaled.scaled_coefficients)
    np.testing.assert_allclose(raw.coefficients, raw.raw_coefficients)
    np.testing.assert_allclose(scaled.raw_coefficients, raw.raw_coefficients)
    np.testing.assert_allclose(scaled.coefficients[1::2], 0.0, atol=1e-14)


def test_matrix_inversion_fit_is_finite_and_feasible():
    kappa = 5.0
    target = lambda x: 1.0 / (2.0 * kappa * x)
    derivative = lambda x: -1.0 / (2.0 * kappa * x**2)
    result = remez(
        target,
        degree=17,
        fit_intervals=[(1.0 / (2.0 * kappa), 1.0)],
        target_scale=1.0 - 1e-6,
        target_derivative=derivative,
    )

    assert np.all(np.isfinite(result.coefficients))
    assert result.metrics.max_constraint_violation <= 1e-10
    assert result.metrics.max_error < 0.2


def test_singular_value_amplification_returns_diagnostic_for_unstable_fit():
    gamma = 0.2
    target = lambda x: x / gamma
    derivative = lambda x: np.full_like(x, 1.0 / gamma)
    with pytest.warns(RemezConvergenceWarning):
        result = remez(
            target,
            degree=9,
            fit_intervals=[(0.0, gamma)],
            work_intervals=[(0.0, gamma - 1.33e-4)],
            target_derivative=derivative,
        )

    assert not result.converged
    assert np.all(np.isfinite(result.coefficients))
    assert result.metrics.max_constraint_violation <= 1e-10
    assert result.message != "Converged."


def test_work_intervals_must_be_inside_fit_intervals():
    with pytest.raises(ValueError, match="subsets"):
        ConstrainedRemezFitter(
            target=lambda x: x,
            fit_intervals=[(0.0, 0.5)],
            work_intervals=[(0.0, 0.75)],
        )


def test_iteration_limit_warns_and_records_status():
    options = RemezOptions(
        max_iterations=1,
        max_exchange_iterations=1,
        fit_grid_size=101,
        constraint_grid_size=101,
        root_grid_size=101,
        metrics_grid_size=101,
    )
    with pytest.warns(RemezConvergenceWarning):
        result = remez(
            threshold_target,
            degree=8,
            fit_intervals=[(0.0, 0.45), (0.55, 1.0)],
            options=options,
        )
    assert not result.converged
    assert "iteration limit" in result.message


def test_strict_iteration_limit_raises():
    options = RemezOptions(
        max_iterations=1,
        max_exchange_iterations=1,
        fit_grid_size=101,
        constraint_grid_size=101,
        root_grid_size=101,
        metrics_grid_size=101,
    )
    with pytest.raises(RemezConvergenceError, match="iteration limit"):
        remez(
            threshold_target,
            degree=8,
            fit_intervals=[(0.0, 0.45), (0.55, 1.0)],
            options=options,
            strict=True,
        )


def test_plot_helper_returns_two_axes():
    result = remez(
        lambda x: 0.5 * x,
        degree=1,
        fit_intervals=[(0.0, 1.0)],
        target_derivative=lambda x: np.full_like(x, 0.5),
        strict=True,
    )
    figure, axes = plot_remez_result(result, lambda x: 0.5 * x, points=101)

    assert len(axes) == 2
    figure.canvas.draw()
    plt.close(figure)


def test_result_variants_are_evaluated_explicitly():
    result = remez(
        threshold_target,
        degree=8,
        fit_intervals=[(0.0, 0.45), (0.55, 1.0)],
        target_scale=1.0 - 1e-6,
    )
    x = np.linspace(0.0, 1.0, 31)

    np.testing.assert_allclose(
        result.evaluate(x, "raw"),
        np.polynomial.chebyshev.chebval(x, result.raw_coefficients),
    )
    np.testing.assert_allclose(
        result.evaluate(x, "scaled"),
        np.polynomial.chebyshev.chebval(x, result.scaled_coefficients),
    )
    np.testing.assert_allclose(
        result.evaluate(x),
        np.polynomial.chebyshev.chebval(x, result.coefficients),
    )
    with pytest.raises(ValueError, match="variant"):
        result.evaluate(x, "unknown")


def test_custom_weight_derivatives_bounds_and_initial_extremals():
    fitter = ConstrainedRemezFitter(
        target=lambda x: 0.25 * x,
        fit_intervals=[(0.0, 1.0)],
        lower_bound=-0.8,
        upper_bound=0.8,
        weight=lambda x: 1.0 + x,
        target_derivative=lambda x: np.full_like(x, 0.25),
        weight_derivative=lambda x: np.ones_like(x),
    )

    result = fitter.fit(
        degree=1,
        strict=True,
        initial_extremals=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(result.coefficients, [0.0, 0.25], atol=1e-12)
    assert result.metrics.max_constraint_violation == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"relative_tolerance": 0}, "relative_tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"fit_grid_size": True}, "fit_grid_size"),
        ({"exchange_tolerance": 0}, "exchange_tolerance"),
    ],
)
def test_remez_options_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RemezOptions(**kwargs).validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fit_intervals": []}, "nonempty"),
        ({"fit_intervals": [(0.5, 0.0)]}, "ordered"),
        ({"fit_intervals": [(-0.1, 0.5)]}, r"\[0, 1\]"),
        ({"fit_intervals": [(0.0, 1.0)], "target_scale": 0}, "target_scale"),
        (
            {"fit_intervals": [(0.0, 1.0)], "lower_bound": 1, "upper_bound": -1},
            "lower_bound",
        ),
    ],
)
def test_remez_input_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConstrainedRemezFitter(target=lambda x: x, **kwargs)


def test_remez_initial_extremal_and_degree_validation():
    fitter = ConstrainedRemezFitter(lambda x: x, [(0.0, 1.0)])

    with pytest.raises(ValueError, match="degree"):
        fitter.fit(0)
    with pytest.raises(ValueError, match="initial_extremals"):
        fitter.fit(1, initial_extremals=[1.1])


def test_plot_helper_supports_custom_axes_and_log_scale():
    result = remez(
        lambda x: 0.5 * x,
        degree=1,
        fit_intervals=[(0.0, 1.0)],
        target_derivative=lambda x: np.full_like(x, 0.5),
        strict=True,
    )
    figure, axes = plt.subplots(1, 2)

    returned_figure, returned_axes = plot_remez_result(
        result,
        lambda x: 0.5 * x + 0.01,
        axes=axes,
        error_scale="log",
        points=101,
    )

    assert returned_figure is figure
    assert returned_axes == tuple(axes)
    assert axes[1].get_yscale() == "log"
    with pytest.raises(ValueError, match="error_scale"):
        plot_remez_result(result, lambda x: x, error_scale="invalid")
    with pytest.raises(ValueError, match="points"):
        plot_remez_result(result, lambda x: x, points=1)
    with pytest.raises(ValueError, match="exactly two"):
        plot_remez_result(result, lambda x: x, axes=[axes[0]])
    plt.close(figure)
