"""Tests for the utils module."""

import numpy as np
import pytest
from qsppack import get_entry, get_unitary, reduced_to_full, chebyshev_to_func
from qsppack.utils import (
    F,
    F_Jacobian,
    get_pim_sym,
    get_pim_sym_real,
    get_unitary_sym,
)

def test_get_unitary():
    """Test get_unitary function with basic inputs."""
    phase = np.array([0.0, np.pi/4])
    x = 0.5
    
    result = get_unitary(phase, x)
    
    assert isinstance(result, float)
    assert -1 <= result <= 1  # Result should be a real number between -1 and 1

def test_reduced_to_full():
    """Test reduced_to_full function with basic inputs."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 1
    targetPre = True
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm)  # For odd parity
    assert np.all(np.isfinite(result))  # Check for valid numbers

def test_chebyshev_to_func():
    """Test chebyshev_to_func with basic inputs."""
    x = np.array([0.0, 0.5, 1.0])
    coef = np.array([1.0])
    parity = 1
    partialcoef = True
    
    result = chebyshev_to_func(x, coef, parity, partialcoef)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == len(x)
    assert np.all(np.isfinite(result))  # Check for valid numbers


def test_chebyshev_to_func_accepts_scalar_input():
    """The scalar input documented by chebyshev_to_func should work."""
    result = chebyshev_to_func(0.5, np.array([1.0]), 1, True)

    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize('parity', [0, 1])
@pytest.mark.parametrize('use_real', [False, True])
def test_phase_map_matches_jacobian_value(parity, use_real):
    """F and F_Jacobian must evaluate the same symmetric QSP map."""
    phi = np.array([0.12, -0.08, 0.03])
    opts = {'useReal': use_real}

    value = F(phi, parity, opts)
    jacobian_value, _ = F_Jacobian(phi, parity, opts)

    np.testing.assert_allclose(value, jacobian_value, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('parity', [0, 1])
def test_real_and_complex_phase_maps_agree(parity):
    """Real and complex implementations should encode the same map."""
    phi = np.array([0.12, -0.08, 0.03])
    x = 0.37

    assert get_pim_sym_real(phi, x, parity) == pytest.approx(
        get_pim_sym(phi, x, parity), abs=1e-12
    )


@pytest.mark.parametrize('parity', [0, 1])
def test_symmetric_unitary_matches_full_phase_evaluation(parity):
    """The reduced L-BFGS representation should match the public full one."""
    reduced_phi = np.array([0.12, -0.08, 0.03])
    lbfgs_phi = reduced_phi.copy()
    if parity == 0:
        lbfgs_phi[0] *= 2
    x = 0.37

    symmetric_value = np.real(get_unitary_sym(lbfgs_phi, x, parity)[0, 0])
    full_phi = reduced_to_full(reduced_phi, parity, True)
    full_value = get_unitary(full_phi, x)

    assert symmetric_value == pytest.approx(full_value, abs=1e-12)


def test_get_entry_does_not_mutate_full_phases():
    phases = np.array([0.1, 0.2, 0.1])
    original = phases.copy()

    get_entry(
        np.linspace(-1.0, 1.0, 5),
        phases,
        {'typePhi': 'full', 'targetPre': False, 'parity': 0},
    )

    assert phases == pytest.approx(original)


@pytest.mark.parametrize('parity', [0, 1])
def test_phase_map_jacobian_matches_finite_difference(parity):
    """F_Jacobian should differentiate the coefficient map returned by F."""
    phi = np.array([0.12, -0.08, 0.03])
    opts = {'useReal': True}
    _, jacobian = F_Jacobian(phi, parity, opts)
    step = 1e-7
    finite_difference = np.empty_like(jacobian)

    for column in range(len(phi)):
        perturbation = np.zeros_like(phi)
        perturbation[column] = step
        finite_difference[:, column] = (
            F(phi + perturbation, parity, opts)
            - F(phi - perturbation, parity, opts)
        ) / (2 * step)

    np.testing.assert_allclose(jacobian, finite_difference, atol=1e-8, rtol=1e-8)

def test_reduced_to_full_even_parity():
    """Test reduced_to_full function with even parity."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = True
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_False():
    """Test reduced_to_full function with even parity and targetPre set to False."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = False
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_True():
    """Test reduced_to_full function with even parity and targetPre set to True."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = True
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_None():
    """Test reduced_to_full function with even parity and targetPre set to None."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = None
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_empty():
    """Test reduced_to_full function with even parity and targetPre set to empty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = []
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_zero():
    """Test reduced_to_full function with even parity and targetPre set to zero."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 0
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_one():
    """Test reduced_to_full function with even parity and targetPre set to one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 1
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_two():
    """Test reduced_to_full function with even parity and targetPre set to two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 2
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_three():
    """Test reduced_to_full function with even parity and targetPre set to three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 3
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_four():
    """Test reduced_to_full function with even parity and targetPre set to four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 4
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_five():
    """Test reduced_to_full function with even parity and targetPre set to five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 5
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_six():
    """Test reduced_to_full function with even parity and targetPre set to six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 6
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seven():
    """Test reduced_to_full function with even parity and targetPre set to seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 7
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eight():
    """Test reduced_to_full function with even parity and targetPre set to eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 8
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_nine():
    """Test reduced_to_full function with even parity and targetPre set to nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 9
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ten():
    """Test reduced_to_full function with even parity and targetPre set to ten."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 10
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eleven():
    """Test reduced_to_full function with even parity and targetPre set to eleven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 11
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twelve():
    """Test reduced_to_full function with even parity and targetPre set to twelve."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 12
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirteen():
    """Test reduced_to_full function with even parity and targetPre set to thirteen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 13
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fourteen():
    """Test reduced_to_full function with even parity and targetPre set to fourteen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 14
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifteen():
    """Test reduced_to_full function with even parity and targetPre set to fifteen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 15
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixteen():
    """Test reduced_to_full function with even parity and targetPre set to sixteen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 16
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventeen():
    """Test reduced_to_full function with even parity and targetPre set to seventeen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 17
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighteen():
    """Test reduced_to_full function with even parity and targetPre set to eighteen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 18
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_nineteen():
    """Test reduced_to_full function with even parity and targetPre set to nineteen."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 19
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty():
    """Test reduced_to_full function with even parity and targetPre set to twenty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 20
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_one():
    """Test reduced_to_full function with even parity and targetPre set to twenty-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 21
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_two():
    """Test reduced_to_full function with even parity and targetPre set to twenty-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 22
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_three():
    """Test reduced_to_full function with even parity and targetPre set to twenty-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 23
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_four():
    """Test reduced_to_full function with even parity and targetPre set to twenty-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 24
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_five():
    """Test reduced_to_full function with even parity and targetPre set to twenty-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 25
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_six():
    """Test reduced_to_full function with even parity and targetPre set to twenty-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 26
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_seven():
    """Test reduced_to_full function with even parity and targetPre set to twenty-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 27
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_eight():
    """Test reduced_to_full function with even parity and targetPre set to twenty-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 28
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_twenty_nine():
    """Test reduced_to_full function with even parity and targetPre set to twenty-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 29
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty():
    """Test reduced_to_full function with even parity and targetPre set to thirty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 30
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_one():
    """Test reduced_to_full function with even parity and targetPre set to thirty-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 31
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_two():
    """Test reduced_to_full function with even parity and targetPre set to thirty-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 32
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_three():
    """Test reduced_to_full function with even parity and targetPre set to thirty-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 33
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_four():
    """Test reduced_to_full function with even parity and targetPre set to thirty-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 34
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_five():
    """Test reduced_to_full function with even parity and targetPre set to thirty-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 35
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_six():
    """Test reduced_to_full function with even parity and targetPre set to thirty-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 36
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_seven():
    """Test reduced_to_full function with even parity and targetPre set to thirty-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 37
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_eight():
    """Test reduced_to_full function with even parity and targetPre set to thirty-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 38
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_thirty_nine():
    """Test reduced_to_full function with even parity and targetPre set to thirty-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 39
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty():
    """Test reduced_to_full function with even parity and targetPre set to forty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 40
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_one():
    """Test reduced_to_full function with even parity and targetPre set to forty-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 41
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_two():
    """Test reduced_to_full function with even parity and targetPre set to forty-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 42
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_three():
    """Test reduced_to_full function with even parity and targetPre set to forty-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 43
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_four():
    """Test reduced_to_full function with even parity and targetPre set to forty-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 44
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_five():
    """Test reduced_to_full function with even parity and targetPre set to forty-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 45
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_six():
    """Test reduced_to_full function with even parity and targetPre set to forty-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 46
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_seven():
    """Test reduced_to_full function with even parity and targetPre set to forty-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 47
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_eight():
    """Test reduced_to_full function with even parity and targetPre set to forty-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 48
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_forty_nine():
    """Test reduced_to_full function with even parity and targetPre set to forty-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 49
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty():
    """Test reduced_to_full function with even parity and targetPre set to fifty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 50
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_one():
    """Test reduced_to_full function with even parity and targetPre set to fifty-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 51
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_two():
    """Test reduced_to_full function with even parity and targetPre set to fifty-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 52
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_three():
    """Test reduced_to_full function with even parity and targetPre set to fifty-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 53
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_four():
    """Test reduced_to_full function with even parity and targetPre set to fifty-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 54
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_five():
    """Test reduced_to_full function with even parity and targetPre set to fifty-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 55
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_six():
    """Test reduced_to_full function with even parity and targetPre set to fifty-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 56
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_seven():
    """Test reduced_to_full function with even parity and targetPre set to fifty-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 57
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_eight():
    """Test reduced_to_full function with even parity and targetPre set to fifty-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 58
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_fifty_nine():
    """Test reduced_to_full function with even parity and targetPre set to fifty-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 59
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty():
    """Test reduced_to_full function with even parity and targetPre set to sixty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 60
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_one():
    """Test reduced_to_full function with even parity and targetPre set to sixty-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 61
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_two():
    """Test reduced_to_full function with even parity and targetPre set to sixty-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 62
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_three():
    """Test reduced_to_full function with even parity and targetPre set to sixty-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 63
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_four():
    """Test reduced_to_full function with even parity and targetPre set to sixty-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 64
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_five():
    """Test reduced_to_full function with even parity and targetPre set to sixty-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 65
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_six():
    """Test reduced_to_full function with even parity and targetPre set to sixty-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 66
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_seven():
    """Test reduced_to_full function with even parity and targetPre set to sixty-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 67
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_eight():
    """Test reduced_to_full function with even parity and targetPre set to sixty-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 68
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_sixty_nine():
    """Test reduced_to_full function with even parity and targetPre set to sixty-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 69
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy():
    """Test reduced_to_full function with even parity and targetPre set to seventy."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 70
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_one():
    """Test reduced_to_full function with even parity and targetPre set to seventy-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 71
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_two():
    """Test reduced_to_full function with even parity and targetPre set to seventy-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 72
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_three():
    """Test reduced_to_full function with even parity and targetPre set to seventy-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 73
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_four():
    """Test reduced_to_full function with even parity and targetPre set to seventy-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 74
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_five():
    """Test reduced_to_full function with even parity and targetPre set to seventy-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 75
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_six():
    """Test reduced_to_full function with even parity and targetPre set to seventy-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 76
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_seven():
    """Test reduced_to_full function with even parity and targetPre set to seventy-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 77
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_eight():
    """Test reduced_to_full function with even parity and targetPre set to seventy-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 78
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_seventy_nine():
    """Test reduced_to_full function with even parity and targetPre set to seventy-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 79
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty():
    """Test reduced_to_full function with even parity and targetPre set to eighty."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 80
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_one():
    """Test reduced_to_full function with even parity and targetPre set to eighty-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 81
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_two():
    """Test reduced_to_full function with even parity and targetPre set to eighty-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 82
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_three():
    """Test reduced_to_full function with even parity and targetPre set to eighty-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 83
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_four():
    """Test reduced_to_full function with even parity and targetPre set to eighty-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 84
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_five():
    """Test reduced_to_full function with even parity and targetPre set to eighty-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 85
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_six():
    """Test reduced_to_full function with even parity and targetPre set to eighty-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 86
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_seven():
    """Test reduced_to_full function with even parity and targetPre set to eighty-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 87
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_eight():
    """Test reduced_to_full function with even parity and targetPre set to eighty-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 88
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_eighty_nine():
    """Test reduced_to_full function with even parity and targetPre set to eighty-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 89
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety():
    """Test reduced_to_full function with even parity and targetPre set to ninety."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 90
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_one():
    """Test reduced_to_full function with even parity and targetPre set to ninety-one."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 91
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_two():
    """Test reduced_to_full function with even parity and targetPre set to ninety-two."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 92
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_three():
    """Test reduced_to_full function with even parity and targetPre set to ninety-three."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 93
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_four():
    """Test reduced_to_full function with even parity and targetPre set to ninety-four."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 94
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_five():
    """Test reduced_to_full function with even parity and targetPre set to ninety-five."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 95
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_six():
    """Test reduced_to_full function with even parity and targetPre set to ninety-six."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 96
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_seven():
    """Test reduced_to_full function with even parity and targetPre set to ninety-seven."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 97
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_eight():
    """Test reduced_to_full function with even parity and targetPre set to ninety-eight."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 98
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_ninety_nine():
    """Test reduced_to_full function with even parity and targetPre set to ninety-nine."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 99
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))

def test_reduced_to_full_even_parity_with_targetPre_one_hundred():
    """Test reduced_to_full function with even parity and targetPre set to one hundred."""
    phi_cm = np.array([0.0, np.pi/4])
    parity = 0
    targetPre = 100
    
    result = reduced_to_full(phi_cm, parity, targetPre)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 2 * len(phi_cm) - 1  # For even parity
    assert np.all(np.isfinite(result))
