# import necessary dependencies
import numpy as np
from numpy.polynomial import chebyshev as cheb
from matplotlib import pyplot as plt
from qsppack.nlfa import b_from_cheb, weiss, inverse_nonlinear_FFT, forward_nonlinear_FFT
from qsppack.utils import cvx_poly_coef

# fix degree and target function
deg = 101  # ACHTUNG: recovered_coeffs only works for odd parity for now
a = 0.2
epsil = 1e-4
targ = lambda x: (1-epsil) * x / a
parity = deg % 2
tolerance = 1e-8

# perform coefficient optimization
opts = {
        'intervals': [0, a],
        'objnorm': np.inf,
        'epsil': epsil,
            'npts': deg,
        'isplot': False,
        'fscale': 1,
        'method': 'cvxpy'
    }
coef_full = cvx_poly_coef(targ, deg, opts)

# plot fit
x_values = np.linspace(-1, 1, 1000)
targ_values = targ(x_values)
coef_values = cheb.chebval(x_values, coef_full)
eps = np.max(np.max(np.abs(coef_values))**2 - 1, 0)
plt.plot(x_values, targ_values, 'b-', label='True')
plt.plot(x_values, coef_values, 'r-', label='Fitting Polynomial')
plt.plot(x_values, coef_values / np.sqrt(1+eps), 'g--', label='Scaled Polynomial')
plt.grid(True, alpha=0.3)
plt.xlim([-1, 1])
plt.ylim([-1.1, 1.1])
plt.legend()
plt.show()