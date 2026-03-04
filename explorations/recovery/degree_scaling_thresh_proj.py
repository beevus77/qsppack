# import necessary dependencies
import numpy as np
from qsppack.utils import cvx_poly_coef, chebyshev_to_func
from qsppack.solver import solve
from qsppack.utils import get_entry
import matplotlib.pyplot as plt
from time import time


delta = 0.05
targ = np.vectorize(lambda x: 1 if np.abs(x) < 0.5 else 0)
deg = 100
parity = deg % 2

opts = {
    'intervals': [0, 0.5-delta, 0.5+delta, 1],
    'objnorm': np.inf,
    'epsil': 0,
    'npts': 100,
    'fscale': 1,
    'maxiter': 100,
    'isplot': False,
    'method': 'cvxpy'
}
coef_full = cvx_poly_coef(targ, deg, opts)
coef = coef_full[parity::2]


opts.update({
    'N': 2**9,
    'method': 'NLFT',
    'targetPre': False,
    'typePhi': 'reduced'})
phi_proc, out = solve(coef, parity, opts)
out['typePhi'] = 'full'


xlist = np.linspace(0, 1, 1000)
func = lambda x: chebyshev_to_func(x, coef, parity, True)
targ_value = targ(xlist)
func_value = func(xlist)
QSP_value = get_entry(xlist, phi_proc, out)


# plt.plot(xlist, targ(xlist)*opts['fscale'], label='True')
# plt.plot(xlist, func_value, label='Polynomial Approximation')
# plt.plot(xlist, QSP_value, label='QSP')
# plt.plot(xlist, np.ones(len(xlist)), "k--", label='Constraint')
# plt.plot(xlist, -np.ones(len(xlist)), "k--")
# plt.xlabel('$x$', fontsize=12)
# plt.grid()
# plt.legend(loc="best")
# plt.ylim([-0.1, 1.1])
# plt.xlim([0, 1])
# plt.show()

mask1 = xlist < 0.5 - delta
mask2 = xlist > 0.5 + delta
plt.plot(xlist[mask1], np.abs(QSP_value[mask1]-func_value[mask1]), color='blue')
plt.plot(xlist[mask2], np.abs(QSP_value[mask2]-func_value[mask2]), color='blue')
plt.yscale("log")
plt.xlim([0, 1])
plt.show()