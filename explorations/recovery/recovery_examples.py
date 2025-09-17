# import necessary dependencies
import numpy as np
from qsppack.utils import cvx_poly_coef
from qsppack.solver import solve
from qsppack.utils import get_entry
import matplotlib.pyplot as plt
from time import time


# initialize array of degrees
degs = 2*np.unique(np.int32(2.**np.linspace(2., 7.5, 40)))+1
print(f'degs = {degs}')
rec_errs = np.zeros(len(degs))
std_errs = np.zeros(len(degs))
times_rec = np.zeros(len(degs))
times_std = np.zeros(len(degs))



for i, deg in enumerate(degs):
    a = 0.2
    targ = lambda x: x / a
    parity = deg % 2

    print(f'Doing recovery method without oversampling for deg = {deg}...')

    opts = {
        'intervals': [0, a],
        'objnorm': np.inf,
        'epsil': 0,
        'npts': deg,
        'isplot': False,
        'fscale': 1,
        'method': 'cvxpy'
    }
    start_time = time()
    coef_full = cvx_poly_coef(targ, deg, opts)
    coef = coef_full[parity::2]


    opts.update({
        'N': 2**10,
        'method': 'NLFT',
        'targetPre': False,
        'typePhi': 'reduced'})
    phi_proc, out = solve(coef, parity, opts)
    times_rec[i] = time() - start_time
    out['typePhi'] = 'full'


    test_xlist = np.linspace(0, a, 1000)
    targ_value = targ(test_xlist)
    QSP_value = get_entry(test_xlist, phi_proc, out)
    rec_errs[i] = np.linalg.norm(QSP_value - targ_value, np.inf)
    print(f'The recovered error is {rec_errs[i]}\n')


    print(f'Doing standard method with oversampling for deg = {deg}...')

    opts = {
        'intervals': [0, a],
        'objnorm': np.inf,
        'epsil': 0,
        'npts': 2000,
        'isplot': False,
        'fscale': 1,
        'method': 'cvxpy'
    }
    start_time = time()
    coef_full = cvx_poly_coef(targ, deg, opts)
    coef = coef_full[parity::2]


    opts.update({
        'N': 2**10,
        'method': 'NLFT',
        'targetPre': False,
        'typePhi': 'reduced'})
    phi_proc, out = solve(coef, parity, opts)
    times_std[i] = time() - start_time
    out['typePhi'] = 'full'


    test_xlist = np.linspace(0, a, 1000)
    targ_value = targ(test_xlist)
    QSP_value = get_entry(test_xlist, phi_proc, out)
    std_errs[i] = np.linalg.norm(QSP_value - targ_value, np.inf)
    print(f'The standard error is {std_errs[i]}\n')


np.savez('recovered_errs_times4.npz', degs=degs, rec_errs=rec_errs, std_errs=std_errs, times_rec=times_rec, times_std=times_std)

plt.loglog(degs, rec_errs, '--', label='Recovered', color='red')
plt.loglog(degs, std_errs, '--', label='Oversampled', color='blue')
plt.scatter(degs, rec_errs, color='red', marker='x')
plt.scatter(degs, std_errs, color='blue', marker='x')
plt.xlabel('Degree')
plt.ylabel('Error')
plt.legend(loc='best')
plt.title('Oversampled vs Recovered Error Asymptotic Scaling')
plt.grid()
plt.show()

plt.loglog(degs, times_rec, '--', label='Recovered', color='red')
plt.loglog(degs, times_std, '--', label='Oversampled', color='blue')
plt.scatter(degs, times_rec, color='red', marker='x')
plt.scatter(degs, times_std, color='blue', marker='x')
plt.xlabel('Degree')
plt.ylabel('Time')
plt.legend(loc='best')
plt.title('Oversampled vs Recovered Time Asymptotic Scaling')
plt.grid()
plt.show()



