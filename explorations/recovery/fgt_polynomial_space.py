# import necessary dependencies
import numpy as np
from qsppack.nlfa import b_from_cheb, weiss, inverse_nonlinear_FFT, forward_nonlinear_FFT
from qsppack.utils import cvx_poly_coef
import os
import json
import pandas as pd
import time
import csv
import argparse


def recovered_coeffs(coefs, parity, N):
    """
    Recovered coefficients from the original coefficients using the NLFA algorithms.
    
    Args:
        coefs: Coefficients to recover
        parity: Parity of the polynomial
        N: Parameter for weiss function
    """
    b_coeffs = b_from_cheb(coefs[parity::2], parity)
    a_coeffs = weiss(b_coeffs, N)
    gammas, _, _ = inverse_nonlinear_FFT(a_coeffs, b_coeffs)
    new_a, new_b = forward_nonlinear_FFT(gammas)

    new_coeffs = np.zeros(len(coefs))
    new_coeffs[1::2] = new_b[int(len(new_b)/2-1)::-1] + new_b[int(len(new_b)/2)::]
    return new_coeffs


# Parse command line arguments
parser = argparse.ArgumentParser(description='Find ground truth for polynomial space recovery')
parser.add_argument('--degree', type=int, default=101, help='Degree of polynomial (default: 101)')
parser.add_argument('--epsilon', type=float, default=1e-4, help='Epsilon value (default: 1e-4)')
parser.add_argument('--N', type=int, default=2**15, help='N parameter for weiss function (default: 2^15 = 32768)')
args = parser.parse_args()

# fix degree and target function
deg = args.degree  # ACHTUNG: recovered_coeffs only works for odd parity for now
a = 0.2
epsil = args.epsilon
targ = lambda x: (1-epsil) * x / a
parity = deg % 2
tolerance = 1e-8


# Define expected CSV columns
expected_columns = ['npts', 'degree', 'parity', 'convergence_diff', 'iteration_time', 'coefs_json', 'coefs_recovered_json']

# Setup CSV logging
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)
# Format epsilon for filename (e.g., 1e-4 -> epsil4)
epsil_exp = 0 if epsil == 0 else int(-np.log10(epsil))
N_exp = int(np.log2(args.N))
csv_filename = os.path.join(data_dir, f"fgt_polynomial_space_convergence_deg_{deg}_epsil{epsil_exp}_N{N_exp}.csv")

# Initialize CSV with headers if it doesn't exist
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(expected_columns)
    print(f"Created new CSV file: {csv_filename}")

# Determine starting npts and sequence
def get_next_npts(current_npts, deg):
    """Get next npts value following exponential growth pattern"""
    if current_npts == deg:
        # First iteration: go to next power of 2 above deg
        return 2 ** (int(np.log2(deg)) + 1)
    else:
        # Subsequent iterations: double the current value
        return current_npts * 2

# Initialize convergence tracking
coef_prev = None
iteration = 0
current_npts = deg

# Load existing data to check for duplicates (tolerate extra columns)
existing_data = None
if os.path.exists(csv_filename):
    existing_data = pd.read_csv(csv_filename)
    print(f"Found existing CSV with {len(existing_data)} entries")

    existing_columns = set(existing_data.columns)
    expected_columns_set = set(expected_columns)

    # Proceed if all required columns are present; allow extra columns like 'constraint_violated'
    if not expected_columns_set.issubset(existing_columns):
        missing = sorted(list(expected_columns_set - existing_columns))
        print("Warning: CSV is missing required columns:", missing)
        print("Creating a new CSV with required headers; keeping the old file as backup.")
        backup_filename = csv_filename + ".bak"
        os.replace(csv_filename, backup_filename)
        print(f"Backed up old CSV to: {backup_filename}")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(expected_columns)
        existing_data = None
        existing_npts = []
    else:
        existing_npts = existing_data['npts'].tolist()
        print(f"Existing npts values: {existing_npts}")
else:
    existing_npts = []

print(f"Starting convergence loop for degree {deg}")
print(f"Tolerance: {tolerance}")
print(f"CSV file: {csv_filename}")
print("-" * 50)

while current_npts < 600000:
    iteration += 1
    print(f"Iteration {iteration}: npts = {current_npts}")
    
    # Check if this npts value has already been computed
    if current_npts in existing_npts:
        print(f"  npts={current_npts} already computed, skipping...")
        # Get the existing data for this npts
        existing_row = existing_data[existing_data['npts'] == current_npts].iloc[0]
        print(f"  Previous convergence diff: {existing_row['convergence_diff']:.2e}")
        print(f"  Previous iteration time: {existing_row['iteration_time']:.2f} seconds")
        
        # Update coef_prev for convergence checking
        coef_prev = np.array(json.loads(existing_row['coefs_json']))
        current_npts = get_next_npts(current_npts, deg)
        print("-" * 50)
        continue
    
    # Time the iteration
    start_time = time.time()
    
    # do convex optimization with current npts
    opts = {
        'intervals': [0, a],
        'objnorm': np.inf,
        'epsil': epsil,
            'npts': current_npts,
        'isplot': False,
        'fscale': 1,
        'method': 'cvxpy'
    }

    coef_full = cvx_poly_coef(targ, deg, opts)
    coef_curr = coef_full[parity::2]

    coef_recovered = recovered_coeffs(coef_full, parity, args.N)

    iteration_time = time.time() - start_time
    
    # Calculate convergence difference
    if coef_prev is not None:
        convergence_diff = np.linalg.norm(coef_curr - coef_prev)
        print(f"  Convergence diff: {convergence_diff:.2e}")
        print(f"  Iteration time: {iteration_time:.2f} seconds")
        
        # Check convergence
        if convergence_diff < tolerance:
            print(f"  CONVERGED! Difference {convergence_diff:.2e} < tolerance {tolerance}")
            # Store the final converged iteration
            print(f"  Preparing to append final converged data to CSV...")
            new_row = {
                'npts': current_npts,
                'degree': deg,
                'parity': parity,
                'convergence_diff': convergence_diff,
                'iteration_time': iteration_time,
                'coefs_json': json.dumps(coef_curr.tolist()),
                'coefs_recovered_json': json.dumps(coef_recovered.tolist())
            }
            print(f"  Appending final converged data to CSV...")
            # Determine header columns to honor existing extra columns (e.g., constraint_violated)
            if existing_data is not None:
                header_columns = list(existing_data.columns)
            else:
                header_columns = expected_columns
            # Build a row dict including placeholders for any extra columns
            row_dict = {col: new_row[col] if col in new_row else '' for col in header_columns}
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header_columns, quoting=csv.QUOTE_ALL)
                writer.writerow(row_dict)
            print(f"  Final converged data successfully appended to CSV")
            break
    else:
        convergence_diff = np.inf
        print(f"  Iteration time: {iteration_time:.2f} seconds")
    
    # Prepare data for CSV
    print(f"  Preparing to append data to CSV...")
    new_row = {
        'npts': current_npts,
        'degree': deg,
        'parity': parity,
        'convergence_diff': convergence_diff,
        'iteration_time': iteration_time,
        'coefs_json': json.dumps(coef_curr.tolist()),
        'coefs_recovered_json': json.dumps(coef_recovered.tolist())
    }
    
    # Append to CSV immediately for fault tolerance
    print(f"  Appending data to CSV...")
    # Determine header columns to honor existing extra columns (e.g., constraint_violated)
    if existing_data is not None:
        header_columns = list(existing_data.columns)
    else:
        header_columns = expected_columns
    # Build a row dict including placeholders for any extra columns
    row_dict = {col: new_row[col] if col in new_row else '' for col in header_columns}
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header_columns, quoting=csv.QUOTE_ALL)
        writer.writerow(row_dict)
    print(f"  Data successfully appended to CSV")
    
    # Update for next iteration
    coef_prev = coef_curr.copy()
    current_npts = get_next_npts(current_npts, deg)
    
    print("-" * 50)

print(f"\nConvergence complete!")
print(f"Final results saved to: {csv_filename}")

# Display available npts values
final_data = pd.read_csv(csv_filename)
print(f"Available npts values: {final_data['npts'].tolist()}")

print(f"To plot results, run: python plot_fit_polynomial_space.py [npts_value]")