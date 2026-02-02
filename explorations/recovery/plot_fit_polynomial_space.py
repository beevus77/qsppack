# import necessary dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
import os
import pandas as pd
import json
import argparse

# Set LaTeX fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12


def chebval_dct(c, M):
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
    c_pad[:len(c)] = c
    # DCT-I synthesis expects half-scaled interior coefficients
    if M > 2:
        c_pad[1:M-1] *= 0.5
    return dct(c_pad, type=1, norm=None)


def plot_fit_from_csv(csv_filename, npts_value=None):
    """
    Plot polynomial approximation and recovered polynomial against true function
    using data from CSV file produced by fgt_polynomial_space.py.

    Parameters:
    -----------
    csv_filename : str
        Path to the .csv file containing the saved data
    npts_value : int, optional
        Specific npts value to plot. If None, uses the latest/most recent value.
    """
    # Load CSV data
    df = pd.read_csv(csv_filename)

    # Select row based on npts_value
    if npts_value is None:
        # Use the latest/most recent npts value
        selected_row = df.iloc[-1]
        print(f"Using latest npts value: {selected_row['npts']}")
    else:
        # Find specific npts value
        matching_rows = df[df['npts'] == npts_value]
        if len(matching_rows) == 0:
            available_npts = df['npts'].tolist()
            print(f"Error: npts value {npts_value} not found in CSV.")
            print(f"Available npts values: {available_npts}")
            return
        selected_row = matching_rows.iloc[0]
        print(f"Using npts value: {selected_row['npts']}")

    # Extract data from selected row
    parity = int(selected_row['parity'])
    degree = int(selected_row['degree'])
    npts = int(selected_row['npts'])
    convergence_diff = float(selected_row['convergence_diff'])
    coef_full = np.zeros(degree+1)
    coef_full[parity::2] = np.array(json.loads(selected_row['coefs_json']))
    coef_recovered_full = np.array(json.loads(selected_row['coefs_recovered_json']))

    # Define target function (same as in fgt scripts)
    a = 0.2
    targ = lambda x: 0.99 * x / a

    # Generate plotting data on Chebyshev nodes (1 million samples)
    M = 10_000
    print("Computing Chebyshev nodes...")
    xlist = np.cos(np.pi * np.arange(M) / (M - 1))
    print("Evaluating target function...")
    targ_value = targ(xlist)
    # Original polynomial approximation built from full coefficients via DCT-I
    print("Evaluating original polynomial...")
    func_value = chebval_dct(coef_full, M)
    # Recovered polynomial evaluated via DCT-I
    print("Evaluating recovered polynomial...")
    recovered_value = chebval_dct(coef_recovered_full, M)

    # Create the plot
    print("Creating plot...")
    plt.figure(figsize=(12, 8))
    # Use colorblind-friendly colors: black, blue (safe), and orange
    plt.plot(xlist, targ_value, label='Target', color='black', linewidth=2)
    plt.plot(xlist, func_value, label='Polynomial Approximation', color='#0072B2', linewidth=2)  # Blue
    plt.plot(xlist, recovered_value, label='Retraction', color='#E69F00', linewidth=2, linestyle='--')  # Orange

    plt.grid(True, alpha=0.3)
    plt.xlim([-1, 1])
    plt.ylim([-1.1, 1.1])
    plt.legend(loc='best', framealpha=1)
    # plt.title(f'Polynomial-Space Fit (degree={degree}, npts={npts}, conv_diff={convergence_diff:.2e})')
    plt.xlabel(r'$x$')
    # plt.ylabel(r'$f(x)$')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Plot results from fgt_polynomial_space convergence CSV')
    parser.add_argument('npts', type=int, nargs='?', default=None,
                       help='Specific npts value to plot (default: latest available)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file (default: auto-detect)')

    args = parser.parse_args()

    # Determine data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Determine CSV filename
    if args.csv:
        csv_filename = args.csv
    else:
        # Auto-detect CSV file produced by fgt_polynomial_space.py
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            print("Please run fgt_polynomial_space.py first to generate the data file.")
            raise SystemExit(1)
        csv_files = [
            f for f in os.listdir(data_dir)
            if f.startswith('fgt_polynomial_space_convergence_deg_') and f.endswith('.csv')
        ]
        if not csv_files:
            print(f"No fgt_polynomial_space convergence CSV files found in {data_dir}")
            print("Please run fgt_polynomial_space.py first to generate the data file.")
            raise SystemExit(1)
        # Choose the first match (consistent with existing plot script behavior)
        csv_filename = os.path.join(data_dir, csv_files[0])
        print(f"Auto-detected CSV file: {csv_filename}")

    # Check if CSV exists
    if not os.path.exists(csv_filename):
        print(f"CSV file not found: {csv_filename}")
        print("Please run fgt_polynomial_space.py first to generate the data file.")
        raise SystemExit(1)

    # Plot from CSV
    plot_fit_from_csv(csv_filename, args.npts)


