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


def load_fit_data(csv_filename, npts_value=None):
    """
    Load polynomial approximation data from CSV file.

    Parameters:
    -----------
    csv_filename : str
        Path to the .csv file containing the saved data
    npts_value : int, optional
        Specific npts value to load. If None, uses the latest/most recent value.

    Returns:
    --------
    dict : Dictionary containing extracted data (coef_full, coef_recovered_full, metadata)
    """
    # Load CSV data
    df = pd.read_csv(csv_filename)

    # Select row based on npts_value
    if npts_value is None:
        # Use the latest/most recent npts value
        selected_row = df.iloc[-1]
        print(f"CSV: {os.path.basename(csv_filename)} - Using latest npts value: {selected_row['npts']}")
    else:
        # Find specific npts value
        matching_rows = df[df['npts'] == npts_value]
        if len(matching_rows) == 0:
            available_npts = df['npts'].tolist()
            print(f"Error: npts value {npts_value} not found in {csv_filename}")
            print(f"Available npts values: {available_npts}")
            return None
        selected_row = matching_rows.iloc[0]
        print(f"CSV: {os.path.basename(csv_filename)} - Using npts value: {selected_row['npts']}")

    # Extract data from selected row
    parity = int(selected_row['parity'])
    degree = int(selected_row['degree'])
    npts = int(selected_row['npts'])
    convergence_diff = float(selected_row['convergence_diff'])
    coef_full = np.zeros(degree+1)
    coef_full[parity::2] = np.array(json.loads(selected_row['coefs_json']))
    coef_recovered_full = np.array(json.loads(selected_row['coefs_recovered_json']))

    return {
        'degree': degree,
        'npts': npts,
        'parity': parity,
        'convergence_diff': convergence_diff,
        'coef_full': coef_full,
        'coef_recovered_full': coef_recovered_full
    }


def plot_comparison(data_deg25, data_deg101):
    """
    Plot comparison of polynomial fits for both degree 25 and degree 101.

    Parameters:
    -----------
    data_deg25 : dict
        Data dictionary for degree 25 fit
    data_deg101 : dict
        Data dictionary for degree 101 fit
    """
    # Define target function (same as in fgt scripts)
    a = 0.2
    targ = lambda x: 0.99 * x / a

    # Generate plotting data on Chebyshev nodes
    M = 10_000
    print("Computing Chebyshev nodes...")
    xlist = np.cos(np.pi * np.arange(M) / (M - 1))
    print("Evaluating target function...")
    targ_value = targ(xlist)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot degree 25
    print("Evaluating degree 25 polynomials...")
    func_value_25 = chebval_dct(data_deg25['coef_full'], M)
    recovered_value_25 = chebval_dct(data_deg25['coef_recovered_full'], M)

    axes[0].plot(xlist, targ_value, label='True', linewidth=2, color='black')
    axes[0].plot(xlist, func_value_25, label='Polynomial Approximation', linewidth=2, alpha=0.8)
    axes[0].plot(xlist, recovered_value_25, label='Recovered (polynomial space)', linewidth=2, alpha=0.8, linestyle='--')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([-1, 1])
    axes[0].set_ylim([-1.1, 1.1])
    axes[0].legend(loc='best')
    axes[0].set_title(f'Degree 25 (npts={data_deg25["npts"]}, conv_diff={data_deg25["convergence_diff"]:.2e})')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$f(x)$')

    # Plot degree 101
    print("Evaluating degree 101 polynomials...")
    func_value_101 = chebval_dct(data_deg101['coef_full'], M)
    recovered_value_101 = chebval_dct(data_deg101['coef_recovered_full'], M)

    axes[1].plot(xlist, targ_value, label='True', linewidth=2, color='black')
    axes[1].plot(xlist, func_value_101, label='Polynomial Approximation', linewidth=2, alpha=0.8)
    axes[1].plot(xlist, recovered_value_101, label='Recovered (polynomial space)', linewidth=2, alpha=0.8, linestyle='--')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([-1, 1])
    axes[1].set_ylim([-1.1, 1.1])
    axes[1].legend(loc='best')
    axes[1].set_title(f'Degree 101 (npts={data_deg101["npts"]}, conv_diff={data_deg101["convergence_diff"]:.2e})')
    axes[1].set_xlabel(r'$x$')
    axes[1].set_ylabel(r'$f(x)$')

    plt.tight_layout()
    plt.show()

    # Also create an overlay comparison plot
    print("Creating overlay comparison plot...")
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(xlist, targ_value, label='True', linewidth=2.5, color='black', zorder=5)
    ax.plot(xlist, recovered_value_25, label='Recovered Deg 25', linewidth=2, alpha=0.8, linestyle='--')
    ax.plot(xlist, recovered_value_101, label='Recovered Deg 101', linewidth=2, alpha=0.8, linestyle=':')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.1, 1.1])
    ax.legend(loc='best')
    ax.set_title(f'Recovered Polynomial Comparison: Degree 25 vs 101')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Compare polynomial-space fits for degree 25 and degree 101',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default CSV files (auto-detect first available for each degree)
  python compare_poly_fit.py
  
  # Specify specific CSV files for each degree
  python compare_poly_fit.py --csv25 data/fgt_polynomial_space_convergence_deg_25_epsil4_N15.csv \\
                             --csv101 data/fgt_polynomial_space_convergence_deg_101_epsil4_N16.csv
  
  # Specify npts values for each degree
  python compare_poly_fit.py --npts25 25 --npts101 101
        """
    )
    parser.add_argument('--npts25', type=int, default=None,
                       help='Specific npts value for degree 25 (default: latest available)')
    parser.add_argument('--npts101', type=int, default=None,
                       help='Specific npts value for degree 101 (default: latest available)')
    parser.add_argument('--csv25', type=str, default=None,
                       help='Path to CSV file for degree 25 (default: auto-detect)')
    parser.add_argument('--csv101', type=str, default=None,
                       help='Path to CSV file for degree 101 (default: auto-detect)')

    args = parser.parse_args()

    # Determine data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Find or validate CSV for degree 25
    if args.csv25:
        csv_filename_25 = args.csv25
        if not os.path.isabs(csv_filename_25):
            csv_filename_25 = os.path.join(data_dir, csv_filename_25)
    else:
        # Auto-detect degree 25 CSV
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            print("Please ensure data files exist.")
            raise SystemExit(1)
        csv_files_25 = [
            f for f in os.listdir(data_dir)
            if f.startswith('fgt_polynomial_space_convergence_deg_25_epsil4_') and f.endswith('.csv')
        ]
        if not csv_files_25:
            print(f"No degree 25 CSV files found in {data_dir}")
            print("Please generate the data files first.")
            raise SystemExit(1)
        csv_filename_25 = os.path.join(data_dir, csv_files_25[0])
        print(f"Auto-detected degree 25 CSV: {csv_filename_25}")

    # Find or validate CSV for degree 101
    if args.csv101:
        csv_filename_101 = args.csv101
        if not os.path.isabs(csv_filename_101):
            csv_filename_101 = os.path.join(data_dir, csv_filename_101)
    else:
        # Auto-detect degree 101 CSV
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            print("Please ensure data files exist.")
            raise SystemExit(1)
        csv_files_101 = [
            f for f in os.listdir(data_dir)
            if f.startswith('fgt_polynomial_space_convergence_deg_101_epsil4_') and f.endswith('.csv')
        ]
        if not csv_files_101:
            print(f"No degree 101 CSV files found in {data_dir}")
            print("Please generate the data files first.")
            raise SystemExit(1)
        csv_filename_101 = os.path.join(data_dir, csv_files_101[0])
        print(f"Auto-detected degree 101 CSV: {csv_filename_101}")

    # Check if CSV files exist
    if not os.path.exists(csv_filename_25):
        print(f"CSV file not found: {csv_filename_25}")
        raise SystemExit(1)
    if not os.path.exists(csv_filename_101):
        print(f"CSV file not found: {csv_filename_101}")
        raise SystemExit(1)

    # Load data from both CSVs
    print("\n=== Loading Degree 25 Data ===")
    data_deg25 = load_fit_data(csv_filename_25, args.npts25)
    if data_deg25 is None:
        raise SystemExit(1)

    print("\n=== Loading Degree 101 Data ===")
    data_deg101 = load_fit_data(csv_filename_101, args.npts101)
    if data_deg101 is None:
        raise SystemExit(1)

    # Create comparison plots
    print("\n=== Creating Comparison Plots ===")
    plot_comparison(data_deg25, data_deg101)

    print("\nPlotting complete!")

