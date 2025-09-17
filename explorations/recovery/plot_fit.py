# import necessary dependencies
import numpy as np
import matplotlib.pyplot as plt
from qsppack.utils import chebyshev_to_func
from qsppack.utils import get_entry
import os
import pandas as pd
import json
import argparse


def plot_fit_from_csv(csv_filename, npts_value=None):
    """
    Plot polynomial approximation and QSP against true function using data from CSV file.
    
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
    angles = np.array(json.loads(selected_row['angles_json']))
    coef = np.array(json.loads(selected_row['coefs_json']))
    parity = selected_row['parity']
    degree = selected_row['degree']
    npts = selected_row['npts']
    convergence_diff = selected_row['convergence_diff']
    
    # Load the complete out dictionary
    out = json.loads(selected_row['out_json'])
    
    # Define target function (same as in find_ground_truth.py)
    a = 0.2
    targ = lambda x: x / a
    
    # Generate plotting data
    xlist = np.linspace(-1, 1, 1000)
    targ_value = targ(xlist)
    func_value = chebyshev_to_func(xlist, coef, parity, True)
    
    QSP_value = get_entry(xlist, angles, out)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(xlist, targ_value, label='True', linewidth=2)
    plt.plot(xlist, func_value, label='Polynomial Approximation', linewidth=2)
    plt.plot(xlist, QSP_value, label='QSP', linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.xlim([-1, 1])
    plt.ylim([-1.1, 1.1])
    plt.legend(loc='best')
    plt.title(f'QSP Fit Comparison (degree={degree}, npts={npts}, conv_diff={convergence_diff:.2e})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.tight_layout()
    plt.show()


def plot_fit_from_npz(npz_filename):
    """
    Plot polynomial approximation and QSP against true function using data from npz file.
    
    Parameters:
    -----------
    npz_filename : str
        Path to the .npz file containing the saved data
    """
    # Load data from npz file
    data = np.load(npz_filename, allow_pickle=True)
    angles = data['angles']
    coef = data['coefs']
    parity = data['parity']
    out = data['out'].item()  # Convert numpy array back to dict
    degree = data['degree']
    npts = data['npts']
    
    # Define target function (same as in find_ground_truth.py)
    a = 0.2
    targ = lambda x: x / a
    
    # Generate plotting data
    xlist = np.linspace(-1, 1, 1000)
    targ_value = targ(xlist)
    func_value = chebyshev_to_func(xlist, coef, parity, True)
    QSP_value = get_entry(xlist, angles, out)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(xlist, targ_value, label='True')
    plt.plot(xlist, func_value, label='Polynomial Approximation')
    plt.plot(xlist, QSP_value, label='QSP')
    plt.grid()
    plt.xlim([-1, 1])
    plt.ylim([-1.1, 1.1])
    plt.legend(loc='best')
    plt.title(f'QSP Fit Comparison (degree={degree}, npts={npts})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Plot QSP fit results from CSV data')
    parser.add_argument('npts', type=int, nargs='?', default=None,
                       help='Specific npts value to plot (default: latest available)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV file (default: auto-detect)')
    parser.add_argument('--npz', type=str, default=None,
                       help='Path to NPZ file (legacy mode)')
    
    args = parser.parse_args()
    
    # Determine data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # Handle NPZ mode (legacy)
    if args.npz:
        if os.path.exists(args.npz):
            plot_fit_from_npz(args.npz)
        else:
            print(f"NPZ file not found: {args.npz}")
        exit()
    
    # Determine CSV filename
    if args.csv:
        csv_filename = args.csv
    else:
        # Auto-detect CSV file (look for ground_truth_convergence_deg_*.csv)
        csv_files = [f for f in os.listdir(data_dir) if f.startswith('ground_truth_convergence_deg_') and f.endswith('.csv')]
        if not csv_files:
            print(f"No convergence CSV files found in {data_dir}")
            print("Please run find_ground_truth.py first to generate the data file.")
            exit()
        csv_filename = os.path.join(data_dir, csv_files[0])
        print(f"Auto-detected CSV file: {csv_filename}")
    
    # Check if CSV exists
    if not os.path.exists(csv_filename):
        print(f"CSV file not found: {csv_filename}")
        print("Please run find_ground_truth.py first to generate the data file.")
        exit()
    
    # Plot from CSV
    plot_fit_from_csv(csv_filename, args.npts)
