#!/usr/bin/env python3
"""
Plot convergence analysis for QSP recovery.
Shows l_∞ norm difference between QSP values and ground truth polynomial.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import argparse
from qsppack.solver import solve
from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry

# Set fonts (LaTeX disabled to avoid compatibility issues)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def plot_recovery_convergence(csv_filename, deg=101):
    """
    Plot convergence analysis showing l_∞ norm difference between QSP values 
    and ground truth polynomial approximation.
    
    Args:
        csv_filename: Path to CSV file with convergence data
        deg: Degree of polynomial (for title)
    """
    # Load CSV data
    data = pd.read_csv(csv_filename)
    
    # Sort by npts to ensure proper order
    data = data.sort_values('npts').reset_index(drop=True)
    
    # Get ground truth (highest npts)
    ground_truth_idx = data['npts'].idxmax()
    ground_truth_row = data.iloc[ground_truth_idx]
    
    # Extract ground truth data
    ground_truth_coefs = np.array(json.loads(ground_truth_row['coefs_json']))
    ground_truth_npts = ground_truth_row['npts']
    
    print(f"Using ground truth from npts = {ground_truth_npts}")
    
    # Create ground truth polynomial function
    ground_truth_parity = ground_truth_row['parity']
    ground_truth_func = lambda x: chebyshev_to_func(x, ground_truth_coefs, ground_truth_parity, True)
    
    # Sample points for evaluation
    n_samples = 10000
    x_samples = np.linspace(-1, 1, n_samples)
    
    # Evaluate ground truth polynomial at sample points
    ground_truth_values = ground_truth_func(x_samples)
    
    # Prepare data for plotting
    npts_values = []
    l_inf_diffs = []
    constraint_violations = []
    
    print("Computing convergence metrics...")
    
    for idx, row in data.iterrows():
        npts = row['npts']
        print(f"  Processing npts = {npts}")
        
        # Extract data for this npts
        angles = np.array(json.loads(row['angles_json']))
        coefs = np.array(json.loads(row['coefs_json']))
        out = json.loads(row['out_json'])
        
        # Get QSP values at sample points
        QSP_values = get_entry(x_samples, angles, out)
        
        # Compute l_∞ norm difference
        l_inf_diff = np.max(np.abs(QSP_values - ground_truth_values))
        
        # Check constraint violations (polynomial approximation)
        parity = row['parity']
        poly_func = lambda x: chebyshev_to_func(x, coefs, parity, True)
        poly_values = poly_func(x_samples)
        max_poly_abs = np.max(np.abs(poly_values))
        constraint_violated = max_poly_abs > 1.0
        
        # Store results
        npts_values.append(npts)
        l_inf_diffs.append(l_inf_diff)
        constraint_violations.append(constraint_violated)
        
        print(f"    l_∞ diff: {l_inf_diff:.2e}, constraint violated: {constraint_violated}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points with color coding
    for i, (npts, diff, violated) in enumerate(zip(npts_values, l_inf_diffs, constraint_violations)):
        color = 'red' if violated else 'blue'
        ax.loglog(npts, diff, 'x', color=color, markersize=8, markeredgewidth=2)
    
    # Connect points with dashed line
    ax.loglog(npts_values, l_inf_diffs, 'k--', alpha=0.7, linewidth=1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('npts', fontsize=14)
    ax.set_ylabel('l_∞ norm difference', fontsize=14)
    ax.set_title(f'QSP Recovery Convergence Analysis (degree {deg})', fontsize=16, pad=20)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='red', linestyle='None', 
               markersize=8, markeredgewidth=2, label='Constraints Violated'),
        Line2D([0], [0], marker='x', color='blue', linestyle='None', 
               markersize=8, markeredgewidth=2, label='Constraints Satisfied')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_filename = csv_filename.replace('.csv', '_convergence.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    # Show plot
    plt.show()
    
    # Print summary
    print(f"\nConvergence Summary:")
    print(f"Ground truth npts: {ground_truth_npts}")
    print(f"Final l_∞ difference: {l_inf_diffs[-1]:.2e}")
    print(f"Constraint violations: {sum(constraint_violations)}/{len(constraint_violations)}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Plot QSP recovery convergence analysis')
    parser.add_argument('--csv', type=str, help='CSV file path (auto-detected if not provided)')
    parser.add_argument('--deg', type=int, default=101, help='Degree of polynomial (default: 101)')
    
    args = parser.parse_args()
    
    # Auto-detect CSV file if not provided
    if args.csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        csv_filename = os.path.join(data_dir, f"ground_truth_convergence_deg_{args.deg}.csv")
        
        if not os.path.exists(csv_filename):
            print(f"Error: CSV file not found: {csv_filename}")
            print("Please specify --csv path or ensure data exists in data/ directory")
            return
    else:
        csv_filename = args.csv
    
    print(f"Using CSV file: {csv_filename}")
    plot_recovery_convergence(csv_filename, args.deg)

if __name__ == "__main__":
    main()
