#!/usr/bin/env python3
"""
Plot convergence analysis in polynomial coefficient space.
Shows 2-norm differences between coefficients and ground truth.
Constraints are checked by evaluating via DCT-I at Chebyshev nodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import argparse
from scipy.fft import dct
import csv

# Set fonts (LaTeX disabled to avoid compatibility issues)
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


def plot_recovery_convergence_polyspace(csv_filename, deg=101, M=10_000_000, plot_type=2):
    """
    Plot convergence analysis in coefficient space using data from
    fgt_polynomial_space.py CSV.

    Args:
        csv_filename: Path to CSV file with convergence data
        deg: Degree of polynomial (for title)
        M: Number of Chebyshev nodes for constraint evaluation
        plot_type: Plot type - 2 for 2-norm differences, 'infty' for infinity norm differences
    """
    # Load CSV data
    data = pd.read_csv(csv_filename)

    # Ensure cache columns exist
    for col in ['constraint_violated_coef', 'constraint_violated_rec', 'sup_diff', 'sup_diff_rec', 'max_abs_coef', 'max_abs_rec']:
        if col not in data.columns:
            data[col] = np.nan

    # Sort by npts to ensure proper order
    data = data.sort_values('npts').reset_index(drop=True)

    # Ground truth: use largest npts row's original coefficients (expanded to full)
    ground_truth_idx = data['npts'].idxmax()
    gt_row = data.iloc[ground_truth_idx]
    gt_parity = int(gt_row['parity'])
    gt_degree = int(gt_row['degree'])
    gt_coef_full = np.zeros(gt_degree+1)
    gt_coef_full[gt_parity::2] = np.array(json.loads(gt_row['coefs_json']))

    print(f"Using ground truth from npts = {int(gt_row['npts'])}")

    npts_values = []
    diff_coef_vs_gt = []           # ||coef_full - gt||_2 or ||vals_coef - vals_gt||_∞
    diff_recovered_vs_gt = []      # ||coef_recovered_full - gt||_2 or ||vals_rec - vals_gt||_∞
    constraint_violations_coef = []  # True if constraint violated for vals_coef
    constraint_violations_rec = []   # True if constraint violated for vals_rec

    print("Computing convergence metrics in coefficient space...")

    modified = False
    vals_gt = None  # Will compute once and reuse
    for idx, row in data.iterrows():
        npts = int(row['npts'])
        degree = int(row['degree'])
        parity = int(row['parity'])

        # Expand reduced coefficients to full-length by parity
        coef_full = np.zeros(degree+1)
        coef_full[parity::2] = np.array(json.loads(row['coefs_json']))
        coef_recovered_full = np.array(json.loads(row['coefs_recovered_json']))

        # Compute 2-norm differences to ground truth coefficients (match sizes if needed)
        # If degrees differ (they shouldn't), compare over min length then include tail
        max_len = max(len(gt_coef_full), len(coef_full), len(coef_recovered_full))
        def pad_to(v, L):
            w = np.zeros(L)
            w[:len(v)] = v
            return w
        gt_pad = pad_to(gt_coef_full, max_len)
        coef_pad = pad_to(coef_full, max_len)
        rec_pad = pad_to(coef_recovered_full, max_len)

        diff_coef = np.linalg.norm(coef_pad - gt_pad)
        diff_rec = np.linalg.norm(rec_pad - gt_pad)

        # Constraint check via DCT-I at Chebyshev nodes
        # Check if we need to compute anything (constraint violations or sup norms)
        need_compute = (pd.isna(row.get('constraint_violated_coef', np.nan)) or 
                       pd.isna(row.get('constraint_violated_rec', np.nan)) or
                       pd.isna(row.get('sup_diff', np.nan)) or 
                       pd.isna(row.get('sup_diff_rec', np.nan)) or
                       pd.isna(row.get('max_abs_coef', np.nan)) or
                       pd.isna(row.get('max_abs_rec', np.nan)))
        
        if not need_compute:
            # Use cached values
            violated_coef = bool(row['constraint_violated_coef'])
            violated_rec = bool(row['constraint_violated_rec'])
            sup_diff = float(row['sup_diff'])
            sup_diff_rec = float(row['sup_diff_rec'])
            max_abs_coef = float(row['max_abs_coef'])
            max_abs_rec = float(row['max_abs_rec'])
        else:
            print("Checking constraint violations...")
            # Compute ground truth values once
            if vals_gt is None:
                print(f"  Computing ground truth values at {M} Chebyshev nodes...")
                vals_gt = chebval_dct(gt_coef_full, M)
            
            try:
                vals_coef = chebval_dct(coef_full, M)
                vals_rec = chebval_dct(coef_recovered_full, M)
                # Compute max absolute values first
                max_abs_coef = np.max(np.abs(vals_coef))
                max_abs_rec = np.max(np.abs(vals_rec))
                # Check constraints using the pre-computed max values
                violated_coef = max_abs_coef > 1.0
                violated_rec = max_abs_rec > 1.0
                # Compute sup norm differences
                sup_diff = np.max(np.abs(vals_coef - vals_gt))
                sup_diff_rec = np.max(np.abs(vals_rec - vals_gt))
            except MemoryError:
                print("Warning: MemoryError during DCT evaluation; reducing M to 1,000,000 for this step.")
                M_small = 1_000_000
                if vals_gt is None:
                    vals_gt = chebval_dct(gt_coef_full, M_small)
                vals_coef = chebval_dct(coef_full, M_small)
                vals_rec = chebval_dct(coef_recovered_full, M_small)
                # Compute max absolute values first
                max_abs_coef = np.max(np.abs(vals_coef))
                max_abs_rec = np.max(np.abs(vals_rec))
                # Check constraints using the pre-computed max values
                violated_coef = max_abs_coef > 1.0
                violated_rec = max_abs_rec > 1.0
                # Compute sup norm differences
                sup_diff = np.max(np.abs(vals_coef - vals_gt))
                sup_diff_rec = np.max(np.abs(vals_rec - vals_gt))
            
            # Cache computed values
            data.at[idx, 'constraint_violated_coef'] = violated_coef
            data.at[idx, 'constraint_violated_rec'] = violated_rec
            data.at[idx, 'max_abs_coef'] = max_abs_coef
            data.at[idx, 'max_abs_rec'] = max_abs_rec
            data.at[idx, 'sup_diff'] = sup_diff
            data.at[idx, 'sup_diff_rec'] = sup_diff_rec
            modified = True

        # Store results (use appropriate metric based on plot_type)
        npts_values.append(npts)
        if plot_type == 'infty':
            diff_coef_vs_gt.append(sup_diff)
            diff_recovered_vs_gt.append(sup_diff_rec)
        else:  # plot_type == 2
            diff_coef_vs_gt.append(diff_coef)
            diff_recovered_vs_gt.append(diff_rec)
        constraint_violations_coef.append(violated_coef)
        constraint_violations_rec.append(violated_rec)

        print(f"  npts={npts}: ||coef-gt||_2={diff_coef:.2e}, ||rec-gt||_2={diff_rec:.2e}")
        print(f"    coef violated={violated_coef}, rec violated={violated_rec}")
        print(f"    max_abs_coef={max_abs_coef:.2e}, max_abs_rec={max_abs_rec:.2e}")
        print(f"    sup_diff={sup_diff:.2e}, sup_diff_rec={sup_diff_rec:.2e}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color coding based on constraint violation - separate colors for coef and rec
    for npts, d1, d2, violated_coef, violated_rec in zip(npts_values, diff_coef_vs_gt, diff_recovered_vs_gt, constraint_violations_coef, constraint_violations_rec):
        color_coef = 'red' if violated_coef else 'blue'
        color_rec = 'red' if violated_rec else 'blue'
        ax.loglog(npts, d1, 'x', color=color_coef, markersize=8, markeredgewidth=2)
        ax.loglog(npts, d2, 'o', color=color_rec, markersize=5, markeredgewidth=1, fillstyle='none')

    # Connect points with dashed lines for each series
    if plot_type == 'infty':
        ax.loglog(npts_values, diff_coef_vs_gt, 'k--', alpha=0.7, linewidth=1, label='max difference from ground truth')
        ax.loglog(npts_values, diff_recovered_vs_gt, 'k-.', alpha=0.7, linewidth=1, label='max difference from ground truth, recovered')
    else:
        ax.loglog(npts_values, diff_coef_vs_gt, 'k--', alpha=0.7, linewidth=1, label='||coef - gt||_2')
        ax.loglog(npts_values, diff_recovered_vs_gt, 'k-.', alpha=0.7, linewidth=1, label='||recovered - gt||_2')

    # Grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('npts', fontsize=14)
    if plot_type == 'infty':
        ax.set_ylabel('Function ∞-norm difference', fontsize=14)
        ax.set_title(f'Recovery Convergence (∞-norm, degree {deg})', fontsize=16, pad=20)
    else:
        ax.set_ylabel('Coefficient 2-norm difference', fontsize=14)
        ax.set_title(f'Polynomial-Space Recovery Convergence (2-norm, degree {deg})', fontsize=16, pad=20)

    # Legend for color coding
    from matplotlib.lines import Line2D
    if plot_type == 'infty':
        legend_elements = [
            Line2D([0], [0], marker='x', color='blue', linestyle='None', markersize=8, markeredgewidth=2, label='Coef: Constraints Satisfied'),
            Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=8, markeredgewidth=2, label='Coef: Constraints Violated'),
            Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=5, fillstyle='none', label='Rec: Constraints Satisfied'),
            Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=5, fillstyle='none', label='Rec: Constraints Violated'),
            Line2D([0], [0], linestyle='--', color='k', label='max difference from ground truth'),
            Line2D([0], [0], linestyle='-.', color='k', label='max difference from ground truth, recovered')
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker='x', color='blue', linestyle='None', markersize=8, markeredgewidth=2, label='Coef: Constraints Satisfied'),
            Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=8, markeredgewidth=2, label='Coef: Constraints Violated'),
            Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=5, fillstyle='none', label='Rec: Constraints Satisfied'),
            Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=5, fillstyle='none', label='Rec: Constraints Violated'),
            Line2D([0], [0], linestyle='--', color='k', label='||coef - gt||_2'),
            Line2D([0], [0], linestyle='-.', color='k', label='||recovered - gt||_2')
        ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()

    # Save plot
    if plot_type == 'infty':
        plot_filename = csv_filename.replace('.csv', '_inf_convergence.png')
    else:
        plot_filename = csv_filename.replace('.csv', '_polyspace_convergence.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")

    # If we computed any constraints, persist them to the CSV
    if modified:
        data.to_csv(csv_filename, index=False, quoting=csv.QUOTE_ALL)
        print(f"Cached constraint violations saved back to CSV: {csv_filename}")

    plt.show()

    # Print summary
    print("\nConvergence Summary (coefficient space):")
    print(f"Ground truth degree: {gt_degree}")
    print(f"Final ||coef-gt||_2: {diff_coef_vs_gt[-1]:.2e}")
    print(f"Final ||recovered-gt||_2: {diff_recovered_vs_gt[-1]:.2e}")
    print(f"Coef constraint violations: {sum(constraint_violations_coef)}/{len(constraint_violations_coef)}")
    print(f"Rec constraint violations: {sum(constraint_violations_rec)}/{len(constraint_violations_rec)}")


def plot_max_constraint_violation(csv_filename, deg=101):
    """
    Plot maximum constraint violation values (max_abs_coef) vs npts.
    
    Args:
        csv_filename: Path to CSV file with convergence data
        deg: Degree of polynomial (for title)
    """
    # Load CSV data
    data = pd.read_csv(csv_filename)
    
    # Ensure required columns exist
    for col in ['constraint_violated_coef', 'max_abs_coef']:
        if col not in data.columns:
            data[col] = np.nan
    
    # Sort by npts to ensure proper order
    data = data.sort_values('npts').reset_index(drop=True)
    
    print(f"Using CSV file: {csv_filename}")
    print("Plotting maximum constraint violations...")
    
    npts_values = []
    max_abs_coef_values = []
    constraint_violations_coef = []
    
    for idx, row in data.iterrows():
        npts = int(row['npts'])
        max_abs_coef = float(row['max_abs_coef']) if not pd.isna(row['max_abs_coef']) else 0.0
        violated_coef = bool(row['constraint_violated_coef']) if not pd.isna(row['constraint_violated_coef']) else False
        
        npts_values.append(npts)
        max_abs_coef_values.append(max_abs_coef)
        constraint_violations_coef.append(violated_coef)
        
        print(f"  npts={npts}: max_abs_coef={max_abs_coef:.2e}, violated={violated_coef}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color coding based on constraint violation
    for npts, max_abs, violated in zip(npts_values, max_abs_coef_values, constraint_violations_coef):
        color = 'red' if violated else 'blue'
        ax.loglog(npts, max_abs, 'x', color=color, markersize=8, markeredgewidth=2)
    
    # Connect points with dashed line
    ax.loglog(npts_values, max_abs_coef_values, 'k--', alpha=0.7, linewidth=1, label='max |coef|')
    
    # Grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('npts', fontsize=14)
    ax.set_ylabel('Maximum |coefficient value|', fontsize=14)
    ax.set_title(f'Maximum Constraint Violations (degree {deg})', fontsize=16, pad=20)
    
    # Legend for color coding
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='blue', linestyle='None', markersize=8, markeredgewidth=2, label='Constraints Satisfied'),
        Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=8, markeredgewidth=2, label='Constraints Violated'),
        Line2D([0], [0], linestyle='--', color='k', label='max |coef|')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = csv_filename.replace('.csv', '_max_constraint_violations.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    plt.show()
    
    # Print summary
    print("\nConstraint Violation Summary:")
    print(f"Total points: {len(npts_values)}")
    print(f"Constraint violations: {sum(constraint_violations_coef)}/{len(constraint_violations_coef)}")
    print(f"Max violation value: {max(max_abs_coef_values):.2e}")
    print(f"Min violation value: {min(max_abs_coef_values):.2e}")


def main():
    parser = argparse.ArgumentParser(description='Plot polynomial-space recovery convergence analysis')
    parser.add_argument('--csv', type=str, help='CSV file path (auto-detected if not provided)')
    parser.add_argument('--deg', type=int, default=101, help='Degree of polynomial (default: 101)')
    parser.add_argument('--M', type=int, default=10_000_000, help='Number of Chebyshev nodes for constraint check')
    parser.add_argument('--plot', type=str, default='2', choices=['2', 'infty'], 
                       help='Plot type: 2 for 2-norm differences (default), infty for infinity norm differences')
    parser.add_argument('--max-violations', action='store_true', 
                       help='Plot maximum constraint violations instead of convergence analysis')

    args = parser.parse_args()

    # Auto-detect CSV file if not provided
    if args.csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        csv_filename = os.path.join(data_dir, f"fgt_polynomial_space_convergence_deg_{args.deg}_N15.csv")

        if not os.path.exists(csv_filename):
            print(f"Error: CSV file not found: {csv_filename}")
            print("Please specify --csv path or ensure data exists in data/ directory")
            return
    else:
        csv_filename = args.csv

    print(f"Using CSV file: {csv_filename}")
    
    # Always run the main convergence analysis
    plot_recovery_convergence_polyspace(csv_filename, args.deg, args.M, args.plot)
    
    # Additionally plot max constraint violations if requested
    if args.max_violations:
        plot_max_constraint_violation(csv_filename, args.deg)


if __name__ == "__main__":
    main()


