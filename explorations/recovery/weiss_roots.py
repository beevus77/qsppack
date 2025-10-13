#!/usr/bin/env python3
"""
Script to find and plot zeros of 1 - |b(z)|^2 on the unit circle.

This script reads Chebyshev polynomial coefficients from a CSV file, converts them
to polynomial coefficients b(z) on the unit circle using b_from_cheb, computes
1 - |b(z)|^2, and plots the zeros of this function.
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qsppack.nlfa import b_from_cheb


def extract_N_from_filename(filename):
    """
    Extract the N exponent from the filename pattern (e.g., N15 means 2^15).
    
    Parameters
    ----------
    filename : str
        Filename containing the pattern N{number}
        
    Returns
    -------
    N : int
        The value 2^exponent, or None if not found
    """
    match = re.search(r'_N(\d+)', filename)
    if match:
        exponent = int(match.group(1))
        return 2 ** exponent
    return None


def load_coefficients_from_csv(csv_file, npts):
    """
    Load Chebyshev coefficients from CSV file for a specific npts value.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file containing Chebyshev coefficients
    npts : int
        Number of points value to extract from CSV
        
    Returns
    -------
    coefs : np.ndarray
        Chebyshev coefficients
    parity : int
        Parity of the polynomial
    degree : int
        Degree of the polynomial
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Filter for the specified npts value
    row = df[df['npts'] == npts]
    
    if len(row) == 0:
        available_npts = df['npts'].unique()
        raise ValueError(
            f"No row found with npts={npts}. "
            f"Available npts values: {sorted(available_npts)}"
        )
    
    # Extract the first matching row
    row = row.iloc[0]
    
    # Parse coefficients and parity (use original coefs, not recovered)
    coefs = np.array(json.loads(row['coefs_json']))
    parity = int(row['parity'])
    degree = int(row['degree'])
    
    return coefs, parity, degree


def evaluate_laurent_polynomial_at_point(b_coefs, z):
    """
    Evaluate 1 - |b(z)|^2 at a given point z.
    
    Parameters
    ----------
    b_coefs : np.ndarray
        Coefficients of polynomial b(z) on unit circle
    z : complex
        Point at which to evaluate
        
    Returns
    -------
    value : complex
        The value of 1 - |b(z)|^2 at z
    """
    # Evaluate b(z)
    b_z = np.polyval(b_coefs[::-1], z)
    
    # Compute 1 - |b(z)|^2
    return 1.0 - np.abs(b_z)**2


def find_zeros(coefs, parity):
    """
    Convert Chebyshev coefficients to unit circle polynomial and find zeros of 1 - |b(z)|^2.
    
    Parameters
    ----------
    coefs : np.ndarray
        Chebyshev coefficients
    parity : int
        Parity of the polynomial
        
    Returns
    -------
    zeros : np.ndarray
        Complex zeros of 1 - |b(z)|^2
    b_coefs : np.ndarray
        Coefficients of polynomial b(z) on unit circle
    """
    # Convert to polynomial on unit circle
    b_coefs = b_from_cheb(coefs, parity)
    n = len(b_coefs)
    
    # Compute b^*(z) = conjugate of b with reversed coefficients
    # b^*(z) corresponds to z^(n-1) * conj(b(1/z))
    b_star_coefs = np.conj(b_coefs[::-1])
    
    # Compute |b(z)|^2 = b(z) * b^*(z) using convolution
    # This produces a Laurent polynomial with powers from z^{-(n-1)} to z^{n-1}
    b_squared_laurent = np.convolve(b_coefs, b_star_coefs)
    
    # The result has length 2n-1, centered at index n-1 (corresponding to z^0)
    # Compute 1 - |b(z)|^2
    one_minus_b_squared = -b_squared_laurent
    one_minus_b_squared[n-1] += 1.0  # Add 1 at the z^0 coefficient
    
    # Find zeros of the Laurent polynomial
    # Since this is a Laurent polynomial from z^{-(n-1)} to z^{n-1},
    # we can convert to regular polynomial by treating it as is
    # The coefficients are ordered from lowest to highest power
    # For numpy.roots, we need highest to lowest power
    zeros = np.roots(one_minus_b_squared[::-1])
    
    return zeros, b_coefs


def find_nearest_to_roots_of_unity(zeros, roots_of_unity):
    """
    Find the zero that is closest to any Nth root of unity.
    
    Parameters
    ----------
    zeros : np.ndarray
        Complex zeros of the polynomial
    roots_of_unity : np.ndarray
        Precomputed Nth roots of unity
        
    Returns
    -------
    nearest_zero : complex
        The zero closest to a root of unity
    nearest_root_of_unity : complex
        The root of unity closest to this zero
    min_distance : float
        The distance between them
    """
    min_distance = np.inf
    nearest_zero = None
    nearest_root_of_unity = None
    
    for zero in zeros:
        # Compute distances to all roots of unity
        distances = np.abs(zero - roots_of_unity)
        min_dist_for_this_zero = np.min(distances)
        
        if min_dist_for_this_zero < min_distance:
            min_distance = min_dist_for_this_zero
            nearest_zero = zero
            nearest_root_of_unity = roots_of_unity[np.argmin(distances)]
    
    return nearest_zero, nearest_root_of_unity, min_distance


def plot_zeros_on_unit_circle(zeros, degree, npts, roots_of_unity=None):
    """
    Plot the zeros of the polynomial on the complex plane with unit circle.
    
    Parameters
    ----------
    zeros : np.ndarray
        Complex zeros to plot
    degree : int
        Degree of the polynomial
    npts : int
        Number of points value from CSV
    roots_of_unity : np.ndarray, optional
        If provided, plot these precomputed roots of unity
    """
    # Set up LaTeX fonts before creating figure
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        xlabel = r'$\mathrm{Re}(z)$'
        ylabel = r'$\mathrm{Im}(z)$'
        title = rf'Zeros of $1 - |b(z)|^2$ on Unit Circle (degree = {degree}, npts = {npts})'
    except:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        xlabel = 'Re(z)'
        ylabel = 'Im(z)'
        title = f'Zeros of 1 - |b(z)|^2 on Unit Circle (degree = {degree}, npts = {npts})'
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Nth roots of unity if provided
    if roots_of_unity is not None:
        N = len(roots_of_unity)
        ax.scatter(roots_of_unity.real, roots_of_unity.imag, 
                   c='#457B9D', s=5, alpha=0.6, 
                   label=f'${N}$th roots of unity', zorder=3)
    else:
        # Draw unit circle only if roots of unity are not plotted
        theta = np.linspace(0, 2*np.pi, 1000)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, 
                alpha=0.4, label='Unit circle')
    
    # Plot zeros with pretty colors
    ax.scatter(zeros.real, zeros.imag, c='#E63946', s=80, alpha=0.8, 
               edgecolors='#A4161A', linewidth=1.5, label='Zeros', zorder=5)
    
    # Set equal aspect ratio and styling
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18, pad=20)
    ax.legend(fontsize=14, loc='upper right')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to parse arguments and orchestrate the workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Find and plot zeros of 1 - |b(z)|^2 on unit circle'
    )
    parser.add_argument('csv_file', type=str, 
                        help='Path to CSV file containing Chebyshev coefficients')
    parser.add_argument('npts', type=int, 
                        help='Number of points (npts value) to extract from CSV')
    args = parser.parse_args()
    
    # Load coefficients from CSV
    coefs, parity, degree = load_coefficients_from_csv(args.csv_file, args.npts)
    print(f"Loaded coefficients: {len(coefs)} Chebyshev coefficients")
    print(f"Degree: {degree}, Parity: {parity}, npts: {args.npts}")
    
    # Extract N from filename and compute roots of unity once
    N = extract_N_from_filename(args.csv_file)
    roots_of_unity = None
    if N is not None:
        print(f"Extracted N = {N} from filename")
        print(f"Computing {N} roots of unity...")
        roots_of_unity = np.exp(2j * np.pi * np.arange(N) / N)
    
    # Find zeros
    zeros, b_coefs = find_zeros(coefs, parity)
    print(f"Converted to {len(b_coefs)} coefficients on unit circle")
    print(f"Found {len(zeros)} zeros of 1 - |b(z)|^2")
    
    # Find nearest zero to roots of unity if available
    if roots_of_unity is not None:
        nearest_zero, nearest_rou, min_dist = find_nearest_to_roots_of_unity(zeros, roots_of_unity)
        print(f"\nNearest zero to roots of unity:")
        print(f"  Zero coordinates: {nearest_zero.real:.10f} + {nearest_zero.imag:.10f}i")
        print(f"  Distance to nearest root of unity: {min_dist:.10e}")
        
        # Evaluate the Laurent polynomial at the nearest root of unity
        value_at_rou = evaluate_laurent_polynomial_at_point(b_coefs, nearest_rou)
        print(f"  Value of 1 - |b(z)|^2 at nearest root of unity: {value_at_rou:.10e}")
    
    # Plot results
    plot_zeros_on_unit_circle(zeros, degree, args.npts, roots_of_unity=roots_of_unity)


if __name__ == '__main__':
    main()

