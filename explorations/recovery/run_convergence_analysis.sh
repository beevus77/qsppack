#!/bin/bash
# Convergence analysis script for polynomial space recovery
# Runs fgt_polynomial_space.py with different parameters and then plots results

set -e  # Exit on error

echo "=========================================="
echo "Starting Convergence Analysis (Attempt 2)"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# # Run 1: degree=25, N=2^15, epsilon=1e-4 (default)
# echo "=========================================="
# echo "Run 1: degree=25, N=2^15 (32768), epsilon=1e-4"
# echo "=========================================="
# python fgt_polynomial_space.py --degree 25 --N 32768
# echo ""

# # Run 2: degree=25, N=2^16, epsilon=0
# echo "=========================================="
# echo "Run 2: degree=25, N=2^16 (65536), epsilon=0"
# echo "=========================================="
# python fgt_polynomial_space.py --degree 25 --N 65536 --epsilon 0
# echo ""

# # Run 3: degree=151, N=2^16, epsilon=1e-4 (default)
# echo "=========================================="
# echo "Run 3: degree=151, N=2^16 (65536), epsilon=1e-4"
# echo "=========================================="
# python fgt_polynomial_space.py --degree 151 --N 65536
# echo ""

# echo "=========================================="
# echo "All computations complete! Now generating plots..."
# echo "=========================================="
# echo ""

# Plot 1a: degree=25, N=2^15, epsilon=1e-4, plot_type=2 (default)
echo "=========================================="
echo "Plot 1a: degree=25, N=15, epsilon=4, 2-norm"
echo "=========================================="
python plot_recovery_conv_polynomial_space.py --csv data/fgt_polynomial_space_convergence_deg_25_epsil4_N15.csv --deg 25
echo ""

# Plot 1b: degree=25, N=2^15, epsilon=1e-4, plot_type=infty
echo "=========================================="
echo "Plot 1b: degree=25, N=15, epsilon=4, infty-norm"
echo "=========================================="
python plot_recovery_conv_polynomial_space.py --csv data/fgt_polynomial_space_convergence_deg_25_epsil4_N15.csv --deg 25 --plot infty
echo ""

# Plot 2a: degree=25, N=2^16, epsilon=0, plot_type=2 (default)
echo "=========================================="
echo "Plot 2a: degree=25, N=16, epsilon=0, 2-norm"
echo "=========================================="
python plot_recovery_conv_polynomial_space.py --csv data/fgt_polynomial_space_convergence_deg_25_epsil0_N16.csv --deg 25
echo ""

# Plot 2b: degree=25, N=2^16, epsilon=0, plot_type=infty
echo "=========================================="
echo "Plot 2b: degree=25, N=16, epsilon=0, infty-norm"
echo "=========================================="
python plot_recovery_conv_polynomial_space.py --csv data/fgt_polynomial_space_convergence_deg_25_epsil0_N16.csv --deg 25 --plot infty
echo ""

# Plot 3a: degree=151, N=2^16, epsilon=1e-4, plot_type=2 (default)
echo "=========================================="
echo "Plot 3a: degree=151, N=16, epsilon=4, 2-norm"
echo "=========================================="
python plot_recovery_conv_polynomial_space.py --csv data/fgt_polynomial_space_convergence_deg_151_epsil4_N16.csv --deg 151
echo ""

# # Plot 3b: degree=151, N=2^16, epsilon=1e-4, plot_type=infty
# echo "=========================================="
# echo "Plot 3b: degree=151, N=16, epsilon=4, infty-norm"
# echo "=========================================="
# python plot_recovery_conv_polynomial_space.py --csv data/fgt_polynomial_space_convergence_deg_151_epsil4_N16.csv --deg 151 --plot infty
# echo ""

echo "=========================================="
echo "All analysis complete!"
echo "=========================================="
echo ""
echo "Generated CSV files:"
echo "  - data/fgt_polynomial_space_convergence_deg_25_epsil4_N15.csv"
echo "  - data/fgt_polynomial_space_convergence_deg_25_epsil0_N16.csv"
echo "  - data/fgt_polynomial_space_convergence_deg_151_epsil4_N16.csv"
echo ""
echo "Generated plots are saved in the data/ directory with appropriate suffixes."

