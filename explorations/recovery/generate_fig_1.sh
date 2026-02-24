#!/bin/bash
# Generate data and plot for Figure 1: max constraint violation vs npts.
#
# Usage:
#   ./generate_fig_1.sh           # generate data if needed (or resume missing npts), then plot
#   ./generate_fig_1.sh --force   # force full regeneration of data, then plot

set -e

# Directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATA_FILE="$SCRIPT_DIR/data/fig_1_constraint_violation_vs_npts.csv"

FORCE=0
if [[ "$1" == "--force" || "$1" == "-f" ]]; then
  FORCE=1
fi

if [[ "$FORCE" -eq 1 ]]; then
  echo "Forcing full regeneration (removing existing data)..."
  rm -f "$DATA_FILE"
fi

echo "Generating Fig. 1 data (will resume missing npts if file exists)..."
python generate_fig_1_data.py

# Plot is written to figures/fig_1_2.pdf (fig_1.pdf is left unchanged for comparison).
echo "Generating Fig. 1 plot..."
python plot_fig_1.py

echo "Done."

