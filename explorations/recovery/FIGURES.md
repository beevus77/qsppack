# Recovery Figure Registry

Run these commands from `explorations/recovery` with the `qsppack` Python environment active.
The plotting commands below reproduce the figure filenames and command-line settings used in the current draft.

## Fig 1

Figure file:
`figures/fig_1_2.pdf`

Data file:
`data/fig_1_constraint_violation_vs_npts.csv`

Generate data:

```bash
python generate_fig_1_data.py \
  --deg 101 \
  --a 0.2 \
  --epsil 0.0 \
  --out data/fig_1_constraint_violation_vs_npts.csv
```

Generate figure:

```bash
python plot_fig_1.py \
  --csv data/fig_1_constraint_violation_vs_npts.csv \
  --output figures/fig_1_2.pdf
```

## Fig 5

Figure file:
`figures/retraction_error_comp.pdf`

Data file:
`data/fgt_polynomial_space_convergence_deg_101_N18.csv`

Generate data:

```bash
python fgt_polynomial_space.py \
  --degree 101 \
  --problem-type uniform_sv_amp \
  --a 0.2 \
  --epsilon 0.0 \
  --N $((2**18))
```

Note: `data/fgt_polynomial_space_convergence_deg_101_N18.csv` is a legacy filename used by the plotting commands below. This data should be generated with `--epsilon 0.0`; the legacy filename does not record the problem type or epsilon. The current `fgt_polynomial_space.py` naming scheme would include `uniform_sv_amp` and `epsil0` in the generated filename for these parameters.

Generate figure:

```bash
python plot_fit_polynomial_space.py 128 \
  --csv data/fgt_polynomial_space_convergence_deg_101_N18.csv \
  --ploterror \
  --output figures/retraction_error_comp.pdf
```

## Fig 6

Figure file:
`figures/fgt_polynomial_space_convergence_deg_101_N18_two_panels_polyspace_convergence.pdf`

Data files:
`data/fgt_polynomial_space_convergence_deg_101_N18.csv`
`data/fgt_polynomial_space_convergence_deg_101_epsil4_N16.csv`

Generate left-panel data:

```bash
python fgt_polynomial_space.py \
  --degree 101 \
  --problem-type uniform_sv_amp \
  --a 0.2 \
  --epsilon 0.0 \
  --N $((2**18))
```

Generate right-panel data:

```bash
python fgt_polynomial_space.py \
  --degree 101 \
  --problem-type uniform_sv_amp \
  --a 0.2 \
  --epsilon 1e-4 \
  --N $((2**16))
```

Note: these are legacy data filenames used by the plotting command. The left-panel `N18` data should be generated with `--epsilon 0.0`; the legacy filename does not record the problem type or epsilon. The current `fgt_polynomial_space.py` default output names include `uniform_sv_amp` and `epsil*` in the basename.

Generate figure:

```bash
python plot_recovery_conv_polynomial_space.py \
  --csv data/fgt_polynomial_space_convergence_deg_101_N18.csv \
  --csv2 data/fgt_polynomial_space_convergence_deg_101_epsil4_N16.csv \
  --plot 2 \
  --output figures/fgt_polynomial_space_convergence_deg_101_N18_two_panels_polyspace_convergence.pdf
```

## Fig 7

Figure file:
`figures/degree_scaling_uniform_sv_amp_npts19_solver2.pdf`

Data file:
`data/degree_scaling_uniform_sv_amp_npts19_solver2.csv`

Generate data:

```bash
python degree_scaling_data.py \
  --problem-type uniform_sv_amp \
  --npts $((2**19)) \
  --a 0.2 \
  --epsilon 0.0 \
  --N $((2**14)) \
  --solver2
```

Note: the existing `data/degree_scaling_uniform_sv_amp_npts19_solver2.csv` records `N_weiss = 16384`, so this figure corresponds to `--N $((2**14))`.

Generate figure:

```bash
python degree_scaling_plot.py \
  --csv data/degree_scaling_uniform_sv_amp_npts19_solver2.csv \
  --output figures/degree_scaling_uniform_sv_amp_npts19_solver2.pdf
```

## Fig 8

Figure file:
`figures/mat_inv_ex.pdf`

Data file:
None. This figure computes the matrix-inversion polynomial, windowed polynomial, and retractions directly inside `mat_inv_ex.py`.

Generate data:

```bash
# No separate data-generation step.
```

Generate figure:

```bash
python mat_inv_ex.py \
  --a 0.1 \
  --n 51 \
  --N_weiss $((2**12)) \
  --method all \
  --error_plot \
  --delta 0.01 \
  --a-window 0.095 \
  --output figures/mat_inv_ex.pdf
```

## Fig 9

Figure file:
`figures/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil0_N18_two_panels_polyspace_convergence.pdf`

Data files:
`data/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil0_N18.csv`
`data/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil4_N16.csv`

Generate left-panel data:

```bash
python fgt_polynomial_space.py \
  --degree 101 \
  --problem-type mat_inv \
  --a 0.2 \
  --epsilon 0.0 \
  --N $((2**18))
```

Generate right-panel data (use `--epsilon 1e-4` so the CSV basename contains `epsil4`; the plot script titles the right subplot as epsilon = 0.0001):

```bash
python fgt_polynomial_space.py \
  --degree 101 \
  --problem-type mat_inv \
  --a 0.2 \
  --epsilon 1e-4 \
  --N $((2**16))
```

Note: same `a` and `--problem-type mat_inv` for both panels; `--epsilon 0.0` vs `1e-4` and `--N` mirror Fig 6’s left/right layout (N18 vs N16). CSV basenames follow `fgt_polynomial_space.py` (`mat_inv`, `epsil0` / `epsil4`).

Generate figure:

```bash
python plot_recovery_conv_polynomial_space.py \
  --csv data/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil0_N18.csv \
  --csv2 data/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil4_N16.csv \
  --plot 2 \
  --output figures/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil0_N18_two_panels_polyspace_convergence.pdf
```

## Fig 10

Figure file:
`figures/degree_scaling_mat_inv_npts19.pdf`

Data file:
`data/degree_scaling_mat_inv_npts19.csv`

Generate data:

```bash
python degree_scaling_data.py \
  --problem-type mat_inv \
  --npts $((2**19)) \
  --a 0.2 \
  --epsilon 0.0 \
  --N $((2**14))
```

Note: the checked-in `data/degree_scaling_mat_inv_npts19.csv` records `N_weiss = 16384` and fixed discretization `npts = 524288` for each degree row, matching `--npts $((2**19))` and `--N $((2**14))`. This is the matrix-inversion target `a/x` on `[a, 1]` (same family as Fig 9), without `--solver2`.

Generate figure:

```bash
python degree_scaling_plot.py \
  --csv data/degree_scaling_mat_inv_npts19.csv \
  --output figures/degree_scaling_mat_inv_npts19.pdf
```

## Fig 11

Figure file:
`figures/thresh_proj_inset.pdf`

Data file:
None. This figure fits the threshold-projection target and retracts with Weiss/NLFT inside `plot_thresh_proj_ex.py` (same pipeline as the former manual PowerPoint crop of the full-curve plot).

Generate data:

```bash
# No separate data-generation step.
```

Generate figure (degree 128, `npts = 256`, matplotlib `inset_axes` + `mark_inset` with a tall inset on the left so the `1 ± 1e-5` band is readable; default data limits `x` in `[0.2, 0.45]` and `y` in `[1 - 1e-5, 1 + 1.1e-5]` (override with `--inset-xmin`, `--inset-xmax`, `--inset-ymin`, `--inset-ymax`). X-axis label size matches Fig 8; legend is slightly smaller and fixed to the upper right.

```bash
python plot_thresh_proj_ex.py \
  --degree 128 \
  --npts 256 \
  --zoom-inset \
  --output figures/thresh_proj_inset.pdf
```

## Fig 12

Figure file:
`figures/fgt_polynomial_space_convergence_deg_128_thresh_proj_epsil0_N15_two_panels_polyspace_convergence.pdf`

Data files:
`data/fgt_polynomial_space_convergence_deg_128_thresh_proj_epsil0_N15.csv`
`data/fgt_polynomial_space_convergence_deg_128_thresh_proj_epsil4_N15.csv`

Generate left-panel data:

```bash
python fgt_polynomial_space.py \
  --degree 128 \
  --problem-type thresh_proj \
  --a 0.2 \
  --epsilon 0.0 \
  --N $((2**15))
```

Generate right-panel data (use `--epsilon 1e-4` so the CSV basename contains `epsil4`; subplot title follows that token like Fig 9):

```bash
python fgt_polynomial_space.py \
  --degree 128 \
  --problem-type thresh_proj \
  --a 0.2 \
  --epsilon 1e-4 \
  --N $((2**15))
```

Note: threshold-projection target on `[0, 1]` as in `fgt_polynomial_space.py`; the `thresh_proj` target does not use `--a`, but `--a 0.2` is kept for consistency with other runs. Both panels use `N15`. Typography matches Fig 9: both figures use `plot_recovery_conv_polynomial_space.py`, whose `plt.rcParams` and label/title/legend constants match `mat_inv_ex.py` (24 / 26 / 28 / 18 pt as in those files).

Generate figure:

```bash
python plot_recovery_conv_polynomial_space.py \
  --csv data/fgt_polynomial_space_convergence_deg_128_thresh_proj_epsil0_N15.csv \
  --csv2 data/fgt_polynomial_space_convergence_deg_128_thresh_proj_epsil4_N15.csv \
  --plot 2 \
  --output figures/fgt_polynomial_space_convergence_deg_128_thresh_proj_epsil0_N15_two_panels_polyspace_convergence.pdf
```

## Fig 13

Figure file:
`figures/degree_scaling_thresh_proj_npts19.pdf`

Data file:
`data/degree_scaling_thresh_proj_npts19.csv`

Generate data (fixed convex discretization `npts = 2^19` for every degree in the sweep; Weiss parameter `N_weiss = 2^14` inside `degree_scaling_thresh_proj.py`):

```bash
python degree_scaling_thresh_proj.py \
  --npts $((2**19))
```

Note: the checked-in CSV records `npts = 524288` and `N_weiss = 16384` on each row. Re-running without `--force` only fills missing degrees. The summary figure matches Fig 10 legend text and marker sizes (`degree_scaling_plot.py`). It plots degrees with `exp2 <= 8` only (largest plotted degree 256 = `2^8`); use `--max-exp 8` explicitly, or omit it when `--npts` is `2**19` and the script applies that default. Summary typography matches Fig 10 (base/tick 20 pt, axis labels 22 pt, titles 24 pt, legend 14 pt, `loc="best"`).

Generate figure (after data exist; avoids re-running optimization if the CSV is complete):

```bash
python degree_scaling_thresh_proj.py \
  --npts $((2**19)) \
  --max-exp 8 \
  --output figures/degree_scaling_thresh_proj_npts19.pdf
```

To refresh only the PDF (no data run):  
`python -c "from degree_scaling_thresh_proj import plot_summary; plot_summary('data/degree_scaling_thresh_proj_npts19.csv', 'figures/degree_scaling_thresh_proj_npts19.pdf', max_exp=8)"`
