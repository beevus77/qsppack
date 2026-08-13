# Explorations Cleanup Manifest

## Purpose

This manifest separates supported package material from paper reproduction and
research history. It records what can leave the advertised branch after the
research snapshot is created. It does not authorize deleting the only copy of a
paper input.

Inventory date: 2026-08-13.

Current tracked inventory:

| Area | Files | Approximate size | Classification |
| --- | ---: | ---: | --- |
| `explorations/recovery` | 153 | 44.2 MB | Paper staging mixed with obsolete outputs |
| `explorations/notebooks` | 14 | 2.5 MB | Archive only |
| `explorations/figures` | 20 | 1.2 MB | Generated archive output |
| `explorations/Remez_QSP_Filter` | 2 | 0.8 MB | Superseded implementation |
| `explorations/data` | 4 | 0.01 MB | Archive only |

Of the 194 tracked files, 85 are rendered figures, 36 are CSV files, and 13 are
notebooks. The largest file is `recovery/plots/zeros.eps` at 27 MB.

## Supported Surface

These are maintained outside `explorations` and must remain on the advertised
branch:

- `qsppack/remez.py` and `tests/test_remez.py`
- `qsppack/retraction.py`, the FFT-based `qsppack.nlfa.weiss`, and
  `tests/test_retraction.py`
- `docs/source/remez_tutorial.ipynb`
- `docs/source/retraction_tutorial.ipynb`
- Their API pages and documentation navigation

Exploration scripts should be migrated to these APIs before being retained as
public examples. New public documentation must not import exploration modules or
read exploration-only CSV files.

## Publication Blockers

1. Figure 9 references
   `data/fgt_polynomial_space_convergence_deg_101_mat_inv_epsil4_N16.csv`, but
   that file is absent. The checked-in alternative is the `N15` dataset. Generate
   and validate the documented `N16` input or deliberately revise the figure and
   registry to use `N15`.
2. Figure 7 has multiple checked-in candidates. Based on the parity and solver
   corrections, the expected canonical candidate is
   `degree_scaling_uniform_sv_amp_npts19_clarabel_odd_Nweiss16.csv`, but this
   designation must be confirmed against the paper source before older datasets
   are removed.
3. Figure 5 is now reproduced through the public retraction notebook. Compare its
   rendered values and styling with the paper asset before removing its legacy
   CSV and plotting scripts.
4. The package version remains `0.3.12`; versioning and publication are explicitly
   outside this cleanup phase.

## Paper Reproduction Staging

Retain these files until each row is reproduced by a maintained notebook or a
small dedicated script outside `explorations`. Generated PDFs are comparison
artifacts, not source inputs.

| Figure | Source programs | Required data | Status |
| --- | --- | --- | --- |
| 1 | `generate_fig_1_data.py`, `plot_fig_1.py` | `fig_1_constraint_violation_vs_npts.csv` | Retain |
| 5 | `fgt_polynomial_space.py`, `plot_fit_polynomial_space.py` | `fgt_polynomial_space_convergence_deg_101_N18.csv` | Migrated to retraction tutorial; visual comparison pending |
| 6 | `fgt_polynomial_space.py`, `plot_recovery_conv_polynomial_space.py` | Figure 5 CSV plus `fgt_polynomial_space_convergence_deg_101_epsil4_N16.csv` | Retain |
| 7 | `degree_scaling_data_odd.py`, `degree_scaling_plot.py` | Expected Clarabel odd-parity `Nweiss16` CSV | Canonical choice pending |
| 8 | `mat_inv_ex.py` | Computed directly | Retain until public API rewrite |
| 9 | `fgt_polynomial_space.py`, `plot_recovery_conv_polynomial_space.py` | Matrix-inversion `epsil0_N18` and missing `epsil4_N16` CSVs | Blocked by missing input |
| 10 | `degree_scaling_data.py`, `degree_scaling_plot.py` | `degree_scaling_mat_inv_npts19.csv` | Retain |
| 11 | `plot_thresh_proj_ex.py` | Computed directly | Retain until public API rewrite |
| 12 | `fgt_polynomial_space.py`, `plot_recovery_conv_polynomial_space.py` | Threshold-projection `epsil0_N15` and `epsil4_N15` CSVs | Retain |
| 13 | `degree_scaling_thresh_proj.py` | `degree_scaling_thresh_proj_npts19.csv` | Retain |

`recovery/FIGURES.md` remains the detailed command registry during migration.

## Migrate Then Remove

These scripts contain useful workflows but duplicate package functionality or
combine numerical work with paper styling:

- `recovery/fgt_polynomial_space.py`: replace its local `recovered_coeffs` with
  `qsppack.retract` while its convergence figures remain staged.
- `recovery/mat_inv_ex.py`: replace direct low-level NLFA composition with
  `qsppack.retract`, then move the explanatory workflow into documentation.
- `recovery/degree_scaling_data.py`, `degree_scaling_data_odd.py`,
  `degree_scaling_thresh_proj.py`, and `rerun_odd_retraction.py`: reduce to
  reproducible benchmark drivers using public APIs.
- `recovery/plot_recovery_conv_polynomial_space.py` and
  `degree_scaling_plot.py`: retain only while the paper requires exact plot
  regeneration; do not promote publication styling into the package API.
- `recovery/generate_fig_1_data.py`, `plot_fig_1.py`, and
  `plot_thresh_proj_ex.py`: either turn into concise documentation examples or
  archive after the paper assets are frozen.

## Archive Only

Preserve these on the research snapshot branch, but omit them from the advertised
branch after the archive gate:

- All of `explorations/notebooks`, `explorations/figures`, and
  `explorations/data`. These record NLFT stability, optimization landscapes,
  perturbation experiments, and generated outputs, but are not supported
  examples.
- `explorations/Remez_QSP_Filter` and
  `explorations/recovery/Remez_QSP_Filter`. Both are superseded by
  `qsppack/remez.py`; their notebooks, pickle, and rendered PDFs are historical.
- `recovery/constrainapprox.py`, `constrainapprox.md`,
  `degree_scaling_constrainapprox.py`, exact-bound SDP data, and method-comparison
  plots. This is a separate experimental method, not the advertised Remez or
  retraction API.
- `recovery/degree 51 poor convergence`, `output_conv_analysis.txt`,
  `nlft_examples.ipynb`, `other_constraint_recovery.py`,
  `recovery_examples.py`, `weiss_roots.py`, and the ad hoc comparison scripts.
- Noncanonical convergence and degree-scaling parameter sweeps after the
  canonical paper datasets are verified.

## Delete From Advertised Branch

After the archive gate, remove these generated or machine-local artifacts rather
than moving them elsewhere in the public tree:

- Every tracked `.DS_Store` file and `.mplconfig` directory
- `recovery/plots` in full, including the 27 MB `zeros.eps`
- Intermediate PNG/PDF/SVG/PPTX outputs not referenced by the final paper or docs
- Empty/header-only CSV files and superseded parameter sweeps
- `__pycache__`, notebook checkpoints, and transient logs

The root `.gitignore` now prevents new `.DS_Store` and `.mplconfig` files.

## Removal Gate

Removal from the advertised branch should happen only after all of these checks:

1. Create and push a named research snapshot branch or tag containing the current
   exploration history.
2. Resolve the Figure 7 canonical dataset and Figure 9 missing input.
3. Execute the Remez and retraction tutorials from a clean wheel installation.
4. Regenerate each retained paper figure from the registry and compare it with
   the paper asset.
5. Replace low-level retraction composition in any retained scripts with
   `qsppack.retract`.
6. Remove archive-only and generated files in one reviewable commit.
7. Build docs with warnings as errors and run the package test suite again.

PyPI publication, version increments, and the upstream PR remain separate,
explicitly user-triggered steps.
