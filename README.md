# LF_SMF_HMF_ESR

Code for the paper *"The functional form of galaxy and halo luminosity
and mass functions"* (Ford, Desmond, Bartlett & Ferreira 2026).

This repository applies **Exhaustive Symbolic Regression**
([ESR](https://github.com/DeaglanBartlett/ESR)) to discover optimal fitting
functions for the galaxy luminosity function (LF), stellar mass function
(SMF), and halo mass function (HMF), ranking them by description length and
subjecting them to physicality checks.

## Requirements

### Python packages

```
numpy
scipy
matplotlib
sympy
mpi4py
psutil
prettytable
pytexit
numdifftools
```

### ESR library

The [Exhaustive Symbolic Regression](https://github.com/DeaglanBartlett/ESR)
library must be installed and importable.

## Configuration

Before running any script, edit the following machine-specific paths to match your setup:

**ESR library path (local-machine scripts).** Set to the directory
containing the ESR package on your machine:

```python
sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')  # <-- edit this
```

Appears in: `compute_combined_DL.py`, `fit_literature_all_sims.py`,
`hmf_covariance_analysis.py`, `build_fiducial_table.py`, `Pareto_plotter.py`.

**ESR library path (cluster scripts).** Same meaning, but the cluster
scripts hardcode a different path that you likewise need to update:

```python
sys.path.insert(0, '/mnt/zfsusers/ameliaford/original_ESR/ESR')  # <-- edit this
```

Appears in: `run_hmf_fiducial_step1.py`, `run_fiducial_hmf.py`,
`run_fiducial_hmf_re.py`, `run_fiducial_hmf_recovery.py`. (`sample_top_200.py`
has a commented-out placeholder on line 38 — uncomment and edit only if ESR
is not already on your `PYTHONPATH`.)

**ESR function-library path (`run_hmf_fiducial_step1.py` only).** On first
use, edit line 69 to point to the `base_e_maths` function-library directory
inside your ESR installation:

```python
fn_dir = '/mnt/zfsusers/ameliaford/original_ESR/ESR/esr/function_library/base_e_maths'
```
(Edit this line to match your ESR install.)

All other paths in the repository are relative and do not require editing.

## Data

All input data are in the `data/` directory.

### LF and SMF

Galaxy luminosity and stellar mass function data from
[Bernardi et al. (2013)](https://doi.org/10.1093/mnras/stt1191), using
SDSS r-band photometry. Four datasets:

| Dataset | File | Description |
|---------|------|-------------|
| LF Sérsic | `data/LF_Ser_L.txt` | Single-Sérsic luminosity function |
| LF cmodel | `data/LF_cmodel.txt` | cmodel photometry LF |
| SMF Sérsic | `data/SMF_Ser_M.txt` | Sérsic stellar mass function |
| SMF cmodel | `data/SMF_cmodel_M.txt` | cmodel stellar mass function |

Each file has 4 space-delimited columns:

| Column | LF | SMF | Units |
|--------|----|-----|-------|
| 1 | x = L / (10^9 L_sun) | x = M_* / (10^9 M_sun) | ESR fitting variable |
| 2 | log(phi) | log(phi) | Mpc^-3 dex^-1 |
| 3 | sigma(log phi) | sigma(log phi) | Poisson uncertainty |
| 4 | V_eff | V_eff | Mpc^3 |

Column 1 is the ESR variable `x` that appears directly in all fitted
functions. The original Bernardi et al. data has been pre-converted to
these units.

### HMF

Halo mass functions from 100 realisations of the
[Quijote](https://quijote-simulations.readthedocs.io) N-body simulations
(LCDM, Friends-of-Friends halos, linking length b=0.2). Data files:

- `data/hmf_files/hmf_<sim>.dat` (sim = 0, 1, ..., 99): 5 columns (see header):

| Column | Name | Description |
|--------|------|-------------|
| 1 | x = sigma | Mass variance (ESR fitting variable) |
| 2 | counts | Halo counts in this mass bin |
| 3 | \|d ln sigma / d log M\| | Derivative factor (`log` is base-10) |
| 4 | norm | = \|d ln sigma / d log M\| * (rho_m / M) * V_eff * d(log M) |
| 5 | log(M / (M_sun/h)) | Halo mass bin centre |

- `data/mass_variance_multiplier.txt`: mapping between halo mass and mass
  variance. 3 columns: log(M/(M_sun/h)), sigma,
  \|d ln sigma / d log M\| * (rho_m/M) / V_eff.

The ESR variable `x` for the HMF is `sigma`. The fitted quantity is the
multiplicity function f(sigma), related to the number density n(M) via
n(M) d(log M) = f(sigma) * (rho_m / M) * |d(ln sigma)|.

## Scripts

The analysis proceeds in several stages. Below they are grouped by function.

### 1. Fitting

| Script | Description |
|--------|-------------|
| `create_LF_cmodel_L_data.py` | LF cmodel: magnitude → luminosity |
| `fit_all.py` | ESR pipeline / literature fits for any dataset |
| `sample_top_200.py` | Three-stage HMF pipeline (`step1`/`step2`/`step3`) |
| `fit_literature_all_sims.py` | Fit Press-Schechter / Warren / Tinker to all 100 Quijote sims |
| `build_final_functions.py` | Assemble `*_final_functions.txt` |
| `build_searchcomp.py` | Function → search-complexity mapping |

Flag: `fit_literature_all_sims.py` and `build_searchcomp.py` default to the
fiducial (restricted) range; pass `--extended` for the full-range version.

Typical invocations:
```
python3 create_LF_cmodel_L_data.py
mpirun -np N python3 fit_all.py <dataset> esr <complexity>
python3 fit_all.py <dataset> paper
python3 sample_top_200.py step1|step2|step3
python3 fit_literature_all_sims.py [--extended]
python3 build_final_functions.py <dataset> [options]
python3 build_searchcomp.py [--extended]
```

### 2. Analysis

| Script | Description |
|--------|-------------|
| `compute_combined_DL.py` | Combined DL across 100 HMF sims |
| `physicality_checks.py` | Asymptotics / monotonicity / positivity / integrability tests |
| `find_PS_like_functions.py` | Identify ESR functions with PS-like 1/σ behaviour |
| `hmf_covariance_analysis.py` | Empirical covariance of HMF counts |
| `fisher_det_analysis.py` | Diagonal vs full-det Fisher code length |
| `fit_double_schechter.py` | Double-Schechter comparison for LF/SMF |
| `param_uncertainties.py` | Fisher parameter uncertainties for Tables 1–3 |
| `propagated_impact.py` | ESR-vs-literature impact on ρ★, ρL, N(>L), n(>M_h) |

Flag: `compute_combined_DL.py`, `find_PS_like_functions.py`, and
`hmf_covariance_analysis.py` default to the fiducial (restricted) range used
in the main text; pass `--extended` for the full-range (appendix) version.
`find_PS_like_functions.py` in fiducial mode produces the "26 PS-like in
top 200" number quoted in Sec 4.3. The fiducial `hmf_covariance_analysis.py`
is the 15-bin analysis quoted in Sec 4.5 (median var/mean ≈ 0.95,
max |ρ| ≈ 0.34); it writes `hmf_covariance_results[_fiducial].txt` and
`Final_Plots/hmf_correlation_matrix[_fiducial].pdf`.

Results files written by these scripts: `fisher_det_results.txt`,
`double_schechter_results.txt`, `param_uncertainties_results.txt`,
`propagated_impact_results.txt`.

### 3. Plotting

| Script | Description |
|--------|-------------|
| `function_plotter.py` | Best-fit curves + residuals + δ-NLL |
| `Pareto_plotter.py` | δ-DL / δ-NLL Pareto fronts with broken x-axes |
| `nll_contributions.py` | Per-bin δ-NLL contributions |
| `extrapolation_plotter.py` | Functions extrapolated beyond data range |
| `extrapolation_HMF_sigma.py` | HMF extrapolation in σ-space (helper) |
| `histogram_and_stacked_plots.py` | Ranking histogram + stacked bar chart |
| `veff_plotter.py` | Effective survey volume vs L/M |

All plot outputs are saved to `Final_Plots/` (the analysis scripts also
write diagnostic PNGs into `Plots/`). Both directories are created
automatically on first use by `setup_paths.py`, which each plotting script
imports.

### 4. HMF restricted mass range (fiducial analysis)

The fiducial HMF results in the paper exclude the two lowest-mass bins
(log(M/h⁻¹M☉) < 12.6, i.e. ≲40 particles per halo) and rerun the full ESR
search on this restricted range. The scripts below implement this pipeline:

| Script | Description | Runs on |
|--------|-------------|---------|
| `run_hmf_fiducial_step1.py` | Full ESR (complexities 4–10) on 10 representative sims | Cluster (MPI) |
| `run_hmf_fiducial_step2.py` | Identify top 200 by mean rank → `top_500_fiducial.txt` | Cluster |
| `run_fiducial_hmf.py` | Refit top 200 to all 100 sims | Cluster (MPI) |
| `run_fiducial_hmf_recovery.py` | Recovery for timed-out sims + re/im-stripped fits | Cluster (MPI) |
| `run_fiducial_hmf_re.py` | Fit functions with sympy `re()`/`im()` artifacts | Cluster (MPI) |
| `build_fiducial_table.py` | Paper table data | Local |
| `fiducial_checks_and_plots.py` | Physicality checks + restricted-range plots | Local |
| `generate_extended_appendix.py` | Appendix HMF Pareto + extrapolation plots | Local |

`generate_extended_appendix.py` produces the full-range Pareto +
extrapolation figures for Appendix A, with enlarged fonts matching Fig A1.
It imports `make_single_panel_figure` from `Pareto_plotter.py`.

Notes on the plotting scripts:

- `Pareto_plotter.py` exposes `make_single_panel_figure`, `plot_pareto`,
  `load_pareto_data`, `load_ps_like_for_hmf` etc. as importable utilities;
  its driver code is gated behind `if __name__ == '__main__':` so importing
  has no side effects.
- `extrapolation_plotter.py` overlays the best PS-like ESR function (rank
  14, Eq 10) on both HMF panels of `extrapolation_behaviour.pdf`.
- `fiducial_checks_and_plots.py` adds that PS-like function to all three
  panels of `HMF_functions_fiducial.pdf` (Fig 5).
- `function_plotter.py` accepts a `skip_sources` argument so the Fig 1
  panels can drop `DblSch.`/`Ber.orig` without editing the
  final-functions files.

## Intermediate data files

Produced by the fitting scripts and consumed by plotting/analysis scripts:

| File | Produced by | Description |
|------|-------------|-------------|
| `*_final_functions.txt` | `build_final_functions.py` | Top ESR + literature functions with parameters |
| `top_500_all.txt` | `sample_top_200.py step1` | Top 500 unique HMF functions (full range) |
| `top_500_fiducial.txt` | `run_hmf_fiducial_step2.py` | Top 500 (restricted range) |
| `hmf_data/hmf_<sim>_data/final_all.txt` | `sample_top_200.py step2` | Per-sim fits (full range) |
| `hmf_data/hmf_<sim>_data/final_all_fiducial.txt` | `run_fiducial_hmf.py` | Per-sim fits (restricted range) |
| `hmf_combined_DL_fiducial.txt` | `compute_combined_DL.py` | Combined DL ranking (fiducial) |
| `hmf_combined_DL.txt` | `compute_combined_DL.py --extended` | Combined DL ranking (full range) |
| `literature_fits_fiducial.txt` | `fit_literature_all_sims.py` | Per-sim literature fits (fiducial) |
| `literature_combined_DL_fiducial.txt` | `fit_literature_all_sims.py` | Combined literature DL (fiducial) |
| `literature_fits_all_sims.txt` | `fit_literature_all_sims.py --extended` | Per-sim literature fits (full range) |
| `literature_combined_DL.txt` | `fit_literature_all_sims.py --extended` | Combined literature DL (full range) |
| `ordered_gold.txt` | `sample_top_200.py step3` | Rank tally (full range) |
| `ordered_gold_fiducial.txt` | `fiducial_checks_and_plots.py` | Rank tally (restricted range) |
| `all_paper_fitting_data.txt` | `fit_all.py` (paper mode) | Literature fits per dataset |
| `hmf_50_final_functions_fiducial.txt` | `build_final_functions.py` ¹ | Per-complexity best + literature for sim 50 |
| `hmf_fiducial_searchcomp.txt` | `build_searchcomp.py` | Function → search-complexity (fiducial) |
| `hmf_func_gencomp.txt` ² | `build_searchcomp.py --extended` | Function → search-complexity (full range) |

¹ Exact command:
```
python3 build_final_functions.py hmf_50 \
    --combined hmf_combined_DL_fiducial.txt \
    --esr-dir hmf_fiducial_50_data \
    --outfile hmf_50_final_functions_fiducial.txt
```

² Consumed by `Pareto_plotter.py` and `generate_extended_appendix.py`.

`*_final_functions.txt` is semicolon-delimited:
`source;complexity;DL;NLL;plot_fcn;blank_fcn` (see *Key file format* below).

## Workflow

A typical end-to-end workflow:

```
1. Prepare data
   python3 create_LF_cmodel_L_data.py

2. Run ESR fits (cluster, one job per complexity per dataset)
   # LF/SMF: 4 datasets × complexities 6..10
   mpirun -np 64 python3 fit_all.py LF_Ser_L esr 6
   mpirun -np 64 python3 fit_all.py LF_Ser_L esr 7
   ...
   mpirun -np 64 python3 fit_all.py LF_Ser_L esr 10
   # (repeat for LF_cmodel_L, SMF_Ser_M, SMF_cmodel_M)

   # HMF: each Quijote sim × complexities 4..10 (100 sims × 7 comps = 700 jobs)
   for sim in $(seq 0 99); do
       for comp in 4 5 6 7 8 9 10; do
           mpirun -np 64 python3 fit_all.py hmf_${sim} esr ${comp}
       done
   done

3. Fit literature functions
   python3 fit_all.py LF_Ser_L paper
   python3 fit_literature_all_sims.py          # HMF literature fits

4. HMF pipeline (identify top functions, refit all sims)
   python3 sample_top_200.py step1
   mpirun -np 100 python3 sample_top_200.py step2
   python3 compute_combined_DL.py

5. Assemble *_final_functions.txt for each dataset
   python3 build_final_functions.py LF_Ser_L
   python3 build_final_functions.py LF_cmodel_L
   python3 build_final_functions.py SMF_Ser_M
   python3 build_final_functions.py SMF_cmodel_M
   python3 build_final_functions.py hmf_50 --combined hmf_combined_DL.txt

6. Fiducial (fiducial) HMF pipeline — main text (§4 of the paper)
   (Cluster: run_hmf_fiducial_step{1,2}.py, run_fiducial_hmf*.py)
   python3 compute_combined_DL.py
   python3 fit_literature_all_sims.py
   python3 build_final_functions.py hmf_50 --combined hmf_combined_DL_fiducial.txt \
       --esr-dir hmf_fiducial_50_data --outfile hmf_50_final_functions_fiducial.txt
   python3 build_searchcomp.py                 # -> hmf_fiducial_searchcomp.txt
   python3 build_fiducial_table.py
   python3 fiducial_checks_and_plots.py         # Figs 5, 6 + physicality checks
   python3 find_PS_like_functions.py           # "26 PS-like in top 200"

7. Other analysis and plots
   python3 physicality_checks.py
   python3 function_plotter.py
   python3 nll_contributions.py
   python3 Pareto_plotter.py
   python3 extrapolation_plotter.py
   python3 veff_plotter.py
   python3 histogram_and_stacked_plots.py      # or sample_top_200.py step3

7b. Full-range (extended) HMF — Appendix A of the paper
   python3 compute_combined_DL.py --extended
   python3 fit_literature_all_sims.py --extended
   python3 build_searchcomp.py --extended      # -> hmf_func_gencomp.txt
   python3 generate_extended_appendix.py      # Appendix A figures
   python3 find_PS_like_functions.py --extended

8. Covariance, Fisher and propagated-impact diagnostics — §4.5, §5
   # fiducial 15-bin covariance (needs hmf_50_final_functions_fiducial.txt from §6)
   python3 hmf_covariance_analysis.py
   python3 hmf_covariance_analysis.py --extended  # full-range (appendix)
   python3 fisher_det_analysis.py               # diagonal vs full-det DL
   python3 fit_double_schechter.py              # double-Schechter comparison
   python3 param_uncertainties.py               # Fisher uncertainties for Tables 1–3
   python3 propagated_impact.py                 # ρ★, ρL, N(>L), n(>M_h)
```

## Key file format: `*_final_functions.txt`

Plotting and analysis scripts read function definitions from
semicolon-delimited files with 6 columns:

```
source;complexity;DL;NLL;plot_fcn;blank_fcn
```
- **source**: `ESR`, `ESR_C` (combined-ranked), `ESR_T` (top-ranked),
  `Sch.`, `Ber.`, `P.Sch.`, `War.`, `Tin.`
- **complexity**: integer (number of operators in tree representation)
- **DL**: description length (nats)
- **NLL**: negative log-likelihood
- **plot_fcn**: evaluable Python string with numeric parameter values
  substituted, e.g. `0.122/53.8*(x/53.8)**(-0.289)*exp(-x/53.8)`
- **blank_fcn**: symbolic form with parameter placeholders, e.g.
  `a0/a1*(x/a1)**a2*exp(-x/a1)`

## Citation

If you use this code, please cite:

```
Ford, Desmond, Bartlett & Ferreira (2026),
"The functional form of galaxy and halo luminosity and mass functions"
arXiv:XXXX.XXXXX
```

and the ESR algorithm:

```
Bartlett, Desmond & Ferreira (2023),
"Exhaustive Symbolic Regression",
IEEE Transactions on Evolutionary Computation
doi:10.1109/TEVC.2023.3319309
```

## Contact

In case of questions or comments, email Harry Desmond (harry.desmond@port.ac.uk)
