# LF_SMF_HMF_ESR

Code for the paper *"The functional form of galaxy and halo luminosity and mass functions"* (Ford, Desmond, Bartlett & Ferreira 2026).

This repository applies **Exhaustive Symbolic Regression** ([ESR](https://github.com/DeaglanBartlett/ESR)) to discover optimal fitting functions for the galaxy luminosity function (LF), stellar mass function (SMF), and halo mass function (HMF), ranking them by description length and subjecting them to physicality checks.

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

The [Exhaustive Symbolic Regression](https://github.com/DeaglanBartlett/ESR) library must be installed and importable.
Some scripts contain a hardcoded `sys.path.insert` pointing to a local ESR installation; update this path to match your setup.

## Data

All input data are in the `data/` directory.

### LF and SMF

Galaxy luminosity and stellar mass function data from [Bernardi et al. (2013)](https://doi.org/10.1093/mnras/stt1191), using SDSS r-band photometry. Four datasets:

| Dataset | File | Description |
|---------|------|-------------|
| LF Sersic | `data/LF_Ser_L.txt` | Single-Sersic luminosity function |
| LF cmodel | `data/LF_cmodel.txt` | cmodel photometry LF |
| SMF Sersic | `data/SMF_Ser_M.txt` | Sersic stellar mass function |
| SMF cmodel | `data/SMF_cmodel_M.txt` | cmodel stellar mass function |

Each file has 4 space-delimited columns:

| Column | LF | SMF | Units |
|--------|----|-----|-------|
| 1 | x = L / (10^9 L_sun) | x = M_* / (10^9 M_sun) | ESR fitting variable |
| 2 | log10(phi) | log10(phi) | Mpc^-3 dex^-1 |
| 3 | sigma(log10 phi) | sigma(log10 phi) | Poisson uncertainty |
| 4 | V_eff | V_eff | Mpc^3 |

Column 1 is the ESR variable `x` that appears directly in all fitted functions. The original Bernardi et al. data has been pre-converted to these units.

### HMF

Halo mass functions from 100 realisations of the [Quijote](https://quijote-simulations.readthedocs.io) N-body simulations (LCDM, Friends-of-Friends halos, linking length b=0.2). Data files:

- `data/hmf_files/hmf_<sim>.dat` (sim = 0, 1, ..., 99): 5 columns (see header):

| Column | Name | Description |
|--------|------|-------------|
| 1 | x = sigma | Mass variance (ESR fitting variable) |
| 2 | counts | Halo counts in this mass bin |
| 3 | \|d ln sigma / d log M\| | Derivative factor |
| 4 | norm | = \|d ln sigma / d log M\| * (rho_m / M) * V_eff * d(log M) |
| 5 | log10(M / (M_sun/h)) | Halo mass bin centre |

- `data/mass_variance_multiplier.txt`: mapping between halo mass and mass variance (3 columns: log10(M/(M_sun/h)), sigma, \|d ln sigma / d log M\| * (rho_m/M) / V_eff).

The ESR variable `x` for the HMF is `sigma`. The fitted quantity is the multiplicity function f(sigma), related to the number density n(M) via n(M) d(log M) = f(sigma) * (rho_m / M) * |d(ln sigma)|.

## Scripts

The analysis proceeds in several stages. Below they are grouped by function.

### 1. Fitting

| Script | Description | Usage |
|--------|-------------|-------|
| `create_LF_cmodel_L_data.py` | Convert LF cmodel data from magnitude to luminosity space for ESR | `python3 create_LF_cmodel_L_data.py` |
| `fit_all.py` | Run ESR pipeline or fit literature functions to any dataset | `mpirun -np N python3 fit_all.py <dataset> esr <complexity>` or `python3 fit_all.py <dataset> paper` |
| `sample_top_200.py` | Three-stage HMF pipeline: identify top functions, refit to all 100 sims, plot results | `python3 sample_top_200.py step1\|step2\|step3` |
| `fit_literature_all_sims.py` | Fit Press-Schechter, Warren, Tinker to all 100 Quijote sims | `python3 fit_literature_all_sims.py` |
| `build_final_functions.py` | Assemble `*_final_functions.txt` from ESR and literature outputs | `python3 build_final_functions.py <dataset> [options]` |

### 2. Analysis

| Script | Description | Usage |
|--------|-------------|-------|
| `compute_combined_DL.py` | Compute combined description length across 100 HMF sims | `python3 compute_combined_DL.py` |
| `physicality_checks.py` | Test functions for asymptotics, monotonicity, positivity, integrability | `python3 physicality_checks.py` |
| `find_PS_like_functions.py` | Identify ESR functions with Press-Schechter-like 1/sigma behaviour | `python3 find_PS_like_functions.py` |

### 3. Plotting

| Script | Description | Output |
|--------|-------------|--------|
| `function_plotter.py` | Best-fit functions overlaid on data with residuals and delta-NLL panels | `LF_SMF_functions.pdf`, `HMF_functions.pdf` |
| `Pareto_plotter.py` | Pareto fronts (delta-DL and delta-NLL vs complexity) with broken x-axes | `Pareto_all.pdf` + standalone panels |
| `nll_contributions.py` | Per-bin delta-NLL contributions for literature vs ESR functions | `LF_functions_NLL.pdf`, `SMF_functions_NLL.pdf`, `HMF_50_functions_NLL.pdf` |
| `extrapolation_plotter.py` | Top ESR + literature functions extrapolated beyond data range | `extrapolation_behaviour.pdf` |
| `extrapolation_HMF_sigma.py` | HMF extrapolation in sigma-space (helper for extrapolation_plotter) | `HMF_extrapolation_sigma.pdf` |
| `histogram_and_stacked_plots.py` | Histogram and stacked bar chart of HMF function rankings | `ranked_histogram.pdf`, `stacked_rank.pdf` |
| `veff_plotter.py` | Effective survey volume vs luminosity/mass | `Veff.pdf` |

All plot outputs are saved to a `Final_Plots/` directory (created if absent).

## Intermediate data files

Several intermediate files are produced by the fitting scripts and consumed by plotting/analysis scripts:

| File | Produced by | Description |
|------|-------------|-------------|
| `*_final_functions.txt` | `build_final_functions.py` | Semicolon-delimited file listing top ESR + literature functions with parameters (source; complexity; DL; NLL; plot_fcn; blank_fcn) |
| `top_500_all.txt` | `sample_top_200.py` step1 | Top 500 unique HMF functions ranked across 10 sims |
| `hmf_data/hmf_<sim>_data/final_all.txt` | `sample_top_200.py` step2 | Per-sim fit results for the top 200 functions |
| `hmf_combined_DL.txt` | `compute_combined_DL.py` | Combined DL ranking across 100 sims |
| `literature_fits_all_sims.txt` | `fit_literature_all_sims.py` | Per-sim literature function fit results |
| `literature_combined_DL.txt` | `fit_literature_all_sims.py` | Combined DL for literature functions |
| `ordered_gold.txt` | `sample_top_200.py` step3 | Rank tally data for stacked bar chart |
| `all_paper_fitting_data.txt` | `fit_all.py` (paper mode) | Literature function fit results per dataset |

## Workflow

A typical end-to-end workflow:

```
1. Prepare data
   python3 create_LF_cmodel_L_data.py

2. Run ESR fits (cluster, one job per complexity per dataset)
   mpirun -np 64 python3 fit_all.py LF_Ser_L esr 6
   mpirun -np 64 python3 fit_all.py LF_Ser_L esr 7
   ...
   mpirun -np 64 python3 fit_all.py LF_Ser_L esr 10

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

6. Analysis and plots
   python3 physicality_checks.py
   python3 find_PS_like_functions.py
   python3 function_plotter.py
   python3 nll_contributions.py
   python3 Pareto_plotter.py
   python3 extrapolation_plotter.py
   python3 veff_plotter.py
   python3 histogram_and_stacked_plots.py      # or sample_top_200.py step3
```

## Key file format: `*_final_functions.txt`

The plotting and analysis scripts read function definitions from semicolon-delimited files with 6 columns:

```
source;complexity;DL;NLL;plot_fcn;blank_fcn
```
- **source**: `ESR`, `ESR_C` (combined-ranked), `ESR_T` (top-ranked), `Sch.`, `Ber.`, `P.Sch.`, `War.`, `Tin.`
- **complexity**: integer (number of operators in tree representation)
- **DL**: description length (nats)
- **NLL**: negative log-likelihood
- **plot_fcn**: evaluable Python string with numeric parameter values substituted (e.g. `0.122/53.8*(x/53.8)**(-0.289)*exp(-x/53.8)`)
- **blank_fcn**: symbolic form with parameter placeholders (e.g. `a0/a1*(x/a1)**a2*exp(-x/a1)`)

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
