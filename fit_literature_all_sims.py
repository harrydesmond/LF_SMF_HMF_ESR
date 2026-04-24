"""
Fit literature HMF functions (Press-Schechter, Warren, Tinker) to all 100 Quijote sims.

The hmf_*.dat files have 5 columns: sigma, counts, col2, normalization, log10(M).
The ESR fits predict f(sigma) and NLL = sum(f*norm - counts*log(f*norm)).

Usage:
    python3 fit_literature_all_sims.py              # fiducial / restricted range (main text)
    python3 fit_literature_all_sims.py --extended   # full range (appendix)
"""

import argparse
import sys
sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import math
import itertools
from scipy.optimize import minimize
import numdifftools as nd
from esr.generation.generator import string_to_node, aifeyn_complexity


def load_hmf_data(sim, fiducial=False):
    filepath = f'data/hmf_files/hmf_{sim}.dat'
    data = np.loadtxt(filepath)
    if fiducial:
        data = data[2:]  # drop first 2 rows (lowest mass / highest sigma)
    return data[:, 0], data[:, 1], data[:, 3]  # sigma, counts, norm


# ── Literature functions: f(sigma) ──

def press_schechter(x, params=None):
    delta_c = 1.686
    return np.sqrt(2.0 / np.pi) * (delta_c / x) * np.exp(-0.5 * (delta_c / x)**2)

def warren(x, params):
    a0, a1, a2, a3 = params
    return a0 * (np.power(x, a2) + a1) * np.exp(-a3 * np.power(x, -2.0))

def tinker(x, params):
    a0, a1, a2, a3 = params
    return a0 * (np.power(x / a2, -a1) + 1.0) * np.exp(-a3 * np.power(x, -2.0))


def poisson_nll(params, sigma, counts, norm, func):
    f = func(sigma, params)
    ypred = f * norm
    if np.any(ypred <= 0) or np.any(~np.isfinite(ypred)):
        return np.inf
    nll = np.sum(ypred - counts * np.log(ypred))
    if not np.isfinite(nll):
        return np.inf
    return nll


def compute_codelen(params, sigma, counts, norm, func):
    k = len(params)
    if k == 0:
        return 0.0
    def nll_func(p):
        return poisson_nll(p, sigma, counts, norm, func)

    # Match ESR's robust Hessian strategy: try default, then sweep step sizes
    d_list = [1e-5, 10**(-5.5), 10**(-4.5), 1e-6, 1e-4, 10**(-6.5),
              10**(-3.5), 1e-7, 1e-3, 10**(-7.5), 10**(-2.5), 1e-8, 1e-2]
    method_list = ["central", "forward", "backward"]

    Fisher_diag = None
    try:
        H = nd.Hessian(nll_func)(params)
        Fisher_diag = np.array([H[i, i] for i in range(k)])
    except Exception:
        pass

    def _is_good(Fd, p):
        if Fd is None:
            return False
        if np.any(Fd <= 0) or np.any(~np.isfinite(Fd)):
            return False
        return True

    if not _is_good(Fisher_diag, params):
        for d2, meth in itertools.product(d_list, method_list):
            try:
                step = np.abs(d2 * params) + 1e-15
                H = nd.Hessian(nll_func, step=step, method=meth)(params)
                Fd_tmp = np.array([H[i, i] for i in range(k)])
                if _is_good(Fd_tmp, params):
                    Fisher_diag = Fd_tmp
                    break
            except Exception:
                continue

    if not _is_good(Fisher_diag, params):
        return np.nan

    Delta = np.sqrt(12.0 / Fisher_diag)
    Nsteps = np.abs(params) / Delta
    mask = Nsteps >= 1
    k_eff = int(np.sum(mask))
    if k_eff == 0:
        return 0.0
    codelen = (-k_eff / 2.0 * math.log(3.0)
               + np.sum(0.5 * np.log(Fisher_diag[mask]) + np.log(np.abs(params[mask]))))
    return codelen


def fit_function(sigma, counts, norm, func, bounds, p0_base, n_restarts=10):
    """Fit using L-BFGS-B with perturbations around a known good starting point."""
    best_nll = np.inf
    best_params = None
    rng = np.random.RandomState(42)

    # Try the base starting point first
    for scale in [1.0, 0.95, 1.05, 0.9, 1.1]:
        p0 = p0_base * scale
        try:
            result = minimize(poisson_nll, p0, args=(sigma, counts, norm, func),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 10000, 'ftol': 1e-15})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            continue

    # Additional random perturbations
    for _ in range(n_restarts):
        p0 = p0_base * (1 + 0.3 * rng.randn(len(p0_base)))
        p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            result = minimize(poisson_nll, p0, args=(sigma, counts, norm, func),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 10000, 'ftol': 1e-15})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except Exception:
            continue

    return best_nll, best_params


# ── aifeynman ──

basis_functions = [["x", "a"],
                   ["inv", "exp", "log", "abs"],
                   ["+", "*", "-", "/", "pow"]]

def get_aifeynman(esr_string):
    try:
        expr, nodes, complexity = string_to_node(esr_string, basis_functions, evalf=True)
        labels = nodes.to_list(basis_functions)
        labels = [lab.lower() if lab not in ['Mul', 'Add'] else
                  ('*' if lab == 'Mul' else '+') for lab in labels]
        nparam = sum(1 for i in range(10) if f'a{i}' in esr_string)
        param_list = [f'a{i}' for i in range(nparam)]
        return aifeyn_complexity(labels, param_list)
    except Exception as e:
        print(f"  Warning: aifeynman failed for {esr_string[:60]}: {e}")
        return None


# Known good parameters from sim 50 (hmf_50_final_functions.txt; extended fits)
LITERATURE_FUNCTIONS = {
    'P.Sch.': {
        'func': press_schechter,
        'nparam': 0,
        'bounds': [],
        'p0': np.array([]),
        'esr_string': 'pow(2/3.141592,0.5)*(1.686/x)*exp(-0.5*pow(1.686/x,2))',
    },
    'War.': {
        'func': warren,
        'nparam': 4,
        'bounds': [(0.01, 100), (-10, 10), (-5, 5), (0.01, 10)],
        'p0': np.array([3.954, -0.851, -0.080, 0.811]),
        'esr_string': 'a0*(pow(x,a2)+a1)*exp(-a3*pow(x,-2))',
    },
    'Tin.': {
        'func': tinker,
        'nparam': 4,
        'bounds': [(1e-6, 10), (0.01, 5), (1, 100000), (0.01, 10)],
        'p0': np.array([0.00234, 0.794, 1224, 0.926]),
        'esr_string': 'a0*(pow(x/a2,-a1)+1)*exp(-a3*pow(x,-2))',
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--extended', action='store_true',
                        help='Use full-range HMF data (keep all bins). Default: fiducial (drop 2 lowest-mass bins).')
    args = parser.parse_args()
    fiducial = not args.extended

    if fiducial:
        per_sim_out = 'literature_fits_fiducial.txt'
        combined_out = 'literature_combined_DL_fiducial.txt'
        mode_label = 'FIDUCIAL'
    else:
        per_sim_out = 'literature_fits_all_sims.txt'
        combined_out = 'literature_combined_DL.txt'
        mode_label = 'EXTENDED'

    sim_numbers = sorted([
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir('data/hmf_files')
        if f.startswith('hmf_') and f.endswith('.dat')
    ])
    print(f"Found {len(sim_numbers)} simulations: {sim_numbers[0]}..{sim_numbers[-1]}")

    if not fiducial:
        # Validate against known ESR fit for sim 50 (extended reference values)
        print("\nValidating data format against known ESR fit (sim 50)...")
        sigma50, counts50, norm50 = load_hmf_data(50, fiducial=False)
        a = [0.65734271, 1.70541354, 1.63275111, 2.78494662]
        ypred = np.abs(a[0])**np.exp(np.abs(a[1])**(np.abs(a[2] - sigma50)**a[3])) * norm50
        nll_check = np.sum(ypred - counts50 * np.log(ypred))
        print(f"  Computed NLL = {nll_check:.2f}, expected = -35593614.30")
        w_params = np.array([3.954474, -0.851404, -0.08000, 0.810767])
        nll_w = poisson_nll(w_params, sigma50, counts50, norm50, warren)
        print(f"  Warren sim50 NLL = {nll_w:.2f}, expected = -35593571.50")

    # Compute aifeynman
    print("\nComputing aifeynman...")
    aifeyn_values = {}
    for name, info in LITERATURE_FUNCTIONS.items():
        af = get_aifeynman(info['esr_string'])
        aifeyn_values[name] = af
        print(f"  {name}: aifeynman = {af}")

    # Fit
    print(f"\nFitting literature functions to {len(sim_numbers)} sims ({mode_label} data)...")
    results = {name: {} for name in LITERATURE_FUNCTIONS}

    for sim_idx, sim in enumerate(sim_numbers):
        try:
            sigma, counts, norm = load_hmf_data(sim, fiducial=fiducial)
        except Exception as e:
            print(f"  Could not load sim {sim}: {e}")
            continue

        for name, info in LITERATURE_FUNCTIONS.items():
            func = info['func']

            if info['nparam'] == 0:
                f = func(sigma)
                ypred = f * norm
                if np.any(ypred <= 0):
                    continue
                nll = np.sum(ypred - counts * np.log(ypred))
                params = np.array([])
                codelen = 0.0
            else:
                nll, params = fit_function(sigma, counts, norm, func,
                                           info['bounds'], info['p0'])
                if params is None:
                    print(f"  {name} sim {sim}: fitting failed")
                    continue
                codelen = compute_codelen(params, sigma, counts, norm, func)

            if np.isfinite(nll) and np.isfinite(codelen):
                aifeyn = aifeyn_values[name]
                dl = nll + codelen + aifeyn
                results[name][sim] = {
                    'nll': -nll,
                    'dl': -dl,
                    'codelen': codelen,
                    'params': params.copy(),
                }

                # Warm start from last good fit
                if info['nparam'] > 0:
                    info['p0'] = params.copy()

        if (sim_idx + 1) % 10 == 0 or sim_idx == 0:
            print(f"  Completed {sim_idx+1}/{len(sim_numbers)} sims", flush=True)

    if not fiducial:
        # Validate sim 50 (extended only: reference values are for extended fits)
        print("\nValidation against hmf_50_final_functions.txt:")
        for name in LITERATURE_FUNCTIONS:
            if 50 in results[name]:
                r = results[name][50]
                pstr = ', '.join(f'{p:.6f}' for p in r['params']) if len(r['params']) > 0 else 'none'
                print(f"  {name}: NLL={r['nll']:.2f}, DL={r['dl']:.2f}, params=[{pstr}]")

    # Combined DL
    print(f"\n{'='*80}")
    print(f"Combined results across simulations ({mode_label})")
    print(f"{'='*80}")

    for name, info in LITERATURE_FUNCTIONS.items():
        sim_results = results[name]
        n_sims = len(sim_results)
        if n_sims == 0:
            print(f"\n{name}: no valid fits")
            continue

        aifeyn = aifeyn_values[name]
        sum_DL = sum(r['dl'] for r in sim_results.values())
        sum_NLL = sum(r['nll'] for r in sim_results.values())
        DL_combined = sum_DL + (n_sims - 1) * aifeyn

        print(f"\n{name}:")
        print(f"  n_sims = {n_sims}")
        print(f"  aifeynman = {aifeyn:.6f}")
        print(f"  sum_NLL = {sum_NLL:.2f}")
        print(f"  sum_DL = {sum_DL:.2f}")
        print(f"  DL_combined = {DL_combined:.2f}")

        if info['nparam'] > 0:
            all_params = np.array([sim_results[s]['params'] for s in sorted(sim_results.keys())])
            print(f"  Parameter statistics across {n_sims} sims:")
            for j in range(info['nparam']):
                vals = all_params[:, j]
                print(f"    a{j}: mean={np.mean(vals):.6f}, "
                      f"16th={np.percentile(vals, 16):.6f}, "
                      f"84th={np.percentile(vals, 84):.6f}")

    # Save per-sim fits
    with open(per_sim_out, 'w') as f:
        f.write('# name;sim;DL;NLL;codelen;params\n')
        for name in LITERATURE_FUNCTIONS:
            for sim in sorted(results[name].keys()):
                r = results[name][sim]
                pstr = ' '.join(f'{p:.10f}' for p in r['params']) if len(r['params']) > 0 else 'none'
                f.write(f"{name};{sim};{r['dl']:.6f};{r['nll']:.6f};{r['codelen']:.6f};{pstr}\n")
    print(f"\nPer-sim results saved to {per_sim_out}")

    # Save combined DL
    with open(combined_out, 'w') as f:
        f.write('# name;DL_combined;sum_NLL;sum_DL;aifeynman;n_sims\n')
        for name in LITERATURE_FUNCTIONS:
            sim_results = results[name]
            n_sims = len(sim_results)
            if n_sims == 0:
                continue
            aifeyn = aifeyn_values[name]
            sum_DL = sum(r['dl'] for r in sim_results.values())
            sum_NLL = sum(r['nll'] for r in sim_results.values())
            DL_combined = sum_DL + (n_sims - 1) * aifeyn
            f.write(f"{name};{DL_combined:.6f};{sum_NLL:.6f};{sum_DL:.6f};{aifeyn:.6f};{n_sims}\n")
    print(f"Combined results saved to {combined_out}")


if __name__ == '__main__':
    main()
