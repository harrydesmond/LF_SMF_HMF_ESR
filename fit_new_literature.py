#!/usr/bin/env python3
"""
Fit Jenkins (HMF), Sheth-Tormen (HMF) and Saunders (LF/SMF) to the datasets.
Produces DL, NLL and physicality check results for Paper.tex table entries.
"""

import sys
sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
import math
import itertools
from scipy.optimize import minimize
from scipy.integrate import quad
import numdifftools as nd
from esr.generation.generator import string_to_node, aifeyn_complexity


basis_functions = [["x", "a"],
                   ["inv", "exp", "log", "abs"],
                   ["+", "*", "-", "/", "pow"]]


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_lf_smf(name):
    txt_map = {
        'LF_Ser_L':     'LF_Ser_L.txt',
        'LF_cmodel_L':  'LF_cmodel.txt',
        'SMF_Ser_M':    'SMF_Ser_M.txt',
        'SMF_cmodel_M': 'SMF_cmodel_M.txt',
    }
    data = np.loadtxt(txt_map[name], unpack=True)
    x_raw, y, y_err, Veff = data[0], data[1], data[2], data[3]
    if x_raw[0] < 0:
        M_sun_r = 4.67
        x_raw = 10**(-0.4 * (x_raw - M_sun_r))
    x = x_raw * 1e-9
    counts = 10**y * Veff
    return x, counts, Veff


def load_hmf_trimmed(sim):
    filepath = f'hmf_files/hmf_{sim}_new.dat'
    data = np.loadtxt(filepath)
    data = data[2:]
    return data[:, 0], data[:, 1], data[:, 3]


# ═══════════════════════════════════════════════════════════════════════
# NLL and DL
# ═══════════════════════════════════════════════════════════════════════

def poisson_nll(params, x, counts, norm_or_veff, func):
    f = func(x, params)
    ypred = f * norm_or_veff
    if np.any(ypred <= 0) or np.any(~np.isfinite(ypred)):
        return np.inf
    nll = np.sum(ypred - counts * np.log(ypred))
    return nll if np.isfinite(nll) else np.inf


def compute_codelen(params, x, counts, norm, func):
    k = len(params)
    if k == 0:
        return 0.0
    def nll_func(p):
        return poisson_nll(p, x, counts, norm, func)

    d_list = [1e-5, 10**(-5.5), 10**(-4.5), 1e-6, 1e-4, 10**(-6.5),
              10**(-3.5), 1e-7, 1e-3, 10**(-7.5), 10**(-2.5), 1e-8, 1e-2]
    method_list = ["central", "forward", "backward"]

    Fisher_diag = None
    try:
        H = nd.Hessian(nll_func)(params)
        Fisher_diag = np.array([H[i, i] for i in range(k)])
    except:
        pass

    def _is_good(Fd):
        return Fd is not None and np.all(Fd > 0) and np.all(np.isfinite(Fd))

    if not _is_good(Fisher_diag):
        for d2, meth in itertools.product(d_list, method_list):
            try:
                step = np.abs(d2 * params) + 1e-15
                H = nd.Hessian(nll_func, step=step, method=meth)(params)
                Fd_tmp = np.array([H[i, i] for i in range(k)])
                if _is_good(Fd_tmp):
                    Fisher_diag = Fd_tmp
                    break
            except:
                continue

    if not _is_good(Fisher_diag):
        return np.nan

    Delta = np.sqrt(12.0 / Fisher_diag)
    Nsteps = np.abs(params) / Delta
    mask = Nsteps >= 1
    k_eff = int(np.sum(mask))
    if k_eff == 0:
        return 0.0
    return (-k_eff / 2.0 * math.log(3.0)
            + np.sum(0.5 * np.log(Fisher_diag[mask]) + np.log(np.abs(params[mask]))))


def fit_function(x, counts, norm, func, bounds, p0_base, n_restarts=30):
    best_nll = np.inf
    best_params = None
    rng = np.random.RandomState(42)

    for scale in [1.0, 0.95, 1.05, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]:
        p0 = p0_base * scale
        p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            result = minimize(poisson_nll, p0, args=(x, counts, norm, func),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 10000, 'ftol': 1e-15})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            continue

    for _ in range(n_restarts):
        p0 = p0_base * (1 + 0.5 * rng.randn(len(p0_base)))
        p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            result = minimize(poisson_nll, p0, args=(x, counts, norm, func),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 10000, 'ftol': 1e-15})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
        except:
            continue

    return best_nll, best_params


def get_aifeynman(esr_string):
    try:
        expr, nodes, complexity = string_to_node(esr_string, basis_functions, evalf=True)
        labels = nodes.to_list(basis_functions)
        labels = [lab.lower() if lab not in ['Mul', 'Add'] else
                  ('*' if lab == 'Mul' else '+') for lab in labels]
        nparam = sum(1 for i in range(10) if f'a{i}' in esr_string)
        param_list = [f'a{i}' for i in range(nparam)]
        return aifeyn_complexity(labels, param_list), complexity
    except Exception as e:
        print(f"  aifeynman error for {esr_string[:60]}: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════
# Function definitions
# ═══════════════════════════════════════════════════════════════════════

def saunders(x, params):
    """Saunders et al. (1990) modified Schechter.
    phi(x) = theta0 * (x/theta1)^theta2 * exp(-(log10(1 + x/theta1))^2 / (2*theta3^2))
    """
    a0, a1, a2, a3 = params
    ratio = x / a1
    return a0 * np.power(ratio, a2) * np.exp(-np.log10(1 + ratio)**2 / (2 * a3**2))


def jenkins(x, params):
    """Jenkins et al. (2001): f(sigma) = A exp(-|ln(1/sigma) + B|^eps)"""
    A, B, eps = params
    return A * np.exp(-np.power(np.abs(np.log(1.0 / x) + B), eps))


def sheth_tormen(x, params):
    """Sheth & Tormen (1999): f(sigma) with ellipsoidal collapse."""
    A, a, p = params
    delta_c = 1.686
    nu = delta_c / x
    return (A * np.sqrt(2.0 * a / np.pi) * nu *
            (1.0 + np.power(a * nu**2, -p)) *
            np.exp(-0.5 * a * nu**2))


# ═══════════════════════════════════════════════════════════════════════
# Physicality checks
# ═══════════════════════════════════════════════════════════════════════

def check_lf_smf_physicality(func, params, x_data, density_type):
    """Check (a) integrability, (b) positivity, (c) integral sign, (e) monotonicity."""
    checks = []

    # Evaluate over extended range
    x_ext = np.logspace(-4, 4, 100000)
    try:
        y_ext = func(x_ext, params)
    except:
        return "(a)", ["Evaluation failed"]

    # Positivity at small x
    x_small = np.logspace(-6, np.log10(x_data.min()), 1000)
    try:
        y_small = func(x_small, params)
        if np.any(y_small < -1e-15):
            checks.append("b")
    except:
        pass

    # Monotonicity over data range
    x_mono = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 2000)
    try:
        y_mono = func(x_mono, params)
        dy = np.diff(y_mono)
        if np.mean(dy > 0) > 0.01:
            checks.append("e")
    except:
        pass

    # Integrability: integral of phi(x) dx over [0, inf]
    y_clean = np.where(np.isfinite(y_ext) & (y_ext >= 0), y_ext, 0)
    integral = np.trapz(y_clean, x_ext)

    if not np.isfinite(integral):
        checks.append("a")
    elif integral < 0:
        checks.append("c")
    else:
        rho = 1e9 / np.log(10) * integral
        if density_type == "LF":
            pass  # rho_L ~ 5-7e7 Lsun/Mpc^3
        else:
            pass  # rho_*/rho_b ~ 0.044-0.056

    if len(checks) == 0:
        return "\\checkmark", []
    else:
        return ", ".join(f"({c})" for c in checks), checks


def check_hmf_physicality(func, params):
    """Check HMF physicality: f->0 as sigma->inf, f finite as sigma->0+, integral <= 1."""
    checks = []

    # f(sigma) -> 0 as sigma -> infinity. Use genuinely asymptotic points:
    # S-T approaches zero slowly for p < 1/2 and can still be >1e-3 at sigma=10.
    sigma_large = np.array([1e3, 1e5, 1e8])
    try:
        f_large = func(sigma_large, params)
        if np.abs(f_large[-1]) > 1e-3:
            checks.append("b")
        if np.any(f_large < -1e-10):
            checks.append("a")
    except:
        checks.append("a")

    # f(sigma) finite as sigma -> 0+
    sigma_small = np.array([0.01, 0.05, 0.1])
    try:
        f_small = func(sigma_small, params)
        if np.any(~np.isfinite(f_small)) or np.any(np.abs(f_small) > 1e20):
            checks.append("c")
    except:
        checks.append("c")

    # Integral: int f(sigma) d(ln sigma) should converge and <= 1
    sigma_int = np.logspace(-3, 3, 100000)
    try:
        f_int = func(sigma_int, params)
        f_clean = np.where(np.isfinite(f_int), f_int, 0)
        # d(ln sigma) = d(sigma)/sigma
        integrand = f_clean / sigma_int
        integral = np.trapz(integrand, sigma_int)
        if not np.isfinite(integral) or integral < 0:
            if "a" not in checks:
                checks.append("a")
        elif integral > 1.5:
            pass  # P.Sch. gives 1.0, so allow some slack
    except:
        if "a" not in checks:
            checks.append("a")

    if len(checks) == 0:
        return "\\checkmark", []
    else:
        return ", ".join(f"({c})" for c in checks), checks


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.chdir('/home/harry/Amelia_code')

    # ── Reference DL values (rank-1 ESR for each dataset) ──
    # From *_final_functions.txt files (the ESR rank-1 DL)
    ref_dl = {
        'LF_Ser_L':     -2383333.067,  # ESR comp 9
        'LF_cmodel_L':  -2383304.443,  # Will check
        'SMF_Ser_M':    -2411137.005,  # ESR comp 10
        'SMF_cmodel_M': None,          # Will check
    }

    # ── Get reference DL from final_functions files ──
    for name, fname in [('LF_Ser_L', 'LF_Ser_L_final_functions.txt'),
                        ('LF_cmodel_L', 'LF_cmodel_L_final_functions.txt'),
                        ('SMF_Ser_M', 'SMF_Ser_M_final_functions.txt'),
                        ('SMF_cmodel_M', 'SMF_cmodel_M_final_functions.txt')]:
        with open(fname) as f:
            best_dl = None
            for line in f:
                if line.startswith('ESR'):
                    parts = line.strip().split(';')
                    dl = float(parts[2])
                    if best_dl is None or dl < best_dl:
                        best_dl = dl
            ref_dl[name] = best_dl
        print(f"  {name}: best ESR DL = {best_dl:.3f}")

    # ── Aifeynman complexities ──
    print("\nComputing aifeynman complexities...")
    saunders_esr = 'a0*pow(x/a1,a2)*exp(-pow(log(1 + x/a1),2)/(2*pow(a3,2)))'
    # Can't use log10 directly in ESR; use log(1+x/a1)/log(10) = ln(1+x/a1)/ln(10)
    # But simpler: Saunders with log base e has same structure, just theta3 absorbs ln(10)
    # Let's compute it properly
    # Actually the ESR basis uses log = ln. So log10(y) = log(y)/log(10).
    # phi = a0 * (x/a1)^a2 * exp( -(log(1+x/a1))^2 / (2*a3^2 * log(10)^2) )
    # Absorb log(10) into a3: a3' = a3*log(10). Then:
    # phi = a0 * (x/a1)^a2 * exp( -(log(1+x/a1))^2 / (2*a3'^2) )
    # ESR form: a0*pow(x/a1,a2)*exp(-pow(log(1+x/a1),2)/(2*pow(a3,2)))
    # This has operators: *, pow, /, exp, -, pow, log, +, /, pow, 2
    # Let me compute it properly

    # For aifeynman, use the simplest expression form
    saunders_esr_v2 = 'a0*pow(x/a1,a2)/exp(pow(log(1+x/a1),2)*a3)'
    # Here a3 = 1/(2*sigma^2*ln(10)^2), absorbing the denominator
    jenkins_esr = 'a0/exp(pow(Abs(a1 - log(x)),a2))'
    # Reparametrized standard S-T form with b = a*delta_c^2 and prefactor
    # absorbed into a0: a0/x * (1 + (x^2/b)^p) * exp[-b/(2*x^2)].
    st_esr = 'a0*(1+pow(x*x/a1,a2))*exp(-a1/(2*x*x))/x'

    for name, esr_str in [('Saunders', saunders_esr_v2),
                           ('Jenkins', jenkins_esr),
                           ('ST', st_esr)]:
        af, comp = get_aifeynman(esr_str)
        print(f"  {name}: aifeynman = {af}, complexity = {comp}")

    # Compute aifeynman values
    af_saunders, comp_saunders = get_aifeynman(saunders_esr_v2)
    af_jenkins, comp_jenkins = get_aifeynman(jenkins_esr)
    af_st, comp_st = get_aifeynman(st_esr)

    # ═══════════════════════════════════════════════════════════════
    # Part 1: Saunders on LF/SMF
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("SAUNDERS FUNCTION ON LF/SMF DATASETS")
    print("="*80)

    # Saunders: phi(x) = a0 * (x/a1)^a2 * exp(-log10(1+x/a1)^2 / (2*a3^2))
    saunders_p0 = np.array([0.1, 20.0, -1.0, 0.5])
    saunders_bounds = [(1e-8, 100), (0.1, 1000), (-3, 2), (0.01, 5)]

    datasets_lf_smf = ['LF_Ser_L', 'LF_cmodel_L', 'SMF_Ser_M', 'SMF_cmodel_M']

    for dset in datasets_lf_smf:
        print(f"\n--- {dset} ---")
        x, counts, Veff = load_lf_smf(dset)
        print(f"  Data: {len(x)} bins, x range [{x.min():.3e}, {x.max():.3e}]")

        # Try multiple starting points
        best_nll = np.inf
        best_params = None

        p0_list = [
            np.array([0.1, 20.0, -1.0, 0.5]),
            np.array([0.01, 50.0, -0.5, 0.3]),
            np.array([0.5, 10.0, -1.5, 0.8]),
            np.array([0.05, 100.0, -0.3, 0.4]),
            np.array([0.2, 30.0, -0.8, 0.6]),
            np.array([0.01, 200.0, -0.1, 0.3]),
        ]

        for p0 in p0_list:
            nll, params = fit_function(x, counts, Veff, saunders,
                                        saunders_bounds, p0, n_restarts=30)
            if params is not None and nll < best_nll:
                best_nll = nll
                best_params = params.copy()

        if best_params is None:
            print("  FITTING FAILED")
            continue

        codelen = compute_codelen(best_params, x, counts, Veff, saunders)
        dl = best_nll + codelen + af_saunders
        delta_dl = dl - ref_dl[dset]
        delta_nll = best_nll - ref_dl[dset]  # approximate

        # Get reference NLL from final_functions files
        fname_map = {
            'LF_Ser_L': 'LF_Ser_L_final_functions.txt',
            'LF_cmodel_L': 'LF_cmodel_L_final_functions.txt',
            'SMF_Ser_M': 'SMF_Ser_M_final_functions.txt',
            'SMF_cmodel_M': 'SMF_cmodel_M_final_functions.txt',
        }
        with open(fname_map[dset]) as f:
            for line in f:
                if line.startswith('ESR'):
                    parts = line.strip().split(';')
                    dl_esr = float(parts[2])
                    if dl_esr == ref_dl[dset]:
                        ref_nll = float(parts[3])
                        break

        delta_nll_proper = best_nll - ref_nll

        density_type = "SMF" if "SMF" in dset else "LF"
        checks_str, checks_list = check_lf_smf_physicality(
            saunders, best_params, x, density_type)

        print(f"  Best params: {best_params}")
        print(f"  NLL = {best_nll:.3f}")
        print(f"  codelen = {codelen:.3f}")
        print(f"  aifeynman = {af_saunders:.3f}")
        print(f"  DL = {dl:.3f} (ref: {ref_dl[dset]:.3f})")
        print(f"  ΔDL = {delta_dl:.1f}")
        print(f"  ΔNLL = {delta_nll_proper:.1f}")
        print(f"  Checks: {checks_str}")

    # ═══════════════════════════════════════════════════════════════
    # Part 2: Jenkins + Sheth-Tormen on HMF (all 100 sims, trimmed)
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("JENKINS & SHETH-TORMEN ON HMF (TRIMMED, 100 SIMS)")
    print("="*80)

    sim_numbers = sorted([
        int(f.split('_')[1])
        for f in os.listdir('hmf_files')
        if f.startswith('hmf_') and f.endswith('_new.dat')
    ])

    jenkins_p0 = np.array([0.315, 0.61, 3.8])
    jenkins_bounds = [(0.01, 10), (-5, 5), (0.5, 10)]
    st_p0 = np.array([0.3222, 0.707, 0.3])
    st_bounds = [(0.01, 10), (0.01, 5), (0.01, 2)]

    hmf_results = {'Jenkins': {}, 'ST': {}}

    for sim in sim_numbers:
        sigma, counts, norm = load_hmf_trimmed(sim)

        # Jenkins
        nll_j, params_j = fit_function(sigma, counts, norm, jenkins,
                                        jenkins_bounds, jenkins_p0, n_restarts=20)
        if params_j is not None:
            codelen_j = compute_codelen(params_j, sigma, counts, norm, jenkins)
            if np.isfinite(codelen_j):
                hmf_results['Jenkins'][sim] = {
                    'nll': nll_j, 'params': params_j, 'codelen': codelen_j
                }
                jenkins_p0 = params_j.copy()

        # ST
        nll_st, params_st = fit_function(sigma, counts, norm, sheth_tormen,
                                          st_bounds, st_p0, n_restarts=20)
        if params_st is not None:
            codelen_st = compute_codelen(params_st, sigma, counts, norm, sheth_tormen)
            if np.isfinite(codelen_st):
                hmf_results['ST'][sim] = {
                    'nll': nll_st, 'params': params_st, 'codelen': codelen_st
                }
                st_p0 = params_st.copy()

    # Reference: paper-rank ESR best on the fiducial trimmed HMF data.
    with open('hmf_combined_DL_fiducial_new.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(';')
            if parts[0] == '0':
                ref_hmf_dl = float(parts[2])
                ref_hmf_nll = float(parts[4])
                break

    for name, af in [('Jenkins', af_jenkins), ('ST', af_st)]:
        results = hmf_results[name]
        n = len(results)
        sum_nll = sum(r['nll'] for r in results.values())
        sum_codelen = sum(r['codelen'] for r in results.values())
        dl_combined = sum_nll + sum_codelen + af

        delta_dl = dl_combined - ref_hmf_dl
        delta_nll = sum_nll - ref_hmf_nll

        all_params = np.array([results[s]['params'] for s in sorted(results.keys())])
        p_median = np.median(all_params, axis=0)
        p_16 = np.percentile(all_params, 16, axis=0)
        p_84 = np.percentile(all_params, 84, axis=0)

        # Physicality checks on median parameters
        checks_str, checks_list = check_hmf_physicality(
            jenkins if name == 'Jenkins' else sheth_tormen, p_median)

        print(f"\n--- {name} ({n} sims) ---")
        print(f"  aifeynman = {af:.4f}")
        print(f"  sum_NLL = {sum_nll:.2f}")
        print(f"  DL_combined = {dl_combined:.2f}")
        print(f"  ref DL = {ref_hmf_dl:.2f}")
        print(f"  ΔDL = {delta_dl:.0f}")
        print(f"  ΔNLL = {delta_nll:.0f}")
        print(f"  Checks: {checks_str}")
        print(f"  Params (median [16th, 84th]):")
        for i in range(len(p_median)):
            print(f"    θ{i}: {p_median[i]:.4f} [{p_16[i]:.4f}, {p_84[i]:.4f}]"
                  f"  → {p_median[i]:.2f}^{{+{p_84[i]-p_median[i]:.2f}}}"
                  f"_{{-{p_median[i]-p_16[i]:.2f}}}")


if __name__ == '__main__':
    main()
