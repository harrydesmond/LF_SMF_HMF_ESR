"""
Fit Sheth-Tormen and Jenkins to the UNTRIMMED sim-50 HMF, to add them to the
appendix-B figures (B1/B2/B3). Self-contained copies of the exact fitting math
used by fit_literature_all_sims.py / fit_new_literature.py (no ESR/aifeynman
import). The data-independent aifeynman codelength is recovered from the
existing TRIMMED fits, then DL = NLL + codelen + aifeynman, matching the
convention in hmf_50_final_functions_extended.txt.
"""
import os
import math
import itertools
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_hmf_untrimmed(sim):
    data = np.loadtxt(f'hmf_files/hmf_{sim}_new.dat')
    return data[:, 0], data[:, 1], data[:, 3]          # sigma, counts, norm


def load_hmf_trimmed(sim):
    data = np.loadtxt(f'hmf_files/hmf_{sim}_new.dat')[2:]
    return data[:, 0], data[:, 1], data[:, 3]


# ── functions (exact copies) ──
def press_schechter(x, params=None):
    dc = 1.686
    return np.sqrt(2.0 / np.pi) * (dc / x) * np.exp(-0.5 * (dc / x)**2)

def warren(x, params):
    a0, a1, a2, a3 = params
    return a0 * (np.power(x, a2) + a1) * np.exp(-a3 * np.power(x, -2.0))

def tinker(x, params):
    a0, a1, a2, a3 = params
    return a0 * (np.power(x / a2, -a1) + 1.0) * np.exp(-a3 * np.power(x, -2.0))

def jenkins(x, params):
    a0, a1, a2 = params
    return a0 * np.exp(-np.power(np.abs(np.log(1.0 / x) + a1), a2))

def sheth_tormen(x, params):
    a0, a1, a2 = params
    dc = 1.686
    nu = dc / x
    return (a0 * np.sqrt(2.0 * a1 / np.pi) * nu *
            (1.0 + np.power(a1 * nu**2, -a2)) * np.exp(-0.5 * a1 * nu**2))


def poisson_nll(params, sigma, counts, norm, func):
    f = func(sigma, params)
    ypred = f * norm
    if np.any(ypred <= 0) or np.any(~np.isfinite(ypred)):
        return np.inf
    nll = np.sum(ypred - counts * np.log(ypred))
    return nll if np.isfinite(nll) else np.inf


def compute_codelen(params, sigma, counts, norm, func):
    k = len(params)
    if k == 0:
        return 0.0
    def nll_func(p):
        return poisson_nll(p, sigma, counts, norm, func)
    d_list = [1e-5, 10**(-5.5), 10**(-4.5), 1e-6, 1e-4, 10**(-6.5),
              10**(-3.5), 1e-7, 1e-3, 10**(-7.5), 10**(-2.5), 1e-8, 1e-2]
    method_list = ["central", "forward", "backward"]
    Fisher_diag = None
    try:
        H = nd.Hessian(nll_func)(params)
        Fisher_diag = np.array([H[i, i] for i in range(k)])
    except Exception:
        pass
    def _good(Fd):
        return Fd is not None and np.all(Fd > 0) and np.all(np.isfinite(Fd))
    if not _good(Fisher_diag):
        for d2, meth in itertools.product(d_list, method_list):
            try:
                step = np.abs(d2 * params) + 1e-15
                H = nd.Hessian(nll_func, step=step, method=meth)(params)
                Fd = np.array([H[i, i] for i in range(k)])
                if _good(Fd):
                    Fisher_diag = Fd
                    break
            except Exception:
                continue
    if not _good(Fisher_diag):
        return np.nan
    Delta = np.sqrt(12.0 / Fisher_diag)
    mask = (np.abs(params) / Delta) >= 1
    k_eff = int(np.sum(mask))
    if k_eff == 0:
        return 0.0
    return (-k_eff / 2.0 * math.log(3.0)
            + np.sum(0.5 * np.log(Fisher_diag[mask]) + np.log(np.abs(params[mask]))))


def fit_function(sigma, counts, norm, func, bounds, p0_base, n_restarts=20):
    best_nll, best_params = np.inf, None
    rng = np.random.RandomState(42)
    for scale in [1.0, 0.95, 1.05, 0.9, 1.1, 0.8, 1.2]:
        p0 = np.clip(p0_base * scale, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            r = minimize(poisson_nll, p0, args=(sigma, counts, norm, func),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 10000, 'ftol': 1e-15})
            if r.fun < best_nll:
                best_nll, best_params = r.fun, r.x.copy()
        except Exception:
            pass
    for _ in range(n_restarts):
        p0 = np.clip(p0_base * (1 + 0.5 * rng.randn(len(p0_base))),
                     [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            r = minimize(poisson_nll, p0, args=(sigma, counts, norm, func),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 10000, 'ftol': 1e-15})
            if r.fun < best_nll:
                best_nll, best_params = r.fun, r.x.copy()
        except Exception:
            pass
    return best_nll, best_params


# ════════════════════════════════════════════════════════════════════
sigma_u, counts_u, norm_u = load_hmf_untrimmed(50)
sigma_t, counts_t, norm_t = load_hmf_trimmed(50)
print(f"untrimmed bins: {len(sigma_u)}, trimmed bins: {len(sigma_t)}")

# ── Validation: reproduce existing untrimmed P.Sch/War/Tin NLL ──
print("\n=== VALIDATION (untrimmed sim-50 NLL vs hmf_50_final_functions_extended.txt) ===")
val = {
    'P.Sch.': (press_schechter, None, -35298722.42040113),
    'War.':   (warren, [3.9544740027300525, -0.8514044830451555,
                        -0.08000126097738508, 0.810767329759749], -35593571.50343531),
    'Tin.':   (tinker, [0.0023429021772495826, 0.7944154039239852,
                        1223.5821328359439, 0.9264052682953166], -35593532.574881725),
}
for nm, (fn, pars, nll_file) in val.items():
    p = np.array(pars) if pars is not None else None
    nll = poisson_nll(p, sigma_u, counts_u, norm_u, fn)
    print(f"  {nm:7s} NLL computed = {nll:.4f}   file = {nll_file:.4f}   "
          f"diff = {nll - nll_file:+.4f}")

# ── Recover aifeynman term from TRIMMED fits ──
# trimmed file rows (params baked into the function strings):
jen_p_trim = np.array([0.31880758, 0.60138646, 3.31853711])
st_p_trim  = np.array([0.32594737, 0.64429254, 0.22974108])
jen_DL_trim, jen_NLL_trim = -14352880.0,  -14352895.713
st_DL_trim,  st_NLL_trim  = -14352886.3,  -14352955.397

# sanity: NLL on trimmed data with trimmed params should match the file
jen_nll_check = poisson_nll(jen_p_trim, sigma_t, counts_t, norm_t, jenkins)
st_nll_check  = poisson_nll(st_p_trim,  sigma_t, counts_t, norm_t, sheth_tormen)
print("\n=== aifeynman recovery (trimmed cross-check) ===")
print(f"  Jen trimmed NLL: computed {jen_nll_check:.3f}  file {jen_NLL_trim:.3f}  "
      f"diff {jen_nll_check - jen_NLL_trim:+.3f}")
print(f"  S-T trimmed NLL: computed {st_nll_check:.3f}  file {st_NLL_trim:.3f}  "
      f"diff {st_nll_check - st_NLL_trim:+.3f}")

cl_jen_trim = compute_codelen(jen_p_trim, sigma_t, counts_t, norm_t, jenkins)
cl_st_trim  = compute_codelen(st_p_trim,  sigma_t, counts_t, norm_t, sheth_tormen)
af_jen = (jen_DL_trim - jen_NLL_trim) - cl_jen_trim
af_st  = (st_DL_trim  - st_NLL_trim)  - cl_st_trim
print(f"  Jenkins: codelen_trim={cl_jen_trim:.3f}  ->  aifeynman={af_jen:.3f}")
print(f"  S-T:     codelen_trim={cl_st_trim:.3f}  ->  aifeynman={af_st:.3f}")

# ── Fit untrimmed ──
print("\n=== UNTRIMMED sim-50 fits ===")
jen_nll_u, jen_p_u = fit_function(sigma_u, counts_u, norm_u, jenkins,
                                  [(0.01, 10), (-5, 5), (0.5, 10)],
                                  np.array([0.315, 0.61, 3.8]))
cl_jen_u = compute_codelen(jen_p_u, sigma_u, counts_u, norm_u, jenkins)
jen_DL_u = jen_nll_u + cl_jen_u + af_jen

st_nll_u, st_p_u = fit_function(sigma_u, counts_u, norm_u, sheth_tormen,
                                [(0.01, 10), (0.01, 5), (0.01, 2)],
                                np.array([0.3222, 0.707, 0.3]))
cl_st_u = compute_codelen(st_p_u, sigma_u, counts_u, norm_u, sheth_tormen)
st_DL_u = st_nll_u + cl_st_u + af_st

# best ESR untrimmed DL/NLL (for delta context)
esr_best_DL, esr_best_NLL = -35593573.032832876, -3.5593614e7

print(f"\nJenkins (comp 13): params={jen_p_u}")
print(f"   NLL={jen_nll_u:.3f}  codelen={cl_jen_u:.3f}  DL={jen_DL_u:.3f}"
      f"   dDL_vs_ESR={jen_DL_u - esr_best_DL:.1f}")
print(f"Sheth-Tormen (comp 24): params={st_p_u}")
print(f"   NLL={st_nll_u:.3f}  codelen={cl_st_u:.3f}  DL={st_DL_u:.3f}"
      f"   dDL_vs_ESR={st_DL_u - esr_best_DL:.1f}")

# ── emit the rows for hmf_50_final_functions_extended.txt ──
jen_plot = (f"{jen_p_u[0]:.8f}*exp(-pow(Abs(log(1.0/x)+{jen_p_u[1]:.8f}),{jen_p_u[2]:.8f}))")
jen_blank = "a0*exp(-pow(Abs(log(1.0/x)+a1),a2))"
st_plot = (f"{st_p_u[0]:.8f}*pow(2*{st_p_u[1]:.8f}/3.141592653589793,0.5)*(1.686/x)*"
           f"(1+pow({st_p_u[1]:.8f}*(1.686/x)**2,-{st_p_u[2]:.8f}))*"
           f"exp(-0.5*{st_p_u[1]:.8f}*(1.686/x)**2)")
st_blank = ("a0*pow(2*a1/3.141592653589793,0.5)*(1.686/x)*"
            "(1+pow(a1*(1.686/x)**2,-a2))*exp(-0.5*a1*(1.686/x)**2)")
print("\n=== ROWS for hmf_50_final_functions_extended.txt ===")
print(f"Jen.;13;{jen_DL_u:.6f};{jen_nll_u:.6f};{jen_plot};{jen_blank}")
print(f"S-T.;24;{st_DL_u:.6f};{st_nll_u:.6f};{st_plot};{st_blank}")

# ── params for the plotting scripts ──
print("\n=== PARAMS for generate_untrimmed_appendix.py ===")
print(f"lit_params['S-T.'] = np.array([{st_p_u[0]:.8f}, {st_p_u[1]:.8f}, {st_p_u[2]:.8f}])")
print(f"lit_params['Jen.'] = np.array([{jen_p_u[0]:.8f}, {jen_p_u[1]:.8f}, {jen_p_u[2]:.8f}])")
