"""
fit_double_schechter.py

Fits the double Schechter function to LF/SMF datasets and compares with
single Schechter and top ESR functions (Sec 4.3 of the paper; see
Eq~(eq:dbl_schechter)).

Double Schechter:
  phi(x) = exp(-x/theta2) * [theta0*(x/theta2)^theta3 + theta1*(x/theta2)^theta4] / theta2

Parameters: theta0=phi1, theta1=phi2, theta2=x*, theta3=alpha1, theta4=alpha2

Inputs:
  - data/LF_Ser_L.txt, data/LF_cmodel_L.dat,
    data/SMF_Ser_M.txt, data/SMF_cmodel_M.txt
    (4-column x, log10(phi), sigma, Veff)
  - Hardcoded single-Schechter/ESR/Bernardi reference NLL/DL values
    (constants at the top of the module; see paper's Tables 1–2).

Outputs:
  - double_schechter_results.txt  (per-dataset best-fit parameters, DL,
                                    physicality checks, comparisons)

Dependencies:
  numpy, scipy, numdifftools
"""

import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
import warnings
warnings.filterwarnings('ignore')

# ─── Dataset definitions ────────────────────────────────────────────────────

DATASETS = {
    'LF_Ser': {
        'path': 'data/LF_Ser_L.txt',
        'label': 'LF Sérsic',
        'type': 'LF',
        # Single Schechter: phi1/xstar * (x/xstar)^alpha * exp(-x/xstar)
        'sch_params': [0.12161616645372644, 53.78580921307795, -0.2892224099059145],
        'sch_NLL': -2380229.290684023,
        'sch_DL': -2380278.3676490863,
        'esr1_NLL': -2383333.06698875,
        'esr1_DL': -2383374.4,
        'ber_NLL': -2383296.765368,
        'ber_DL': -2383399.7914435384,
    },
    'LF_cmodel': {
        'path': 'data/LF_cmodel_L.dat',
        'label': 'LF cmodel',
        'type': 'LF',
        'sch_params': [0.12437228929785779, 28.321015012755353, -0.03056437139826267],
        'sch_NLL': -2418362.3392734877,
        'sch_DL': -2418409.205030131,
        'esr1_NLL': -2418867.81442375,
        'esr1_DL': -2418909.0,
        'ber_NLL': -2418826.677790933,
        'ber_DL': -2418935.645171763,
    },
    'SMF_Ser': {
        'path': 'data/SMF_Ser_M.txt',
        'label': 'SMF Sérsic',
        'type': 'SMF',
        'sch_params': [0.6472797677857043, 184.46982394144908, -0.2688701129621677],
        'sch_NLL': -2406940.975603176,
        'sch_DL': -2406990.2041110764,
        'esr1_NLL': -2411137.005073308,
        'esr1_DL': -2411177.5,
        'ber_NLL': -2411341.4877431737,
        'ber_DL': -2411442.2765243794,
    },
    'SMF_cmodel': {
        'path': 'data/SMF_cmodel_M.txt',
        'label': 'SMF cmodel',
        'type': 'SMF',
        'sch_params': [0.5973521552420994, 92.57115561723623, -0.09324696509273002],
        'sch_NLL': -2341073.3537972597,
        'sch_DL': -2341121.4623185387,
        'esr1_NLL': -2342298.12060375,
        'esr1_DL': -2342338.9,
        'ber_NLL': -2342485.502726237,
        'ber_DL': -2342587.8520432464,
    },
}

# ─── Double Schechter model ──────────────────────────────────────────────────

def double_schechter(x, theta):
    """
    phi(x) = exp(-x/theta2) * [theta0*(x/theta2)^theta3 + theta1*(x/theta2)^theta4] / theta2
    theta = [phi1, phi2, x*, alpha1, alpha2]
    """
    phi1, phi2, xstar, alpha1, alpha2 = theta
    if xstar <= 0:
        return np.full_like(x, np.inf)
    u = x / xstar
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        term1 = phi1 * np.power(np.abs(u), alpha1)
        term2 = phi2 * np.power(np.abs(u), alpha2)
        phi = np.exp(-u) * (term1 + term2) / xstar
    return phi


def nll_fn(theta, x, N, Veff):
    """Poisson NLL: sum(lambda - N*ln(lambda)) where lambda = phi(x)*Veff"""
    phi = double_schechter(x, theta)
    lam = phi * Veff
    if np.any(lam <= 0) or not np.all(np.isfinite(lam)):
        return 1e30
    return np.sum(lam - N * np.log(lam))


# ─── Description length calculation ─────────────────────────────────────────

def compute_DL(theta_best, x, N, Veff, nll_val):
    """
    DL = NLL + codelen + aifeynman
    codelen = -k/2*ln(3) + sum_i [0.5*ln(H_ii) + ln|theta_i|]
    aifeynman ~ structural complexity contribution

    For double Schechter:
      Expression: exp(-x/a2)*(a0*pow(x/a2,a3)+a1*pow(x/a2,a4))/a2
      Structural nodes (non-parameter): exp, neg, div, x, a2, mul, pow, div, x, a2,
                                        add, mul, pow, div, x, a2, div, a2
      ~20 structural nodes, 7 unique operators (exp, -, /, x, *, pow, +)
      aifeynman = 20 * ln(7)
    """
    k = len(theta_best)

    # Numerical Hessian diagonal using numdifftools
    # Use relative step size, robust against parameters of very different scales
    step = 1e-4 * np.abs(theta_best).clip(min=1e-8)
    try:
        def f_scalar(t):
            return nll_fn(t, x, N, Veff)
        hess_fn = nd.Hessian(f_scalar, step=step)
        H = hess_fn(theta_best)
        H_diag = np.diag(H)
        # Take abs to handle near-zero or slightly negative diagonal entries
        H_diag = np.abs(H_diag).clip(min=1e-30)
    except Exception as e:
        print(f"  WARNING: Hessian failed ({e}), using fallback")
        H_diag = np.ones(k)

    codelen = (-k / 2.0 * np.log(3)
               + np.sum(0.5 * np.log(H_diag) + np.log(np.abs(theta_best).clip(min=1e-300))))

    # Structural complexity: ~20 nodes, 7 unique operators
    n_nodes = 20
    n_ops = 7  # exp, -, *, pow, +, /, x
    aifeynman = n_nodes * np.log(n_ops)

    DL = nll_val + codelen + aifeynman

    return DL, codelen, aifeynman, H_diag


# ─── Physicality checks ──────────────────────────────────────────────────────

def physicality_checks(theta_best, dataset_type):
    """
    1. Asymptotic: phi -> 0 as x -> infinity
    2. Non-negative: check over [1e-4, 1e4]
    3. Integral convergence: integral of phi(x) dx over [1e-4, 1e5]
    4. SMF only: rho_* = integral of x * phi(x) dx * 1e9 M_sun < Omega_b * rho_crit
    """
    from scipy.integrate import quad

    results = {}

    # 1. Asymptotic
    xstar = abs(theta_best[2])
    phi_large = double_schechter(np.array([1e6 * xstar]), theta_best)[0]
    results['asymptotic_ok'] = bool(np.isfinite(phi_large) and phi_large < 1e-100)
    results['phi_at_large_x'] = float(phi_large)

    # 2. Non-negativity over [1e-4, 1e4]
    x_check = np.logspace(-4, 4, 10000)
    phi_check = double_schechter(x_check, theta_best)
    results['nonneg_ok'] = bool(np.all(phi_check >= 0))
    results['min_phi'] = float(np.min(phi_check))

    # 3. Integral of phi dx
    def integrand_phi(xv):
        return double_schechter(np.array([xv]), theta_best)[0]

    try:
        integral_phi, err_phi = quad(integrand_phi, 1e-4, 1e5,
                                     limit=500, epsabs=1e-10, epsrel=1e-6)
        results['integral_phi'] = float(integral_phi)
        results['integral_phi_err'] = float(err_phi)
        results['integral_converges'] = bool(np.isfinite(integral_phi))
    except Exception:
        results['integral_phi'] = np.nan
        results['integral_phi_err'] = np.nan
        results['integral_converges'] = False

    # 4. SMF: stellar mass density
    # phi(x) is in Mpc^-3 per unit x, where x = M*/(1e9 M_sun).
    # The data label is phi/(Mpc^-3 dex^-1); since d(log10 x) = dx/(x*ln10),
    # the physical stellar mass density is:
    #   rho_* = 1e9 M_sun * integral phi(x) dx / ln(10)   [M_sun/Mpc^3]
    # (This avoids the x*phi(x) form which requires a different unit interpretation.)
    # Omega_b * rho_crit ~ 0.049 * 2.775e11 * 0.7^2 = 6.66e9 M_sun/Mpc^3
    if dataset_type == 'SMF':
        def integrand_phi_only(xv):
            return double_schechter(np.array([xv]), theta_best)[0]
        try:
            integral_phi_smf, err_phi_smf = quad(integrand_phi_only, 1e-4, 1e5,
                                                  limit=500, epsabs=1e-10, epsrel=1e-6)
            rho_star = 1e9 * integral_phi_smf / np.log(10)  # M_sun/Mpc^3
            rho_b_limit = 0.049 * 2.775e11 * 0.7**2         # ~ 6.66e9 M_sun/Mpc^3
            results['rho_star'] = float(rho_star)
            results['rho_star_err'] = float(1e9 * err_phi_smf / np.log(10))
            results['rho_b_limit'] = rho_b_limit
            results['rho_ok'] = bool(rho_star < rho_b_limit)
        except Exception:
            results['rho_star'] = np.nan
            results['rho_star_err'] = np.nan
            results['rho_ok'] = False

    return results


# ─── Main fitting routine ────────────────────────────────────────────────────

def load_data(path):
    data = np.loadtxt(path, comments='#')
    x_raw = data[:, 0]
    log10_phi = data[:, 1]
    sigma = data[:, 2]
    Veff = data[:, 3]
    # Scale x if values are in raw solar luminosity/mass units (> 1e6)
    x = x_raw * 1e-9 if np.median(x_raw) > 1e6 else x_raw.copy()
    return x, log10_phi, sigma, Veff


def fit_dataset(key, info):
    import time
    print(f"\n{'='*60}")
    print(f"Dataset: {info['label']}")
    print(f"{'='*60}")

    x, log10_phi, sigma, Veff = load_data(info['path'])
    N = 10**log10_phi * Veff

    print(f"  N data points: {len(x)}")
    print(f"  x range: [{x.min():.3g}, {x.max():.3g}]")

    phi1_sch, xstar_sch, alpha_sch = info['sch_params']
    phi_amp = phi1_sch / xstar_sch   # amplitude of single Schechter

    rng = np.random.default_rng(42)

    # Build starting points
    # Start 0: from single Schechter (second component very small)
    starts = [
        [phi_amp, 1e-3 * phi_amp, xstar_sch, alpha_sch, alpha_sch - 1.0],
        [phi_amp, 1e-2 * phi_amp, xstar_sch, alpha_sch - 0.5, alpha_sch - 2.0],
        [phi_amp * 0.5, phi_amp * 0.5, xstar_sch, alpha_sch, alpha_sch - 1.5],
    ]

    # Random starts around single Schechter solution
    for _ in range(22):
        p = [
            phi_amp * 10**rng.uniform(-1.5, 1.5),
            phi_amp * 10**rng.uniform(-3.0, 0.5),
            xstar_sch * 10**rng.uniform(-0.5, 0.5),
            alpha_sch + rng.uniform(-1.5, 1.5),
            alpha_sch + rng.uniform(-3.0, 2.0),
        ]
        starts.append(p)

    # Physically motivated starts (double power law shape)
    for _ in range(5):
        a1 = rng.uniform(-0.5, 0.5)
        a2 = rng.uniform(-2.5, -0.5)
        xs = xstar_sch * 10**rng.uniform(-0.3, 0.3)
        p1 = phi_amp * 10**rng.uniform(-0.5, 0.5)
        p2 = phi_amp * 10**rng.uniform(-2.0, 0.0)
        starts.append([p1, p2, xs, a1, a2])

    best_nll = np.inf
    best_theta = None
    t0 = time.time()

    for i, p0 in enumerate(starts):
        try:
            res = minimize(
                lambda t: nll_fn(t, x, N, Veff), p0,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9,
                         'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass

    print(f"  Multi-start Nelder-Mead: {len(starts)} starts, {time.time()-t0:.1f}s")

    # Final refinement on best found
    if best_theta is not None:
        try:
            res2 = minimize(
                lambda t: nll_fn(t, x, N, Veff), best_theta,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if np.isfinite(res2.fun) and res2.fun < best_nll:
                best_nll = res2.fun
                best_theta = res2.x.copy()
        except Exception:
            pass

    if best_theta is None:
        print("  ERROR: all minimizations failed!")
        return None

    print(f"\n  Best-fit parameters:")
    print(f"    phi1   (theta0) = {best_theta[0]:.6e}")
    print(f"    phi2   (theta1) = {best_theta[1]:.6e}")
    print(f"    x*     (theta2) = {best_theta[2]:.6e}")
    print(f"    alpha1 (theta3) = {best_theta[3]:.6f}")
    print(f"    alpha2 (theta4) = {best_theta[4]:.6f}")
    print(f"  NLL = {best_nll:.6f}")

    # Comparisons
    print(f"\n  NLL comparisons:")
    print(f"    Delta NLL (vs single Schechter): {best_nll - info['sch_NLL']:+.2f}")
    print(f"    Delta NLL (vs ESR rank 1):       {best_nll - info['esr1_NLL']:+.2f}")
    print(f"    Delta NLL (vs Bernardi):          {best_nll - info['ber_NLL']:+.2f}")

    # Description length
    DL, codelen, aifeynman, H_diag = compute_DL(best_theta, x, N, Veff, best_nll)
    print(f"\n  Description length:")
    print(f"    NLL          = {best_nll:.4f}")
    print(f"    codelen      = {codelen:.4f}")
    print(f"    aifeynman    = {aifeynman:.4f}  (20 nodes * ln(7))")
    print(f"    DL (total)   = {DL:.4f}")
    print(f"    Hessian diag = {H_diag}")

    print(f"\n  DL comparisons:")
    print(f"    Delta DL (vs single Schechter): {DL - info['sch_DL']:+.2f}")
    print(f"    Delta DL (vs ESR rank 1):       {DL - info['esr1_DL']:+.2f}")
    print(f"    Delta DL (vs Bernardi):          {DL - info['ber_DL']:+.2f}")

    # Physicality checks
    phys = physicality_checks(best_theta, info['type'])
    print(f"\n  Physicality checks:")
    print(f"    Asymptotic -> 0:  {'PASS' if phys['asymptotic_ok'] else 'FAIL'}"
          f"  (phi at 1e6*x* = {phys['phi_at_large_x']:.2e})")
    print(f"    Non-negative:     {'PASS' if phys['nonneg_ok'] else 'FAIL'}"
          f"  (min phi = {phys['min_phi']:.2e})")
    print(f"    Integral finite:  {'PASS' if phys['integral_converges'] else 'FAIL'}"
          f"  (integral = {phys['integral_phi']:.4e} +/- {phys['integral_phi_err']:.2e} Mpc^-3)")
    if info['type'] == 'SMF':
        print(f"    rho_* < Omega_b*rho_crit: {'PASS' if phys['rho_ok'] else 'FAIL'}"
              f"  (rho_* = {phys['rho_star']:.3e}, limit = {phys['rho_b_limit']:.3e} M_sun/Mpc^3)"
              f"  [= 1e9/ln(10) * integral phi dx]")

    return {
        'key': key,
        'label': info['label'],
        'theta': best_theta,
        'nll': best_nll,
        'DL': DL,
        'codelen': codelen,
        'aifeynman': aifeynman,
        'H_diag': H_diag,
        'phys': phys,
        'sch_nll': info['sch_NLL'],
        'sch_DL': info['sch_DL'],
        'esr1_nll': info['esr1_NLL'],
        'esr1_DL': info['esr1_DL'],
        'ber_nll': info['ber_NLL'],
        'ber_DL': info['ber_DL'],
    }


# ─── Save results ────────────────────────────────────────────────────────────

def save_results(all_results, outpath):
    lines = []
    lines.append("Double Schechter Function Fits to LF/SMF Datasets")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Model: phi(x) = exp(-x/theta2) *")
    lines.append("       [theta0*(x/theta2)^theta3 + theta1*(x/theta2)^theta4] / theta2")
    lines.append("Parameters: theta0=phi1, theta1=phi2, theta2=x*, theta3=alpha1, theta4=alpha2")
    lines.append("")
    lines.append("Description length: DL = NLL + codelen + aifeynman")
    lines.append("  codelen = -k/2*ln(3) + sum_i[0.5*ln(H_ii) + ln|theta_i|]  (k=5)")
    lines.append("  aifeynman = 20 * ln(7) = 38.92")
    lines.append("  Note: aifeynman is estimated from tree structure; exact value")
    lines.append("  requires ESR tree computation. 20 structural nodes, 7 unique operators.")
    lines.append("")

    for r in all_results:
        if r is None:
            continue
        lines.append(f"{'─'*60}")
        lines.append(f"Dataset: {r['label']}")
        lines.append(f"{'─'*60}")
        lines.append("Best-fit parameters:")
        lines.append(f"  phi1   (theta0) = {r['theta'][0]:.8e}")
        lines.append(f"  phi2   (theta1) = {r['theta'][1]:.8e}")
        lines.append(f"  x*     (theta2) = {r['theta'][2]:.8e}")
        lines.append(f"  alpha1 (theta3) = {r['theta'][3]:.8f}")
        lines.append(f"  alpha2 (theta4) = {r['theta'][4]:.8f}")
        lines.append("")
        lines.append("Description length components:")
        lines.append(f"  NLL        = {r['nll']:.4f}")
        lines.append(f"  codelen    = {r['codelen']:.4f}")
        lines.append(f"  aifeynman  = {r['aifeynman']:.4f}  (~20 nodes * ln(7))")
        lines.append(f"  DL (total) = {r['DL']:.4f}")
        lines.append(f"  Hessian diagonal = {r['H_diag']}")
        lines.append("")
        lines.append("NLL comparisons:")
        lines.append(f"  Single Schechter (k=3):  NLL = {r['sch_nll']:.4f}  DL = {r['sch_DL']:.4f}")
        lines.append(f"  Double Schechter (k=5):  NLL = {r['nll']:.4f}  DL = {r['DL']:.4f}")
        lines.append(f"  ESR rank 1 (k=4):        NLL = {r['esr1_nll']:.4f}  DL = {r['esr1_DL']:.4f}")
        lines.append(f"  Bernardi (k=7):          NLL = {r['ber_nll']:.4f}  DL = {r['ber_DL']:.4f}")
        lines.append("")
        lines.append("Delta NLL (Dbl.Sch. - reference):")
        lines.append(f"  vs Single Schechter: {r['nll'] - r['sch_nll']:+.2f}")
        lines.append(f"  vs ESR rank 1:       {r['nll'] - r['esr1_nll']:+.2f}")
        lines.append(f"  vs Bernardi:          {r['nll'] - r['ber_nll']:+.2f}")
        lines.append("Delta DL (Dbl.Sch. - reference):")
        lines.append(f"  vs Single Schechter: {r['DL'] - r['sch_DL']:+.2f}")
        lines.append(f"  vs ESR rank 1:       {r['DL'] - r['esr1_DL']:+.2f}")
        lines.append(f"  vs Bernardi:          {r['DL'] - r['ber_DL']:+.2f}")
        lines.append("")
        phys = r['phys']
        lines.append("Physicality checks:")
        lines.append(f"  Asymptotic -> 0:  {'PASS' if phys['asymptotic_ok'] else 'FAIL'}"
                     f"  (phi at 1e6*x* = {phys['phi_at_large_x']:.2e})")
        lines.append(f"  Non-negative:     {'PASS' if phys['nonneg_ok'] else 'FAIL'}"
                     f"  (min phi = {phys['min_phi']:.2e})")
        lines.append(f"  Integral finite:  {'PASS' if phys['integral_converges'] else 'FAIL'}"
                     f"  (integral = {phys['integral_phi']:.4e} +/- {phys['integral_phi_err']:.2e} Mpc^-3)")
        if 'rho_star' in phys:
            lines.append(f"  rho_* < Omega_b*rho_crit: {'PASS' if phys['rho_ok'] else 'FAIL'}"
                         f"  (rho_* = {phys['rho_star']:.3e}, limit = {phys['rho_b_limit']:.3e} M_sun/Mpc^3)")
            lines.append(f"    [rho_* = 1e9/ln(10) * integral phi(x) dx; per-dex convention]")
        lines.append("")

    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nResults saved to {outpath}")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('Plots', exist_ok=True)

    all_results = []
    for key, info in DATASETS.items():
        if not os.path.exists(info['path']):
            print(f"WARNING: {info['path']} not found, skipping.")
            all_results.append(None)
            continue
        result = fit_dataset(key, info)
        all_results.append(result)

    save_results(all_results, 'double_schechter_results.txt')
    print("\nDone.")
