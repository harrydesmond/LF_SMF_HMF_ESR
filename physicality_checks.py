"""
Physicality checks on best-fit ESR and paper functions.

For each function in the *_final_functions.txt files, checks:
1. Asymptotic behaviour: f(x) -> 0 as x -> inf, f(x) finite as x -> 0+
2. Monotonicity: f(x) should be monotonically decreasing over the data range
3. Positivity: f(x) >= 0 over the data range
4. Integrability: integral of f(x) should converge, with physical density bounds

Integration uses a log-uniform grid (200k points) for accurate quadrature across
the wide dynamic range of x. Convergence is verified by comparing integrals over
a base range [x_min, x_max] and an extended range [x_min/10, x_max*10].

Usage:
    python3 physicality_checks.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.special import gamma as scipy_gamma


def evaluate_function(fcn_str, x_vals):
    """Evaluate a function string at given x values.

    Args:
        fcn_str: Function string using numpy syntax (from column 5 of final_functions files)
        x_vals: Array of x values

    Returns:
        Array of y values, or None if evaluation fails
    """
    x = x_vals
    try:
        y = eval(fcn_str, {"__builtins__": {}}, {
            "x": x,
            "np": np,
            "pow": np.power,
            "exp": np.exp,
            "log": np.log,
            "Abs": np.abs,
            "abs": np.abs,
            "gamma": scipy_gamma,
        })
        return np.array(y, dtype=float)
    except Exception:
        return None


def check_asymptotics(fcn_str, label, x_range):
    """Check behaviour at extremes of x range.

    Tests at multiple points beyond the data range to catch non-monotonic
    pathologies, not just at a single extrapolated point.
    """
    issues = []

    # Check x -> large at several points: 10x, 100x, 1000x beyond data max
    for factor in [10, 100, 1000]:
        x_high = np.array([x_range[1] * factor])
        y_high = evaluate_function(fcn_str, x_high)
        if y_high is not None:
            val = y_high[0]
            if not np.isfinite(val):
                issues.append(f"  x -> {x_high[0]:.1e}: DIVERGES ({val})")
                break
            elif abs(val) > 1e-6 and factor == 1000:
                issues.append(f"  x -> {x_high[0]:.1e}: does not vanish (f = {val:.3e})")
        else:
            issues.append(f"  x -> {x_high[0]:.1e}: evaluation failed")
            break

    # Check x -> small at several points
    for factor in [0.1, 0.01, 0.001]:
        x_low = np.array([x_range[0] * factor])
        y_low = evaluate_function(fcn_str, x_low)
        if y_low is not None:
            val = y_low[0]
            if not np.isfinite(val):
                issues.append(f"  x -> {x_low[0]:.1e}: NOT FINITE ({val})")
                break
            elif val < -1e-10:
                issues.append(f"  x -> {x_low[0]:.1e}: NEGATIVE (f = {val:.3e})")
                break

    return issues


def check_monotonicity(fcn_str, x_range, n_points=5000):
    """Check if function is monotonically decreasing over data range.

    Uses a log-uniform grid so that the low-x region (where the function
    is steepest) is well-resolved.
    """
    x = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), n_points)
    y = evaluate_function(fcn_str, x)

    if y is None:
        return ["  Monotonicity: evaluation failed"]

    issues = []
    dy = np.diff(y)
    n_increasing = np.sum(dy > 0)

    if n_increasing > 0:
        frac = n_increasing / len(dy)
        if frac > 0.01:
            idx = np.where(dy > 0)[0]
            x_inc_start = x[idx[0]]
            x_inc_end = x[idx[-1] + 1]
            issues.append(f"  Non-monotonic: increasing in {frac*100:.1f}% of range "
                         f"(x ~ {x_inc_start:.2e} to {x_inc_end:.2e})")

    # Check for negative values
    n_negative = np.sum(y < -1e-15)  # tolerance for floating-point noise
    if n_negative > 0:
        frac = n_negative / len(y)
        idx = np.where(y < -1e-15)[0]
        issues.append(f"  Negative values: {frac*100:.1f}% of range "
                     f"(x ~ {x[idx[0]]:.2e} to {x[idx[-1]]:.2e})")

    return issues


def check_integrability(fcn_str, x_range, density_type=None, n_points=200000):
    """Check if the function integrates to a finite value, and compare to
    physical bounds if density_type is specified.

    Uses a log-uniform grid for accurate quadrature across the wide dynamic
    range.  The physical integration limits are:
      - x_min = 1e-4  (L or M* ~ 10^5 Lsun/Msun, below any observed galaxy)
      - x_max = 1e4   (L or M* ~ 10^13 Lsun/Msun, above any observed galaxy)
    Convergence is checked by extending to [1e-5, 1e5].

    For the HMF, the physically relevant integral is the mass fraction in
    collapsed objects: integral f(sigma) d(ln sigma) = integral f(sigma)/sigma dsigma.
    This must converge and be <= 1 for self-consistency. The plain integral
    f dsigma is NOT the right test for the HMF (e.g. Press-Schechter has
    f ~ delta_c/sigma which gives a divergent dsigma integral but a convergent
    d(ln sigma) integral).

    The density integral for LF/SMF is:
      rho = (10^9 / ln 10) * integral(f(x) dx)
    where x = L/10^9 Lsun or M*/10^9 Msun, and the 10^9/ln(10) factor
    arises because phi is given per dex (d log10) and x absorbs the 10^9
    unit scaling.

    Assumptions and caveats:
      - The functions are extrapolated beyond the data range for this integral.
        Functions with additive constant offsets (e.g. ESR9 LF Sersic with
        offset -1.27e-5) give a tiny negative tail at very large x; this is
        negligible over the physical range but would dominate if integrated
        to x -> infinity.
      - Schechter functions have f(x) ~ x^alpha as x -> 0 with -1 < alpha < 0,
        so they diverge pointwise but remain integrable.
      - rho_* is the zeroth-moment integral (number density weighted by mass);
        the density comparison assumes the observed functions extend to
        arbitrarily low masses/luminosities.

    Args:
        fcn_str: Function string
        x_range: (x_min, x_max) of data range
        density_type: 'SMF', 'LF', or 'HMF' to select the appropriate integral
        n_points: Number of integration points (log-spaced)
    """
    # Physical integration limits
    x_int_min = 1e-4   # L or M* ~ 10^5
    x_int_max = 1e4    # L or M* ~ 10^13

    x = np.logspace(np.log10(x_int_min), np.log10(x_int_max), n_points)
    y = evaluate_function(fcn_str, x)

    if y is None:
        return ["  Integration: evaluation failed"]

    issues = []

    y_clean = np.where(np.isfinite(y), y, 0)

    # For HMF, the relevant integral is f d(ln sigma) = f/sigma dsigma
    if density_type == 'HMF':
        integrand = y_clean / x
        integral = np.trapz(integrand, x)
        integral_label = "int f d(ln sigma)"
    else:
        integral = np.trapz(y_clean, x)
        integral_label = "int f dx"

    if not np.isfinite(integral):
        issues.append(f"  {integral_label} DIVERGES over [{x_int_min:.0e}, {x_int_max:.0e}]")
        return issues
    elif integral < 0:
        issues.append(f"  {integral_label} is NEGATIVE: {integral:.3e}")
        return issues

    # Convergence check: extend range progressively (10x, 1000x, 10^6x)
    # A single 10x extension can miss tiny non-zero asymptotes
    converged = False
    prev_integral = integral
    for factor in [10, 1000, 1e6]:
        x_ext = np.logspace(np.log10(x_int_min / factor),
                            np.log10(x_int_max * factor), n_points)
        y_ext = evaluate_function(fcn_str, x_ext)
        if y_ext is None:
            break
        y_ext_clean = np.where(np.isfinite(y_ext), y_ext, 0)
        if density_type == 'HMF':
            integrand_ext = y_ext_clean / x_ext
            integral_ext = np.trapz(integrand_ext, x_ext)
        else:
            integral_ext = np.trapz(y_ext_clean, x_ext)
        if not np.isfinite(integral_ext) or integral_ext < 0:
            issues.append(f"  {integral_label} DIVERGES when range extended "
                         f"{factor:.0e}x ({prev_integral:.3e} -> {integral_ext:.3e})")
            return issues
        if integral > 0:
            change_frac = abs(integral_ext - integral) / integral
            if change_frac > 0.1:
                issues.append(f"  {integral_label} not well-converged: changes by "
                             f"{change_frac*100:.1f}% when range extended {factor:.0e}x "
                             f"({integral:.3e} -> {integral_ext:.3e})")
                return issues
        prev_integral = integral_ext
    else:
        converged = True
        if integral > 0:
            final_frac = abs(prev_integral - integral) / integral
            issues.append(f"  {integral_label} = {integral:.4e} (converged to "
                         f"{final_frac*100:.1f}% over [{x_int_min:.0e}, {x_int_max:.0e}])")

    # Physical density check for SMF and LF
    if density_type in ('SMF', 'LF') and converged:
        rho = 1e9 / np.log(10) * integral

        if density_type == 'SMF':
            Omega_b = 0.049
            rho_crit = 1.27e11  # Msun Mpc^-3
            rho_baryon = Omega_b * rho_crit  # ~ 6.2e9 Msun Mpc^-3
            ratio = rho / rho_baryon
            issues.append(f"  Stellar mass density: rho_* = {rho:.3e} Msun Mpc^-3")
            issues.append(f"  rho_*/rho_b = {ratio:.4f}  "
                         f"(Omega_b rho_crit = {rho_baryon:.2e})")
            if rho > rho_baryon:
                issues.append(f"  WARNING: rho_* EXCEEDS baryon budget by {ratio:.1f}x")
            else:
                issues.append(f"  Physical: well within baryon budget")

        elif density_type == 'LF':
            rho_L_obs = 2e8  # Lsun Mpc^-3, approximate observed upper bound
            issues.append(f"  Luminosity density: rho_L = {rho:.3e} Lsun Mpc^-3")
            issues.append(f"  (observed ~ 1-2e8 Lsun Mpc^-3)")
            if rho > 10 * rho_L_obs:
                issues.append(f"  WARNING: rho_L unreasonably large")
            else:
                issues.append(f"  Physical: within plausible range")

    # Physical check for HMF: mass fraction must be <= 1
    if density_type == 'HMF' and converged:
        issues.append(f"  Mass fraction in haloes: {integral:.4f}")
        if integral > 1:
            issues.append(f"  WARNING: mass fraction exceeds 1")
        else:
            issues.append(f"  Physical: mass fraction <= 1")

    return issues


def analyse_dataset(filename, dataset_name, x_range, density_type=None):
    """Run all checks on functions from a final_functions file.

    Args:
        filename: Path to *_final_functions.txt
        dataset_name: Display name
        x_range: (x_min, x_max) of data range
        density_type: 'SMF', 'LF', or None for physical density comparison
    """
    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"  x range: [{x_range[0]:.2e}, {x_range[1]:.2e}]")
    print(f"{'='*70}")

    try:
        source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt(
            filename, dtype=str, delimiter=';', unpack=True)
    except Exception as e:
        print(f"  Could not load {filename}: {e}")
        return

    for i in range(len(source)):
        label = f"{source[i].strip()} (comp {comp[i].strip()})"
        fcn = plot_fcn[i].strip()
        template = blank_fcn[i].strip()

        print(f"\n--- {label} ---")
        print(f"  Template: {template}")
        print(f"  Fitted:   {fcn[:80]}{'...' if len(fcn) > 80 else ''}")

        # Run checks
        issues = []
        issues.extend(check_asymptotics(fcn, label, x_range))
        if density_type in ('LF', 'SMF'):
            # Monotonicity check only for LF/SMF; HMF f(sigma) peaks near sigma~1
            issues.extend(check_monotonicity(fcn, x_range))
        issues.extend(check_integrability(fcn, x_range, density_type=density_type))

        if not issues:
            print("  ALL CHECKS PASSED")
        else:
            for issue in issues:
                print(issue)


if __name__ == '__main__':
    # HMF: x = sigma (mass variance), typical range ~0.2 to ~2
    analyse_dataset('hmf_50_final_functions.txt',
                    'HMF (sim 50) — x = σ (mass variance)',
                    x_range=(0.2, 2.0), density_type='HMF')

    # LF: x = L / 10^9 L_sun
    analyse_dataset('LF_cmodel_L_final_functions.txt',
                    'LF cmodel — x = L / 10⁹ L☉',
                    x_range=(0.01, 100.0), density_type='LF')

    analyse_dataset('LF_Ser_L_final_functions.txt',
                    'LF Sersic — x = L / 10⁹ L☉',
                    x_range=(0.01, 100.0), density_type='LF')

    # SMF: x = M* / 10^9 M_sun
    analyse_dataset('SMF_cmodel_M_final_functions.txt',
                    'SMF cmodel — x = M★ / 10⁹ M☉',
                    x_range=(0.01, 100.0), density_type='SMF')

    analyse_dataset('SMF_Ser_M_final_functions.txt',
                    'SMF Sersic — x = M★ / 10⁹ M☉',
                    x_range=(0.01, 100.0), density_type='SMF')
