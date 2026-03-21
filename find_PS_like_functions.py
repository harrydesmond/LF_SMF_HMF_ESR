#!/usr/bin/env python3
"""
Find ESR HMF functions with Press-Schechter-like behaviour at large sigma.

Press-Schechter: f(sigma) ~ 1/sigma as sigma -> infinity.
Test: sigma * f(sigma) -> positive constant as sigma -> infinity.
"""

import numpy as np
import re
import ast

def eval_fcn(func_str, x, params):
    """Evaluate a symbolic function string with numpy."""
    s = func_str
    s = s.replace('pow', 'np.power')
    s = s.replace('exp', 'np.exp')
    s = s.replace('log', 'np.log')
    s = s.replace('Abs', 'np.abs')
    # Replace parameter names
    for i, p in enumerate(params):
        s = s.replace(f'a{i}', str(p))
    s = s.replace('x', 'x_val')
    x_val = x
    with np.errstate(all='ignore'):
        try:
            result = eval(s)
            return result
        except:
            return np.nan

def parse_params(param_str):
    """Parse parameter string like '[0.65 1.69 1.63 2.78]'."""
    param_str = param_str.strip()
    # Handle numpy-style arrays with spaces
    inner = param_str.strip('[]')
    parts = inner.split()
    return [float(p) for p in parts]

def check_ps_like(func_str, params, sigma_vals=[100, 1000, 10000]):
    """
    Check if f(sigma) ~ 1/sigma at large sigma.
    Returns (is_ps_like, details_dict) where details_dict has sigma*f values.
    """
    products = {}
    f_vals = {}
    for s in sigma_vals:
        with np.errstate(all='ignore'):
            f = eval_fcn(func_str, float(s), params)
        if f is None or not np.isfinite(f):
            return False, {}
        f_vals[s] = f
        products[s] = s * f

    # Check criteria:
    # 1. f(sigma) > 0 for all test sigmas
    # 2. sigma * f(sigma) converges to a positive constant
    #    i.e., the products at 100, 1000, 10000 are similar and positive

    if any(f_vals[s] <= 0 for s in sigma_vals):
        return False, {'f_vals': f_vals, 'products': products, 'reason': 'f <= 0'}

    # Check convergence: products should be within a factor of ~3 of each other
    # and positive
    prods = [products[s] for s in sigma_vals]
    if any(p <= 0 for p in prods):
        return False, {'f_vals': f_vals, 'products': products, 'reason': 'sigma*f <= 0'}

    # Check that the ratio between consecutive products is close to 1
    # (indicating convergence, not growth or decay)
    ratio_1 = prods[1] / prods[0]
    ratio_2 = prods[2] / prods[1]

    # For true 1/sigma behaviour, ratios should be ~1
    # Allow some tolerance: between 0.1 and 10 for the first ratio,
    # and tightening for larger sigma
    converging = (0.1 < ratio_1 < 10) and (0.1 < ratio_2 < 10)

    # Also check that f is actually decaying (not constant or growing fast)
    # f should decrease with sigma
    f_decaying = f_vals[1000] < f_vals[100] and f_vals[10000] < f_vals[1000]

    # The product sigma*f should stabilize
    # For 1/sigma: product is constant
    # For 1/sigma^alpha with alpha near 1: product ~ sigma^(1-alpha)
    # We want alpha close to 1, so allow some power law
    # Let's compute effective alpha from f(1000)/f(100) ~ (1000/100)^(-alpha)
    if f_vals[100] > 0 and f_vals[1000] > 0:
        alpha_est = -np.log(f_vals[1000] / f_vals[100]) / np.log(10)
    else:
        alpha_est = np.nan

    is_ps = converging and f_decaying and np.isfinite(alpha_est) and (0.5 < alpha_est < 1.5)

    return is_ps, {
        'f_vals': f_vals,
        'products': products,
        'alpha_est': alpha_est,
        'ratio_1': ratio_1,
        'ratio_2': ratio_2,
    }

def main():
    results_combined = []
    results_per_sim = []

    # ==========================================
    # Part 1: Check combined DL file (top 200)
    # ==========================================
    print("=" * 80)
    print("PART 1: Checking hmf_combined_DL.txt (top 200 functions)")
    print("=" * 80)

    combined_file = '/home/harry/Amelia_code/hmf_combined_DL.txt'
    with open(combined_file) as fh:
        lines = fh.readlines()

    # For these functions we don't have per-function params in the combined file.
    # We need to get params from the per-sim files.
    # First, build a map of function -> params from per-sim files.
    func_params_map = {}  # func_str -> list of (sim_id, params, DL, NLL)

    sims = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for sim in sims:
        filepath = f'/home/harry/Amelia_code/hmf_data/hmf_{sim}_data/final_all.txt'
        try:
            with open(filepath) as fh:
                sim_lines = fh.readlines()
        except FileNotFoundError:
            continue
        for line in sim_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            if len(parts) < 4:
                continue
            rank = parts[0]
            func_str = parts[1]
            try:
                dl = float(parts[2])
                nll = float(parts[3])
            except:
                continue
            param_str = parts[4] if len(parts) > 4 else '[]'
            try:
                params = parse_params(param_str)
            except:
                continue

            if func_str not in func_params_map:
                func_params_map[func_str] = []
            func_params_map[func_str].append((sim, params, dl, nll))

    # Now check combined functions
    combined_funcs = {}
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split(';')
        rank = int(parts[0])
        func_str = parts[1]
        dl_combined = float(parts[2])
        delta_dl = float(parts[3])
        combined_funcs[func_str] = {
            'rank': rank,
            'dl_combined': dl_combined,
            'delta_dl': delta_dl,
        }

    ps_like_combined = []

    for func_str, info in combined_funcs.items():
        if func_str not in func_params_map:
            continue
        # Test with params from each sim
        any_ps = False
        sim_results = []
        for sim, params, dl, nll in func_params_map[func_str]:
            is_ps, details = check_ps_like(func_str, params)
            if is_ps:
                any_ps = True
                sim_results.append((sim, params, dl, nll, details))

        if any_ps:
            ps_like_combined.append({
                'func': func_str,
                'rank': info['rank'],
                'dl_combined': info['dl_combined'],
                'delta_dl': info['delta_dl'],
                'sim_results': sim_results,
            })

    # Sort by combined DL (more negative = better)
    ps_like_combined.sort(key=lambda x: -x['dl_combined'])  # most negative first

    print(f"\nFound {len(ps_like_combined)} PS-like functions in top 200:\n")
    for item in ps_like_combined:
        print(f"  Rank {item['rank']:3d} | DL_combined = {item['dl_combined']:.1f} | delta_DL = {item['delta_dl']:.1f}")
        print(f"           | f(σ) = {item['func']}")
        print(f"           | PS-like in {len(item['sim_results'])}/{len(func_params_map.get(item['func'], []))} sims tested")
        # Show details from first PS-like sim
        sr = item['sim_results'][0]
        det = sr[4]
        print(f"           | Example (sim {sr[0]}): params = {sr[1]}")
        print(f"           |   α_est = {det['alpha_est']:.3f}, σ·f(σ) at σ=100,1000,10000: "
              f"{det['products'][100]:.4e}, {det['products'][1000]:.4e}, {det['products'][10000]:.4e}")
        if len(item['sim_results']) > 1:
            alphas = [r[4]['alpha_est'] for r in item['sim_results']]
            print(f"           |   α range across sims: [{min(alphas):.3f}, {max(alphas):.3f}]")
        print()

    # ==========================================
    # Part 2: Check ALL functions in per-sim files
    # ==========================================
    print("=" * 80)
    print("PART 2: Checking all functions in per-realisation files")
    print("=" * 80)

    # Track unique PS-like functions across all sims
    ps_like_all = {}  # func_str -> list of (sim, params, dl, nll, details)

    for sim in sims:
        filepath = f'/home/harry/Amelia_code/hmf_data/hmf_{sim}_data/final_all.txt'
        try:
            with open(filepath) as fh:
                sim_lines = fh.readlines()
        except FileNotFoundError:
            continue

        count = 0
        for line in sim_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            if len(parts) < 4:
                continue
            rank = int(parts[0])
            func_str = parts[1]
            try:
                dl = float(parts[2])
                nll = float(parts[3])
            except:
                continue
            param_str = parts[4] if len(parts) > 4 else '[]'
            try:
                params = parse_params(param_str)
            except:
                continue

            is_ps, details = check_ps_like(func_str, params)
            if is_ps:
                count += 1
                if func_str not in ps_like_all:
                    ps_like_all[func_str] = []
                ps_like_all[func_str].append((sim, params, dl, nll, details))

        print(f"  Sim {sim:2d}: {count} PS-like functions found")

    # Sort by best single-sim DL
    ps_like_sorted = []
    for func_str, entries in ps_like_all.items():
        best_dl = min(e[2] for e in entries)  # most negative DL
        in_combined = func_str in combined_funcs
        combined_info = combined_funcs.get(func_str, {})
        ps_like_sorted.append({
            'func': func_str,
            'entries': entries,
            'best_dl': best_dl,
            'n_sims': len(entries),
            'in_combined': in_combined,
            'combined_rank': combined_info.get('rank', None),
            'dl_combined': combined_info.get('dl_combined', None),
        })

    ps_like_sorted.sort(key=lambda x: x['best_dl'])  # most negative first

    print(f"\n{len(ps_like_sorted)} unique PS-like functions found across all sims:\n")
    print("-" * 120)

    for item in ps_like_sorted:
        combined_str = ""
        if item['in_combined']:
            combined_str = f" | Combined rank {item['combined_rank']}, DL_comb = {item['dl_combined']:.1f}"

        print(f"  f(σ) = {item['func']}")
        print(f"    Appears in {item['n_sims']}/10 sims | Best single-sim DL = {item['best_dl']:.1f}{combined_str}")

        # Show per-sim details
        for sim, params, dl, nll, det in sorted(item['entries'], key=lambda e: e[2]):
            print(f"      Sim {sim:2d}: DL={dl:.1f}, NLL={nll:.1f}, params={params}")
            print(f"              α_est={det['alpha_est']:.3f}, σ·f: {det['products'][100]:.4e}, {det['products'][1000]:.4e}, {det['products'][10000]:.4e}")
        print()

    # ==========================================
    # Summary
    # ==========================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal unique PS-like functions (all sources): {len(ps_like_sorted)}")
    print(f"Of these, {sum(1 for x in ps_like_sorted if x['in_combined'])} appear in top-200 combined ranking")

    # Highlight the most interesting ones
    print("\n--- Functions appearing in >=5 sims (most robust) ---")
    robust = [x for x in ps_like_sorted if x['n_sims'] >= 5]
    for item in robust:
        alphas = [e[4]['alpha_est'] for e in item['entries']]
        print(f"  {item['func']}")
        print(f"    {item['n_sims']}/10 sims, α_est = {np.mean(alphas):.3f} ± {np.std(alphas):.3f}")
        if item['in_combined']:
            print(f"    Combined rank {item['combined_rank']}, DL_combined = {item['dl_combined']:.1f}")
        print()

    # Also specifically look for the known function
    print("\n--- Known PS-like function check ---")
    known = '1/(x + exp(pow(x,a0)))'
    if known in ps_like_all:
        print(f"  {known}: found in {len(ps_like_all[known])} sims")
        for sim, params, dl, nll, det in ps_like_all[known]:
            print(f"    Sim {sim}: a0={params[0]:.4f}, DL={dl:.1f}, α_est={det['alpha_est']:.3f}")
    else:
        # Try checking it manually with typical params
        print(f"  {known}: not found in per-sim files with PS-like behaviour")
        print("  Checking manually with a0=-1.905...")
        is_ps, det = check_ps_like(known, [-1.905])
        print(f"    is_ps={is_ps}, details={det}")

if __name__ == '__main__':
    main()
