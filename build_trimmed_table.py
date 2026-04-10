"""
Build data for the trimmed HMF table (tab:HMF_trimmed) in Paper.tex.

Reads: hmf_combined_DL_trimmed_new.txt, literature_combined_DL_trimmed.txt,
       literature_fits_trimmed.txt, hmf_data/hmf_*_data/final_all_trimmed.txt
"""

import sys
sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')

import numpy as np
import os
from esr.generation.generator import string_to_node, aifeyn_complexity

basis_functions = [["x", "a"],
                   ["inv", "exp", "log", "abs"],
                   ["+", "*", "-", "/", "pow"]]


def get_gencomp(func_str):
    try:
        expr, nodes, complexity = string_to_node(func_str, basis_functions, evalf=True)
        return complexity
    except:
        return None


def main():
    # Load ESR combined DL results
    esr_results = []
    with open('hmf_combined_DL_trimmed_new.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(';')
            esr_results.append({
                'rank': int(parts[0]),
                'function': parts[1],
                'DL_combined': float(parts[2]),
                'delta_DL': float(parts[3]),
                'sum_NLL': float(parts[4]),
                'sum_DL': float(parts[5]),
                'aifeyn': float(parts[6]),
                'gencomp': int(parts[7]) if parts[7] != '-1' else None,
                'n_sims': int(parts[8]),
            })

    best_DL = esr_results[0]['DL_combined']

    # Load literature combined DL
    lit_results = {}
    with open('literature_combined_DL_trimmed.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(';')
            name = parts[0]
            lit_results[name] = {
                'DL_combined': float(parts[1]),
                'sum_NLL': float(parts[2]),
                'sum_DL': float(parts[3]),
                'aifeyn': float(parts[4]),
                'n_sims': int(parts[5]),
            }

    # Load literature per-sim parameters
    lit_params = {}
    with open('literature_fits_trimmed.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(';')
            name = parts[0]
            if name not in lit_params:
                lit_params[name] = []
            pstr = parts[5]
            if pstr != 'none':
                params = [float(x) for x in pstr.split()]
                lit_params[name].append(params)

    # Load ESR per-sim parameters for top 8 functions
    top_n = 8
    top_funcs = [r['function'] for r in esr_results[:top_n]]

    # Collect parameters across all sims
    func_params = {f: [] for f in top_funcs}
    data_dir = 'hmf_data'
    for sim in range(100):
        filepath = os.path.join(data_dir, f'hmf_{sim}_data/final_all_trimmed.txt')
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) < 5:
                    continue
                func = parts[1]
                if func in func_params:
                    # Parse params from string like "[ 0.82681409  0.4562324  -2.21343193]"
                    pstr = parts[4].strip()
                    try:
                        pstr = pstr.replace('[', '').replace(']', '')
                        params = [float(x) for x in pstr.split()]
                        func_params[func].append(params)
                    except:
                        pass

    # Print table data
    print("=" * 120)
    print("TRIMMED HMF TABLE DATA")
    print("=" * 120)

    # ESR best NLL for delta_NLL computation
    best_NLL = esr_results[0]['sum_NLL']

    print(f"\n{'Rank':>4} {'Comp':>4} {'ΔDL':>8} {'ΔNLL':>8} {'n':>3}  {'Params':60s}  Function")
    print("-" * 120)

    for i, r in enumerate(esr_results[:top_n]):
        delta_DL = r['DL_combined'] - best_DL
        delta_NLL = r['sum_NLL'] - best_NLL
        gc = r['gencomp'] if r['gencomp'] else '?'

        # Format parameters
        params = func_params.get(r['function'], [])
        if params:
            arr = np.array(params)
            n_params = arr.shape[1]
            pstr_parts = []
            for j in range(n_params):
                med = np.median(arr[:, j])
                p16 = np.percentile(arr[:, j], 16)
                p84 = np.percentile(arr[:, j], 84)
                pstr_parts.append(f"θ{j}={med:.2f}(+{p84-med:.2f}/-{med-p16:.2f})")
            pstr = ', '.join(pstr_parts)
        else:
            pstr = "no params collected"

        print(f"{i+1:4d} {gc:>4} {delta_DL:8.0f} {delta_NLL:8.0f} {r['n_sims']:3d}  {pstr:60s}  {r['function'][:50]}")

    print("\nLiterature fits:")
    print("-" * 120)

    for name in ['P.Sch.', 'War.', 'Tin.']:
        lr = lit_results[name]
        delta_DL = -(lr['DL_combined'] - (-best_DL))  # Note: lit DL is positive, ESR is negative
        # Actually: ESR DL_combined is negative (more negative = better)
        # Lit DL_combined is positive (from the sign convention in fit_literature)
        # delta_DL = lit_DL - best_ESR_DL, but signs are different...
        # Let me be more careful. ESR stores DL as negative (better = more negative).
        # Literature stores as positive (the raw DL value).
        # Combined DL for literature: sum of -DL per sim + (n-1)*aifeyn
        # For ESR: sum of DL per sim (already negative) - (n-1)*aifeyn
        # The comparison: ESR DL_combined = -1433112815, Lit DL_combined = +1433110590
        # So Lit is worse by: (-best_ESR) - lit = 1433112815 - 1433110590 = 2225
        # Wait that says ESR is 2225 better. delta_DL should be positive for lit.

        delta_DL = (-best_DL) - lr['DL_combined']
        delta_NLL_lit = (-best_NLL) - (-lr['sum_NLL'])

        params = lit_params.get(name, [])
        if params:
            arr = np.array(params)
            n_params = arr.shape[1]
            pstr_parts = []
            for j in range(n_params):
                med = np.median(arr[:, j])
                p16 = np.percentile(arr[:, j], 16)
                p84 = np.percentile(arr[:, j], 84)
                pstr_parts.append(f"θ{j}={med:.2f}(+{p84-med:.2f}/-{med-p16:.2f})")
            pstr = ', '.join(pstr_parts)
        else:
            pstr = "δc = 1.686"

        print(f"{'':>4} {name:>4} {delta_DL:8.0f} {delta_NLL_lit:8.0f} {lr['n_sims']:3d}  {pstr}")

    # Also print comparison with untrimmed
    print("\n\nCOMPARISON: Untrimmed vs Trimmed")
    print("=" * 80)

    # Load untrimmed results
    untrimmed_results = []
    with open('hmf_combined_DL.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(';')
            untrimmed_results.append({
                'function': parts[1],
                'DL_combined': float(parts[2]),
            })

    untrimmed_lit = {}
    with open('literature_combined_DL.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(';')
            untrimmed_lit[parts[0]] = float(parts[1])

    print(f"\nUntrimmed best ESR DL: {untrimmed_results[0]['DL_combined']:.2f}")
    print(f"  Warren ΔDL: {(-untrimmed_results[0]['DL_combined']) - untrimmed_lit['War.']:.0f}")
    print(f"  Tinker ΔDL: {(-untrimmed_results[0]['DL_combined']) - untrimmed_lit['Tin.']:.0f}")

    print(f"\nTrimmed best ESR DL: {best_DL:.2f}")
    print(f"  Warren ΔDL: {(-best_DL) - lit_results['War.']['DL_combined']:.0f}")
    print(f"  Tinker ΔDL: {(-best_DL) - lit_results['Tin.']['DL_combined']:.0f}")

    print(f"\nUntrimmed rank-1: {untrimmed_results[0]['function']}")
    print(f"Trimmed rank-1:   {esr_results[0]['function']}")

    # Check if trimmed rank-1 appears in untrimmed
    for i, r in enumerate(untrimmed_results):
        if r['function'] == esr_results[0]['function']:
            print(f"  (Trimmed rank-1 was untrimmed rank {i+1}, ΔDL = {r['DL_combined'] - untrimmed_results[0]['DL_combined']:.0f})")
            break


if __name__ == '__main__':
    main()
