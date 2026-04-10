"""
Compute combined DL for the new trimmed HMF functions (from full ESR rerun)
across all 100 simulations.

For independent datasets, the combined DL is:
    DL_combined = sum_i(NLL_i) + sum_i(codelength_i) + aifeynman
                = sum_i(DL_i) - (N-1) * aifeynman

Reads from: hmf_data/hmf_{sim}_data/final_all_trimmed.txt
Functions from: top_500_trimmed.txt (first 200)
Output: hmf_combined_DL_trimmed_new.txt

Also includes Warren and Tinker literature fits from the existing
hmf_combined_DL_trimmed.txt for comparison.
"""

import sys
sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')

import numpy as np
import os
from esr.generation.generator import string_to_node, aifeyn_complexity

basis_functions = [["x", "a"],       # type0: nullary
                   ["inv","exp","log","abs"],  # type1: unary
                   ["+", "*", "-", "/", "pow"]]  # type2: binary


def get_aifeynman(func_str):
    """Compute aifeynman for a function string using ESR's tree representation."""
    try:
        expr, nodes, complexity = string_to_node(func_str, basis_functions, evalf=True)
        labels = nodes.to_list(basis_functions)
        labels = [lab.lower() if lab not in ['Mul', 'Add'] else (
            '*' if lab == 'Mul' else '+') for lab in labels]
        nparam = sum(1 for i in range(10) if f'a{i}' in func_str)
        param_list = [f'a{i}' for i in range(nparam)]
        return aifeyn_complexity(labels, param_list)
    except Exception as e:
        print(f"  Warning: could not compute aifeynman for {func_str[:60]}...: {e}")
        return None


def get_generation_complexity(func_str):
    """Get the generation complexity (number of nodes in expression tree)."""
    try:
        expr, nodes, complexity = string_to_node(func_str, basis_functions, evalf=True)
        return complexity
    except:
        return None


def main():
    # Load the top 200 functions from trimmed rerun
    with open('top_500_trimmed.txt', 'r') as f:
        all_functions = [line.strip() for line in f if line.strip()]
    functions_200 = all_functions[:200]

    # Find available sims
    data_dir = 'hmf_data'
    sim_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('hmf_') and d.endswith('_data')])
    sim_numbers = sorted([int(d.split('_')[1]) for d in sim_dirs])
    print(f"Found {len(sim_numbers)} simulations: {sim_numbers[0]}..{sim_numbers[-1]}")

    # Load per-sim data from final_all_trimmed.txt
    sim_data = {}
    for sim in sim_numbers:
        filepath = os.path.join(data_dir, f'hmf_{sim}_data/final_all_trimmed.txt')
        if not os.path.exists(filepath):
            continue
        try:
            func_dl = {}
            with open(filepath) as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) < 4:
                        continue
                    func = parts[1]
                    dl = float(parts[2])
                    nll = float(parts[3])
                    func_dl[func] = (dl, nll)
            sim_data[sim] = func_dl
        except Exception as e:
            print(f"  Warning: could not load sim {sim}: {e}")

    print(f"Successfully loaded {len(sim_data)} simulations")

    # Compute aifeynman and generation complexity for each function (once)
    print("\nComputing aifeynman for each function...")
    aifeyn_cache = {}
    gencomp_cache = {}
    for i, func in enumerate(functions_200):
        af = get_aifeynman(func)
        gc = get_generation_complexity(func)
        aifeyn_cache[func] = af
        gencomp_cache[func] = gc
        if i < 5 or (i+1) % 50 == 0:
            af_str = f"{af:.4f}" if af is not None else "None"
            print(f"  [{i+1}/200] aifeyn = {af_str}, gencomp = {gc} for {func[:50]}...")

    # Compute combined DL for each function
    print("\nComputing combined DL...")
    N = len(sim_data)
    results = []

    for func in functions_200:
        aifeyn = aifeyn_cache.get(func)
        if aifeyn is None:
            continue

        DLs = []
        NLLs = []
        for sim in sorted(sim_data.keys()):
            if func in sim_data[sim]:
                dl, nll = sim_data[sim][func]
                if np.isfinite(dl) and np.isfinite(nll) and dl < 0 and nll < 0:
                    DLs.append(dl)
                    NLLs.append(nll)

        n_sims = len(DLs)
        if n_sims == 0:
            continue

        sum_DL = sum(DLs)
        sum_NLL = sum(NLLs)
        DL_combined = sum_DL - (n_sims - 1) * aifeyn

        results.append({
            'function': func,
            'DL_combined': DL_combined,
            'sum_NLL': sum_NLL,
            'sum_DL': sum_DL,
            'aifeyn': aifeyn,
            'gencomp': gencomp_cache.get(func),
            'n_sims': n_sims,
        })

    # Sort by combined DL (lower is better)
    results.sort(key=lambda r: r['DL_combined'])

    # Print results
    print(f"\n{'='*80}")
    print(f"Top 30 functions ranked by combined DL (TRIMMED) across {N} simulations")
    print(f"{'='*80}")
    best_DL = results[0]['DL_combined']
    print(f"{'Rank':>4} {'n_sims':>6} {'DL_combined':>16} {'delta_DL':>10} {'aifeyn':>8} {'comp':>4}  Function")
    print(f"{'-'*4} {'-'*6} {'-'*16} {'-'*10} {'-'*8} {'-'*4}  {'-'*40}")
    for i, r in enumerate(results[:30]):
        delta = r['DL_combined'] - best_DL
        gc = r['gencomp'] if r['gencomp'] is not None else '?'
        print(f"{i+1:4d} {r['n_sims']:6d} {r['DL_combined']:16.2f} {delta:10.2f} {r['aifeyn']:8.4f} {gc:>4}  {r['function'][:60]}")

    # Save full results
    outfile = 'hmf_combined_DL_trimmed_new.txt'
    with open(outfile, 'w') as f:
        f.write('# rank;function;DL_combined;delta_DL;sum_NLL;sum_DL;aifeynman;gencomp;n_sims\n')
        for i, r in enumerate(results):
            delta = r['DL_combined'] - best_DL
            gc = r['gencomp'] if r['gencomp'] is not None else -1
            f.write(f"{i};{r['function']};{r['DL_combined']:.6f};{delta:.6f};{r['sum_NLL']:.6f};{r['sum_DL']:.6f};{r['aifeyn']:.6f};{gc};{r['n_sims']}\n")
    print(f"\nFull results saved to {outfile}")

    # Also compute combined DL for the old (untrimmed) top functions on trimmed data
    # Load the existing untrimmed combined DL to get the literature fits
    print("\n\nLoading literature fits from existing trimmed results...")
    lit_file = 'hmf_combined_DL_trimmed.txt'
    if os.path.exists(lit_file):
        with open(lit_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(';')
                func = parts[1]
                if 'Warren' in func or 'Tinker' in func or 'Press' in func:
                    print(f"  Literature: {func[:60]}  DL={parts[2]}")


if __name__ == '__main__':
    main()
