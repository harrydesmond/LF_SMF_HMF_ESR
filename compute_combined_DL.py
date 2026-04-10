"""
Compute combined DL for the top 200 HMF functions across all 100 simulations.

For independent datasets, the combined DL is:
    DL_combined = sum_i(NLL_i) + sum_i(codelength_i) + aifeynman
                = sum_i(DL_i) - (N-1) * aifeynman

where N is the number of sims, since each per-sim DL_i already includes one
copy of aifeynman: DL_i = NLL_i + codelength_i + aifeynman.

The aifeynman (functional complexity) is computed from the expression tree
using the ESR library.
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
        # Normalise labels to lowercase
        labels = [lab.lower() if lab not in ['Mul', 'Add'] else (
            '*' if lab == 'Mul' else '+') for lab in labels]
        # Count parameters
        nparam = sum(1 for i in range(10) if f'a{i}' in func_str)
        param_list = [f'a{i}' for i in range(nparam)]
        return aifeyn_complexity(labels, param_list)
    except Exception as e:
        print(f"  Warning: could not compute aifeynman for {func_str[:60]}...: {e}")
        return None


def main():
    # Load the top 200 functions
    with open('top_500_all.txt', 'r') as f:
        all_functions = [line.strip() for line in f if line.strip()]
    functions_200 = all_functions[:200]

    # Find available sims
    data_dir = 'hmf_data'
    sim_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('hmf_') and d.endswith('_data')])
    sim_numbers = sorted([int(d.split('_')[1]) for d in sim_dirs])
    print(f"Found {len(sim_numbers)} simulations: {sim_numbers[0]}..{sim_numbers[-1]}")

    # Load per-sim data for all functions
    # For each sim, build a dict: function_string -> (DL, NLL)
    sim_data = {}
    for sim in sim_numbers:
        filepath = os.path.join(data_dir, f'hmf_{sim}_data/final_all.txt')
        if not os.path.exists(filepath):
            continue
        try:
            rankings, funcs, DLs, NLLs = np.loadtxt(
                filepath, dtype=str, delimiter=';', usecols=(0,1,2,3), unpack=True)
            sim_data[sim] = {f: (float(d), float(n)) for f, d, n in zip(funcs, DLs, NLLs)}
        except Exception as e:
            print(f"  Warning: could not load sim {sim}: {e}")

    print(f"Successfully loaded {len(sim_data)} simulations")

    # Compute aifeynman for each function (once)
    print("\nComputing aifeynman for each function...")
    aifeyn_cache = {}
    for i, func in enumerate(functions_200):
        af = get_aifeynman(func)
        aifeyn_cache[func] = af
        if i < 5 or (i+1) % 50 == 0:
            print(f"  [{i+1}/200] aifeyn = {af:.4f} for {func[:50]}...")

    # Compute combined DL for each function
    print("\nComputing combined DL...")
    N = len(sim_data)
    results = []

    for func in functions_200:
        aifeyn = aifeyn_cache.get(func)
        if aifeyn is None:
            continue

        # Gather DL and NLL across all sims where this function appears
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

        # DL_combined = sum(DL_i) - (n_sims - 1) * aifeynman
        sum_DL = sum(DLs)
        sum_NLL = sum(NLLs)
        DL_combined = sum_DL - (n_sims - 1) * aifeyn

        results.append({
            'function': func,
            'DL_combined': DL_combined,
            'sum_NLL': sum_NLL,
            'sum_DL': sum_DL,
            'aifeyn': aifeyn,
            'n_sims': n_sims,
        })

    # Sort by combined DL (lower is better)
    results.sort(key=lambda r: r['DL_combined'])

    # Print results
    print(f"\n{'='*80}")
    print(f"Top 20 functions ranked by combined DL across {N} simulations")
    print(f"{'='*80}")
    best_DL = results[0]['DL_combined']
    print(f"{'Rank':>4} {'n_sims':>6} {'DL_combined':>16} {'delta_DL':>10} {'aifeyn':>8}  Function")
    print(f"{'-'*4} {'-'*6} {'-'*16} {'-'*10} {'-'*8}  {'-'*40}")
    for i, r in enumerate(results[:30]):
        delta = r['DL_combined'] - best_DL
        print(f"{i+1:4d} {r['n_sims']:6d} {r['DL_combined']:16.2f} {delta:10.2f} {r['aifeyn']:8.4f}  {r['function'][:60]}")

    # Save full results
    outfile = 'hmf_combined_DL.txt'
    with open(outfile, 'w') as f:
        f.write('# rank;function;DL_combined;delta_DL;sum_NLL;sum_DL;aifeynman;n_sims\n')
        for i, r in enumerate(results):
            delta = r['DL_combined'] - best_DL
            f.write(f"{i};{r['function']};{r['DL_combined']:.6f};{delta:.6f};{r['sum_NLL']:.6f};{r['sum_DL']:.6f};{r['aifeyn']:.6f};{r['n_sims']}\n")
    print(f"\nFull results saved to {outfile}")


if __name__ == '__main__':
    main()
