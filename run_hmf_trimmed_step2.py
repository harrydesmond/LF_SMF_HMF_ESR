"""Step 2: Identify top 200 unique functions from trimmed ESR results.

After Step 1 completes, this script:
1. Loads the per-complexity results from the 10 representative sims
2. Ranks all unique functions by DL within each sim
3. Averages ranks across sims
4. Selects the 200 functions with lowest mean rank
5. Saves to top_500_trimmed.txt

Usage: python3 run_hmf_trimmed_step2.py
"""

import numpy as np
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

hmf_sims = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def format_function(fcn):
    """Normalise function strings for deduplication (from sample_top_200.py)."""
    fcn = fcn.replace('a0', 'C').replace('a1', 'C').replace('a2', 'C').replace('a3', 'C')
    fcn = fcn.replace('exp(C)', 'C').replace('1/C', 'C')
    fcn = fcn.replace('Abs(C + x)', 'Abs(C - x)')
    fcn = fcn.replace('log(Abs(C))', 'C')
    fcn = fcn.replace(',(C)', ',C')
    fcn = fcn.replace('Abs(1/C)', 'Abs(C)')

    const = 0
    new_fcn = ''
    for char in fcn:
        if char == 'C':
            new_fcn += 'a{}'.format(const)
            const += 1
        else:
            new_fcn += char

    fcn = new_fcn.replace('-a', 'a')
    return fcn


def load_sim_results(sim):
    """Load all complexity results for one sim, return dict: function -> DL."""
    all_funcs = {}
    for comp in range(4, 11):
        filepath = f'hmf_trimmed_{sim}_data/final_{comp}_trimmed.dat'
        if not os.path.exists(filepath):
            print(f"  WARNING: missing {filepath}")
            continue

        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) < 5:
                    continue
                func = parts[1]
                try:
                    dl = float(parts[2])
                    nll = float(parts[4])
                except (ValueError, IndexError):
                    continue

                if not np.isfinite(dl) or not np.isfinite(nll):
                    continue
                if dl > 0 or nll > 0:
                    continue  # invalid

                # Keep the best DL for each unique function
                if func not in all_funcs or dl < all_funcs[func]:
                    all_funcs[func] = dl

    return all_funcs


def main():
    # Load results from all 10 sims
    sim_results = {}
    for sim in hmf_sims:
        print(f"Loading sim {sim}...")
        sim_results[sim] = load_sim_results(sim)
        print(f"  {len(sim_results[sim])} unique functions")

    # Get all functions that appear in at least one sim
    all_functions = set()
    for sim in hmf_sims:
        all_functions.update(sim_results[sim].keys())
    print(f"\nTotal unique functions across all sims: {len(all_functions)}")

    # Deduplicate using format_function
    # Map: formatted -> list of original function strings
    formatted_map = {}
    for func in all_functions:
        fmt = format_function(func)
        if fmt not in formatted_map:
            formatted_map[fmt] = []
        formatted_map[fmt].append(func)

    print(f"After deduplication: {len(formatted_map)} unique formatted functions")

    # For each formatted function group, pick the representative with best average DL
    # Then rank within each sim and compute mean rank

    # First: for each sim, rank all functions by DL
    sim_ranks = {}
    for sim in hmf_sims:
        # Sort by DL (most negative = best)
        sorted_funcs = sorted(sim_results[sim].items(), key=lambda x: x[1])
        rank_dict = {}
        for rank, (func, dl) in enumerate(sorted_funcs):
            rank_dict[func] = rank
        sim_ranks[sim] = rank_dict

    # For each formatted function group, compute mean rank
    func_mean_ranks = []
    for fmt, originals in formatted_map.items():
        # For each sim, find the best rank among all originals in this group
        ranks = []
        for sim in hmf_sims:
            best_rank = None
            for orig in originals:
                if orig in sim_ranks[sim]:
                    r = sim_ranks[sim][orig]
                    if best_rank is None or r < best_rank:
                        best_rank = r
            if best_rank is not None:
                ranks.append(best_rank)

        if len(ranks) < 5:  # require at least 5 sims
            continue

        mean_rank = np.mean(ranks)
        # Pick the original with best DL across all sims as representative
        best_orig = None
        best_dl = np.inf
        for orig in originals:
            for sim in hmf_sims:
                if orig in sim_results[sim] and sim_results[sim][orig] < best_dl:
                    best_dl = sim_results[sim][orig]
                    best_orig = orig

        func_mean_ranks.append((best_orig, mean_rank, len(ranks)))

    # Sort by mean rank
    func_mean_ranks.sort(key=lambda x: x[1])

    # Save top 500 (we'll use top 200 for Step 3)
    outfile = 'top_500_trimmed.txt'
    with open(outfile, 'w') as f:
        for func, mean_rank, n_sims in func_mean_ranks[:500]:
            f.write(func + '\n')

    print(f"\nSaved top 500 to {outfile}")
    print(f"\nTop 20 by mean rank:")
    for i, (func, mr, ns) in enumerate(func_mean_ranks[:20]):
        print(f"  {i+1:3d}  mean_rank={mr:8.1f}  n_sims={ns:2d}  {func[:60]}")


if __name__ == '__main__':
    main()
