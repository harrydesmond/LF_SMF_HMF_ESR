"""Refit top 200 HMF functions on trimmed data (2 lowest-mass bins removed).

This is a modified version of sample_top_200.py's run_fits() that:
1. Creates trimmed data files (dropping first 2 rows from hmf_*.dat)
2. Fits all 200 functions from top_500_all.txt to each sim using ESR's fit_from_string
3. Saves results to hmf_data/hmf_{sim}_data/final_all_trimmed.txt

Usage:
    addqueue -q berg -n 50 -m 4 /usr/local/shared/python/3.11.4/bin/python3 run_trimmed_hmf.py
"""

import sys
sys.path.insert(0, '/mnt/zfsusers/ameliaford/original_ESR/ESR')

import numpy as np
import os
import traceback

from esr.fitting.likelihood import PoissonLikelihood
from esr.fitting.fit_single import fit_from_string

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# All 100 sims
all_sims = list(range(100))

# Distribute sims across MPI ranks
my_sims = [s for i, s in enumerate(all_sims) if i % size == rank]


def create_trimmed_data(sim_id):
    """Create a trimmed data file (drop first 2 rows) in hmf_files/.

    The PoissonLikelihood reads cols 0-3 (sigma, counts, error, Veff).
    We keep all 5 columns so the file format is unchanged, just 2 rows shorter.
    The file is placed alongside the originals so the path can be passed
    directly to PoissonLikelihood (which prepends data_dir).
    """
    src = f'data/hmf_files/hmf_{sim_id}.dat'
    data = np.loadtxt(src)
    trimmed = data[2:]  # drop 2 lowest-mass (highest-sigma) bins
    # Write to the working directory (not data_dir, since we'll pass full path)
    dst_rel = f'hmf_files/hmf_{sim_id}_trimmed.dat'
    dst_abs = os.path.join(os.getcwd(), dst_rel)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    np.savetxt(dst_abs, trimmed, fmt='%.18e')
    return dst_rel, dst_abs


def fit_str(func_str, data_path_rel):
    """Fit a single function string to trimmed data."""
    basis_functions = [["x", "a"],
                       ["inv", "exp", "log", "abs"],
                       ["+", "*", "-", "/", "pow"]]

    nparam = sum(1 for i in range(4) if f'a{i}' in func_str)
    Niter_params = [3040, 3060]
    Nconv_params = [-5, 20]
    Niter_new = int(np.sum(nparam ** np.arange(len(Niter_params)) * np.array(Niter_params)))
    Nconv_new = int(np.sum(nparam ** np.arange(len(Nconv_params)) * np.array(Nconv_params)))

    if Nconv_new <= 0 or Niter_new <= 0 or Nconv_new > Niter_new:
        return np.nan, np.nan, None

    # Pass data_dir='.' so PoissonLikelihood reads from current working directory
    likelihood = PoissonLikelihood(data_path_rel, 'poisson_trimmed',
                                   data_dir='.', fn_set='base_e_maths')

    count = 0
    done = False
    while count < 20 and not done:
        count += 1
        logl, dl, tree, params = fit_from_string(
            func_str, basis_functions, likelihood,
            verbose=False, Niter=Niter_new, Nconv=Nconv_new, return_params=True)
        try:
            float(logl)
            float(dl)
            done = True
        except:
            pass

    if not done:
        return np.nan, np.nan, None

    return logl, dl, params


def run_sim(sim_id):
    """Fit all 200 functions to one trimmed sim."""
    print(f'Rank {rank}: starting sim {sim_id}', flush=True)

    # Create trimmed data file
    data_path_rel, data_path_abs = create_trimmed_data(sim_id)

    # Load top 200 functions
    with open('top_500_trimmed.txt', 'r') as f:
        functions = [line.strip() for line in f][:200]

    data = []
    all_funcs = []
    for idx, func in enumerate(functions):
        try:
            logl, dl, params = fit_str(func, data_path_rel)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: func {idx} ERROR: {e}\n{traceback.format_exc()}', flush=True)
            continue

        if not np.isnan(dl) and func not in all_funcs:
            data.append([func, dl, logl, str(params)])
            all_funcs.append(func)

        if idx % 20 == 0:
            print(f'Rank {rank}, sim {sim_id}: {idx}/200', flush=True)

    # Sort by DL
    if len(data) == 0:
        print(f'Rank {rank}, sim {sim_id}: NO FITS SUCCEEDED', flush=True)
        return

    sorted_data = sorted(data, key=lambda x: x[1])

    outdir = f'hmf_data/hmf_{sim_id}_data'
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'final_all_trimmed.txt')

    with open(outpath, 'w') as f:
        for i, (func, dl, nll, params) in enumerate(sorted_data):
            f.write(f'{i};{func};{dl};{nll};{params}\n')

    print(f'Rank {rank}, sim {sim_id}: DONE ({len(sorted_data)} functions saved)', flush=True)


if __name__ == '__main__':
    for sim_id in my_sims:
        try:
            run_sim(sim_id)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: FAILED with {e}\n{traceback.format_exc()}', flush=True)

    comm.Barrier()
    if rank == 0:
        print('\nAll done!', flush=True)
