"""Follow-up: fit the 17 functions containing re()/im() after stripping to real form.

After the main Step 3 job completes, this fits the stripped versions to all 100 sims
and appends results to the existing final_all_fiducial.txt files.

Usage:
    addqueue -q cmb -n 2x20 -m 4 -c "HMF fiducial step3: re/im functions" \
        /usr/bin/env LD_LIBRARY_PATH=... PYTHONPATH=... python3 run_fiducial_hmf_re.py
"""

import sys
sys.path.insert(0, '/mnt/zfsusers/ameliaford/original_ESR/ESR')

import numpy as np
import os
import re
import traceback

from esr.fitting.likelihood import PoissonLikelihood
from esr.fitting.fit_single import fit_from_string

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

all_sims = list(range(100))
my_sims = [s for i, s in enumerate(all_sims) if i % size == rank]


def strip_re_im(func_str):
    """Strip re() and cos(im()) from function string for real-valued evaluation.

    re(expr) -> expr  (since all inputs are real)
    cos(im(expr)) -> 1  (since im(real) = 0, cos(0) = 1)
    """
    # First remove *cos(im(...)) factors — need to match balanced parens
    # Pattern: *cos(im(...))
    # We do this iteratively since regex can't match balanced parens
    result = func_str

    # Remove cos(im(...)) by finding and removing balanced expressions
    while 'cos(im(' in result:
        idx = result.find('cos(im(')
        # Find the matching closing paren for cos(
        depth = 0
        end = idx + 4  # start after 'cos('
        for i in range(idx + 4, len(result)):
            if result[i] == '(':
                depth += 1
            elif result[i] == ')':
                if depth == 0:
                    end = i + 1
                    break
                depth -= 1
        # Remove the cos(im(...)) and any preceding *
        prefix = result[:idx]
        suffix = result[end:]
        if prefix.endswith('*'):
            prefix = prefix[:-1]
        elif suffix.startswith('*'):
            suffix = suffix[1:]
        result = prefix + suffix

    # Now strip re(...) -> contents
    while 're(' in result:
        idx = result.find('re(')
        # Find matching close paren
        depth = 0
        end = idx + 3  # start after 're('
        for i in range(idx + 3, len(result)):
            if result[i] == '(':
                depth += 1
            elif result[i] == ')':
                if depth == 0:
                    end = i
                    break
                depth -= 1
        inner = result[idx+3:end]
        result = result[:idx] + inner + result[end+1:]

    return result


def create_fiducial_data(sim_id):
    """Create a fiducial data file (drop first 2 rows)."""
    src = f'data/hmf_files/hmf_{sim_id}.dat'
    data = np.loadtxt(src)
    fiducial = data[2:]
    dst_rel = f'hmf_files/hmf_{sim_id}_fiducial.dat'
    dst_abs = os.path.join(os.getcwd(), dst_rel)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    np.savetxt(dst_abs, fiducial, fmt='%.18e')
    return dst_rel


def fit_str(func_str, data_path_rel):
    """Fit a single function string to fiducial data."""
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

    likelihood = PoissonLikelihood(data_path_rel, 'poisson_fiducial',
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


def run_sim(sim_id, re_functions):
    """Fit re/im-stripped functions to one fiducial sim and append to results."""
    print(f'Rank {rank}: starting sim {sim_id} ({len(re_functions)} functions)', flush=True)

    data_path_rel = create_fiducial_data(sim_id)

    # Read existing results to get current count for numbering
    outdir = f'hmf_data/hmf_{sim_id}_data'
    outpath = os.path.join(outdir, 'final_all_fiducial.txt')
    existing_count = 0
    existing_funcs = set()
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    existing_funcs.add(parts[1])
                    existing_count += 1

    new_data = []
    for idx, (orig_func, stripped_func) in enumerate(re_functions):
        # Skip if original already fitted somehow
        if orig_func in existing_funcs or stripped_func in existing_funcs:
            continue

        try:
            logl, dl, params = fit_str(stripped_func, data_path_rel)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: func {idx} ERROR: {e}', flush=True)
            continue

        if not np.isnan(dl):
            # Store with original function name for traceability
            new_data.append([orig_func, dl, logl, str(params)])

    if not new_data:
        print(f'Rank {rank}, sim {sim_id}: no new fits', flush=True)
        return

    # Append to existing file
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, 'a') as f:
        for func, dl, nll, params in new_data:
            f.write(f'{existing_count};{func};{dl};{nll};{params}\n')
            existing_count += 1

    print(f'Rank {rank}, sim {sim_id}: DONE ({len(new_data)} re/im functions appended)', flush=True)


if __name__ == '__main__':
    # Load functions with re/im and create stripped versions
    with open('top_500_fiducial.txt') as f:
        all_functions = [line.strip() for line in f][:200]

    re_functions = []
    for func in all_functions:
        if 're(' in func or 'im(' in func:
            stripped = strip_re_im(func)
            re_functions.append((func, stripped))
            if rank == 0:
                print(f'  {func}  -->  {stripped}', flush=True)

    if rank == 0:
        print(f'\n{len(re_functions)} functions to fit after stripping re/im\n', flush=True)

    for sim_id in my_sims:
        try:
            run_sim(sim_id, re_functions)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: FAILED with {e}\n{traceback.format_exc()}', flush=True)

    comm.Barrier()
    if rank == 0:
        print('\nAll re/im functions done!', flush=True)
