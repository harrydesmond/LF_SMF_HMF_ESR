"""Recovery: re-run sims 41 and 61 with per-function timeout, plus fit re/im functions for all 100 sims.

Combines two tasks:
1. Refit all 200 functions to sims 41 and 61 (with 120s timeout per function)
2. Fit 17 re/im-stripped functions to all 100 sims and append to output files

Usage:
    addqueue -q cmb -n 2x20 -m 4 -c "HMF trimmed: recovery sims 41,61 + re/im" \
        /usr/bin/env LD_LIBRARY_PATH=... PYTHONPATH=... python3 run_trimmed_hmf_recovery.py
"""

import sys
sys.path.insert(0, '/mnt/zfsusers/ameliaford/original_ESR/ESR')

import numpy as np
import os
import traceback
import signal

from esr.fitting.likelihood import PoissonLikelihood
from esr.fitting.fit_single import fit_from_string

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


TIMEOUT_SECONDS = 120  # per-function timeout


class FitTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise FitTimeout("Fit timed out")


def strip_re_im(func_str):
    """Strip re() and cos(im()) from function string for real-valued evaluation."""
    result = func_str

    # Remove cos(im(...)) factors
    while 'cos(im(' in result:
        idx = result.find('cos(im(')
        depth = 0
        end = idx + 4
        for i in range(idx + 4, len(result)):
            if result[i] == '(':
                depth += 1
            elif result[i] == ')':
                if depth == 0:
                    end = i + 1
                    break
                depth -= 1
        prefix = result[:idx]
        suffix = result[end:]
        if prefix.endswith('*'):
            prefix = prefix[:-1]
        elif suffix.startswith('*'):
            suffix = suffix[1:]
        result = prefix + suffix

    # Strip re(...) -> contents
    while 're(' in result:
        idx = result.find('re(')
        depth = 0
        end = idx + 3
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


def create_trimmed_data(sim_id):
    """Create a trimmed data file (drop first 2 rows)."""
    src = f'data/hmf_files/hmf_{sim_id}.dat'
    data = np.loadtxt(src)
    trimmed = data[2:]
    dst_rel = f'hmf_files/hmf_{sim_id}_trimmed.dat'
    dst_abs = os.path.join(os.getcwd(), dst_rel)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    np.savetxt(dst_abs, trimmed, fmt='%.18e')
    return dst_rel


def fit_str(func_str, data_path_rel):
    """Fit a single function string to trimmed data with timeout."""
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

    likelihood = PoissonLikelihood(data_path_rel, 'poisson_trimmed',
                                   data_dir='.', fn_set='base_e_maths')

    # Set alarm for timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
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
    except FitTimeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return np.nan, np.nan, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not done:
        return np.nan, np.nan, None

    return logl, dl, params


def run_full_sim(sim_id, functions):
    """Fit all functions to one sim (for recovery sims 41, 61)."""
    print(f'Rank {rank}: FULL refit sim {sim_id} ({len(functions)} functions)', flush=True)

    data_path_rel = create_trimmed_data(sim_id)

    data = []
    all_funcs = []
    for idx, func in enumerate(functions):
        try:
            logl, dl, params = fit_str(func, data_path_rel)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: func {idx} ERROR: {e}', flush=True)
            continue

        if not np.isnan(dl) and func not in all_funcs:
            data.append([func, dl, logl, str(params)])
            all_funcs.append(func)

        if idx % 20 == 0:
            print(f'Rank {rank}, sim {sim_id}: {idx}/{len(functions)}', flush=True)

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


def run_re_im_sim(sim_id, re_functions):
    """Fit re/im-stripped functions to one sim and append to results."""
    data_path_rel = create_trimmed_data(sim_id)

    outdir = f'hmf_data/hmf_{sim_id}_data'
    outpath = os.path.join(outdir, 'final_all_trimmed.txt')
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
        if orig_func in existing_funcs or stripped_func in existing_funcs:
            continue

        try:
            logl, dl, params = fit_str(stripped_func, data_path_rel)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: re/im func {idx} ERROR: {e}', flush=True)
            continue

        if not np.isnan(dl):
            new_data.append([orig_func, dl, logl, str(params)])

    if not new_data:
        return

    os.makedirs(outdir, exist_ok=True)
    with open(outpath, 'a') as f:
        for func, dl, nll, params in new_data:
            f.write(f'{existing_count};{func};{dl};{nll};{params}\n')
            existing_count += 1

    print(f'Rank {rank}, sim {sim_id}: {len(new_data)} re/im functions appended', flush=True)


if __name__ == '__main__':
    # Load all 200 functions
    with open('top_500_trimmed.txt', 'r') as f:
        all_functions = [line.strip() for line in f][:200]

    # Identify re/im functions
    re_functions = []
    for func in all_functions:
        if 're(' in func or 'im(' in func:
            stripped = strip_re_im(func)
            re_functions.append((func, stripped))
            if rank == 0:
                print(f'  re/im: {func}  -->  {stripped}', flush=True)

    if rank == 0:
        print(f'\n{len(re_functions)} re/im functions to fit\n', flush=True)

    # Phase 1: Recovery sims 41 and 61 (full refit with timeout)
    recovery_sims = [41, 61]
    my_recovery = [s for i, s in enumerate(recovery_sims) if i % size == rank]
    for sim_id in my_recovery:
        try:
            run_full_sim(sim_id, all_functions)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: RECOVERY FAILED: {e}\n{traceback.format_exc()}', flush=True)

    comm.Barrier()
    if rank == 0:
        print('\nPhase 1 (recovery) done. Starting Phase 2 (re/im)...\n', flush=True)

    # Phase 2: re/im functions for all 100 sims
    all_sims = list(range(100))
    my_sims = [s for i, s in enumerate(all_sims) if i % size == rank]
    for sim_id in my_sims:
        try:
            run_re_im_sim(sim_id, re_functions)
        except Exception as e:
            print(f'Rank {rank}, sim {sim_id}: RE/IM FAILED: {e}\n{traceback.format_exc()}', flush=True)

    comm.Barrier()
    if rank == 0:
        print('\nAll done!', flush=True)
