"""Step 1: Run full ESR on trimmed HMF data for 10 representative sims.

Creates trimmed data files (dropping 2 lowest-mass bins from hmf_*_new.dat),
then runs the full ESR pipeline (test_all, test_all_Fisher, match, combine_DL)
for complexities 4-10 on each of the 10 representative sims.

This is the most expensive step: ~1.5M functions at complexity 10.

Usage:
    # Run one sim+comp at a time via CLI args:
    addqueue -q berg -n 20 -m 5 /bin/bash run_trimmed_step1.sh <sim> <comp>

    # Or use the batch submission script run_trimmed_step1_batch.sh
"""

import sys
import os
import shutil
import numpy as np

# CLI: python3 run_hmf_trimmed_step1.py <sim> <comp>
if len(sys.argv) != 3:
    print("Usage: python3 run_hmf_trimmed_step1.py <sim> <comp>")
    print("  sim: Quijote simulation number (0, 10, 20, ..., 90)")
    print("  comp: ESR complexity (4-10)")
    sys.exit(1)

hmf_sim = int(sys.argv[1])
comp = int(sys.argv[2])

sys.path.insert(0, '/mnt/zfsusers/ameliaford/original_ESR/ESR')
os.chdir('/users/hdesmond/Amelia_code')

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create trimmed data file if it doesn't exist
trimmed_path = f'hmf_files/hmf_{hmf_sim}_trimmed.dat'
if rank == 0:
    src = f'/mnt/zfsusers/ameliaford/original_ESR/ESR/hmf_files/hmf_{hmf_sim}_new.dat'
    data = np.loadtxt(src)
    trimmed = data[2:]  # drop 2 lowest-mass (highest-sigma) bins
    os.makedirs('hmf_files', exist_ok=True)
    np.savetxt(trimmed_path, trimmed, fmt='%.18e')
    print(f"Created {trimmed_path}: {len(trimmed)} bins (was {len(data)})", flush=True)

    # Create output directory
    outdir = f'hmf_trimmed_{hmf_sim}_data'
    os.makedirs(outdir, exist_ok=True)

comm.Barrier()

# Import the patched ESR pipeline functions from fit_all_neater
# We need to import carefully to avoid running module-level code
import importlib.util
spec = importlib.util.spec_from_file_location("fit_all_neater",
    "/users/hdesmond/Amelia_code/fit_all_neater.py")

# Actually, the simplest approach: use the ESR library directly
from esr.fitting.likelihood import PoissonLikelihood
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL

# Create likelihood using trimmed data
# Use per-sim run_name to avoid output directory conflicts
run_name = f'poisson_trimmed_sim{hmf_sim}'
likelihood = PoissonLikelihood(trimmed_path, run_name,
                               data_dir='.', fn_set='base_e_maths')

# Use Amelia's function library directly, but create a per-job writable overlay.
# Each job (sim+comp) gets its own isolated dir to avoid race conditions.
amelia_fn_dir = '/mnt/zfsusers/ameliaford/original_ESR/ESR/esr/function_library/base_e_maths'
local_fn_dir = os.path.join(os.getcwd(), f'function_library_sim{hmf_sim}_comp{comp}', 'base_e_maths')

if rank == 0:
    compl_dir = os.path.join(local_fn_dir, f'compl_{comp}')
    os.makedirs(compl_dir, exist_ok=True)

    # Symlink all OTHER complexity dirs to Amelia's (read-only, no conflicts)
    for c in range(1, 12):
        src_dir = os.path.join(amelia_fn_dir, f'compl_{c}')
        dst_dir = os.path.join(local_fn_dir, f'compl_{c}')
        if c != comp and os.path.exists(src_dir) and not os.path.exists(dst_dir):
            os.symlink(src_dir, dst_dir)

    # Copy files for our complexity (use copy, not copy2, to get writable perms)
    src_dir = os.path.join(amelia_fn_dir, f'compl_{comp}')
    for fname in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fname)
        dst_file = os.path.join(compl_dir, fname)
        if os.path.isfile(src_file) and not os.path.exists(dst_file):
            shutil.copy(src_file, dst_file)

    # Force-(re)create the previous_eqns file writable (ESR writes to this)
    prev_eqns = os.path.join(compl_dir, f'previous_eqns_{comp}.txt')
    open(prev_eqns, 'w').close()

comm.Barrier()
likelihood.fn_dir = local_fn_dir

if rank == 0:
    print(f"Running ESR for sim {hmf_sim}, complexity {comp}", flush=True)
    print(f"  Data: {trimmed_path}", flush=True)
    print(f"  fn_dir: {likelihood.fn_dir}", flush=True)
    print(f"  out_dir: {likelihood.out_dir}", flush=True)
    print(f"  temp_dir: {likelihood.temp_dir}", flush=True)

# Run the ESR pipeline stages
try:
    if rank == 0:
        print(f"Stage 1: test_all (fitting all functions)...", flush=True)
    esr.fitting.test_all.main(comp, likelihood, ignore_previous_eqns=True)
    comm.Barrier()

    if rank == 0:
        print(f"Stage 2: test_all_Fisher (computing Hessians)...", flush=True)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    comm.Barrier()

    if rank == 0:
        print(f"Stage 3: match (matching functions)...", flush=True)
    esr.fitting.match.main(comp, likelihood)
    comm.Barrier()

    if rank == 0:
        print(f"Stage 4: combine_DL (ranking)...", flush=True)
    esr.fitting.combine_DL.main(comp, likelihood)
    comm.Barrier()

    # Save output
    if rank == 0:
        src_final = f'{likelihood.out_dir}/final_{comp}.dat'
        dst_final = f'hmf_trimmed_{hmf_sim}_data/final_{comp}_trimmed.dat'
        if os.path.exists(src_final):
            shutil.copy2(src_final, dst_final)
            n_lines = sum(1 for _ in open(dst_final))
            print(f"Saved {dst_final} ({n_lines} functions)", flush=True)
        else:
            print(f"WARNING: {src_final} not found!", flush=True)
            print(f"  out_dir contents: {os.listdir(likelihood.out_dir) if os.path.exists(likelihood.out_dir) else 'DIR NOT FOUND'}", flush=True)

except Exception as e:
    print(f"Rank {rank}: FAILED at sim {hmf_sim}, comp {comp}: {e}", flush=True)
    import traceback
    traceback.print_exc()

comm.Barrier()
if rank == 0:
    print(f"Done: sim {hmf_sim}, comp {comp}", flush=True)
