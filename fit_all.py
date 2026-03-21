"""Central fitting script for running ESR and literature-function fits.

This script performs two main tasks:
  (a) Run the full ESR pipeline (function generation, fitting, Fisher information,
      parameter matching, and description-length calculation) on LF, SMF, or HMF data.
  (b) Fit literature functions (Schechter, Bernardi, Press-Schechter, Warren, Tinker)
      to the same datasets for comparison.

Usage:
    # ESR fits at a given complexity (requires MPI):
    mpirun -np <N> python3 fit_all.py <dataset> esr <complexity>

    # Literature function fits (single process):
    python3 fit_all.py <dataset> paper

Arguments:
    dataset     For LF/SMF: one of LF_Ser_L, LF_cmodel_L, SMF_Ser_M, SMF_cmodel_M.
                For HMF: the Quijote simulation index (e.g. 0, 10, 50).
    complexity  Integer complexity for ESR (typically 3-10).

Inputs:
    - LF/SMF data: LF_and_SMF_data/<dataset>.dat  (4 columns: x, log10(phi), sigma, Veff)
    - HMF data:    hmf_files/hmf_<sim>.dat     (5 columns: logM, counts, ...)
    - ESR function libraries: equation_<basis>_<comp>.txt files

Outputs:
    - ESR results:  <dataset>_data/final_<comp>_new.dat (semicolon-delimited)
    - Paper fits:   <dataset>_data/all_paper_fitting_data.txt

Dependencies:
    numpy, sympy, mpi4py, psutil, prettytable, numdifftools, scipy,
    esr (Exhaustive Symbolic Regression: https://github.com/DeaglanBartlett/ESR)
"""

import sys
hmf_sim = str(sys.argv[1])

runname = 'base_e_maths'

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
import numpy as np
import os
import gc
import sympy
from esr.fitting.likelihood import PoissonLikelihood
from esr.fitting.fit_single import fit_from_string, single_function

from mpi4py import MPI
comm = MPI.COMM_WORLD
global rank
rank = comm.Get_rank()
size = comm.Get_size()

from prettytable import PrettyTable


# --- Memory cleanup utility --- #
def flush_memory(label=""):
    """Force garbage collection and clear sympy caches to reclaim memory between ESR phases."""
    gc.collect()
    try:
        sympy.core.cache.clear_cache()
    except Exception:
        pass
    if rank == 0 and label:
        try:
            import psutil
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            print(f"[Memory] After {label}: {mem_mb:.0f} MB", flush=True)
        except ImportError:
            pass


# --- Memory Watchdog code --- #
import threading
import time
import psutil

# ----------------------------
# Memory Watchdog Definition
# ----------------------------
class MemoryWatchdog(threading.Thread):
    def __init__(self, interval=0.5, threshold_mb=None, verbose=True, constant_update=False):
        super().__init__(daemon=True)
        self.interval = interval
        self.threshold_mb = threshold_mb
        self.verbose = verbose
        self.max_memory = 0
        self.running = True
        self.start_time = None
        self.constant_update = constant_update

    def get_memory_mb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    def run(self):
        self.start_time = time.time()
        if rank == 1:
            print(f"Sleep time: {self.interval}s")
        while self.running:
            elapsed = time.time() - self.start_time
            mem_mb = self.get_memory_mb()

            if self.constant_update == True:
                self.checkpoint()

            if mem_mb > self.max_memory:
                self.max_memory = mem_mb
                if self.verbose and rank == 1:
                    print(f"[Watchdog] NEW PEAK MEMORY: {mem_mb:.2f} MB")# | time elapsed: {elapsed:.2f}s")
                if self.threshold_mb and mem_mb > self.threshold_mb and rank == 1:
                    print(f"[Watchdog]   WARNING: Exceeded {self.threshold_mb} MB")

            time.sleep(self.interval)
    
    def checkpoint(self):
        elapsed = time.time() - self.start_time
        mem_mb = self.get_memory_mb()
        if self.verbose and rank == 1:
            print(f"[Watchdog] {elapsed:.1f} seconds | {mem_mb:.2f} MB")

    def stop(self):
        self.running = False

## To use the above code:
#    watchdog = MemoryWatchdog(interval=0.5)
#    watchdog.start()
#
#    ... code to be monitored ...
#
#    watchdog.stop() 


# -------------------------------------------------------------------
# Monkey-patches for ESR library functions to reduce peak memory usage
# -------------------------------------------------------------------

def _patched_test_all_main(comp, likelihood, tmax=5, pmin=0, pmax=3,
                           try_integration=False, log_opt=False,
                           Niter_params=[40, 60], Nconv_params=[-5, 20],
                           gc_interval=50, **kwargs):
    """Drop-in replacement for esr.fitting.test_all.main with periodic GC.

    Identical logic but calls gc.collect() + sympy cache clear every
    gc_interval equations so that transient sympy/lambdify objects are
    reclaimed promptly instead of accumulating until the loop finishes.
    """
    import esr.generation.simplifier as simplifier

    fcn_list_proc, _, _ = esr.fitting.test_all.get_functions(comp, likelihood)

    max_param = int(max(4, np.floor((comp - 1) / 2)))

    chi2 = np.zeros(len(fcn_list_proc), dtype=np.float32)
    params = np.zeros([len(fcn_list_proc), max_param], dtype=np.float32)

    for i in range(len(fcn_list_proc)):
        if rank == 0:
            print(rank, i, len(fcn_list_proc), flush=True)
        try:
            with simplifier.time_limit(tmax):
                try:
                    chi2[i], params[i, :] = esr.fitting.test_all.optimise_fun(
                        fcn_list_proc[i], likelihood, tmax, pmin, pmax,
                        try_integration=try_integration, log_opt=log_opt,
                        max_param=max_param, Niter_params=Niter_params,
                        Nconv_params=Nconv_params)
                except NameError:
                    if try_integration:
                        chi2[i], params[i, :] = esr.fitting.test_all.optimise_fun(
                            fcn_list_proc[i], likelihood, tmax, pmin, pmax,
                            try_integration=False, log_opt=log_opt,
                            max_param=max_param, Niter_params=Niter_params,
                            Nconv_params=Nconv_params)
                    else:
                        raise NameError
        except:
            chi2[i] = np.nan
            params[i, :] = 0.

        # Periodic garbage collection to prevent sympy/lambdify memory build-up
        if (i + 1) % gc_interval == 0:
            gc.collect()
            try:
                sympy.core.cache.clear_cache()
            except Exception:
                pass

    out_arr = np.transpose(np.vstack([chi2] + [params[:, j] for j in range(max_param)]))
    np.savetxt(likelihood.temp_dir + '/chi2_comp' + str(comp) + 'weights_' + str(rank) + '.dat', out_arr, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "chi2_comp' + str(comp) + 'weights_*.dat" | sort -V` > ' + likelihood.out_dir + '/negloglike_comp' + str(comp) + '.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/chi2_comp' + str(comp) + 'weights_*.dat'
        os.system(string)

    comm.Barrier()


def _patched_test_all_Fisher_main(comp, likelihood, tmax=5,
                                  try_integration=False, gc_interval=10):
    """Drop-in replacement for esr.fitting.test_all_Fisher.main with periodic GC.

    The Fisher/Hessian step is the heaviest per-equation: numdifftools
    creates large closure objects and the fallback loop can try up to
    48 Hessian evaluators.  We gc.collect() very frequently here (every
    gc_interval equations, default 10) to prevent memory accumulation.
    """
    import math
    import itertools
    import numdifftools as nd
    import esr.generation.simplifier as simplifier
    from esr.fitting.sympy_symbols import x, a0

    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')

    if comp >= 8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    fcn_list_proc, data_start, data_end = esr.fitting.test_all.get_functions(comp, likelihood)
    negloglike, params_proc = esr.fitting.test_all_Fisher.load_loglike(comp, likelihood, data_start, data_end)
    negloglike = negloglike.astype(np.float32)
    params_proc = params_proc.astype(np.float32)
    max_param = params_proc.shape[1]

    codelen = np.zeros(len(fcn_list_proc), dtype=np.float32)
    params = np.zeros([len(fcn_list_proc), max_param], dtype=np.float32)
    deriv = np.zeros([len(fcn_list_proc), int(max_param * (max_param + 1) / 2)], dtype=np.float32)

    for i in range(len(fcn_list_proc)):
        if rank == 0:
            print(i, len(fcn_list_proc), flush=True)

        if np.isnan(negloglike[i]) or np.isinf(negloglike[i]):
            codelen[i] = np.nan
            continue

        theta_ML = params_proc[i, :]

        try:
            fcn_i = fcn_list_proc[i].replace('\n', '')
            fcn_i = fcn_list_proc[i].replace('\'', '')
            fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
            params[i, :], negloglike[i], deriv[i, :], codelen[i] = \
                esr.fitting.test_all_Fisher.convert_params(
                    fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param)
        except NameError:
            if try_integration:
                fcn_i = fcn_list_proc[i].replace('\n', '')
                fcn_i = fcn_list_proc[i].replace('\'', '')
                fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                params[i, :], negloglike[i], deriv[i, :], codelen[i] = \
                    esr.fitting.test_all_Fisher.convert_params(
                        fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param)
            else:
                params[i, :] = 0.
                deriv[i, :] = 0.
                codelen[i] = 0
        except:
            params[i, :] = 0.
            deriv[i, :] = 0.
            codelen[i] = 0

        # Aggressive GC every gc_interval equations — numdifftools Hessian
        # objects and sympy closures are the biggest per-equation memory users.
        # At comp 10, the fallback loop in convert_params creates up to 48
        # Hessian evaluators per problematic equation.
        if (i + 1) % gc_interval == 0:
            gc.collect()
            try:
                sympy.core.cache.clear_cache()
            except Exception:
                pass

    out_arr = np.transpose(np.vstack([codelen, negloglike] + [params[:, j] for j in range(max_param)]))
    out_arr_deriv = np.transpose(np.vstack([deriv[:, j] for j in range(deriv.shape[1])]))

    np.savetxt(likelihood.temp_dir + '/codelen_deriv_' + str(comp) + '_' + str(rank) + '.dat', out_arr, fmt='%.7e')
    np.savetxt(likelihood.temp_dir + '/derivs_' + str(comp) + '_' + str(rank) + '.dat', out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        for prefix, infix, out in [
            ('codelen_deriv_', 'codelen_deriv_', 'codelen_comp%d_deriv.dat' % comp),
            ('derivs_', 'derivs_', 'derivs_comp%d.dat' % comp),
        ]:
            string = 'cat `find ' + likelihood.temp_dir + '/ -name "' + prefix + str(comp) + '_*.dat" | sort -V` > ' + likelihood.out_dir + '/' + out
            os.system(string)
            string = 'rm ' + likelihood.temp_dir + '/' + prefix + str(comp) + '_*.dat'
            os.system(string)

    comm.Barrier()


def _load_subs_rank_slice(fname, max_param, data_start, data_end):
    """Memory-efficient replacement for simplifier.load_subs.

    The original load_subs broadcasts the ENTIRE inv_subs list (with
    sympy dict objects for every equation) to every MPI rank.  At
    complexity 10 with base_e_maths this can be millions of entries,
    each containing sympy objects — easily >10 GB per rank.

    This replacement has rank 0 read the raw file, send each rank only
    the raw string rows for its [data_start:data_end] slice, then each
    rank converts strings→sympy dicts locally.  The full list never
    exists in sympy form on any rank.
    """
    import csv
    import ast
    import esr.generation.simplifier as simplifier

    # Rank 0 reads the raw CSV rows and distributes slices as strings
    if rank == 0:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            all_rows = [r for r in reader]
        n_total = len(all_rows)
    else:
        all_rows = None
        n_total = None
    n_total = comm.bcast(n_total, root=0)

    # Each rank tells rank 0 what slice it needs (data_start:data_end)
    if rank == 0:
        # Collect all ranks' slice requests
        slices = [(data_start, data_end)]  # rank 0's own
        for r in range(1, comm.Get_size()):
            slices.append(comm.recv(source=r, tag=78))
        # Send raw string rows to each rank
        my_raw_rows = all_rows[slices[0][0]:slices[0][1]]
        for r in range(1, comm.Get_size()):
            comm.send(all_rows[slices[r][0]:slices[r][1]], dest=r, tag=79)
        del all_rows, slices
        gc.collect()
    else:
        comm.send((data_start, data_end), dest=0, tag=78)
        my_raw_rows = comm.recv(source=0, tag=79)

    # Now each rank converts its raw string rows → sympy dicts locally
    param_list = ['a%i' % i for i in range(max_param)]
    all_a = sympy.symbols(" ".join(param_list), real=True)
    if max_param == 1:
        all_a = [all_a]
    locs = dict(simplifier.sympy_locs)
    if max_param > 0:
        for i in range(len(all_a)):
            locs["a%i" % i] = all_a[i]

    for i in range(len(my_raw_rows)):
        if len(my_raw_rows[i]) == 0:
            continue
        for j in range(len(my_raw_rows[i])):
            s = my_raw_rows[i][j]
            s = s.replace("{", "{'")
            s = s.replace("}", "'}")
            s = s.replace(", ", "', '")
            s = s.replace(": ", "': '")
            if s == 'nan':
                my_raw_rows[i][j] = np.nan
            else:
                d = ast.literal_eval(s)
                k = list(d.keys())
                v = list(d.values())
                k = [sympy.sympify(kk, locals=locs) for kk in k]
                v = [sympy.sympify(vv, locals=locs) for vv in v]
                my_raw_rows[i][j] = dict(zip(k, v))

    comm.Barrier()
    return my_raw_rows


def _patched_match_main(comp, likelihood, tmax=5, try_integration=False, gc_interval=100):
    """Drop-in replacement for esr.fitting.match.main with memory optimisations.

    Key changes vs the original:
    - Uses _load_subs_rank_slice to avoid broadcasting the FULL inv_subs
      list (with millions of sympy dicts) to every rank.
    - More aggressive GC (every 100 equations instead of 500).
    - Clears sympy cache periodically.
    - Explicit deletion of large arrays after the loop.
    """
    import math
    import esr.generation.simplifier as simplifier
    from esr.fitting.sympy_symbols import x, a0

    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')

    def fop(x_val):
        return likelihood.negloglike(x_val, eq_numpy, integrated=integrated)

    def f1(x_val):
        return likelihood.negloglike([x_val], eq_numpy, integrated=integrated)

    invsubs_file = likelihood.fn_dir + "/compl_%i/inv_subs_%i.txt" % (comp, comp)
    match_file = likelihood.fn_dir + "/compl_%i/matches_%i.txt" % (comp, comp)

    fcn_list_proc, data_start, data_end = esr.fitting.test_all.get_functions(comp, likelihood, unique=False)
    negloglike, params_meas = esr.fitting.test_all_Fisher.load_loglike(comp, likelihood, data_start, data_end, split=False)
    negloglike = negloglike.astype(np.float32)
    params_meas = params_meas.astype(np.float32)
    max_param = params_meas.shape[1]

    # Memory-efficient inv_subs loading — only keep this rank's slice
    all_inv_subs_proc = _load_subs_rank_slice(invsubs_file, max_param, data_start, data_end)
    flush_memory("load_subs")

    matches_proc = np.atleast_1d(np.loadtxt(match_file).astype(int))[data_start:data_end]

    # Load Fisher derivatives — kept as float64 because 12./fish overflows float32
    derivs_path = likelihood.out_dir + '/derivs_comp' + str(comp) + '.dat'
    try:
        all_fish = np.loadtxt(derivs_path)
        all_fish = np.atleast_2d(all_fish)
    except Exception:
        all_fish = np.atleast_2d(np.loadtxt(derivs_path))

    codelen = np.zeros(len(fcn_list_proc), dtype=np.float32)
    negloglike_all = np.zeros(len(fcn_list_proc), dtype=np.float32)
    index_arr = np.zeros(len(fcn_list_proc))
    params = np.zeros([len(fcn_list_proc), max_param], dtype=np.float32)

    for i in range(len(fcn_list_proc)):
        if i % 1000 == 0 and rank == 0:
            print(i, len(fcn_list_proc))

        fcn_i = fcn_list_proc[i].replace('\'', '')

        nparams = simplifier.count_params([fcn_i], max_param)[0]

        index = matches_proc[i]
        index_arr[i] = index
        negloglike_all[i] = negloglike[index]

        if np.isnan(negloglike[index]) or np.isinf(negloglike[index]):
            codelen[i] = np.nan
            continue

        if nparams == 0:
            continue
        else:
            k = nparams
            measured = params_meas[index, :nparams]

        fish_measured = all_fish[index, :]

        try:
            p, fish = simplifier.convert_params(measured, fish_measured, all_inv_subs_proc[i], n=max_param)
        except Exception:
            codelen[i] = np.inf
            continue

        if np.sum(fish <= 0) > 0:
            codelen[i] = np.inf
            continue

        Delta = np.zeros(fish.shape)
        m = (fish != 0)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Delta[m] = np.atleast_1d(np.sqrt(12. / fish[m]))
        Delta[~m] = np.inf
        Nsteps = np.atleast_1d(np.abs(np.array(p)))
        m = (Delta != 0) & np.isfinite(Delta)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            Nsteps[m] /= Delta[m]
        Nsteps[~m] = np.nan

        if np.sum(Nsteps < 1) > 0:

            try:
                p[Nsteps < 1] = 0.
            except (IndexError, TypeError):
                p = 0.

            try:
                fcn_i_s, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
                if k == 1:
                    eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                    negloglike_all[i] = f1(p)
                else:
                    nparam = nparams
                    all_a = ' '.join([f'a{j}' for j in range(nparam)])
                    all_a = list(sympy.symbols(all_a, real=True))
                    eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
                    negloglike_all[i] = fop(p)
            except NameError:
                if try_integration:
                    fcn_i_s, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                    if k == 1:
                        eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                        negloglike_all[i] = f1(p)
                    else:
                        nparam = nparams
                        all_a = ' '.join([f'a{j}' for j in range(nparam)])
                        all_a = list(sympy.symbols(all_a, real=True))
                        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
                        negloglike_all[i] = fop(p)
                else:
                    negloglike_all[i] = np.nan
            except:
                negloglike_all[i] = np.nan

            k -= np.sum(Nsteps < 1)

            if k < 0:
                print("This shouldn't have happened", flush=True)
                quit()
            elif k == 0:
                continue

            fish = fish[Nsteps >= 1]
            p = p[Nsteps >= 1]

        try:
            list_p = list(p)
            padded = np.pad(p, (0, max_param - len(p)))
            params[i, :] = np.clip(padded, -3.4e38, 3.4e38).astype(np.float32)
        except:
            p_arr = np.atleast_1d(np.asarray(p))
            if p_arr.size > 0 and np.any(p_arr != 0):
                params[i, :] = 0
                params[i, 0] = np.clip(p_arr.flat[0], -3.4e38, 3.4e38)
            else:
                params[i, :] = np.zeros(max_param, dtype=np.float32)

        assert len(params[i, :]) == max_param

        try:
            codelen[i] = -k / 2. * math.log(3.) + np.sum(0.5 * np.log(fish) + np.log(abs(np.array(p))))
        except:
            codelen[i] = np.nan

        # Periodic GC — more frequent at comp 10 scale
        if (i + 1) % gc_interval == 0:
            gc.collect()
            try:
                sympy.core.cache.clear_cache()
            except Exception:
                pass

    # Free the large arrays we no longer need before writing output
    del all_fish, negloglike, params_meas, all_inv_subs_proc, matches_proc
    gc.collect()
    try:
        sympy.core.cache.clear_cache()
    except Exception:
        pass

    out_arr = np.transpose(np.vstack([negloglike_all, codelen, index_arr] + [params[:, j] for j in range(max_param)]))
    np.savetxt(likelihood.temp_dir + '/codelen_matches_' + str(comp) + '_' + str(rank) + '.dat', out_arr, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_matches_' + str(comp) + '_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_matches_comp' + str(comp) + '.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_matches_' + str(comp) + '_*.dat'
        os.system(string)

    comm.Barrier()


# --- Runs fits on hmf data --- #
def run_hmf_esr_fits(hmf_sim_val, comp):

    watchdog = MemoryWatchdog(interval=0.1)
    watchdog.start()

    likelihood = PoissonLikelihood('hmf_files/hmf_{}.dat'.format(hmf_sim_val), 'poisson_example', data_dir='/mnt/zfsusers/ameliaford/original_ESR/ESR', fn_set='base_e_maths')
#    esr.generation.duplicate_checker.main(runname, comp)
    _patched_test_all_main(comp, likelihood, ignore_previous_eqns=True)
    flush_memory("test_all")
    _patched_test_all_Fisher_main(comp, likelihood)
    flush_memory("test_all_Fisher")
    _patched_match_main(comp, likelihood)
    flush_memory("match")
    esr.fitting.combine_DL.main(comp, likelihood)
    flush_memory("combine_DL")
    #esr.fitting.plot.main(comp, likelihood)

    # --- Save data to hmf_?_data/final_?.dat ---
    with open('fitting/output/output_poisson_example/final_{}.dat'.format(comp), 'r') as reader:
        with open("hmf_{0}_data/final_{1}_new.dat".format(hmf_sim_val,comp), 'w') as writer:
            for line in reader:
                writer.write(line)

    watchdog.stop()
        
# --- Run fits on SMF or LF data --- #
def run_galaxy_esr_fits(sim_name, comp, data_dir='/mnt/zfsusers/ameliaford/original_ESR/ESR'):

    watchdog = MemoryWatchdog(interval=0.1, constant_update=False)
    watchdog.start()

    likelihood = PoissonLikelihood('LF_and_SMF_data/{}.dat'.format(sim_name), 'poisson_example', data_dir=data_dir, fn_set='base_e_maths')
#    esr.generation.duplicate_checker.main(runname, comp)
    # Redirect fn_dir to a local writable copy so ESR can write previous_eqns files
    # while still reading the equation lists from the original (read-only) location
    amelia_fn_dir = '/mnt/zfsusers/ameliaford/original_ESR/ESR/esr/function_library/base_e_maths'
    local_fn_dir = os.path.join(os.getcwd(), 'function_library_local', 'base_e_maths')
    if rank == 0:
        compl_dir = os.path.join(local_fn_dir, 'compl_%i' % comp)
        os.makedirs(compl_dir, exist_ok=True)
        # Symlink the equation files into our local dir so ESR can find them
        # Skip previous_eqns files — ESR will recreate them and needs write access
        for fname in os.listdir(os.path.join(amelia_fn_dir, 'compl_%i' % comp)):
            if fname.startswith('previous_eqns'):
                continue
            src = os.path.join(amelia_fn_dir, 'compl_%i' % comp, fname)
            dst = os.path.join(compl_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        # Remove any stale previous_eqns symlink from a prior run
        prev_eqns = os.path.join(compl_dir, 'previous_eqns_%i.txt' % comp)
        if os.path.islink(prev_eqns):
            os.remove(prev_eqns)
        # Also symlink lower-complexity dirs (needed for ignore_previous_eqns)
        for c in range(1, comp):
            src_dir = os.path.join(amelia_fn_dir, 'compl_%i' % c)
            dst_dir = os.path.join(local_fn_dir, 'compl_%i' % c)
            if os.path.isdir(src_dir) and not os.path.exists(dst_dir):
                os.symlink(src_dir, dst_dir)
    comm.Barrier()
    likelihood.fn_dir = local_fn_dir
    watchdog.checkpoint()
    _patched_test_all_main(comp, likelihood, ignore_previous_eqns=True)
    flush_memory("test_all")
    watchdog.checkpoint()
    _patched_test_all_Fisher_main(comp, likelihood)
    flush_memory("test_all_Fisher")
    watchdog.checkpoint()
    _patched_match_main(comp, likelihood)
    flush_memory("match")
    watchdog.checkpoint()
    esr.fitting.combine_DL.main(comp, likelihood)
    flush_memory("combine_DL")
    watchdog.checkpoint()
##    esr.fitting.plot.main(comp, likelihood)

    # --- Save data to 'LF' or 'SMF' _?_data/final_?.dat ---
    with open('fitting/output/output_poisson_example/final_{}.dat'.format(comp), 'r') as reader:
        with open("{0}_data/final_{1}.dat".format(sim_name,comp), 'w') as writer:
            for line in reader:
                writer.write(line)
   
    watchdog.checkpoint()
    watchdog.stop()

    
# --- Define Paper equations --- #
def def_pap_eq():
    # HMF Schechter
    eq_1_str = "(1.345233369513631/x)*exp(-2.842596/(2*pow(x,2)))"
    eq_1_tree = ['*', '1.345233369513631', '*', 'inv', 'x', 'exp', '*', '-2.842596', 'pow', 'x', '-2']

    # Warren
    eq_2_str = "a0*(pow(x,a2)+a1)*exp(-a3*pow(x,-2))"
    eq_2_tree = ['*', 'a0', '*', '+', 'a1', 'pow', 'x', 'a2', 'exp', '*', '-1', '*', 'a3', 'pow', 'x', '-2']

    # Tinker
    eq_3_str = "a0*(pow(x/a2,-a1)+1)*exp(-a3*pow(x,-2))"
    eq_3_tree = ['*', 'a0', '*', '+', '1', 'pow', '/', 'x', 'a2', '*', '-1', 'a1', 'exp', '*', '-1', '*', 'a3', 'pow', 'x', '-2']

    # Bernardi
    eq_4_str = "a0*pow(x,a1)*exp(-a2*pow(x,a3))-a4*pow(x,a5)*exp(-a6*x)"
    
    return eq_1_str, eq_1_tree, eq_2_str, eq_2_tree,eq_3_str, eq_3_tree, eq_4_str


# --- Fit individual equations --- #
def fit_str(string,basis_functions,hmf_sim,data_dir='/mnt/zfsusers/ameliaford/original_ESR/ESR'):
    nparam = 0
    for i in range(0,9):
        if 'a{}'.format(i) in string:
            nparam += 1
            continue

    Niter_params=[3040,3060]
    Nconv_params=[-5,20]

    if nparam != 0:
        Niter_new = int(np.sum(nparam ** np.arange(len(Niter_params)) * np.array(Niter_params)))
        Nconv_new = int(np.sum(nparam ** np.arange(len(Nconv_params)) * np.array(Nconv_params)))
    else:
        Niter_new = 30
        Nconv_new = 5

    if (Nconv_new <= 0) or (Niter_new <= 0) or (Nconv_new > Niter_new):
        raise ValueError("Nconv and/or Niter have unacceptable values")

    if '_' in hmf_sim:
        likelihood = PoissonLikelihood('LF_and_SMF_data/{}.dat'.format(hmf_sim), 'poisson_example', data_dir=data_dir, fn_set='base_e_maths')

    else:
        likelihood = PoissonLikelihood('hmf_files/hmf_{}.dat'.format(hmf_sim), 'poisson_example', data_dir=data_dir, fn_set='base_e_maths')
    
    count = 0
    done = False
    while count < 20 and done == False:

        count += 1
        logl_lcdm, dl_lcdm, tree, params = fit_from_string(string,
                                                basis_functions,
                                                likelihood,
                                                verbose=False,
                                                log_opt=True,
                                                Niter=Niter_new,
                                                Nconv=Nconv_new,
                                                return_params=True)

        #NOTE: The following actually needs to be given the tree
        #logl_lcdm, dl_lcdm, params = single_function(string, basis_functions, likelihood, Niter=Niter_new, Nconv=Nconv_new, return_params=True)
        
        try:
            a = float(logl_lcdm)
            b = float(dl_lcdm)

            if np.isnan(logl_lcdm) == False and np.isnan(dl_lcdm) == False:
                done = True

            if rank == 1:
                fcn_str = string
                for idx in range(0, len(params)):
                    fcn_str = fcn_str.replace('a{}'.format(idx), str(params[idx]))
                print('{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, string))
                print()
        except:
            pass


    return logl_lcdm, dl_lcdm, params



# --- Fits paper functions ---
def fitting_paper(hmf_sim, data_dir='/mnt/zfsusers/ameliaford/original_ESR/ESR'):
    eq_1_str, eq_1_tree, eq_2_str, eq_2_tree,eq_3_str, eq_3_tree, eq_4_str = def_pap_eq()
    
    # ----- #
    basis_functions = [["x", "a"],  # type0
                ["inv","exp","log","abs", "gamma"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2

    all_paper_data = []

    if 'SMF' not in hmf_sim and 'LF' not in hmf_sim:
        if rank == 1:
            print("HMF")
            print("\nFITTING SCHECHTER")
        logl_lcdm, dl_lcdm, params = fit_str('pow(2/3.141592,0.5)*(1.686/x)*exp(-0.5*pow(1.686/x,2))',basis_functions,hmf_sim,data_dir=data_dir)
        all_paper_data.append('P.Sch.;10;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, 'pow(2/3.141592,0.5)*(1.686/x)*exp(-0.5*pow(1.686/x,2))', 'pow(2/3.141592,0.5)*(\delta_c/x)*exp(-0.5*pow(\delta_c/x,2))'))
        if rank == 1:
            print('P.Sch.;10;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, 'pow(2/3.141592,0.5)*(1.686/x)*exp(-0.5*pow(1.686/x,2))', 'pow(2/3.141592,0.5)*(\delta_c/x)*exp(-0.5*pow(\delta_c/x,2))'))


        # If we allow the 1.686 to become a free parameter...
        logl_lcdm, dl_lcdm, params = fit_str('pow(2/3.141592,0.5)*(a0/x)*exp(-0.5*pow(a0/x,2))',basis_functions,hmf_sim,data_dir=data_dir)
        a = params[0]
        all_paper_data.append('P.Sch.;10;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, 'pow(2/3.141592,0.5)*({0}/x)*exp(-0.5*pow({0}/x,2))'.format(a), 'pow(2/3.141592,0.5)*(a0/x)*exp(-0.5*pow(a0/x,2))'))
        if rank == 1:
            print('P.Sch.;10;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, 'pow(2/3.141592,0.5)*({0}/x)*exp(-0.5*pow({0}/x,2))'.format(a), 'pow(2/3.141592,0.5)*(a0/x)*exp(-0.5*pow(a0/x,2))'))
            print('Should be 1.686, but when fitted as a free parameter is: {}'.format(a))


#        a = params[1]
#        L = -np.log10((params[2]*10**(-10))/np.log10(np.exp(1)))
#    #    L = 1/(params[2]*10**(-10))/np.log10(np.exp(1)) # for x = M not logM
#        n = 10**params[0] * 10**(L*a)
#        
#        fcn = 'n*pow((pow(10,x)/pow(10,L)),a)*exp(-pow(10,x)/pow(10,L))'
#        actual_fcn = fcn.replace('n',str(n)).replace('L',str(L)).replace('a',str(a))
#        all_paper_data.append(format_line(['50', 'Sch.', 10, actual_fcn, logl_lcdm, dl_lcdm, params]))

        if rank == 1:
            print("\nFITTING WARREN")
        logl_lcdm, dl_lcdm, params = fit_str(eq_2_str,basis_functions,hmf_sim,data_dir=data_dir)
        fcn_str = eq_2_str
        for idx in range(0, len(params)):
            fcn_str = fcn_str.replace('a{}'.format(idx), str(params[idx]))
        all_paper_data.append('War.;16;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_2_str))
        if rank==1:
            print('War.;16;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_2_str))

        if rank == 1:
            print("\nFITTING TINKER")
        logl_lcdm, dl_lcdm, params = fit_str(eq_3_str,basis_functions,hmf_sim,data_dir=data_dir)
        fcn_str = eq_3_str
        for idx in range(0, len(params)):
            fcn_str = fcn_str.replace('a{}'.format(idx), str(params[idx]))
        all_paper_data.append('Tin.;20;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_3_str))
        if rank==1:
            print('Tin.;20;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_3_str))

        if rank == 1:
            print("\nFITTING BERNARDI")
        logl_lcdm, dl_lcdm, params = fit_str(eq_4_str,basis_functions,hmf_sim,data_dir=data_dir)
        fcn_str = eq_4_str
        for idx in range(0, len(params)):
            fcn_str = fcn_str.replace('a{}'.format(idx), str(params[idx]))
        all_paper_data.append('Ber.;28;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_4_str))
        if rank==1:
            print('Ber.;28;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_4_str))


    else:
        if rank == 1:
            print('LF or SMF')
        

        # phi   X_star alpha beta  phi_g X_g    gamma
        # "phi_alpha*beta*pow(x/X_star,alpha)*exp(-1*pow(x/X_star,beta))/Gamma(alpha/beta)+phi_gamma*pow(x/X_gamma,gamma)*exp(-x/X_gamma)"

        if 'SMF' in hmf_sim:
            if 'Ser' in hmf_sim:
                # phi   X_star alpha beta  phi_g X_g    gamma
                # 1.040 0.0094 1.665 0.255 0.675 2.7031 0.296
                Ber_fcn = 'log(10)*(1.040*0.01*0.255*pow(x/0.0094,1.665)*exp(-1*pow(x/0.0094,0.255))/Gamma(1.665/0.255)+0.675*0.01*pow(x/2.7031,0.296)*exp(-x/2.7031))'
#                Ber_fcn = 'log(10)*(1.040*1e-2*0.255*pow(x/(0.0094*1e9),1.665)*exp(-1*pow(x/(0.0094*1e9),0.255))/Gamma(1.665/0.255)+0.675*1e-2*pow(x/(2.7031*1e9),0.296)*exp(-x/(2.7031*1e9)))' # Bernardi (SMF_Ser)
            else:
                # phi   X_star alpha beta  phi_g X_g    gamma
                # 0.766 0.4103 1.764 0.384 0.557 4.7802 0.053
                Ber_fcn = 'log(10)*(0.766*0.01*0.384*pow(x/0.4103,1.764)*exp(-1*pow(x/0.4103,0.384))/Gamma(1.764/0.384)+0.557*0.01*pow(x/4.7802,0.053)*exp(-x/4.7802))' # Bernardi (SMF_cmodel)

        else:
            if 'cmodel' in hmf_sim:
                #  phi   X_star alpha beta  phi_g X_g    gamma
                #  0.928 0.3077 1.918 0.433 0.964 1.8763 0.470
                Ber_fcn = '0.928*0.01*0.433*pow(x/0.3077,1.918)*exp(-1*pow(x/0.3077,0.433))/Gamma(1.918/0.433)+0.964*0.01*pow(x/1.8763,0.470)*exp(-x/1.8763)'
            else:
                # phi   X_star alpha beta  phi_g X_g    gamma
                # 1.343 0.0187 1.678 0.300 0.843 0.8722 1.058 0.150
                Ber_fcn = '1.343*0.01*0.300*pow(x/0.0187,1.678)*exp(-1*pow(x/0.0187,0.300))/Gamma(1.678/0.300)+0.843*0.01*pow(x/0.8722,1.058)*exp(-x/0.8722)'
                pass


        if rank == 1:
            print("FITTING SCHECHTER")
        fcn = 'phi/L_star*pow(x/L_star,alpha)*exp(-x/L_star)'
        Sch_fcn = fcn.replace('phi','a0').replace('L_star','a1').replace('alpha','a2')
        logl_lcdm, dl_lcdm, params = fit_str(Sch_fcn,basis_functions,hmf_sim,data_dir=data_dir)
        fcn_str = Sch_fcn
        for idx in range(0, len(params)):
            fcn_str = fcn_str.replace('a{}'.format(idx), str(params[idx]))
        all_paper_data.append('Sch.;10;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, Sch_fcn))
        if rank == 1:
            print('Sch.;10;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, Sch_fcn))

        if rank == 1:
            print("FITTING BERNARDI")
        logl_lcdm, dl_lcdm, params = fit_str(Ber_fcn,basis_functions,hmf_sim,data_dir=data_dir)

        if rank == 1:
            print("FITTING BERNARDI PROPERLY...")
        logl_lcdm, dl_lcdm, params = fit_str(eq_4_str,basis_functions,hmf_sim,data_dir=data_dir)
        fcn_str = eq_4_str
        for idx in range(0, len(params)):
            fcn_str = fcn_str.replace('a{}'.format(idx), str(params[idx]))
        all_paper_data.append('Ber.;28;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_4_str))
        if rank==1:
            print('Ber.;28;{0};{1};{2};{3}'.format(dl_lcdm, logl_lcdm, fcn_str, eq_4_str))


    if rank == 1:
        print("DONE\n")
        print(all_paper_data)
    
        for line in all_paper_data:
            print(line)

    return all_paper_data



# --- Saves all data ---
def save_data(data, hmf_sim):
    if '_' in hmf_sim:
        location = '{}_data'.format(hmf_sim)
    else:
        location = 'hmf_{}_data'.format(hmf_sim)
    with open('{}/all_paper_fitting_data.txt'.format(location), 'w') as f:
        for line in data:
            f.write(line)
            f.write('\n')
    f.close()


# --- Runs ESR and individual fits ---
def fit_and_store_paper_fits(hmf_sim):
    data = fitting_paper(hmf_sim)
    if rank==1:
        print(data)
        print('-----------------')
    save_data(data, hmf_sim)


basis_functions = [["x", "a"],  # type0
            ["inv","exp","log","abs"],  # type1
            ["+", "*", "-", "/", "pow"]]  # type2


## To run galaxy fits:
# comp = 6
# run_galaxy_esr_fits(hmf_sim, comp)

## To run HMF fits:
# comp = 4
# run_hmf_esr_fits(hmf_sim, comp)

## To run paper fits:
# fit_and_store_paper_fits(hmf_sim)
# fitting_paper(hmf_sim)

# --- Command-line mode selector ---
# Usage:
#   python3 fit_all_neater.py <dataset> esr <comp>      # ESR fits at given complexity
#   python3 fit_all_neater.py <dataset> paper            # Paper function fits
# If LF_and_SMF_data/<dataset>.dat exists locally (cwd), use cwd as data_dir;
# otherwise fall back to Amelia's directory.
if len(sys.argv) >= 3:
    mode = sys.argv[2]
    local_data = os.path.join(os.getcwd(), 'LF_and_SMF_data', f'{hmf_sim}.dat')
    if os.path.isfile(local_data):
        data_dir = os.getcwd()
    else:
        data_dir = '/mnt/zfsusers/ameliaford/original_ESR/ESR'
    if mode == 'esr':
        comp = int(sys.argv[3])
        if '_' in hmf_sim:
            os.makedirs(f'{hmf_sim}_data', exist_ok=True)
            run_galaxy_esr_fits(hmf_sim, comp, data_dir=data_dir)
        else:
            run_hmf_esr_fits(hmf_sim, comp)
    elif mode == 'paper':
        # Paper mode is intended as a single-process post-processing run.
        # Running it with multiple MPI ranks duplicates work and can skip
        # saving if rank 1 is absent/present unexpectedly.
        if size != 1:
            if rank == 0:
                print("ERROR: paper mode must be run with exactly one MPI rank/core.", flush=True)
                print("Use: python3 fit_all_neater.py <dataset> paper", flush=True)
            sys.exit(2)
        if '_' in hmf_sim:
            os.makedirs(f'{hmf_sim}_data', exist_ok=True)
        data = fitting_paper(hmf_sim, data_dir=data_dir)
        if rank == 0:
            save_data(data, hmf_sim)

