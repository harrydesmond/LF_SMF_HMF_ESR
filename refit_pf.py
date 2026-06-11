#!/usr/bin/env python3
"""Thorough multi-start re-fit of the top-N ESR HMF functions across all 100 trimmed sims.

Purpose: test whether the paper's combined-DL ranking (hmf_combined_DL_fiducial_new.txt)
is stable under MUCH more thorough optimisation than the original ESR run, which used only
~220 random restarts drawn from the narrow range [0,3] per (function, sim). Here every
function gets, on every sim: (a) cross-sim donor warm-starts (paper params from every sim
where it was fit, plus a propagation pass using this run's own best params), and (b) a wide
random multi-start. This is strictly more thorough and removes the per-sim optimisation
fragility quantified in the phase-1/phase-2 investigation.

Likelihood (poisson_nll) and description-length pieces (compute_codelen) are byte-for-byte
the same as ESR's PoissonLikelihood.negloglike and test_all_Fisher.convert_params (verified),
and the combined metric matches compute_combined_DL.py:
    DL_i        = NLL_i + codelen_i + aifeyn          (per sim)
    DL_combined = sum_i DL_i - (n_sims - 1) * aifeyn
aifeyn is structural (optimisation-independent) and read from aifeynman_cache.json.

Submit with -s (MULTIPROCESSING, not MPI):
    addqueue -q berg -n 1xNW -m 4 -s -c "thorough refit top100" \
        /usr/local/shared/python/3.11.4/bin/python3 thorough_refit.py
"""

import os
import sys
import re
import json
import math
import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
from multiprocessing import Pool, cpu_count

os.chdir('/users/hdesmond/Amelia_code') if os.path.isdir('/users/hdesmond/Amelia_code') else None

# ── Guard: this MUST run as multiprocessing, not MPI. If launched without -s,
#    addqueue starts it as N MPI ranks; detect and abort loudly (avoids silent N-fold redundancy).
try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("FATAL: launched under MPI (size>1). Submit with -s for multiprocessing. Aborting.", flush=True)
        sys.exit(1)
except ImportError:
    pass

# ── Config (env overrides for local testing) ──
TOP_N      = int(os.environ.get('REFIT_TOP_N', 100000))
N_SIMS     = int(os.environ.get('REFIT_N_SIMS', 100))
N_RANDOM   = int(os.environ.get('REFIT_N_RANDOM', 30))
NPROCS     = int(os.environ.get('REFIT_NPROCS', cpu_count()))
ALL_SIMS   = list(range(N_SIMS))
RANK_FILE  = 'pf_candidates.txt'
LIT_FILE   = 'literature_combined_DL_trimmed.txt'
COMP_MAP   = {}   # func -> search complexity (from pf_candidates.txt)
AIFEYN_MAP = {}   # func -> aifeynman (from pf_candidates.txt)

with open('aifeynman_cache.json') as _f:
    AIFEYN = json.load(_f)


# ── Data ──
def load_hmf_trimmed(sim):
    tp = f'hmf_files/hmf_{sim}_trimmed.dat'
    if os.path.exists(tp):
        data = np.loadtxt(tp)             # already trimmed (== new.dat[2:])
    else:
        data = np.loadtxt(f'hmf_files/hmf_{sim}_new.dat')[2:]  # drop 2 lowest-mass bins
    return data[:, 0], data[:, 1], data[:, 3]   # sigma, counts, Veff(norm)


# ── Function string -> numpy callable (same handling as phase-2) ──
def make_callable(func_str):
    safe = func_str.replace("re(", "(")
    while 'cos(im(' in safe:
        idx = safe.find('cos(im(')
        depth = 0
        end = idx + 4
        for i in range(idx + 4, len(safe)):
            if safe[i] == '(':
                depth += 1
            elif safe[i] == ')':
                if depth == 0:
                    end = i + 1
                    break
                depth -= 1
        prefix, suffix = safe[:idx], safe[end:]
        if prefix.endswith('*'):
            prefix = prefix[:-1]
        elif suffix.startswith('*'):
            suffix = suffix[1:]
        safe = prefix + suffix
    # Single-pass replacement of function-call tokens. The negative lookbehind
    # for [\w.] prevents matching inside an already-substituted name (e.g. the
    # "abs" inside "np.abs"); re.sub does not rescan its replacement text, so
    # there is no double-substitution (the earlier bug produced "np.np.abs").
    _map = {'pow': 'np.power(', 'exp': 'np.exp(', 'log': 'np.log(',
            'Abs': 'np.abs(', 'abs': 'np.abs(', 'inv': '1.0/('}
    safe = re.sub(r'(?<![\w.])(pow|exp|log|Abs|abs|inv)\(',
                  lambda m: _map[m.group(1)], safe)

    def f(params, x):
        a0 = params[0] if len(params) > 0 else 0
        a1 = params[1] if len(params) > 1 else 0
        a2 = params[2] if len(params) > 2 else 0
        a3 = params[3] if len(params) > 3 else 0
        with np.errstate(all='ignore'):
            try:
                result = eval(safe)
                return np.where(np.isfinite(result), result, np.nan)
            except Exception:
                return np.full_like(x, np.nan)
    return f


def poisson_nll(params, x, counts, norm, fc):
    f = fc(params, x)
    ypred = f * norm
    if np.any(~np.isfinite(ypred)) or np.any(ypred <= 0):
        return np.inf
    nll = np.sum(ypred - counts * np.log(ypred))
    return nll if np.isfinite(nll) else np.inf


def compute_codelen(params, x, counts, norm, fc):
    k = len(params)
    if k == 0:
        return 0.0

    def nll_func(p):
        return poisson_nll(p, x, counts, norm, fc)

    d_list = [1e-5, 10**(-5.5), 10**(-4.5), 1e-6, 1e-4, 10**(-6.5),
              10**(-3.5), 1e-7, 1e-3, 10**(-7.5), 10**(-2.5), 1e-8, 1e-2]
    method_list = ["central", "forward", "backward"]

    Fisher_diag = None
    try:
        H = nd.Hessian(nll_func)(params)
        Fisher_diag = np.array([H[i, i] for i in range(k)])
    except Exception:
        pass

    def _good(Fd):
        return Fd is not None and np.all(Fd > 0) and np.all(np.isfinite(Fd))

    if not _good(Fisher_diag):
        for d2, meth in itertools.product(d_list, method_list):
            try:
                step = np.abs(d2 * np.asarray(params)) + 1e-15
                H = nd.Hessian(nll_func, step=step, method=meth)(params)
                Fd = np.array([H[i, i] for i in range(k)])
                if _good(Fd):
                    Fisher_diag = Fd
                    break
            except Exception:
                continue

    if not _good(Fisher_diag):
        return np.nan

    Delta = np.sqrt(12.0 / Fisher_diag)
    Nsteps = np.abs(params) / Delta
    mask = Nsteps >= 1
    k_eff = int(np.sum(mask))
    if k_eff == 0:
        return 0.0
    return (-k_eff / 2.0 * math.log(3.0)
            + np.sum(0.5 * np.log(Fisher_diag[mask]) + np.log(np.abs(np.asarray(params)[mask]))))


# ── Multi-start optimisation for one (function, sim) ──
def multistart(fc, nparam, sigma, counts, norm, donors, n_random, rng):
    best_nll = np.inf
    best_p = None

    starts = []
    # cross-sim / paper donor params (truncate or pad to nparam)
    for d in donors:
        d = np.asarray(d, dtype=float)
        if len(d) >= nparam:
            starts.append(d[:nparam])
        elif len(d) > 0:
            starts.append(np.concatenate([d, np.full(nparam - len(d), 1.0)]))
    # wide random starts: mix of moderate and very wide ranges
    n_wide = n_random // 3
    starts += [rng.uniform(-5.0, 5.0, nparam) for _ in range(n_random - n_wide)]
    starts += [rng.uniform(-15.0, 15.0, nparam) for _ in range(n_wide)]

    for p0 in starts:
        try:
            r = minimize(poisson_nll, p0, args=(sigma, counts, norm, fc),
                         method='L-BFGS-B', options={'maxiter': 10000, 'ftol': 1e-15})
            if np.isfinite(r.fun) and r.fun < best_nll:
                best_nll, best_p = r.fun, r.x.copy()
        except Exception:
            pass

    # Nelder-Mead polish of the best basin
    if best_p is not None:
        try:
            r = minimize(poisson_nll, best_p, args=(sigma, counts, norm, fc),
                         method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.isfinite(r.fun) and r.fun < best_nll:
                best_nll, best_p = r.fun, r.x.copy()
        except Exception:
            pass

    return best_nll, best_p


# ── Worker: process one function across all sims (two donor stages) ──
def process_function(func):
    aifeyn = AIFEYN_MAP.get(func, AIFEYN.get(func, 25.0))
    nparam = max(sum(1 for i in range(10) if f'a{i}' in func), 0)
    fc = make_callable(func)
    paper_donors = PAPER_DONORS.get(func, [])

    if nparam == 0:
        # parameter-free: evaluate directly on each sim
        per_sim = []
        for sim in ALL_SIMS:
            sigma, counts, norm = SIM_CACHE[sim]
            nll = poisson_nll(np.array([]), sigma, counts, norm, fc)
            if np.isfinite(nll):
                dl = nll + aifeyn
                per_sim.append((sim, dl, nll, 0.0, []))
        return _combine(func, aifeyn, per_sim)

    rng = np.random.RandomState(12345)

    # Stage A: paper donors + wide random
    stageA = {}
    for sim in ALL_SIMS:
        sigma, counts, norm = SIM_CACHE[sim]
        nll, p = multistart(fc, nparam, sigma, counts, norm, paper_donors, N_RANDOM, rng)
        stageA[sim] = (nll, p)

    # Stage B: propagate this run's own best params across sims as donors
    propagated = [p for (n, p) in stageA.values() if p is not None]
    best = {}
    for sim in ALL_SIMS:
        sigma, counts, norm = SIM_CACHE[sim]
        donorsB = propagated + paper_donors
        nllB, pB = multistart(fc, nparam, sigma, counts, norm, donorsB, N_RANDOM // 2, rng)
        nA, pA = stageA[sim]
        if pB is not None and (pA is None or nllB < nA):
            best[sim] = (nllB, pB)
        else:
            best[sim] = (nA, pA)

    # DL per sim
    per_sim = []
    for sim in ALL_SIMS:
        nll, p = best[sim]
        if p is None or not np.isfinite(nll):
            continue
        sigma, counts, norm = SIM_CACHE[sim]
        codelen = compute_codelen(p, sigma, counts, norm, fc)
        if not np.isfinite(codelen):
            continue
        dl = nll + codelen + aifeyn
        per_sim.append((sim, dl, nll, codelen, list(np.asarray(p))))
    return _combine(func, aifeyn, per_sim)


def _combine(func, aifeyn, per_sim):
    n = len(per_sim)
    if n == 0:
        return {'func': func, 'aifeyn': aifeyn, 'n_sims': 0,
                'DL_combined': np.inf, 'sum_NLL': np.nan, 'sum_DL': np.nan, 'per_sim': []}
    sum_dl = sum(r[1] for r in per_sim)
    sum_nll = sum(r[2] for r in per_sim)
    DL_combined = sum_dl - (n - 1) * aifeyn
    return {'func': func, 'aifeyn': aifeyn, 'n_sims': n,
            'DL_combined': DL_combined, 'sum_NLL': sum_nll, 'sum_DL': sum_dl, 'per_sim': per_sim}


# ── Loaders (run in parent; workers inherit via fork) ──
def load_top_functions():
    funcs = []
    with open(RANK_FILE) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.rstrip('\n').split(';')      # complexity;blank_func;aifeyn
            func = p[1]
            comp = int(p[0])
            # A function can recur at several complexities (different ESR trees that
            # simplify to the same string). Keep its MINIMUM generation complexity
            # (the shortest encoding) and the aifeyn recorded at that complexity --
            # NOT last-wins, which mis-binned recurring functions to a higher
            # complexity (wrong aifeyn too) and dropped genuinely good functions
            # from the low-complexity Pareto front.
            if func not in COMP_MAP or comp < COMP_MAP[func]:
                COMP_MAP[func] = comp
                if len(p) > 2:
                    AIFEYN_MAP[func] = float(p[2])
            funcs.append(func)
    # de-duplicate while preserving order (a function may recur across complexities)
    seen = set(); uniq = []
    for fn in funcs:
        if fn not in seen:
            seen.add(fn); uniq.append(fn)
    return uniq[:TOP_N]


def _parse_params(blob):
    # blob like "[ 0.84935342  0.53839971 -2.23250022]"
    nums = []
    for tok in blob.replace('[', ' ').replace(']', ' ').split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    return np.array(nums) if nums else None


def load_paper_donors(top_funcs):
    top_set = set(top_funcs)
    donors = {fn: [] for fn in top_funcs}
    for sim in ALL_SIMS:
        fp = f'hmf_data/hmf_{sim}_data/final_all_trimmed.txt'
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                parts = line.split(';')
                if len(parts) < 5:
                    continue
                fn = parts[1]
                if fn in top_set:
                    p = _parse_params(parts[4])
                    if p is not None and len(p) > 0:
                        donors[fn].append(p)
    return donors


def load_literature():
    lit = {}
    if not os.path.exists(LIT_FILE):
        return lit
    with open(LIT_FILE) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.strip().split(';')
            # file stores -DL_combined (positive); negate to get DL_combined (lower=better)
            lit[p[0]] = -float(p[1])
    return lit


def load_paper_ranking():
    rank = {}
    order = []
    with open(RANK_FILE) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split(';')
            rank[p[1]] = float(p[2])
            order.append(p[1])
    return rank, order


# Globals populated in main (inherited by forked workers)
SIM_CACHE = {}
PAPER_DONORS = {}


def main():
    global SIM_CACHE, PAPER_DONORS

    top_funcs = load_top_functions()
    print(f"Thorough re-fit: TOP_N={len(top_funcs)} funcs x {len(ALL_SIMS)} sims, "
          f"N_RANDOM={N_RANDOM}, NPROCS={NPROCS}", flush=True)

    print("Loading sim data...", flush=True)
    SIM_CACHE = {sim: load_hmf_trimmed(sim) for sim in ALL_SIMS}
    print("Loading paper donor params...", flush=True)
    PAPER_DONORS = load_paper_donors(top_funcs)
    ndon = np.mean([len(PAPER_DONORS[f]) for f in top_funcs])
    print(f"  mean donor sets/function = {ndon:.1f}", flush=True)

    print(f"Fitting {len(top_funcs)} functions on {NPROCS} processes...", flush=True)
    if NPROCS > 1:
        with Pool(NPROCS) as pool:
            results = pool.map(process_function, top_funcs, chunksize=1)
    else:
        results = [process_function(fn) for fn in top_funcs]

    results.sort(key=lambda r: r['DL_combined'])

    # ── Write combined results (with search complexity) ──
    best_DL = results[0]['DL_combined']
    sum_NLL0 = results[0]['sum_NLL']
    with open('pf_refit_combined.txt', 'w') as f:
        f.write('# rank;function;complexity;DL_combined;delta_DL;sum_NLL;sum_DL;aifeynman;n_sims\n')
        for i, r in enumerate(results):
            c = COMP_MAP.get(r['func'], -1)
            f.write(f"{i};{r['func']};{c};{r['DL_combined']:.6f};"
                    f"{r['DL_combined']-best_DL:.6f};{r['sum_NLL']:.6f};"
                    f"{r['sum_DL']:.6f};{r['aifeyn']:.6f};{r['n_sims']}\n")

    # ── Write per-sim diagnostics ──
    with open('pf_refit_persim.txt', 'w') as f:
        f.write('# function;sim;DL;NLL;codelen;params\n')
        for r in results:
            for (sim, dl, nll, cl, p) in r['per_sim']:
                pstr = ' '.join(f'{v:.10e}' for v in p) if len(p) else 'none'
                f.write(f"{r['func']};{sim};{dl:.6f};{nll:.6f};{cl:.6f};{pstr}\n")

    # ── Per-complexity Pareto-front points (min combined DL among full-100-sim fns) ──
    by_comp = {}
    for r in results:
        if r['n_sims'] < N_SIMS:        # only functions fit on ALL sims define a PF point
            continue
        c = COMP_MAP.get(r['func'], -1)
        if c not in by_comp or r['DL_combined'] < by_comp[c]['DL_combined']:
            by_comp[c] = r
    print("\n" + "=" * 78, flush=True)
    print("  PARETO-FRONT POINTS  (combined over 100 sims, relative to overall best)", flush=True)
    print("=" * 78, flush=True)
    print(f"  overall best: DL_combined={best_DL:.2f}  ({results[0]['func'][:55]})", flush=True)
    print(f"  {'comp':>4} {'dDL':>10} {'dNLL':>10} {'n':>4}  function", flush=True)
    for c in sorted(by_comp):
        r = by_comp[c]
        print(f"  {c:>4} {r['DL_combined']-best_DL:>10.1f} {r['sum_NLL']-sum_NLL0:>10.1f} "
              f"{r['n_sims']:>4}  {r['func'][:55]}", flush=True)
    lit = load_literature()
    if lit:
        print("\n  literature (combined DL, relative to overall best):", flush=True)
        for name in sorted(lit, key=lambda k: lit[k]):
            print(f"     {name:8s} dDL={lit[name]-best_DL:.1f}", flush=True)
    n_full = sum(1 for r in results if r['n_sims'] >= N_SIMS)
    print(f"\n  {n_full}/{len(results)} functions fit on all {N_SIMS} sims.", flush=True)
    print(f"  Saved: pf_refit_combined.txt, pf_refit_persim.txt", flush=True)


if __name__ == '__main__':
    main()
