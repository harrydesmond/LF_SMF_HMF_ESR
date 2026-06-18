#!/usr/bin/env python3
"""Refit the 5 literature HMF forms on the EXTENDED (untrimmed, 18-bin) data on
EQUAL FOOTING with the extended ESR ranking (hmf_combined_DL_diagonal.txt), for
Table B1's literature rows.

This is the extended-data analogue of refit_literature.py and uses the IDENTICAL
method, so the fiducial (Table 3) and extended (Table B1) literature values are
computed the same way:
  * Codelen / NLL / function evaluation reuse refit_pf's exact routines (the same
    ones used for the ESR functions) -> diagonal-Fisher DL, byte-for-byte comparable.
  * Each form is its constant-free ESR tree, so codelen AND aifeyn come from the
    same tree.
  * Press-Schechter: parameter-free (delta_c = 1.686 fixed).
  * Warren/Tinker: codelen at the STORED bounded ML params (no re-opt -- re-optimising
    drifts toward the a0/a2 amplitude/scale degeneracy giving a marginally lower NLL
    but a higher codelen and a WORSE total DL). Stored extended per-sim params come
    from literature_fits_all_sims.txt (the untrimmed all-sims fits).
  * S-T/Jenkins: smooth/stable -> bounded multistart fit from the canonical seed.

The ONLY differences from refit_literature.py are the data range (full 18-bin
hmf_<sim>_new.dat, no [2:] trim) and the reference best (the extended diagonal
combined ranking). dDL/dNLL are printed and written relative to that best.

Run locally (<=4 cores, ~1-2 min) or on Glamdring with -s.
Output -> literature_refit_extended.txt (+ per-sim file).
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
import refit_pf as R                  # identical poisson_nll / compute_codelen / make_callable
from multiprocessing import Pool

N_SIMS   = 100
ALL_SIMS = list(range(N_SIMS))
NPROCS   = int(os.environ.get('LITFIT_NPROCS', 4))

# IDENTICAL forms / aifeyn / bounds to the fiducial refit_literature.py.
LIT = {
    'P.Sch.': dict(esr='pow(2/3.141592653589793,0.5)*(1.686/x)*exp(-0.5*pow(1.686/x,2))', aifeyn=23.567004, comp=10, nparam=0, bounds=None),
    'War.':   dict(esr='a0*(pow(x,a2)+a1)*exp(-a3*pow(x,-2))',                    aifeyn=29.361299, comp=14, nparam=4,
                   bounds=[(0.01, 100), (-10, 10), (-5, 5), (0.01, 10)]),
    'Tin.':   dict(esr='a0*(pow(x/a2,-a1)+1)*exp(-a3*pow(x,-2))',                 aifeyn=42.281978, comp=16, nparam=4,
                   bounds=[(1e-6, 10), (0.01, 5), (1, 100000), (0.01, 10)]),
    'Jen.':   dict(esr='a0/exp(pow(Abs(a1 - log(x)),a2))',                        aifeyn=24.953299, comp=13, nparam=3,
                   bounds=[(0.01, 10), (-5, 5), (0.5, 10)]),
    'S-T.':   dict(esr='a0*(1+pow(x*x/a1,a2))*exp(-a1/(2*x*x))/x',                aifeyn=54.119678, comp=24, nparam=3,
                   bounds=[(1e-3, 10), (0.01, 20), (0.01, 2)]),
}

SEED = {'Jen.': np.array([0.31840, 0.59970, 3.29400]),
        'S-T.': np.array([0.35114, 1.82880, 0.22900])}


def load_hmf_extended(sim):
    """Full (untrimmed) HMF data: all 18 bins, no [2:] trim. cols sigma,counts,norm."""
    data = np.loadtxt(f'hmf_files/hmf_{sim}_new.dat')
    return data[:, 0], data[:, 1], data[:, 3]


def load_stored_extended():
    """Stored EXTENDED per-sim ML params for Warren/Tinker (physical-basin fits)."""
    d = {'War.': {}, 'Tin.': {}}
    with open('literature_fits_all_sims.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.strip().split(';')
            if p[0] in d and len(p) > 5 and p[5] != 'none':
                d[p[0]][int(p[1])] = np.array([float(v) for v in p[5].split()])
    return d


def fit_bounded(fc, bounds, p0, sig, cnt, nrm, rng, n_restarts=25):
    """Bounded multistart Poisson-NLL fit (physical basin); identical to refit_literature.py."""
    best_nll, best_p = np.inf, None
    lo = [b[0] for b in bounds]; hi = [b[1] for b in bounds]
    starts = [p0] + [p0 * s for s in (0.9, 1.1, 0.8, 1.2)]
    starts += [np.clip(p0 * (1 + 0.4 * rng.randn(len(p0))), lo, hi) for _ in range(n_restarts)]
    for s0 in starts:
        s0 = np.clip(np.asarray(s0, float), lo, hi)
        try:
            r = minimize(R.poisson_nll, s0, args=(sig, cnt, nrm, fc),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 10000, 'ftol': 1e-15})
            if np.isfinite(r.fun) and r.fun < best_nll:
                best_nll, best_p = r.fun, r.x.copy()
        except Exception:
            pass
    return best_nll, best_p


SIM = {}
STORED = {}


def work(task):
    name, sim = task
    info = LIT[name]; fc = R.make_callable(info['esr']); npar = info['nparam']
    sig, cnt, nrm = SIM[sim]
    if npar == 0:                                   # PS: parameter-free
        nll = R.poisson_nll(np.array([]), sig, cnt, nrm, fc)
        return (name, sim, nll, 0.0, []) if np.isfinite(nll) else (name, sim, None, None, None)
    if name in STORED:                              # War/Tin: codelen at STORED extended params
        p = STORED[name].get(sim)
        if p is None:
            return (name, sim, None, None, None)
        nll = R.poisson_nll(p, sig, cnt, nrm, fc)
    else:                                           # S-T/Jen: bounded fit from canonical seed
        rng = np.random.RandomState(42 + sim)
        nll, p = fit_bounded(fc, info['bounds'], SEED[name], sig, cnt, nrm, rng)
    if p is None or not np.isfinite(nll):
        return (name, sim, None, None, None)
    cl = R.compute_codelen(p, sig, cnt, nrm, fc)
    if not np.isfinite(cl):
        return (name, sim, None, None, None)
    return (name, sim, nll, cl, list(np.asarray(p)))


def read_diagonal_best():
    """Extended ESR best (rank 0 of hmf_combined_DL_diagonal.txt): magnitudes."""
    with open('hmf_combined_DL_diagonal.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            c = line.split(';')
            return abs(float(c[2])), abs(float(c[4]))   # DL_combined, sum_NLL
    raise RuntimeError('no rows in hmf_combined_DL_diagonal.txt')


def main():
    global SIM, STORED
    print(f"Extended literature refit: {len(LIT)} funcs x {N_SIMS} sims, NPROCS={NPROCS}", flush=True)
    SIM    = {s: load_hmf_extended(s) for s in ALL_SIMS}
    STORED = load_stored_extended()
    best_DL_mag, best_NLL_mag = read_diagonal_best()
    print(f"  extended diagonal best: DL_mag={best_DL_mag:.3f}  NLL_mag={best_NLL_mag:.3f}", flush=True)

    tasks = [(name, sim) for name in LIT for sim in ALL_SIMS]
    with Pool(NPROCS) as pool:
        res = pool.map(work, tasks, chunksize=8)

    by = {name: [] for name in LIT}
    params = {name: [] for name in LIT}
    for name, sim, nll, cl, p in res:
        if nll is None:
            continue
        by[name].append((sim, nll, cl, p))
        if p is not None and len(p):
            params[name].append(p)

    rows, persim = [], []
    print(f"\n  {'name':7s} {'n':>3} {'comp':>4} {'codelen_tot':>11} {'dDL':>10} {'dNLL':>10}", flush=True)
    for name in LIT:
        per = by[name]; aif = LIT[name]['aifeyn']; comp = LIT[name]['comp']; n = len(per)
        sum_nll = sum(r[1] for r in per)
        sum_cl  = sum(r[2] for r in per)
        DL_combined_actual = sum_nll + sum_cl + aif      # structural counted once
        DL_mag  = -DL_combined_actual
        NLL_mag = -sum_nll
        dDL  = best_DL_mag  - DL_mag
        dNLL = best_NLL_mag - NLL_mag
        rows.append((name, DL_mag, NLL_mag, -(sum_nll + sum_cl + n * aif), aif, n, comp, dDL, dNLL))
        for (sim, nll, cl, p) in per:
            pstr = ' '.join(f'{v:.10f}' for v in p) if len(p) else 'none'
            persim.append(f"{name};{sim};{-(nll + cl + aif):.6f};{-nll:.6f};{cl:.6f};{pstr}")
        print(f"  {name:7s} {n:>3} {comp:>4} {sum_cl:>11.2f} {dDL:>10.1f} {dNLL:>10.1f}", flush=True)

    with open('literature_refit_extended.txt', 'w') as f:
        f.write('# name;DL_combined;sum_NLL;sum_DL;aifeynman;n_sims;complexity;dDL;dNLL\n')
        for name, dlm, nllm, sdl, aif, n, comp, dDL, dNLL in rows:
            f.write(f"{name};{dlm:.6f};{nllm:.6f};{sdl:.6f};{aif:.6f};{n};{comp};{dDL:.6f};{dNLL:.6f}\n")
    with open('literature_refit_persim_extended.txt', 'w') as f:
        f.write('# name;sim;DL;NLL;codelen;params\n')
        f.write('\n'.join(persim) + '\n')

    # physical-parameter medians for the Table B1 theta columns (S-T mapped to A,a,p)
    DELTA_C = 1.686
    print("\n  Median params (for Table B1 theta columns):", flush=True)
    for name in LIT:
        if LIT[name]['nparam'] == 0 or not params[name]:
            continue
        P = np.array(params[name])
        if name == 'S-T.':
            A = P[:, 0] / (np.sqrt(2.0 * (P[:, 1] / DELTA_C**2) / np.pi) * DELTA_C)
            a = P[:, 1] / DELTA_C**2
            P = np.column_stack([A, a, P[:, 2]]); labels = ['A', 'a', 'p']
        else:
            labels = [f'th{i}' for i in range(P.shape[1])]
        med = np.median(P, axis=0)
        lo = med - np.percentile(P, 16, axis=0); hi = np.percentile(P, 84, axis=0) - med
        print(f"    {name}: " + ", ".join(f"{l}={m:.4g}(+{h:.2g}/-{l2:.2g})"
                                           for l, m, h, l2 in zip(labels, med, hi, lo)), flush=True)
    print("\n  Saved literature_refit_extended.txt + per-sim file", flush=True)


if __name__ == '__main__':
    main()
