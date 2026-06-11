#!/usr/bin/env python3
"""Fit Sheth-Tormen and Jenkins HMF forms on all 100 EXTENDED (untrimmed) sims,
computed EXACTLY analogously to the fiducial main-text fit (fit_lit_st_jenkins.py):
same reparam (constant-free) forms, same aifeyn (S-T 54.12 / Jenkins 24.95), same
refit_pf likelihood/codelen/multistart. ONLY the data range (full 18-bin
hmf_*_new.dat, no [2:] trim) and the reference best (extended combined ranking)
differ -> gives the combined dDL/dNLL for Table B1 (cf. fiducial Table 3).

Warren/Tinker extended (literature_combined_DL.txt) reconcile to B1 (Warren dDL=4343,
dNLL=3389) relative to the extended best, with minimal-form aifeyn -> same convention.

Output -> literature_st_jenkins_extended.txt (+ per-sim physical-param percentiles
printed for the Table B1 theta columns).

Submit with -s (multiprocessing).
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('/users/hdesmond/Amelia_code')
import numpy as np
from multiprocessing import Pool, cpu_count
import refit_pf as rp   # make_callable, poisson_nll, compute_codelen, multistart

N_SIMS   = 100
ALL_SIMS = list(range(N_SIMS))
N_RANDOM = 50
NPROCS   = int(os.environ.get('LITFIT_NPROCS', min(16, cpu_count())))
DELTA_C  = 1.686

# Identical reparam forms / aifeyn to the fiducial fit (fit_lit_st_jenkins.py).
# Also refit Warren/Tinker (minimal forms, aifeyn 29.36/42.28) to VERIFY B1's
# existing extended values (4343/8358) with the same machinery.
LIT = {
    'War.': dict(
        form='a0*(pow(x,a2)+a1)*exp(-a3*pow(x,-2))',
        aifeyn=29.361299, nparam=4, comp=14,
        donors=[[7.43, -0.92, -0.04, 0.80], [7.64, -0.92, -0.04, 0.81]]),
    'Tin.': dict(
        form='a0*(pow(x/a2,-a1)+1)*exp(-a3*pow(x,-2))',
        aifeyn=42.281978, nparam=4, comp=16,
        donors=[[8.9e-4, 0.80, 5300, 0.93], [1.1e-3, 0.75, 4440, 0.90]]),
    'S-T.': dict(
        form='a0*(1+pow(x*x/a1,a2))*exp(-a1/(2*x*x))/x',
        aifeyn=54.119700, nparam=3, comp=24,
        donors=[[0.35109, 1.82864, 0.2290], [0.35200, 1.83157, 0.22974]]),
    'Jen.': dict(
        form='a0/exp(pow(Abs(a1 - log(x)),a2))',
        aifeyn=24.953299, nparam=3, comp=13,
        donors=[[0.3184, 0.5997, 3.294], [0.31880758, 0.60138646, 3.31853711]]),
}


def load_hmf_extended(sim):
    """Full (untrimmed) HMF data: all bins, no [2:] trim. cols sigma,counts,norm."""
    data = np.loadtxt(f'hmf_files/hmf_{sim}_new.dat')
    return data[:, 0], data[:, 1], data[:, 3]


SIM_CACHE = {sim: load_hmf_extended(sim) for sim in ALL_SIMS}


def st_reparam_to_physical(a0, a1, a2):
    """(a0,a1,a2) of the reparam S-T form -> physical (A, a, p)."""
    p = a2
    a = a1 / DELTA_C**2
    A = a0 / (np.sqrt(2.0 * a / np.pi) * DELTA_C)
    return A, a, p


def work(task):
    name, sim = task
    spec = LIT[name]
    fc = rp.make_callable(spec['form'])
    sigma, counts, norm = SIM_CACHE[sim]
    donors = [np.asarray(d, float) for d in spec['donors']]
    rng = np.random.RandomState(12345 + sim)
    nll, p = rp.multistart(fc, spec['nparam'], sigma, counts, norm, donors, N_RANDOM, rng)
    if p is None or not np.isfinite(nll):
        return (name, sim, None, None, None)
    cl = rp.compute_codelen(p, sigma, counts, norm, fc)
    if not np.isfinite(cl):
        return (name, sim, None, None, None)
    return (name, sim, nll, cl, list(np.asarray(p)))


def main():
    print(f"Fit S-T & Jenkins on {N_SIMS} EXTENDED sims, N_RANDOM={N_RANDOM}, NPROCS={NPROCS}", flush=True)
    tasks = [(name, sim) for name in LIT for sim in ALL_SIMS]
    with Pool(NPROCS) as pool:
        res = pool.map(work, tasks, chunksize=4)

    by = {name: [] for name in LIT}
    params = {name: [] for name in LIT}
    sim50 = {}
    for name, sim, nll, cl, p in res:
        if nll is None:
            continue
        dl = nll + cl + LIT[name]['aifeyn']
        by[name].append((sim, dl, nll, cl))
        params[name].append(p)
        if sim == 50:
            sim50[name] = nll

    KNOWN_NLL50 = {'War.': -35593571.50343531, 'Tin.': -35593532.574881725,
                   'S-T.': -35593529.774345, 'Jen.': -35592714.657771}
    print("\nsim-50 NLL cross-check vs hmf_50_final_functions_extended.txt:", flush=True)
    for name in LIT:
        if name in sim50:
            print(f"  {name}: NLL={sim50[name]:.1f} (known {KNOWN_NLL50[name]:.1f}, "
                  f"diff {sim50[name]-KNOWN_NLL50[name]:+.3f})", flush=True)

    # Extended reference best (comp-10 rank-1 of the extended combined ranking,
    # hmf_combined_DL.txt rank 0 -- hardcoded as that file is local-only).
    best_DL, best_NLL = -3555040871.212088, -3555042797.586628
    # B1 reference values (Warren/Tinker already in the table) for verification
    PAPER = {'War.': (4343, 3389), 'Tin.': (8358, 7433)}

    rows = []
    print("\nCombined over 100 sims (extended; Delta vs comp-10 best):", flush=True)
    for name in LIT:
        per = by[name]
        n = len(per)
        sum_dl = sum(r[1] for r in per)
        sum_nll = sum(r[2] for r in per)
        aif = LIT[name]['aifeyn']
        DL_comb = sum_dl - (n - 1) * aif
        dDL, dNLL = DL_comb - best_DL, sum_nll - best_NLL
        ptag = ''
        if name in PAPER:
            pdl, pnll = PAPER[name]
            ptag = f"  [B1: {pdl}/{pnll}, diff {dDL-pdl:+.1f}/{dNLL-pnll:+.1f}]"
        print(f"  {name}: n={n}  dDL={dDL:.1f}  dNLL={dNLL:.1f}  aifeyn={aif}{ptag}", flush=True)
        rows.append((name, DL_comb, sum_nll, sum_dl, aif, n, LIT[name]['comp']))

        # Physical-parameter percentiles for the Table B1 theta columns
        P = np.array(params[name])
        if name == 'S-T.':
            phys = np.array([st_reparam_to_physical(*row) for row in P])
            labels = ['A(th0)', 'a(th1)', 'p(th2)']
        else:  # War/Tin/Jenkins params are the form's own (physical) thetas
            phys = P
            labels = [f'th{i}' for i in range(P.shape[1])]
        med = np.median(phys, axis=0)
        lo = med - np.percentile(phys, 16, axis=0)
        hi = np.percentile(phys, 84, axis=0) - med
        print(f"    params (median +hi -lo):", flush=True)
        for lab, m, l, h in zip(labels, med, lo, hi):
            print(f"      {lab}: {m:.4g}  +{h:.2g} -{l:.2g}", flush=True)

    with open('literature_st_jenkins_extended.txt', 'w') as f:
        f.write('# name;DL_combined;sum_NLL;sum_DL;aifeynman;n_sims;complexity\n')
        for name, dlc, snll, sdl, aif, n, comp in rows:
            f.write(f"{name};{-dlc:.6f};{-snll:.6f};{-sdl:.6f};{aif:.6f};{n};{comp}\n")
    print("\nSaved literature_st_jenkins_extended.txt", flush=True)


if __name__ == '__main__':
    main()
