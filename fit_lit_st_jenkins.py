#!/usr/bin/env python3
"""Fit Sheth-Tormen and Jenkins HMF forms on all 100 FIDUCIAL (trimmed) sims,
using refit_pf.py's exact likelihood / codelen / data loading so the combined
DL is consistent with the ESR Pareto-front numbers and with Warren/Tinker.

aifeyn (structural term) computed via ESR string_to_aifeyn(replace_floats=False),
validated to reproduce Warren=29.361299 / Tinker=42.281978.

Combined metric (same as refit_pf):
    DL_i        = NLL_i + codelen_i + aifeyn
    DL_combined = sum_i DL_i - (n-1)*aifeyn   (aifeyn counted once)

Output -> literature_st_jenkins_combined.txt  (same columns as
literature_combined_DL_trimmed.txt; DL/NLL stored as +magnitude = -actual).

Submit with -s (multiprocessing):
    addqueue -q <queue> -n 1x16 -m 4 -s -c "fit S-T/Jen fiducial" \
        /usr/local/shared/python/3.11.4/bin/python3 fit_lit_st_jenkins.py
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('/users/hdesmond/Amelia_code')
import numpy as np
from multiprocessing import Pool, cpu_count
import refit_pf as rp   # reuse make_callable, poisson_nll, compute_codelen,
                        # multistart, load_hmf_trimmed (byte-identical machinery)

N_SIMS   = 100
ALL_SIMS = list(range(N_SIMS))
N_RANDOM = 50
NPROCS   = int(os.environ.get('LITFIT_NPROCS', min(16, cpu_count())))

# Use the SAME minimal, constant-free ESR forms as the 28-May audit
# (fit_new_literature.py) -> consistent with Warren/Tinker/Jenkins aifeyn and with
# Table 3. Physical constants (delta_c=1.686, sqrt(2a/pi), 1/2) are absorbed into
# the free params, so aifeyn(S-T)=54.12 (complexity 24), not 83.39 (literal form).
# S-T reparam: a0/x*(1+(x^2/a1)^a2)*exp(-a1/(2x^2)); donor a0=A*sqrt(2a/pi)*dc, a1=a*dc^2, a2=p.
LIT = {
    'S-T.': dict(
        form='a0*(1+pow(x*x/a1,a2))*exp(-a1/(2*x*x))/x',
        aifeyn=54.119700, nparam=3, comp=24,
        donors=[[0.35109, 1.82864, 0.2290], [0.35200, 1.83157, 0.22974]]),
    'Jen.': dict(
        form='a0/exp(pow(Abs(a1 - log(x)),a2))',
        aifeyn=24.953299, nparam=3, comp=13,
        donors=[[0.3184, 0.5997, 3.294], [0.31880758, 0.60138646, 3.31853711]]),
}

SIM_CACHE = {sim: rp.load_hmf_trimmed(sim) for sim in ALL_SIMS}  # fiducial data


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
    print(f"Fitting S-T & Jenkins on {N_SIMS} fiducial sims, N_RANDOM={N_RANDOM}, NPROCS={NPROCS}", flush=True)
    tasks = [(name, sim) for name in LIT for sim in ALL_SIMS]
    with Pool(NPROCS) as pool:
        res = pool.map(work, tasks, chunksize=4)

    by = {name: [] for name in LIT}
    sim50 = {}
    for name, sim, nll, cl, p in res:
        if nll is None:
            continue
        dl = nll + cl + LIT[name]['aifeyn']
        by[name].append((sim, dl, nll, cl))
        if sim == 50:
            sim50[name] = (nll, dl)

    # sim-50 NLL must match the literal-form fit (same function) -> validates the fit.
    KNOWN_NLL50 = {'S-T.': -14352955.397, 'Jen.': -14352895.713}
    print("\nsim-50 NLL cross-check (reparam form must reach same optimum):", flush=True)
    for name in LIT:
        if name in sim50:
            nll, dl = sim50[name]
            kn = KNOWN_NLL50[name]
            print(f"  {name}: NLL={nll:.1f} (known {kn:.1f}, diff {nll-kn:+.3f})", flush=True)

    # Reference best (comp-10 rank-1) for Delta DL / Delta NLL vs paper Table 3
    with open('pf_refit_combined.txt') as f:
        f.readline()
        r0 = f.readline().strip().split(';')
    best_DL, best_NLL = float(r0[3]), float(r0[5])
    PAPER = {'S-T.': (2062, 2261), 'Jen.': (8656, 8789)}

    rows = []
    print("\nCombined over 100 sims (Delta vs comp-10 best; paper = Table 3):", flush=True)
    for name in LIT:
        per = by[name]
        n = len(per)
        sum_dl = sum(r[1] for r in per)
        sum_nll = sum(r[2] for r in per)
        aif = LIT[name]['aifeyn']
        DL_comb = sum_dl - (n - 1) * aif
        dDL, dNLL = DL_comb - best_DL, sum_nll - best_NLL
        pdl, pnll = PAPER[name]
        print(f"  {name}: n={n}  dDL={dDL:.1f} (paper {pdl}, diff {dDL-pdl:+.1f}) | "
              f"dNLL={dNLL:.1f} (paper {pnll}, diff {dNLL-pnll:+.1f})  aifeyn={aif}", flush=True)
        rows.append((name, DL_comb, sum_nll, sum_dl, aif, n, LIT[name]['comp']))

    # Write (positive-magnitude convention, matching literature_combined_DL_trimmed.txt)
    with open('literature_st_jenkins_combined.txt', 'w') as f:
        f.write('# name;DL_combined;sum_NLL;sum_DL;aifeynman;n_sims;complexity\n')
        for name, dlc, snll, sdl, aif, n, comp in rows:
            f.write(f"{name};{-dlc:.6f};{-snll:.6f};{-sdl:.6f};{aif:.6f};{n};{comp}\n")
    print("\nSaved literature_st_jenkins_combined.txt", flush=True)


if __name__ == '__main__':
    main()
