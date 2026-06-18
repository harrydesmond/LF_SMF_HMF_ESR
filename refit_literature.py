#!/usr/bin/env python3
"""Verify + consolidate the 5 literature HMF combined DLs on EQUAL FOOTING with the
fresh ESR PF refit (pf_refit_combined.txt, job 744126).

Why bounded (not refit_pf's unbounded multistart): Warren/Tinker have flat
amplitude/scale degeneracies (a0->~200, a2->~1e5) that an UNBOUNDED optimiser
exploits for a marginally lower NLL but a HIGHER codelen and a WORSE total DL.
The established analysis imposes physical bounds, keeping these forms in their
physical basin (and at their DL minimum). S-T/Jenkins are smooth and stable.

Codelen / NLL / function evaluation all reuse refit_pf's EXACT routines (the same
ones used for the ESR functions), so DL is byte-for-byte comparable. Each form is
given in its constant-free ESR tree, so codelen AND aifeyn come from the SAME tree
(removing the old S-T "2062" bug). Outputs literature_refit_combined.txt (magnitude
convention; +complexity col) + literature_refit_persim.txt, with dDL/dNLL printed
relative to the fresh ESR best.

Verification: War/Tin codelen is recomputed at the STORED per-sim ML params
(literature_fits_fiducial.txt) -> if it matches the stored codelen, there is no
artifact and we keep those fits. S-T/Jen are (re)fit bounded across all 100 sims
(no stored per-sim params existed).
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
import refit_pf as R                       # identical poisson_nll / compute_codelen / make_callable / load_hmf_trimmed
from multiprocessing import Pool

N_SIMS   = 100
ALL_SIMS = list(range(N_SIMS))

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
# Comp. column = the paper's reported display complexities (P.Sch 10, War 14, Tin 16,
# Jen 13, S-T 24); these match Table 3 / the Pareto plot's hardcoded lit_comp and are
# kept for consistency (the DL depends only on aifeyn, which is verified, not on this int).

SEED = {'Jen.': np.array([0.31840, 0.59970, 3.29400]),
        'S-T.': np.array([0.35114, 1.82880, 0.22900])}


def load_stored_persim():
    """Stored per-sim ML params for Warren/Tinker (the bounded fits in the paper)."""
    d = {'War.': {}, 'Tin.': {}}
    with open('literature_fits_fiducial.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.strip().split(';')
            if p[0] in d and len(p) > 5 and p[5] != 'none':
                d[p[0]][int(p[1])] = np.array([float(v) for v in p[5].split()])
    return d


def fit_bounded(fc, bounds, p0, sig, cnt, nrm, rng, n_restarts=25):
    """Bounded multistart Poisson-NLL fit (physical basin)."""
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


def process(name):
    info = LIT[name]; fc = R.make_callable(info['esr']); aif = info['aifeyn']; npar = info['nparam']
    rng = np.random.RandomState(42)

    if npar == 0:                                   # parameter-free
        per_sim = []
        for sim in ALL_SIMS:
            sig, cnt, nrm = SIM[sim]
            nll = R.poisson_nll(np.array([]), sig, cnt, nrm, fc)
            if np.isfinite(nll):
                per_sim.append((sim, nll, 0.0, []))
        return name, per_sim

    per_sim = []
    for sim in ALL_SIMS:
        sig, cnt, nrm = SIM[sim]
        if name in STORED:                          # War/Tin: codelen at the STORED bounded ML params (no re-opt:
            p = STORED[name].get(sim)               #   re-optimising drifts toward the a0/a2 degeneracy -> worse DL)
            if p is None:
                continue
            nll = R.poisson_nll(p, sig, cnt, nrm, fc)
        else:                                       # S-T/Jen: bounded fit from canonical seed
            nll, p = fit_bounded(fc, info['bounds'], SEED[name], sig, cnt, nrm, rng)
        if p is None or not np.isfinite(nll):
            continue
        cl = R.compute_codelen(p, sig, cnt, nrm, fc)
        if not np.isfinite(cl):
            continue
        per_sim.append((sim, nll, cl, list(np.asarray(p))))
    return name, per_sim


SIM = {}
STORED = {}


def main():
    global SIM, STORED
    print(f"Literature verify+refit: {len(LIT)} funcs x {N_SIMS} sims (bounded)", flush=True)
    SIM = {s: R.load_hmf_trimmed(s) for s in ALL_SIMS}
    STORED = load_stored_persim()

    with Pool(min(5, len(LIT))) as pool:
        out = pool.map(process, list(LIT))

    # fresh ESR best (rank-0): magnitudes
    with open('pf_refit_combined.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.split(';')
            best_DL_mag  = abs(float(cols[3]))     # DL_combined
            best_NLL_mag = abs(float(cols[5]))     # sum_NLL
            break

    rows, persim = [], []
    for name, per_sim in out:
        aif = LIT[name]['aifeyn']; comp = LIT[name]['comp']; n = len(per_sim)
        sum_nll = sum(r[1] for r in per_sim)           # actual (negative)
        sum_cl  = sum(r[2] for r in per_sim)
        sum_dl_actual      = sum_nll + sum_cl + n * aif
        DL_combined_actual = sum_dl_actual - (n - 1) * aif
        DL_mag  = -DL_combined_actual
        NLL_mag = -sum_nll
        dDL  = best_DL_mag  - DL_mag
        dNLL = best_NLL_mag - NLL_mag
        rows.append([name, DL_mag, NLL_mag, -sum_dl_actual, aif, n, comp, sum_cl, dDL, dNLL])
        for (sim, nll, cl, p) in per_sim:
            pstr = ' '.join(f'{v:.10f}' for v in p) if len(p) else 'none'
            persim.append(f"{name};{sim};{-(nll + cl + aif):.6f};{-nll:.6f};{cl:.6f};{pstr}")

    with open('literature_refit_combined.txt', 'w') as f:
        f.write('# name;DL_combined;sum_NLL;sum_DL;aifeynman;n_sims;complexity\n')
        for r in rows:
            f.write(f"{r[0]};{r[1]:.6f};{r[2]:.6f};{r[3]:.6f};{r[4]:.6f};{r[5]};{r[6]}\n")
    with open('literature_refit_persim.txt', 'w') as f:
        f.write('# name;sim;DL;NLL;codelen;params\n')
        f.write('\n'.join(persim) + '\n')

    # stored values for comparison: DL_combined_mag, sum_NLL_mag, sum_DL_mag
    stored = {}
    for fn in ('literature_combined_DL_trimmed.txt', 'literature_st_jenkins_combined.txt'):
        for line in open(fn):
            if line.startswith('#') or not line.strip():
                continue
            c = line.strip().split(';')
            stored[c[0]] = (float(c[1]), float(c[2]), float(c[3]))

    print(f"\n  fresh ESR best: DL_mag={best_DL_mag:.3f}  NLL_mag={best_NLL_mag:.3f}", flush=True)
    print(f"  {'name':7s} {'n':>3} {'codelen_tot':>11} {'dDL(new)':>9} {'dNLL(new)':>9} | "
          f"{'dDL(stored)':>11} {'codelen_stored':>14}", flush=True)
    for r in sorted(rows, key=lambda z: z[8]):
        name = r[0]; n = r[5]; sum_cl = r[7]
        if name in stored:
            s_dlmag, s_nllmag, s_sumdlmag = stored[name]
            s_dDL = best_DL_mag - s_dlmag
            s_cl  = s_nllmag - s_sumdlmag - n * r[4]     # stored total codelen
        else:
            s_dDL, s_cl = float('nan'), float('nan')
        print(f"  {name:7s} {n:>3} {sum_cl:>11.2f} {r[8]:>9.1f} {r[9]:>9.1f} | "
              f"{s_dDL:>11.1f} {s_cl:>14.2f}", flush=True)

    print(f"\n  Saved: literature_refit_combined.txt, literature_refit_persim.txt", flush=True)


if __name__ == '__main__':
    main()
