#!/usr/bin/env python3
"""v2 HMF candidate selection — NO MIN_SIMS (obsoleted by design).

WHY: the v1 selection ranked functions by mean DL-rank over 10 sims and required a function
to be fit in >=MIN_SIMS=5 of them. But the per-sim fits come from ESR's cheap enumeration
optimiser (220 restarts in [0,3]), which SILENTLY FAILS to converge for good functions on
some sims (even rank-1 only "fits" 7/10; candidate forms tied with rank-1 fit 2/10). So "n"
measures cheap-optimiser luck, not quality, and MIN_SIMS drops genuinely-competitive functions
(verified: 4 distinct excluded functions would enter Table 3, incl. one ~rank 3 and two PS-like
better than the showcased one). See log.md 2026-06-02.

FIX (this script):
  1. NOMINATE generously: union of the top-K functions by DL in ANY single sim. A good function
     ranks well in the sims it DOES fit (candidate #1 was cheap rank-0 in sim 0), so the union
     catches good-but-fragile forms. Flukes are caught too -- harmless (tier 2 discards them).
  2. FILL: refit every nominee on ALL 10 selection sims, donor-seeded from the sims where the
     cheap fit succeeded (+ a few wide-random restarts). A seed from a working sim makes the
     other sims converge -> NO FAILED FITS -> every candidate has all 10 sims -> n is no longer
     a discriminator -> MIN_SIMS is unnecessary.
  3. RANK by combined NLL over the 10 sims (uniform n=10, so combining is now valid; codelen is
     a small correction deferred to tier 2). Select top-N per complexity.
  Tier 2 = refit_pf.py refines the ~1000 selected across all 100 sims with the full DL.

Output: pf_candidates.txt  (lines: complexity;blank_func;aifeyn)  -- drop-in for refit_pf.py.
Run on Glamdring (reads the full per-complexity search files hmf_trimmed_{sim}_data/final_{c}_trimmed.dat).
    addqueue -q cmb -n 1x32 -m 4 -s -c "pf v2 select (no MIN_SIMS)" \
        /usr/local/shared/python/3.11.4/bin/python3 select_pf_candidates_v2.py
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

if os.path.isdir('/users/hdesmond/Amelia_code'):
    os.chdir('/users/hdesmond/Amelia_code')

import refit_pf as R   # identical make_callable / poisson_nll / load_hmf_trimmed

SIMS  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
COMPS = [int(c) for c in os.environ.get('PF_COMPS', '6,7,8,9,10').split(',')]
TOPK  = int(os.environ.get('PF_TOPK', 3000))    # per-sim nomination depth (union)
TOPN  = int(os.environ.get('PF_TOPN', 200))     # final selection per complexity
NRAND = int(os.environ.get('PF_FILL_NRAND', 6)) # wide-random restarts per (cand, sim) on top of donors
NPROCS = int(os.environ.get('PF_NPROCS', cpu_count()))


def fmtf(fcn):
    """Canonical form for dedup (verbatim from select_pf_candidates.py / run_hmf_trimmed_step2.py)."""
    fcn = fcn.replace('a0', 'C').replace('a1', 'C').replace('a2', 'C').replace('a3', 'C')
    fcn = fcn.replace('exp(C)', 'C').replace('1/C', 'C').replace('Abs(C + x)', 'Abs(C - x)')
    fcn = fcn.replace('log(Abs(C))', 'C').replace(',(C)', ',C').replace('Abs(1/C)', 'Abs(C)')
    k = 0; new = ''
    for ch in fcn:
        if ch == 'C': new += f'a{k}'; k += 1
        else: new += ch
    return new.replace('-a', 'a')


def nparam(s):
    return sum(1 for i in range(4) if f'a{i}' in s)


def load_topk(c, sim):
    """Top-K functions by DL in (comp c, sim). raw -> (dl, nll, aifeyn, params)."""
    fp = f'hmf_trimmed_{sim}_data/final_{c}_trimmed.dat'
    rows = []
    if not os.path.exists(fp):
        return rows
    with open(fp) as f:
        for line in f:
            p = line.rstrip('\n').split(';')
            if len(p) < 8:
                continue
            try:
                dl = float(p[2]); nll = float(p[4]); aif = float(p[6])
            except (ValueError, IndexError):
                continue
            if not (dl < 0 and nll < 0):
                continue
            np_ = nparam(p[1])
            try:
                pars = [float(p[7 + i]) for i in range(np_)]
            except (ValueError, IndexError):
                pars = []
            rows.append((dl, nll, aif, p[1], pars))
    rows.sort(key=lambda r: r[0])      # ascending DL (best first)
    return rows[:TOPK]


def fit_sim(cand):
    """Fit one candidate on ALL 10 sims, donor-seeded (+wide random); return sum_NLL, n_fit."""
    canon, rep, aif, npar, donors = cand
    fc = R.make_callable(rep)
    rng = np.random.RandomState(12345)
    if npar == 0:
        sn = 0.0; n = 0
        for s in SIMS:
            sig, cnt, nrm = SIM[s]
            nll = R.poisson_nll(np.array([]), sig, cnt, nrm, fc)
            if np.isfinite(nll): sn += nll; n += 1
        return (canon, rep, aif, sn, n)
    sn = 0.0; n = 0
    for s in SIMS:
        sig, cnt, nrm = SIM[s]
        best = np.inf
        starts = [np.asarray(d, float) for d in donors if len(d) == npar]
        starts += [rng.uniform(-5, 5, npar) for _ in range(NRAND)]
        for s0 in starts:
            try:
                r = minimize(R.poisson_nll, s0, args=(sig, cnt, nrm, fc),
                             method='L-BFGS-B', options={'maxiter': 10000, 'ftol': 1e-14})
                if np.isfinite(r.fun) and r.fun < best:
                    best = r.fun
            except Exception:
                pass
        if np.isfinite(best):
            sn += best; n += 1
    return (canon, rep, aif, sn, n)


SIM = {}


def main():
    global SIM
    print(f"v2 select: COMPS={COMPS} TOPK={TOPK} TOPN={TOPN} NRAND={NRAND} NPROCS={NPROCS}", flush=True)
    SIM = {s: R.load_hmf_trimmed(s) for s in SIMS}
    out = []
    for c in COMPS:
        # nominate: union of top-K per sim
        buckets = {}   # canon -> dict(rep, repnll, aif, npar, donors[])
        for s in SIMS:
            for (dl, nll, aif, raw, pars) in load_topk(c, s):
                cf = fmtf(raw); b = buckets.get(cf)
                if b is None:
                    buckets[cf] = dict(rep=raw, repnll=nll, aif=aif, npar=nparam(raw), donors=[pars] if pars else [])
                else:
                    if pars: b['donors'].append(pars)
                    if nll < b['repnll']:
                        b['repnll'] = nll; b['rep'] = raw; b['aif'] = aif; b['npar'] = nparam(raw)
        cands = [(cf, b['rep'], b['aif'], b['npar'], b['donors'][:12]) for cf, b in buckets.items()]
        print(f"comp {c}: nominated {len(cands)} unique candidates (union of top-{TOPK}/sim)", flush=True)
        # fill: fit every candidate on all 10 sims (no fails by donor-seeding)
        with Pool(NPROCS) as pool:
            res = pool.map(fit_sim, cands, chunksize=8)
        # rank by combined NLL over the 10 sims (uniform n; no MIN_SIMS)
        res = [r for r in res if r[4] >= 1]
        res.sort(key=lambda r: r[3])     # ascending sum_NLL (most negative = best)
        nfull = sum(1 for r in res if r[4] == len(SIMS))
        sel = res[:TOPN]
        print(f"comp {c}: filled {len(res)} ({nfull} with all 10 sims); selected top-{len(sel)} "
              f"(min n_fit among selected = {min(r[4] for r in sel)})", flush=True)
        for (canon, rep, aif, sn, n) in sel:
            out.append(f"{c};{rep};{aif:.10g}")

    with open('pf_candidates.txt', 'w') as f:
        f.write("# complexity;blank_func;aifeyn  (v2: union-nominate + donor-fill, NO MIN_SIMS; rank by combined NLL over 10 sims)\n")
        f.write('\n'.join(out) + '\n')
    print(f"\nWrote pf_candidates.txt: {len(out)} functions", flush=True)


if __name__ == '__main__':
    main()
