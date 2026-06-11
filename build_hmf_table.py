#!/usr/bin/env python3
"""Rebuild the fiducial HMF Table 3 (tab:HMF_functions) from the fresh authoritative
refit pf_refit_combined.txt (job 744126), with the codelen artifact fixed.

Steps:
  1. Load full-100-sim functions from pf_refit_combined.txt.
  2. Dedup algebraically-equivalent tree representations by combined sum_NLL
     (equivalent reps share sum_NLL to <1e-3; distinct funcs differ by >>1 nat),
     keeping the min-DL representative of each group.
  3. Re-rank by DL_combined; dDL/dNLL relative to rank-1.
  4. Classify PS-like (sigma*f -> const) using sim-50 params (pf_refit_sim50.txt),
     same _check_ps_like as the Pareto plot.
  5. Per-function theta summary (median + 16th/84th pctile) from pf_refit_persim.txt,
     folding exact sign/reciprocal degeneracies onto one branch.
  6. HMF physicality checks (f->0 as sigma->inf; finite as sigma->0+; 0<int f dlnsigma<=1).
  7. Print the ranking + the literature block (from literature_refit_combined.txt).
"""
import os, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import refit_pf as R
from collections import defaultdict
from Pareto_plotter import _check_ps_like      # identical PS-like test as Fig 2

DEDUP_TOL = 0.05     # nat; group functions whose combined sum_NLL match to within this


def load_combined():
    rows = []
    with open('pf_refit_combined.txt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            c = line.rstrip('\n').split(';')
            rows.append(dict(func=c[1], comp=int(c[2]), DL=float(c[3]),
                             sumNLL=float(c[5]), aifeyn=float(c[7]), nsims=int(c[8])))
    return [r for r in rows if r['nsims'] == 100]


def dedup(rows):
    """Keep the min-DL representative per group of equal-sum_NLL (equivalent) forms."""
    rows = sorted(rows, key=lambda r: r['sumNLL'])
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if abs(r['sumNLL'] - cur[-1]['sumNLL']) <= DEDUP_TOL:
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    reps = []
    for g in groups:
        rep = min(g, key=lambda r: r['DL'])
        rep = dict(rep)
        rep['nequiv'] = len(g)
        rep['equiv'] = [x['func'] for x in g if x['func'] != rep['func']]
        reps.append(rep)
    reps.sort(key=lambda r: r['DL'])
    return reps


def load_sim50():
    d = {}
    for l in open('pf_refit_sim50.txt'):
        p = l.strip().split(';')
        if len(p) >= 6:
            try:
                d[p[0]] = [float(v) for v in p[5].split()]
            except ValueError:
                pass
    return d


def load_persim_params():
    d = defaultdict(list)
    for l in open('pf_refit_persim.txt'):
        if l.startswith('#') or not l.strip():
            continue
        p = l.strip().split(';')
        if len(p) >= 6 and p[5] not in ('none', 'nan', ''):
            try:
                d[p[0]].append([float(v) for v in p[5].split()])
            except ValueError:
                pass
    return d


def theta_summary(persim, func):
    """Median + 16th/84th-percentile errorbars across the 100 per-sim ML fits."""
    arr = np.array(persim.get(func, []))
    if arr.size == 0:
        return None
    arr = fold_degenerate_params(func, arr)
    out = []
    for j in range(arr.shape[1]):
        v = arr[:, j]
        med = np.median(v)
        out.append((med, np.percentile(v, 84) - med, med - np.percentile(v, 16)))
    return out


def fold_degenerate_params(func, params):
    """Fold exact parameter branches before reporting cross-realisation scatter.

    ESR's Abs wrappers make e.g. Abs(a0) invariant under a0 -> -a0.  Likewise
    Abs(log(Abs(a0))) is invariant under |a0| -> 1/|a0|.  The optimiser may land
    on either branch in different realisations, so raw percentiles can report a
    spurious two-branch scatter even though the fitted function is identical.
    """
    arr = np.array(params, dtype=float, copy=True)
    if arr.ndim != 2:
        return arr
    for j in range(arr.shape[1]):
        name = f'a{j}'
        if f'Abs(log(Abs({name})))' in func or f'Abs(log(Abs(1/{name})))' in func:
            v = np.abs(arr[:, j])
            with np.errstate(divide='ignore', invalid='ignore'):
                arr[:, j] = np.minimum(v, 1.0 / v)
        elif f'Abs({name})' in func or f'Abs(1/{name})' in func:
            arr[:, j] = np.abs(arr[:, j])
    return arr


def physicality(func, params):
    """HMF physicality checks (mirrors fit_new_literature.check_hmf_physicality,
    the implementation that defines the Table 3 a/b/c codes). Returns
    (passes, code_str, mass_fraction). Accepts slow 1/sigma (PS-like) decay."""
    fc = R.make_callable(func)
    p = np.asarray(params, float)
    codes = []
    # f -> 0 as sigma -> inf (evaluated far out so 1/sigma PS-like decay counts)
    f_large = fc(p, np.array([1e3, 1e5, 1e8]))
    if not np.all(np.isfinite(f_large)) or abs(f_large[-1]) > 1e-3:
        codes.append('b')                              # tends to a non-zero constant / doesn't vanish
    if np.any(f_large < -1e-10):
        codes.append('a')                              # goes negative
    # finite as sigma -> 0+
    f_small = fc(p, np.array([0.01, 0.05, 0.1]))
    if np.any(~np.isfinite(f_small)) or np.any(np.abs(f_small) > 1e20):
        codes.append('c')                              # diverges as sigma -> 0+
    # integrability: 0 < int f dln sigma  (collapsed mass fraction)
    xg = np.logspace(-3, 3, 100000)
    yg = fc(p, xg)
    yg = np.where(np.isfinite(yg), yg, 0.0)
    integral = np.trapz(yg / xg, xg)
    if (not np.isfinite(integral)) or integral < 0:
        if 'a' not in codes:
            codes.append('a')
    # de-dup, keep order a,b,c
    codes = [c for c in ('a', 'b', 'c') if c in codes]
    if not codes:
        return True, '', integral
    return False, ''.join(f'({c})' for c in codes), integral


def main():
    rows = load_combined()
    reps = dedup(rows)
    sim50 = load_sim50()
    persim = load_persim_params()
    best_DL  = reps[0]['DL']
    best_NLL = reps[0]['sumNLL']

    print(f"Loaded {len(rows)} full-100-sim funcs -> {len(reps)} unique after sum_NLL dedup "
          f"(tol {DEDUP_TOL})\n")

    # multi-member dedup groups in the top region (sanity)
    print("Dedup groups with >1 member in top 30:")
    for r in reps[:30]:
        if r['nequiv'] > 1:
            print(f"  rank kept '{r['func'][:48]}'  (+{r['nequiv']-1} equiv, e.g. {r['equiv'][0][:48]})")
    print()

    print(f"{'rk':>3} {'comp':>4} {'dDL':>8} {'dNLL':>8} {'PS':>3} {'chk':>4} {'mass':>6}  function")
    print('-' * 110)
    table = []
    for i, r in enumerate(reps[:40]):
        fn = r['func']
        p50 = sim50.get(fn)
        ps = bool(p50 is not None and _check_ps_like(fn, p50))
        if p50 is not None:
            ok, code, integ = physicality(fn, p50)
        else:
            ok, code, integ = None, '?', float('nan')
        dDL = r['DL'] - best_DL
        dNLL = r['sumNLL'] - best_NLL
        chk = '\\checkmark' if ok else code
        table.append(dict(rank=i+1, func=fn, comp=r['comp'], dDL=dDL, dNLL=dNLL,
                          ps=ps, ok=ok, code=code, mass=integ, theta=theta_summary(persim, fn)))
        print(f"{i+1:>3} {r['comp']:>4} {dDL:>8.1f} {dNLL:>8.1f} {str(ps):>3} {chk:>4} {integ:>6.3f}  {fn[:60]}")

    # theta summaries for the top non-PS-like passing functions + first PS-like passing
    print("\n\nTHETA SUMMARIES (median +84/-16, folded exact degeneracies):")
    for t in table[:25]:
        if t['theta']:
            ts = ', '.join(f"t{j}={m:.3f}(+{hi:.3f}/-{lo:.3f})" for j,(m,hi,lo) in enumerate(t['theta']))
        else:
            ts = '(none)'
        print(f"  rank {t['rank']:>2} ps={int(t['ps'])} chk={t['code'] or 'ok'}: {ts}   <- {t['func'][:46]}")

    # literature block
    print("\n\nLITERATURE (relative to fresh best):")
    bestmag = abs(best_DL); bestnllmag = abs(best_NLL)
    for l in open('literature_refit_combined.txt'):
        if l.startswith('#') or not l.strip():
            continue
        c = l.strip().split(';')
        name, dlmag, nllmag, comp = c[0], float(c[1]), float(c[2]), c[6]
        print(f"  {name:8s} comp {comp:>2}  dDL={bestmag-dlmag:>12.1f}  dNLL={bestnllmag-nllmag:>12.1f}")


if __name__ == '__main__':
    main()
