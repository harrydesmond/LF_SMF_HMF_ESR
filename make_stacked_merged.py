#!/usr/bin/env python3
"""Regenerate the HMF stacked-rank bar charts (Fig 6 = fig:HMF_comp and the
appendix Fig B4 = fig:HMF_comp_full) with algebraically-equivalent function
representations MERGED into single bars, so the chart's function *set* matches the
deduped v2 tables (Table 3 / Table B1), and with shared functions labelled in the
v2 table representation.

Why merge: the per-sim ESR search lists several equivalent tree representations of
the same function (identical per-sim NLL to ~1e-6, differing only in codelen). The
old chart counted each representation as a separate bar (e.g. fiducial D/E/M were all
the same function = v2 rank 21). The v2 tables dedup these; the chart now does too.

Method (mirrors build_hmf_table dedup, applied per realisation):
  1. Per sim, load top-200 (template, DL, NLL) from the per-sim ESR file.
  2. Build global equivalence classes by per-sim NLL agreement (union-find).
  3. Per sim, collapse equivalent reps and re-rank by DL so 5 DISTINCT functions
     fill the top-5; tally rank-1..5 counts per class.
  4. Order classes by total ranks earned (then by 1st, 2nd, ...), label A,B,...
  5. Label: classes matching a v2 combined-pool rep (matched by per-sim NLL) use the
     pool's canonical (min-DL) v2 representation; others use their own min-DL rep.

Outputs (PNG when CHECK=1 for diagnostics, else the manuscript PDFs).
"""
import os, sys, warnings, string
warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')
import setup_paths  # noqa: F401 — ensures Plots/ and Final_Plots/ exist
import numpy as np
from matplotlib import pyplot as plt
from pytexit import py2tex

NLL_TOL = 1e-3          # nat; equiv reps agree to ~2e-9, distinct funcs differ by >~1e-2
DEDUP_TOL = 0.05        # nat; build_hmf_table grouping on the 100-sim sum_NLL
CHECK = os.environ.get('CHECK', '0') == '1'


# ----------------------------------------------------------------------------- #
#  Load per-sim ESR search results (the chart's data source)
# ----------------------------------------------------------------------------- #
def load_persim_figure(persim_name):
    """sim -> list of (template, DL, NLL), top-200 as written by the ESR search."""
    sim_rows = {}
    for sim in range(100):
        path = f'hmf_data/hmf_{sim}_data/{persim_name}'
        try:
            rows = []
            with open(path) as fh:
                for line in fh:
                    p = line.strip().split(';')
                    if len(p) < 4:
                        continue
                    rows.append((p[1], float(p[2]), float(p[3])))  # template, DL, NLL
            sim_rows[sim] = rows
        except FileNotFoundError:
            continue
    return sim_rows


# ----------------------------------------------------------------------------- #
#  Global equivalence classes via per-sim NLL agreement (union-find)
# ----------------------------------------------------------------------------- #
class UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(sim_rows):
    """Return (group_of: template->class_id, members: class_id->set, info per class).

    Equivalence rule: two templates are the same function iff they agree in per-sim
    NLL to <=NLL_TOL on EVERY sim where both appear (a clique test, not a single
    transitive link). Equivalent reps agree to ~2e-9 on all shared sims; distinct
    functions differ by >~1e-2 on essentially every sim, so they can never pass.
    Candidate pairs are seeded from per-sim anchor-bounded NLL clusters (cheap), then
    each candidate is verified globally before being unioned -- this prevents the
    cross-sim chaining that single-link union-find suffers from.
    """
    nll_by = {}                       # template -> {sim: nll}
    for sim, rows in sim_rows.items():
        for t, dl, nll in rows:
            nll_by.setdefault(t, {})[sim] = nll

    cand = set()                      # candidate equivalent pairs (seed)
    for sim, rows in sim_rows.items():
        s = sorted(rows, key=lambda r: r[2])  # by NLL
        i = 0
        while i < len(s):
            anchor = s[i][2]
            j = i + 1
            while j < len(s) and (s[j][2] - anchor) <= NLL_TOL:
                a, b = s[i][0], s[j][0]
                if a != b:
                    cand.add((a, b) if a < b else (b, a))
                j += 1
            i = j

    uf = UF()
    for t in nll_by:
        uf.find(t)
    for a, b in cand:
        shared = set(nll_by[a]) & set(nll_by[b])
        if shared and all(abs(nll_by[a][s] - nll_by[b][s]) <= NLL_TOL for s in shared):
            uf.union(a, b)
    # canonical class id = representative template with the lowest DL anywhere
    best_dl = {}   # template -> min DL seen
    for rows in sim_rows.values():
        for t, dl, nll in rows:
            if t not in best_dl or dl < best_dl[t]:
                best_dl[t] = dl
    members = {}
    for t in uf.p:
        root = uf.find(t)
        members.setdefault(root, set()).add(t)
    # choose display canonical = min-DL member; class id keep as root
    canon = {}
    for root, mem in members.items():
        canon[root] = min(mem, key=lambda t: best_dl[t])
    group_of = {t: uf.find(t) for t in uf.p}
    # validation: confirm members of a class never disagree in NLL within a sim by >TOL
    bad = 0
    for sim, rows in sim_rows.items():
        by_t = {t: nll for t, dl, nll in rows}
        for root, mem in members.items():
            present = [by_t[t] for t in mem if t in by_t]
            if len(present) > 1 and (max(present) - min(present)) > NLL_TOL:
                bad += 1
    return group_of, members, canon, bad


# ----------------------------------------------------------------------------- #
#  Per-sim merged top-5 tally
# ----------------------------------------------------------------------------- #
def count_merged(sim_rows, group_of):
    from collections import defaultdict
    counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    classNLL = {}    # class -> {sim: nll}
    for sim, rows in sim_rows.items():
        s = sorted(rows, key=lambda r: r[1])  # by DL (best first)
        seen, rank = set(), 0
        for t, dl, nll in s:
            g = group_of[t]
            classNLL.setdefault(g, {}).setdefault(sim, nll)
            if g in seen:
                continue
            counts[g][rank] += 1
            seen.add(g)
            rank += 1
            if rank == 5:
                break
    return counts, classNLL


# ----------------------------------------------------------------------------- #
#  v2 combined-pool dedup (for v2 ranks + canonical labels)
# ----------------------------------------------------------------------------- #
def load_pool(combined_file):
    rows = []
    with open(combined_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            c = line.rstrip('\n').split(';')
            rows.append(dict(func=c[1], comp=int(c[2]), DL=float(c[3]),
                             sumNLL=float(c[5]), nsims=int(c[8])))
    rows = [r for r in rows if r['nsims'] == 100]
    rows.sort(key=lambda r: r['sumNLL'])
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if abs(r['sumNLL'] - cur[-1]['sumNLL']) <= DEDUP_TOL:
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    reps = []
    for g in groups:
        rep = min(g, key=lambda r: r['DL']); rep = dict(rep)
        rep['members'] = [x['func'] for x in g]
        reps.append(rep)
    reps.sort(key=lambda r: r['DL'])
    for i, r in enumerate(reps):
        r['rank'] = i + 1
    return reps


def pool_persim_nll(persim_pool_file, want_funcs, ref_sims):
    """{func: {sim: nll}} restricted to reps and reference sims."""
    out = {}
    want = set(want_funcs)
    with open(persim_pool_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.rstrip('\n').split(';')
            if len(p) < 4:
                continue
            fn, sim = p[0], int(p[1])
            if fn in want and sim in ref_sims:
                try:
                    out.setdefault(fn, {})[sim] = float(p[3])
                except ValueError:
                    pass
    return out


def match_rank(classNLL_g, reps, pool_nll):
    """Match a class to a v2 rep by per-sim NLL agreement; return rep or None."""
    # reference sims where this class appears, pick a few
    g_sims = list(classNLL_g.keys())
    for s in (50, 0, 25, 75, 99):
        if s in g_sims:
            target = classNLL_g[s]
            for r in reps:
                pn = pool_nll.get(r['func'], {})
                if s in pn and abs(pn[s] - target) <= NLL_TOL:
                    return r
    return None


# ----------------------------------------------------------------------------- #
#  Label rendering (identical style to make_trimmed_plots / histogram_and_stacked)
# ----------------------------------------------------------------------------- #
def render(tmpl):
    text_fcn = py2tex(tmpl.replace('Abs', 'abs')).replace('$', '').replace(' ', '')
    if r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}' in text_fcn:
        text_fcn = text_fcn.replace(r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}',
                                    r'e^{a0-\left({\frac{|a1|}{x}}\right)^{\left(a2-x\right)}}')
    elif r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}' in text_fcn:
        text_fcn = text_fcn.replace(r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}',
                                    r'{|a0|}^{\eXp{\left[\frac{e^{x^{a1}}}{x}\right]}}')
    text_fcn = (text_fcn.replace('a0', r'\theta_0').replace('a1', r'\theta_1')
                        .replace('a2', r'\theta_2').replace('a3', r'\theta_3')
                        .replace('x', r'\sigma').replace('eXp', 'exp'))
    return text_fcn.replace('e^', ' e^')


# ----------------------------------------------------------------------------- #
#  Build one chart
# ----------------------------------------------------------------------------- #
def build_chart(tag, persim_name, combined_file, persim_pool_file, out_pdf, LABEL_MODE='pool'):
    print(f'\n===== {tag} =====')
    sim_rows = load_persim_figure(persim_name)
    print(f'loaded {len(sim_rows)} sims')
    group_of, members, canon, bad = build_groups(sim_rows)
    if bad:
        print(f'  !! WARNING: {bad} class/sim NLL inconsistencies (tol {NLL_TOL})')
    counts, classNLL = count_merged(sim_rows, group_of)

    items = []
    for g, c in counts.items():
        items.append((sum(c), c, g))
    items.sort(key=lambda x: (-x[0], -x[1][0], -x[1][1], -x[1][2], -x[1][3]))
    total_all = sum(it[0] for it in items)
    print(f'classes appearing in top-5: {len(items)};  total ranks earned = {total_all} (expect 500)')

    reps = load_pool(combined_file)
    pool_nll = pool_persim_nll(persim_pool_file, [r['func'] for r in reps],
                               {0, 25, 50, 75, 99})

    items = items[:26]
    letters = string.ascii_uppercase
    labels, ranks = [], []
    print(f"{'L':>2} {'tot':>4} {'1st':>4} {'v2rank':>7}  label-source")
    for idx, (tot, c, g) in enumerate(items):
        rep = match_rank(classNLL[g], reps, pool_nll)
        rk = rep['rank'] if rep is not None else None
        pool_lab = render(rep['func']) if rep is not None else '(no pool match)'
        own_lab = render(canon[g])
        if LABEL_MODE == 'pool' and rep is not None:
            lab = pool_lab
        else:
            lab = own_lab
        labels.append(lab); ranks.append(rk)
        print(f"{letters[idx]:>2} {tot:>4} {c[0]:>4} {str(rk):>7}  POOL[{rep['func'][:34] if rep else '-'}]  OWN[{canon[g][:34]}]")

    # ---- plot ----
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#6E7B8B', '#445577']
    subs = ['1st', '2nd', '3rd', '4th', '5th']
    n = len(items)
    xp = np.arange(n)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
    ax, ax2 = axs
    bottom = np.zeros(n)
    for i, (col, sub) in enumerate(zip(colors, subs)):
        y = np.array([it[1][i] for it in items])
        ax.bar(xp, y, bottom=bottom, label=sub, color=col)
        bottom += y
    ax.set_xlabel('Functions')
    ax.set_xticks(xp); ax.set_xticklabels(list(letters)[:n])
    ax.set_ylabel('Number of Ranks Earned (in 100 simulations)')
    ax.legend(subs)
    for idx, lab in enumerate(labels):
        ax2.text(0.01, (n - idx) / n, letters[idx] + ': ', fontsize=10, ha='left', va='center')
        ax2.text(0.2, (n - idx) / n, r'${}$'.format(lab), fontsize=10, ha='left', va='center')
    ax2.axis('off')
    plt.tight_layout()
    if CHECK:
        png = f'Plots/stacked_merged_{tag}.png'
        plt.savefig(png, bbox_inches='tight', dpi=150)
        print(f'  saved {png}')
    else:
        plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
        print(f'  saved {out_pdf}')
    plt.close(fig)
    return items, labels, ranks


if __name__ == '__main__':
    # Fiducial Table 3 is built by build_hmf_table from the combined pool, so pool-rep
    # labels reproduce its forms.  Table B1 was built on the diagonal codelen via a
    # different pipeline whose forms match the per-sim min-DL ("own") representations.
    # Each chart is independent; a missing dataset only skips its own figure.
    JOBS = [
        ('fiducial', 'final_all_trimmed.txt', 'pf_refit_combined.txt',
         'pf_refit_persim.txt', 'Final_Plots/stacked_rank_fiducial.pdf', 'pool'),
        ('extended', 'final_all.txt', 'pf_refit_combined_extended.txt',
         'pf_refit_persim_extended.txt', 'Final_Plots/stacked_rank.pdf', 'own'),
    ]
    for tag, persim, combined, persimpool, out, mode in JOBS:
        try:
            build_chart(tag, persim, combined, persimpool, out, LABEL_MODE=mode)
        except FileNotFoundError as e:
            print(f"  [skip {tag}] missing input ({e}); run the {tag} refit first.")
