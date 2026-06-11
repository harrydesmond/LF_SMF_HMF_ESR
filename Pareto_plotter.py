import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')
from esr.generation.generator import string_to_node

# ── PS-like detection (from ps_like_pareto.py) ──────────────────────────────
BASIS = [["x", "a"], ["inv", "exp", "log", "abs"], ["+", "*", "-", "/", "pow"]]


def _eval_fcn(func_str, x, params):
    s = func_str.replace('pow', 'np.power').replace('exp', 'np.exp')
    s = s.replace('log', 'np.log').replace('Abs', 'np.abs')
    for i, p in enumerate(params):
        s = s.replace(f'a{i}', str(p))
    # CLAUDEADD 2026-06-10: replace the variable x ONLY as a whole word; the old bare
    # s.replace('x','x_val') also corrupted the 'x' inside 'np.exp' -> 'np.ex_valp' -> eval
    # -> nan, so EVERY function containing exp() was misclassified as non-PS-like (e.g. the
    # PS-like e^{t0/sigma^2}/(t1+sigma) was wrongly rejected). See log.md 2026-06-10.
    s = re.sub(r'\bx\b', 'x_val', s)
    x_val = x
    with np.errstate(all='ignore'):
        try:
            return float(eval(s))
        except Exception:
            return np.nan


def _check_ps_like(func_str, params, sigma_vals=(100, 1000, 10000)):
    """Return True if f(sigma) ~ 1/sigma at large sigma."""
    f_vals, products = {}, {}
    for s in sigma_vals:
        f = _eval_fcn(func_str, float(s), params)
        if not np.isfinite(f) or f <= 0:
            return False
        f_vals[s] = f
        products[s] = s * f
    prods = [products[s] for s in sigma_vals]
    if any(p <= 0 for p in prods):
        return False
    ratio_1, ratio_2 = prods[1] / prods[0], prods[2] / prods[1]
    converging = (0.1 < ratio_1 < 10) and (0.1 < ratio_2 < 10)
    f_decaying = f_vals[1000] < f_vals[100] and f_vals[10000] < f_vals[1000]
    if f_vals[100] > 0 and f_vals[1000] > 0:
        alpha_est = -np.log(f_vals[1000] / f_vals[100]) / np.log(10)
    else:
        return False
    return converging and f_decaying and np.isfinite(alpha_est) and (0.5 < alpha_est < 1.5)


def load_ps_like_for_hmf(sim=50, variant='extended'):
    """Load PS-like function data for the HMF panel.

    ``variant`` selects the fiducial (16-bin) or extended (18-bin) per-sim data.

    Returns list of dicts with keys: func, complexity, DL, NLL, ps_like.
    DL and NLL are for the given sim (not combined).
    Complexity uses ESR generation complexity (from hmf_func_gencomp.txt),
    NOT string_to_node which can disagree with generation complexity
    due to sympy simplification changing the tree structure.
    """
    # Load generation complexity mapping
    gencomp_map = {}
    try:
        with open('hmf_func_gencomp.txt') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(';')
                gencomp_map[parts[0]] = int(parts[1])
    except FileNotFoundError:
        pass

    # Load per-sim data for all 200 functions (final_all_trimmed.txt = fiducial per-sim data)
    final_all = (f'hmf_data/hmf_{sim}_data/final_all_trimmed.txt' if variant == 'fiducial'
                 else f'hmf_data/hmf_{sim}_data/final_all.txt')
    sim_data = {}  # func_str -> (DL, NLL, params)
    with open(final_all) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            func_str = parts[1]
            dl = float(parts[2])
            nll = float(parts[3])
            param_str = parts[4].strip().strip('[]')
            params = [float(p) for p in param_str.split()]
            sim_data[func_str] = (dl, nll, params)

    # Load combined ranking for rank info
    combined_file = ('hmf_combined_DL_fiducial.txt' if variant == 'fiducial'
                     else 'hmf_combined_DL.txt')
    results = []
    with open(combined_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            func_str = parts[1]
            if func_str not in sim_data:
                continue
            dl, nll, params = sim_data[func_str]
            comp = gencomp_map.get(func_str, 10)
            is_ps = _check_ps_like(func_str, params)
            results.append({
                'func': func_str,
                'complexity': comp,
                'DL': dl,
                'NLL': nll,
                'ps_like': is_ps,
            })
    return results
def coloured_ylabel(fig, ax, fontsize=11, x_offset=0.05, y_offset=None):
    """Set a two-colour y-axis label: blue ΔDL / red ΔNLL."""
    box = ax.get_position()
    x = box.x0 - x_offset
    y_mid = 0.5 * (box.y0 + box.y1)
    if y_offset is None:
        fig_h = fig.get_size_inches()[1]
        y_offset = (fontsize / 72.0) / fig_h * 1.0
    fig.text(x, y_mid + y_offset, r'$\Delta$DL', color='C0', fontsize=fontsize,
             rotation=90, ha='center', va='bottom')
    fig.text(x, y_mid, '/', color='black', fontsize=fontsize,
             rotation=90, ha='center', va='center')
    fig.text(x, y_mid - y_offset, r'$\Delta$NLL', color='C3', fontsize=fontsize,
             rotation=90, ha='center', va='top')


def load_pareto_data(data_set, variant='extended'):
    """Load DL and NLL from final_functions file.

    For the HMF, ``variant`` selects the fiducial (16-bin, main text) or
    extended (18-bin, Appendix B) ranking. Ignored for LF/SMF datasets.
    """
    if data_set == 'hmf_50':
        fname = ('hmf_50_final_functions_fiducial.txt' if variant == 'fiducial'
                 else 'hmf_50_final_functions_extended.txt')
    else:
        fname = '{}_final_functions.txt'.format(data_set)
    source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt(
        fname, dtype=str, delimiter=';', unpack=True)
    DL = DL.astype(float)
    NLL = NLL.astype(float)
    return source, comp.astype(int), DL, NLL


def split_complexity_segments(comp_values, min_gap=5, pad=0.35):
    """Split complexity values into contiguous segments with breaks at gaps >= min_gap."""
    unique_comp = np.unique(np.asarray(comp_values, dtype=int))
    unique_comp.sort()

    if len(unique_comp) == 0:
        return []

    segments = []
    seg_start = unique_comp[0]
    seg_prev = unique_comp[0]

    for val in unique_comp[1:]:
        if val - seg_prev >= min_gap:
            ticks = [c for c in unique_comp if seg_start <= c <= seg_prev]
            segments.append((seg_start - pad, seg_prev + pad, ticks))
            seg_start = val
        seg_prev = val

    ticks = [c for c in unique_comp if seg_start <= c <= seg_prev]
    segments.append((seg_start - pad, seg_prev + pad, ticks))

    return segments


def apply_lower_x_cut(segments, x_min_cut=None):
    """Apply a lower x cutoff to segmented broken-axis ranges."""
    if x_min_cut is None:
        return segments

    cut_segments = []
    for xmin, xmax, ticks in segments:
        if xmax <= x_min_cut:
            continue
        new_xmin = max(xmin, x_min_cut)
        new_ticks = [tick for tick in ticks if tick >= x_min_cut]
        cut_segments.append((new_xmin, xmax, new_ticks))

    return cut_segments if len(cut_segments) > 0 else segments


def add_break_marks(fig, ax_left, ax_right, dx=0.004, dy=0.007, x_shift=-0.0015,
                    sep=0.006):
    """Draw uniform-angle bottom break marks and keep top axis visually continuous."""
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()

    seam_x = 0.5 * (left_box.x1 + right_box.x0) + x_shift
    y_bottom = left_box.y0
    y_top = left_box.y1

    line1 = Line2D([seam_x - dx, seam_x + dx], [y_bottom - dy, y_bottom + dy],
                   transform=fig.transFigure, color='k', lw=1.0, clip_on=False)
    line2 = Line2D([seam_x + sep - dx, seam_x + sep + dx],
                   [y_bottom - dy, y_bottom + dy],
                   transform=fig.transFigure, color='k', lw=1.0, clip_on=False)
    top_connector = Line2D([left_box.x1, right_box.x0], [y_top, y_top],
                           transform=fig.transFigure, color='k', lw=1.0, clip_on=False)
    fig.add_artist(line1)
    fig.add_artist(line2)
    fig.add_artist(top_connector)


def plot_pareto(ax_list, segments, data_set, datasets_info, ps_like_data=None,
                tick_fontsize=11, legend_fontsize=8, show_lit_legend=False,
                xmax_cap_override=None, variant='extended'):
    """Plot Pareto front for a single dataset onto a list of broken-x sub-axes."""
    source, comp, DL, NLL = load_pareto_data(data_set, variant=variant)

    # ESR Pareto: use 'ESR' entries (per-complexity bests); skip 'ESR_C'/'ESR_T' (overall rankings)
    esr_mask = source == 'ESR'
    esr_c_mask = source == 'ESR_C'
    esr_t_mask = source == 'ESR_T'
    esr_comp = comp[esr_mask]
    esr_DL = DL[esr_mask]
    esr_NLL = NLL[esr_mask]

    # Normalise relative to overall best DL (ESR or literature)
    best_idx = np.argmin(DL)
    DL_ref = DL[best_idx]
    NLL_ref = NLL[best_idx]

    esr_DL_rel = esr_DL - DL_ref
    esr_NLL_rel = esr_NLL - NLL_ref

    sort_idx = np.argsort(esr_comp)
    esr_comp = esr_comp[sort_idx]
    esr_DL_rel = esr_DL_rel[sort_idx]
    esr_NLL_rel = esr_NLL_rel[sort_idx]

    # Literature functions (exclude ESR, ESR_C, ESR_T)
    lit_mask = ~(esr_mask | esr_c_mask | esr_t_mask)
    lit_source = source[lit_mask]
    lit_comp = comp[lit_mask]
    lit_DL_rel = DL[lit_mask] - DL_ref
    lit_NLL_rel = NLL[lit_mask] - NLL_ref

    markers = {
        'Sch.': ('x', 'Schechter'),
        'P.Sch.': ('*', 'Press-Sch.'),
        'War.': ('^', 'Warren'),
        'Tin.': ('x', 'Tinker'),
        'Ber.': ('+', 'Bernardi'),
        'Ber. Paper': ('+', 'Bernardi'),
        'Ber.orig': ('1', 'Bernardi (orig.)'),
        'DblSch.': ('d', 'Dbl. Schechter'),
        'Sau.': ('p', 'Saunders'),
        'S-T.': ('h', 'Sheth-Tormen'),
        'Jen.': ('*', 'Jenkins'),
    }

    ds_label = 'LF' if 'LF' in data_set else ('SMF' if 'SMF' in data_set else 'HMF')

    plotted_lit = []

    for i, src in enumerate(lit_source):
        if src in markers:
            marker, name = markers[src]
            key = src.replace(' Paper', '')
            if key not in datasets_info:
                datasets_info[key] = {'marker': marker, 'name': name, 'datasets': set()}
            datasets_info[key]['datasets'].add(ds_label)
            plotted_lit.append((lit_comp[i], lit_DL_rel[i], lit_NLL_rel[i], marker))

    # Compute sensible y-limits excluding extreme outliers
    all_DL_rel = np.concatenate([esr_DL_rel, lit_DL_rel])
    all_NLL_rel = np.concatenate([esr_NLL_rel, lit_NLL_rel])
    all_vals = np.concatenate([all_DL_rel, all_NLL_rel])

    # Exclude extreme values (> 20x median of positive values) for upper limit
    positive = all_vals[all_vals > 0]
    if len(positive) > 2:
        median_val = np.median(positive)
        reasonable = all_vals[(all_vals < 20 * median_val)]
        if len(reasonable) > 0:
            ymax = np.max(reasonable) * 1.15
        else:
            ymax = np.max(all_vals) * 1.15
    else:
        ymax = np.max(np.abs(all_vals)) * 1.15

    # Lower limit: accommodate negative ΔDL (e.g. Bernardi beating ESR)
    ymin_data = np.min(all_vals)
    ymin = min(-ymax * 0.03, ymin_data * 1.15 if ymin_data < 0 else -ymax * 0.03)

    # Per-dataset y-limit overrides
    YMAX_OVERRIDES = {
        'SMF_Ser_M': 600,
        'LF_cmodel_L': 300,
        'SMF_cmodel_M': 600,
        'hmf_50': 150,
    }
    YMIN_OVERRIDES = {
        'SMF_Ser_M': -30,
        'LF_cmodel_L': -35,
    }
    XMAX_OVERRIDES = {
        'hmf_50': 25,  # extend to show Sheth-Tormen at complexity 24
    }
    if data_set in YMAX_OVERRIDES:
        ymax = YMAX_OVERRIDES[data_set]
    if data_set == 'hmf_50' and variant == 'extended':
        ymax = 170  # Fig B2 (extended HMF Pareto): raise upper limit
    if data_set in YMIN_OVERRIDES:
        ymin = YMIN_OVERRIDES[data_set]
    xmax_cap = xmax_cap_override if xmax_cap_override is not None else XMAX_OVERRIDES.get(data_set)
    if xmax_cap is not None:
        segments = [(xmin_s, min(xmax_s, xmax_cap), [t for t in ticks_s if t <= xmax_cap])
                     for xmin_s, xmax_s, ticks_s in segments]

    for ax, (xmin, xmax, ticks) in zip(ax_list, segments):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=tick_fontsize)

        # ESR curves and points
        ax.plot(esr_comp, esr_DL_rel, 'o-', color='C0', ms=5, zorder=5, lw=1.5)
        ax.plot(esr_comp, esr_NLL_rel, 'o-', color='C3', ms=5, zorder=4, lw=1.5)

        # Literature markers
        for x_lit, dl_val, nll_val, marker in plotted_lit:
            in_segment = (xmin <= x_lit <= xmax)
            if not in_segment:
                continue

            # Skip off-scale points entirely
            if ymin <= dl_val <= ymax:
                ax.scatter(x_lit, dl_val, marker=marker, color='C0', s=70, zorder=6,
                           linewidths=1.5)
            if ymin <= nll_val <= ymax:
                ax.scatter(x_lit, nll_val, marker=marker, color='C3', s=70, zorder=6,
                           linewidths=1.5)

        # PS-like overlay (HMF panel only)
        if ps_like_data is not None:
            from collections import defaultdict
            ps_entries = [e for e in ps_like_data
                          if e['ps_like'] and e['complexity'] is not None]
            ps_comps = np.array([e['complexity'] for e in ps_entries])
            ps_DL_rel = np.array([e['DL'] for e in ps_entries]) - DL_ref
            ps_NLL_rel = np.array([e['NLL'] for e in ps_entries]) - NLL_ref

            # PS-like Pareto front (best at each complexity)
            ps_by_comp_dl = defaultdict(list)
            ps_by_comp_nll = defaultdict(list)
            for c, d, n in zip(ps_comps, ps_DL_rel, ps_NLL_rel):
                ps_by_comp_dl[c].append(d)
                ps_by_comp_nll[c].append(n)
            ps_front_comps = sorted(ps_by_comp_dl.keys())
            ps_front_dl = [min(ps_by_comp_dl[c]) for c in ps_front_comps]
            ps_front_nll = [min(ps_by_comp_nll[c]) for c in ps_front_comps]
            frontier_dl_set = set(zip(ps_front_comps, ps_front_dl))
            frontier_nll_set = set(zip(ps_front_comps, ps_front_nll))

            in_seg = (ps_comps >= xmin) & (ps_comps <= xmax)

            # Pareto front lines (drawn on top)
            pf_c = np.array(ps_front_comps)
            pf_dl = np.array(ps_front_dl)
            pf_nll = np.array(ps_front_nll)
            pf_in = (pf_c >= xmin) & (pf_c <= xmax)
            if np.any(pf_in):
                ax.plot(pf_c[pf_in], pf_dl[pf_in], 's-', color='purple',
                        ms=6, zorder=7, lw=1.8)
                ax.plot(pf_c[pf_in], pf_nll[pf_in], 's-', color='darkcyan',
                        ms=6, zorder=7, lw=1.8)

    # Add local legend to the last (rightmost) sub-axis for HMF
    if ps_like_data is not None:
        from matplotlib.lines import Line2D as L2D
        loc_handles = [
            L2D([], [], color='grey', marker='o', linestyle='-', ms=5),
            L2D([], [], color='purple', marker='s', linestyle='-', ms=5, lw=1.8),
            L2D([], [], color='darkcyan', marker='s', linestyle='-', ms=5, lw=1.8),
        ]
        loc_labels = [
            'ESR',
            r'PS-like $\Delta$DL',
            r'PS-like $\Delta$NLL',
        ]
        if show_lit_legend:
            loc_handles += [
                L2D([], [], color='grey', marker='^', linestyle='None', ms=8, markeredgewidth=1.5),
                L2D([], [], color='grey', marker='x', linestyle='None', ms=8, markeredgewidth=1.5),
                L2D([], [], color='grey', marker='h', linestyle='None', ms=8, markeredgewidth=1.5),
                L2D([], [], color='grey', marker='*', linestyle='None', ms=10, markeredgewidth=1.5),
            ]
            loc_labels += ['Warren', 'Tinker', 'Sheth-Tormen', 'Jenkins']
        ncol = 2 if show_lit_legend else 1
        ax_list[-1].legend(loc_handles, loc_labels, loc='upper right', ncol=ncol,
                           fontsize=legend_fontsize, frameon=True, framealpha=0.9)

    # Style seams between neighboring segments (actual marks are drawn later after layout)
    break_pairs = []
    for i in range(len(ax_list) - 1):
        left_ax = ax_list[i]
        right_ax = ax_list[i + 1]
        left_ax.spines['right'].set_visible(False)
        right_ax.spines['left'].set_visible(False)
        right_ax.tick_params(labelleft=False, left=False)
        break_pairs.append((left_ax, right_ax))

    return break_pairs


def make_single_panel_figure(data_set, output_path, ps_like_data=None,
                              label_fontsize=11, tick_fontsize=11, legend_fontsize=8,
                              figsize=(5.6, 4.0), ylabel_x_offset=0.05,
                              hmf_xmax_cut=16.5, variant='extended'):
    """Create one Pareto panel (with broken x-axis if needed) for a single dataset."""
    source, comp, DL, NLL = load_pareto_data(data_set, variant=variant)
    seg_comp = comp[~np.isin(source, ['ESR_C', 'ESR_T'])]
    if ps_like_data is not None:
        ps_comps_extra = [e['complexity'] for e in ps_like_data
                          if e['ps_like'] and e['complexity'] is not None]
        all_comps = np.concatenate([seg_comp, np.array(ps_comps_extra, dtype=int)])
        segments = split_complexity_segments(all_comps, min_gap=3, pad=0.35)
    else:
        segments = split_complexity_segments(seg_comp, min_gap=3, pad=0.35)
    segments = apply_lower_x_cut(segments, LOWER_X_CUTS.get(data_set))

    # HMF: cap upper x and insert tick 15 in the 14-16 segment
    if data_set == 'hmf_50':
        xmax_cut = hmf_xmax_cut
        new_segs = []
        for xmin_s, xmax_s, ticks_s in segments:
            if xmin_s > xmax_cut:
                continue
            new_segs.append((xmin_s, min(xmax_s, xmax_cut), [t for t in ticks_s if t <= xmax_cut]))
        segments = new_segs
        for j, (xmin_s, xmax_s, ticks_s) in enumerate(segments):
            if 14 <= xmax_s and 15 not in ticks_s and xmin_s <= 15 <= xmax_s:
                ticks_s.append(15)
                ticks_s.sort()
                segments[j] = (xmin_s, xmax_s, ticks_s)

    fig = plt.figure(figsize=figsize)
    width_ratios = [max(xmax - xmin, 0.5) for xmin, xmax, _ in segments]
    inner = fig.add_gridspec(1, len(segments), wspace=0.05, width_ratios=width_ratios)

    ax_list = []
    for j in range(len(segments)):
        if j == 0:
            ax = fig.add_subplot(inner[0, j])
        else:
            ax = fig.add_subplot(inner[0, j], sharey=ax_list[0])
        ax_list.append(ax)

    break_pairs = plot_pareto(ax_list, segments, data_set, datasets_info={},
                              ps_like_data=ps_like_data,
                              tick_fontsize=tick_fontsize,
                              legend_fontsize=legend_fontsize,
                              show_lit_legend=True,
                              xmax_cap_override=(hmf_xmax_cut if data_set == 'hmf_50'
                                                 else None),
                              variant=variant)

    fig.subplots_adjust(top=0.97, left=0.18, right=0.98, bottom=0.17)
    coloured_ylabel(fig, ax_list[0], fontsize=label_fontsize, x_offset=ylabel_x_offset)
    fig_w, fig_h = fig.get_size_inches()
    sx, sy = 12.0 / fig_w, 11.0 / fig_h
    for left_ax, right_ax in break_pairs:
        add_break_marks(fig, left_ax, right_ax,
                        dx=0.004 * sx, dy=0.007 * sy, sep=0.006 * sx)

    left_box = ax_list[0].get_position()
    right_box = ax_list[-1].get_position()
    x_center = 0.5 * (left_box.x0 + right_box.x1)
    y_pos = left_box.y0 - 0.055
    fig.text(x_center, y_pos, 'Complexity', ha='center', va='top', fontsize=label_fontsize)

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def draw_hmf_pf_panel(fig, cell, datasets_info, tick_fontsize=11, legend_fontsize=8):
    """Draw the HMF Pareto panel from the TRUE combined-over-100-sims PF front.

    Data: the v2 no-MIN_SIMS candidate pool (top-3000-per-realisation nomination,
    donor-filled on the 10 selection realisations, then top 200 per complexity)
    re-fit on all 100 realisations (pf_refit_combined.txt). This replaces the
    earlier panel, which showed a single realisation (sim 50). DeltaDL/DeltaNLL
    are combined over all 100 sims relative to the overall-best (comp-10 rank-1)
    function, matching the combined ranking of Table~\\ref{tab:HMF_functions}.

    Returns (ax_list, segments, break_pairs) and registers the literature markers
    in datasets_info (under 'HMF') for the unified legend.
    """
    from collections import defaultdict
    from matplotlib.lines import Line2D as L2D

    rows = [l.strip().split(';') for l in open('pf_refit_combined.txt')
            if l.strip() and not l.startswith('#')]
    func = [r[1] for r in rows]
    comp = np.array([int(r[2]) for r in rows])
    DLc = np.array([float(r[3]) for r in rows])
    NLLc = np.array([float(r[5]) for r in rows])
    nsims = np.array([int(r[8]) for r in rows])
    best_DL, best_NLL = DLc[0], NLLc[0]

    # ESR Pareto front: minimum combined DeltaDL per complexity (full-100-sim funcs)
    esr_c, esr_dDL, esr_dNLL = [], [], []
    for c in sorted(set(comp.tolist())):
        m = (comp == c) & (nsims == 100)
        idx = np.where(m)[0][np.argmin(DLc[m])]
        esr_c.append(c); esr_dDL.append(DLc[idx] - best_DL); esr_dNLL.append(NLLc[idx] - best_NLL)
    esr_c = np.array(esr_c); esr_dDL = np.array(esr_dDL); esr_dNLL = np.array(esr_dNLL)

    # PS-like front: classify candidates with sim-50 params, min combined DL/NLL per comp
    def _pp(b):
        b = b.strip()
        if b in ('none', 'nan', ''):
            return None
        try:
            return [float(x) for x in b.split()]
        except ValueError:
            return None
    sim50 = {}
    for l in open('pf_refit_sim50.txt'):
        p = l.strip().split(';')
        if len(p) >= 6:
            sim50[p[0]] = _pp(p[5])
    ps_dl, ps_nll = defaultdict(list), defaultdict(list)
    for i, fn in enumerate(func):
        if nsims[i] != 100:
            continue
        pr = sim50.get(fn)
        if pr is not None and _check_ps_like(fn, pr):
            ps_dl[comp[i]].append(DLc[i] - best_DL)
            ps_nll[comp[i]].append(NLLc[i] - best_NLL)
    ps_front_c = np.array(sorted(ps_dl))
    ps_front_dl = np.array([min(ps_dl[c]) for c in ps_front_c])
    ps_front_nll = np.array([min(ps_nll[c]) for c in ps_front_c])

    # Literature (combined; files store +magnitude = -DL_combined). S-T/Jenkins from
    # the dedicated extended-style fiducial refit (consistent reparam forms).
    lit_comp = {'P.Sch.': 10, 'Jen.': 13, 'War.': 14, 'Tin.': 16, 'S-T.': 24}
    lit_mk = {'P.Sch.': ('*', 'Press-Sch.'), 'War.': ('^', 'Warren'),
              'Tin.': ('x', 'Tinker'), 'S-T.': ('h', 'Sheth-Tormen'),
              'Jen.': ('*', 'Jenkins')}
    lit = {}
    for fname in ('literature_combined_DL_trimmed.txt', 'literature_st_jenkins_combined.txt'):
        for l in open(fname):
            l = l.strip()
            if not l or l.startswith('#'):
                continue
            p = l.split(';')
            lit[p[0]] = (lit_comp[p[0]], -float(p[1]) - best_DL, -float(p[2]) - best_NLL)

    # Broken-axis segments: lower cut 6.8 (drop comp 6), upper cut 25, tick 15 inserted
    all_comps = np.concatenate([esr_c, ps_front_c, np.array([lit[n][0] for n in lit])])
    segments = split_complexity_segments(all_comps, min_gap=3, pad=0.35)
    segments = apply_lower_x_cut(segments, 6.8)
    segments = [(xn, min(xm, 25), [t for t in tk if t <= 25])
                for xn, xm, tk in segments if xn <= 25]
    for j, (xn, xm, tk) in enumerate(segments):
        if xn <= 15 <= xm and 15 not in tk:
            segments[j] = (xn, xm, sorted(tk + [15]))

    ymax = 15000.0
    ymin = -0.03 * ymax

    width_ratios = [max(xm - xn, 0.5) for xn, xm, _ in segments]
    inner = cell.subgridspec(1, len(segments), wspace=0.05, width_ratios=width_ratios)
    ax_list = []
    for j in range(len(segments)):
        ax = fig.add_subplot(inner[0, j], sharey=ax_list[0] if ax_list else None)
        ax_list.append(ax)

    for ax, (xn, xm, tk) in zip(ax_list, segments):
        ax.set_xlim(xn, xm)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(tk)
        ax.tick_params(labelsize=tick_fontsize)
        # ESR front (full arrays so the off-scale comp-6 point clips to a near-vertical line)
        ax.plot(esr_c, esr_dDL, 'o-', color='C0', ms=5, lw=1.5, zorder=5)
        ax.plot(esr_c, esr_dNLL, 'o-', color='C3', ms=5, lw=1.5, zorder=4)
        # PS-like front (purple / darkcyan), same treatment
        ax.plot(ps_front_c, ps_front_dl, 's-', color='purple', ms=6, lw=1.8, zorder=7)
        ax.plot(ps_front_c, ps_front_nll, 's-', color='darkcyan', ms=6, lw=1.8, zorder=7)
        # Literature markers (blue DL + red NLL), off-scale skipped
        for name, (lc, dl, nll) in lit.items():
            if not (xn <= lc <= xm):
                continue
            mk = lit_mk[name][0]
            if ymin <= dl <= ymax:
                ax.scatter(lc, dl, marker=mk, color='C0', s=70, linewidths=1.5, zorder=6)
            if ymin <= nll <= ymax:
                ax.scatter(lc, nll, marker=mk, color='C3', s=70, linewidths=1.5, zorder=6)
            if name != 'P.Sch.':   # P.Sch off-scale -> not in legend
                if name not in datasets_info:
                    datasets_info[name] = {'marker': mk, 'name': lit_mk[name][1], 'datasets': set()}
                datasets_info[name]['datasets'].add('HMF')

    break_pairs = []
    for i in range(len(ax_list) - 1):
        ax_list[i].spines['right'].set_visible(False)
        ax_list[i + 1].spines['left'].set_visible(False)
        ax_list[i + 1].tick_params(labelleft=False, left=False)
        break_pairs.append((ax_list[i], ax_list[i + 1]))

    # Local PS-like legend (ESR + literature are in the unified legend);
    # placed in the top-right segment so it sits at the panel's top right
    ax_list[-1].legend(
        [L2D([], [], color='purple', marker='s', ls='-', ms=5, lw=1.8),
         L2D([], [], color='darkcyan', marker='s', ls='-', ms=5, lw=1.8)],
        [r'PS-like $\Delta$DL', r'PS-like $\Delta$NLL'],
        loc='upper right', fontsize=legend_fontsize, frameon=True, framealpha=0.9)

    return ax_list, segments, break_pairs


# ── Extended (Appendix B) combined-PF HMF panel ──────────────────────────────
# TRUE extended best — normalisation reference, kept consistent with Table B1.
# CLAUDEADD 2026-06-10: switched from the ~/ESR det-I codelen (-3555040871.21) to the
# PAPER diagonal-Fisher codelen (rank-1 recomputed; jobs 744636/744652) to match the
# rebuilt Table B1 and the main-text DL formula. NLL unchanged (fits identical).
TRUE_BEST_DL_EXT = -3555040281.634192
TRUE_BEST_NLL_EXT = -3555042797.586635


def _load_gencomp(path='hmf_func_gencomp.txt'):
    """function -> ESR generation complexity (default 10 if absent)."""
    m = {}
    try:
        for line in open(path):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split(';')
            m[p[0]] = int(p[1])
    except FileNotFoundError:
        pass
    return m


def draw_hmf_pf_panel_extended(fig, cell, tick_fontsize=11, legend_fontsize=8,
                               ymax=11000.0):
    """Draw the EXTENDED-range HMF Pareto panel (Fig.~B2): the analogue of
    draw_hmf_pf_panel() for the full (18-bin) mass range.

    ESR front = per-complexity minimum combined DeltaDL over the MERGE of
      (i) the fiducial PF candidates re-fit on the extended data
          (pf_refit_combined_extended.txt, v2 job 744570) — gives comp 6-10
          coverage and the PS-like classification; and
      (ii) the retained true-extended ranking (hmf_combined_DL_diagonal.txt), which
          supplies the genuine best functions (comp 9-10) that are absent from
          the fiducial candidate pool.
    Everything is normalised to the TRUE extended best (rank-0 of
    hmf_combined_DL_diagonal.txt) so the panel stays consistent with Table~B1.

    Returns (ax_list, segments, break_pairs).
    """
    from collections import defaultdict

    best_DL, best_NLL = TRUE_BEST_DL_EXT, TRUE_BEST_NLL_EXT
    gencomp = _load_gencomp()

    # (i) fiducial candidates re-fit on extended data
    cand = []
    for l in open('pf_refit_combined_extended.txt'):
        l = l.strip()
        if not l or l.startswith('#'):
            continue
        p = l.split(';')
        cand.append((p[1], int(p[2]), float(p[3]), float(p[5]), int(p[8])))
    # (ii) retained true-extended ranking (complexities via the generation map)
    # CLAUDEADD 2026-06-10: use the diagonal-codelen recompute (hmf_combined_DL_diagonal.txt,
    # top-25 true-ranking funcs re-fit to the paper codelen) instead of the det-I
    # hmf_combined_DL.txt, so the front matches the rebuilt Table B1.
    true_rows = []
    true_nlls = []
    for l in open('hmf_combined_DL_diagonal.txt'):
        l = l.strip()
        if not l or l.startswith('#'):
            continue
        p = l.split(';')
        true_rows.append((p[1], gencomp.get(p[1], 10), float(p[2]), float(p[4]), int(p[7])))
        true_nlls.append(float(p[4]))

    # Merged ESR front: per-complexity min combined DL across both sources
    # (full-100-sim functions only). If a candidate has the same NLL as one of
    # the retained true-ranking rows it is an equivalent tree representative;
    # prefer the diagonal Table-B1 row so the plotted front cannot go below its
    # own normalisation because of a mixed structural-code convention.
    front = {}
    for func, c, dl, nll, n in true_rows:
        if n != 100:
            continue
        if c not in front or dl < front[c][0]:
            front[c] = (dl, nll)
    for func, c, dl, nll, n in cand:
        if n != 100:
            continue
        if any(abs(nll - tnll) <= 0.05 for tnll in true_nlls):
            continue
        if c not in front or dl < front[c][0]:
            front[c] = (dl, nll)
    esr_c = np.array(sorted(front))
    esr_dDL = np.array([front[c][0] - best_DL for c in esr_c])
    esr_dNLL = np.array([front[c][1] - best_NLL for c in esr_c])

    # PS-like front: classify candidates with extended sim-50 params
    def _pp(b):
        b = b.strip()
        if b in ('none', 'nan', ''):
            return None
        try:
            return [float(x) for x in b.split()]
        except ValueError:
            return None
    sim50 = {}
    for l in open('pf_refit_persim_extended.txt'):
        p = l.strip().split(';')
        if len(p) >= 6 and p[1] == '50':
            sim50[p[0]] = _pp(p[5])
    ps_dl, ps_nll = defaultdict(list), defaultdict(list)
    for func, c, dl, nll, n in cand:
        if n != 100:
            continue
        pr = sim50.get(func)
        if pr is not None and _check_ps_like(func, pr):
            ps_dl[c].append(dl - best_DL)
            ps_nll[c].append(nll - best_NLL)
    ps_front_c = np.array(sorted(ps_dl))
    ps_front_dl = np.array([min(ps_dl[c]) for c in ps_front_c])
    ps_front_nll = np.array([min(ps_nll[c]) for c in ps_front_c])

    # Literature (files store +magnitude = -DL_combined). War/Tin/P.Sch from
    # literature_combined_DL.txt (read last -> wins for the War/Tin duplicates),
    # S-T/Jenkins from literature_st_jenkins_extended.txt; matches Table~B1.
    lit_comp = {'P.Sch.': 10, 'Jen.': 13, 'War.': 14, 'Tin.': 16, 'S-T.': 24}
    lit_mk = {'P.Sch.': ('*', 'Press-Sch.'), 'War.': ('^', 'Warren'),
              'Tin.': ('x', 'Tinker'), 'S-T.': ('h', 'Sheth-Tormen'),
              'Jen.': ('*', 'Jenkins')}
    lit = {}
    for fname in ('literature_st_jenkins_extended.txt', 'literature_combined_DL.txt'):
        for l in open(fname):
            l = l.strip()
            if not l or l.startswith('#'):
                continue
            p = l.split(';')
            if p[0] not in lit_comp:
                continue
            lit[p[0]] = (lit_comp[p[0]], -float(p[1]) - best_DL, -float(p[2]) - best_NLL)

    # Broken-axis segments (mirror fiducial, but start just before comp 8 so
    # off-scale low-complexity tails do not dominate the panel edge).
    all_comps = np.concatenate([esr_c, ps_front_c, np.array([lit[n][0] for n in lit])])
    segments = split_complexity_segments(all_comps, min_gap=3, pad=0.35)
    segments = apply_lower_x_cut(segments, 7.8)
    segments = [(xn, min(xm, 25), [t for t in tk if t <= 25])
                for xn, xm, tk in segments if xn <= 25]
    for j, (xn, xm, tk) in enumerate(segments):
        if xn <= 15 <= xm and 15 not in tk:
            segments[j] = (xn, xm, sorted(tk + [15]))

    ymin = -0.03 * ymax

    width_ratios = [max(xm - xn, 0.5) for xn, xm, _ in segments]
    inner = cell.subgridspec(1, len(segments), wspace=0.05, width_ratios=width_ratios)
    ax_list = []
    for j in range(len(segments)):
        ax = fig.add_subplot(inner[0, j], sharey=ax_list[0] if ax_list else None)
        ax_list.append(ax)

    for ax, (xn, xm, tk) in zip(ax_list, segments):
        ax.set_xlim(xn, xm)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(tk)
        ax.tick_params(labelsize=tick_fontsize)
        # ESR + PS-like fronts (full arrays so off-scale low-comp points clip to
        # near-vertical lines, as in the fiducial panel)
        ax.plot(esr_c, esr_dDL, 'o-', color='C0', ms=5, lw=1.5, zorder=5)
        ax.plot(esr_c, esr_dNLL, 'o-', color='C3', ms=5, lw=1.5, zorder=4)
        ax.plot(ps_front_c, ps_front_dl, 's-', color='purple', ms=6, lw=1.8, zorder=7)
        ax.plot(ps_front_c, ps_front_nll, 's-', color='darkcyan', ms=6, lw=1.8, zorder=7)
        # Literature markers (blue DL + red NLL); off-scale (P.Sch, Jenkins) skipped
        for name, (lc, dl, nll) in lit.items():
            if not (xn <= lc <= xm):
                continue
            mk = lit_mk[name][0]
            if ymin <= dl <= ymax:
                ax.scatter(lc, dl, marker=mk, color='C0', s=70, linewidths=1.5, zorder=6)
            if ymin <= nll <= ymax:
                ax.scatter(lc, nll, marker=mk, color='C3', s=70, linewidths=1.5, zorder=6)

    break_pairs = []
    for i in range(len(ax_list) - 1):
        ax_list[i].spines['right'].set_visible(False)
        ax_list[i + 1].spines['left'].set_visible(False)
        ax_list[i + 1].tick_params(labelleft=False, left=False)
        break_pairs.append((ax_list[i], ax_list[i + 1]))

    return ax_list, segments, break_pairs


def make_hmf_pf_extended_figure(output_path, label_fontsize=16, tick_fontsize=14,
                                legend_fontsize=12, figsize=(6.8, 5.0),
                                ylabel_x_offset=0.10, ymax=11000.0):
    """Standalone single-panel figure for the extended HMF Pareto front (Fig.~B2)."""
    from matplotlib.lines import Line2D as L2D

    fig = plt.figure(figsize=figsize)
    cell = fig.add_gridspec(1, 1)[0, 0]
    ax_list, segments, break_pairs = draw_hmf_pf_panel_extended(
        fig, cell, tick_fontsize=tick_fontsize, legend_fontsize=legend_fontsize,
        ymax=ymax)

    fig.subplots_adjust(top=0.86, left=0.18, right=0.98, bottom=0.15)

    # Top legend (covers everything; ESR shown grey, DL/NLL distinguished by y-label)
    handles = [
        L2D([], [], color='grey', marker='o', ls='-', ms=5),
        L2D([], [], color='purple', marker='s', ls='-', ms=5, lw=1.8),
        L2D([], [], color='darkcyan', marker='s', ls='-', ms=5, lw=1.8),
        L2D([], [], color='grey', marker='^', ls='None', ms=8, markeredgewidth=1.5),
        L2D([], [], color='grey', marker='x', ls='None', ms=8, markeredgewidth=1.5),
        L2D([], [], color='grey', marker='h', ls='None', ms=8, markeredgewidth=1.5),
    ]
    labels = ['ESR', r'PS-like $\Delta$DL', r'PS-like $\Delta$NLL',
              'Warren', 'Tinker', 'Sheth-Tormen']
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=legend_fontsize,
               frameon=True, bbox_to_anchor=(0.58, 1.0))

    coloured_ylabel(fig, ax_list[0], fontsize=label_fontsize, x_offset=ylabel_x_offset)
    fig_w, fig_h = fig.get_size_inches()
    sx, sy = 12.0 / fig_w, 11.0 / fig_h
    for left_ax, right_ax in break_pairs:
        add_break_marks(fig, left_ax, right_ax,
                        dx=0.004 * sx, dy=0.007 * sy, sep=0.006 * sx)

    left_box = ax_list[0].get_position()
    right_box = ax_list[-1].get_position()
    x_center = 0.5 * (left_box.x0 + right_box.x1)
    fig.text(x_center, left_box.y0 - 0.06, 'Complexity', ha='center', va='top',
             fontsize=label_fontsize)

    import os
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    png = output_path.replace('Final_Plots', 'Plots').replace('.pdf', '.png')
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


# All datasets
all_datasets = [
    ('LF_Ser_L', 'LF Sérsic'),
    ('LF_cmodel_L', 'LF cmodel'),
    ('SMF_Ser_M', 'SMF Sérsic'),
    ('SMF_cmodel_M', 'SMF cmodel'),
    ('hmf_50', 'HMF'),
]

LOWER_X_CUTS = {
    'LF_Ser_L': 6.8,
    'LF_cmodel_L': 6.8,
    'SMF_Ser_M': 6.8,
    'hmf_50': 6.8,
    'SMF_cmodel_M': 5.8,
}

if __name__ == '__main__':
    # Pre-load PS-like data for the HMF panel (main text = fiducial mass bins)
    print("Loading PS-like data for HMF panel...")
    hmf_ps_data = load_ps_like_for_hmf(sim=50, variant='fiducial')
    n_ps = sum(1 for e in hmf_ps_data if e['ps_like'])
    print(f"  Found {n_ps} PS-like functions among top 200")

    fig = plt.figure(figsize=(12, 11))
    outer = fig.add_gridspec(3, 2, wspace=0.25, hspace=0.35)

    datasets_info = {}
    panel_axes = []
    all_break_pairs = []

    for i, (ds, title) in enumerate(all_datasets):
        if ds == 'hmf_50':
            # HMF panel: TRUE combined-over-100-sims PF front (centred, bottom row)
            cell = outer[2, :].subgridspec(1, 3, wspace=0, width_ratios=[1, 2, 1])[0, 1]
            ax_list, segments, break_pairs = draw_hmf_pf_panel(fig, cell, datasets_info)
            all_break_pairs.extend(break_pairs)
            panel_axes.append((ax_list, segments))
            continue
        # Main text uses the fiducial (16-bin) HMF ranking
        variant = 'fiducial' if ds == 'hmf_50' else 'extended'
        source, comp, DL, NLL = load_pareto_data(ds, variant=variant)
        # Segments/ticks driven by plotted functions only (skip ESR_C/ESR_T overall rankings)
        seg_comp = comp[~np.isin(source, ['ESR_C', 'ESR_T'])]
        # For HMF, include PS-like complexities in segment computation
        if ds == 'hmf_50' and hmf_ps_data is not None:
            ps_comps_extra = [e['complexity'] for e in hmf_ps_data
                              if e['ps_like'] and e['complexity'] is not None]
            all_comps = np.concatenate([seg_comp, np.array(ps_comps_extra, dtype=int)])
            segments = split_complexity_segments(all_comps, min_gap=3, pad=0.35)
        else:
            segments = split_complexity_segments(seg_comp, min_gap=3, pad=0.35)
        segments = apply_lower_x_cut(segments, LOWER_X_CUTS.get(ds))

        # Apply upper x cut (HMF: extend to show Sheth-Tormen at complexity 24)
        UPPER_X_CUTS = {'hmf_50': 25}
        if ds in UPPER_X_CUTS:
            xmax_cut = UPPER_X_CUTS[ds]
            new_segs = []
            for xmin_s, xmax_s, ticks_s in segments:
                if xmin_s > xmax_cut:
                    continue
                new_segs.append((xmin_s, min(xmax_s, xmax_cut), [t for t in ticks_s if t <= xmax_cut]))
            segments = new_segs
            # Add tick 15 to the segment containing 14-16
            for j, (xmin_s, xmax_s, ticks_s) in enumerate(segments):
                if 14 <= xmax_s and 15 not in ticks_s and xmin_s <= 15 <= xmax_s:
                    ticks_s.append(15)
                    ticks_s.sort()
                    segments[j] = (xmin_s, xmax_s, ticks_s)

        # For LF/SMF, fill in tick 22 between the Saunders (21) and Bernardi-orig (23) points
        if ds != 'hmf_50':
            for j, (xmin_s, xmax_s, ticks_s) in enumerate(segments):
                if xmin_s <= 22 <= xmax_s and 22 not in ticks_s:
                    ticks_s = sorted(ticks_s + [22])
                    segments[j] = (xmin_s, xmax_s, ticks_s)

        if i < 4:
            row, col = divmod(i, 2)
            cell = outer[row, col]
        else:
            # HMF: centred in bottom row spanning both columns
            # Use a 3-column wrapper: [padding, panel, padding] to centre
            cell = outer[2, :]
            wrapper = cell.subgridspec(1, 3, wspace=0, width_ratios=[1, 2, 1])
            cell = wrapper[0, 1]

        width_ratios = [max(xmax - xmin, 0.5) for xmin, xmax, _ in segments]
        inner = cell.subgridspec(1, len(segments), wspace=0.05, width_ratios=width_ratios)

        ax_list = []
        for j in range(len(segments)):
            if j == 0:
                ax = fig.add_subplot(inner[0, j])
            else:
                ax = fig.add_subplot(inner[0, j], sharey=ax_list[0])
            ax_list.append(ax)

        ps_data = hmf_ps_data if ds == 'hmf_50' else None
        break_pairs = plot_pareto(ax_list, segments, ds, datasets_info,
                                  ps_like_data=ps_data, variant=variant)
        all_break_pairs.extend(break_pairs)
        panel_axes.append((ax_list, segments))

    # Build unified legend
    leg_handles = []
    leg_labels = []

    leg_handles.append(Line2D([], [], color='grey', marker='o', linestyle='-', ms=5))
    leg_labels.append('ESR')

    for key in ['Ber.', 'Ber.orig', 'DblSch.', 'Sau.', 'War.', 'Tin.', 'S-T.', 'Jen.']:
        if key in datasets_info:
            info = datasets_info[key]
            ds_str = ', '.join(sorted(info['datasets']))
            leg_handles.append(Line2D([], [], color='grey', marker=info['marker'],
                                      linestyle='None', ms=8, markeredgewidth=1.5))
            leg_labels.append(f"{info['name']}")

    fig.legend(leg_handles, leg_labels, loc='upper center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 1.03), frameon=True)

    fig.subplots_adjust(top=0.955, left=0.08, right=0.98, bottom=0.06)

    # Y-axis labels (HMF has 5-digit tick numbers -> nudge its label further left)
    for (ax_list, _), (ds, _title) in zip(panel_axes, all_datasets):
        x_off = 0.07 if ds == 'hmf_50' else 0.05
        coloured_ylabel(fig, ax_list[0], fontsize=11, x_offset=x_off)

    # Panel titles centred across full broken-axis span
    for (ax_list, _segs), (_, title) in zip(panel_axes, all_datasets):
        left_box = ax_list[0].get_position()
        right_box = ax_list[-1].get_position()
        x_center = 0.5 * (left_box.x0 + right_box.x1)
        y_pos = left_box.y1 - 0.005
        fig.text(x_center, y_pos, title, ha='center', va='top', fontsize=13, fontweight='bold')

    # Draw break marks after final layout so they sit exactly on axis seams
    for left_ax, right_ax in all_break_pairs:
        add_break_marks(fig, left_ax, right_ax)

    # Centered x-label under each full broken-axis panel
    for ax_list, _ in panel_axes:
        left_box = ax_list[0].get_position()
        right_box = ax_list[-1].get_position()
        x_center = 0.5 * (left_box.x0 + right_box.x1)
        y_pos = left_box.y0 - 0.03
        fig.text(x_center, y_pos, 'Complexity', ha='center', va='top', fontsize=11)

    # Force integer ticks on all axes (must be done after layout is finalized)
    for ax_list, segs in panel_axes:
        for ax, (xmin, xmax, ticks) in zip(ax_list, segs):
            ax.set_xticks(ticks)

    plt.savefig('Final_Plots/Pareto_all.pdf', dpi=200, bbox_inches='tight')

    # Standalone panels (not used in paper) — save to Plots/
    make_single_panel_figure('LF_Ser_L', 'Plots/Pareto_LF_Sérsic.pdf')
    make_single_panel_figure('LF_cmodel_L', 'Plots/Pareto_LF_cmodel.pdf')
    make_single_panel_figure('SMF_Ser_M', 'Plots/Pareto_SMF_Sérsic.pdf')
    make_single_panel_figure('SMF_cmodel_M', 'Plots/Pareto_SMF_cmodel.pdf')
    make_single_panel_figure('hmf_50', 'Plots/Pareto_HMF.pdf', ps_like_data=hmf_ps_data,
                             variant='fiducial')

    plt.show()
    plt.clf()
