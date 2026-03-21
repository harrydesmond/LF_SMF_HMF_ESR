"""Plot Pareto fronts of delta-DL and delta-NLL vs complexity for all datasets.

For each dataset, shows the best ESR function at each complexity as connected
points, with literature functions as isolated markers. Supports broken x-axes
where complexity gaps exist (e.g. between complexity 10 and 16+).

Produces a combined 5-panel figure (2x2 grid for LF/SMF + centred HMF panel)
and 5 standalone single-panel figures.

Figures produced:
    - Final_Plots/Pareto_all.pdf               (combined 5-panel)
    - Final_Plots/Pareto_LF_Sersic.pdf         (standalone)
    - Final_Plots/Pareto_LF_cmodel.pdf         (standalone)
    - Final_Plots/Pareto_SMF_Sersic.pdf        (standalone)
    - Final_Plots/Pareto_SMF_cmodel.pdf        (standalone)
    - Final_Plots/Pareto_HMF.pdf               (standalone)

Inputs:
    - *_final_functions.txt : function definitions with DL, NLL, complexity
      (semicolon-delimited: source, complexity, DL, NLL, plot_fcn, blank_fcn)

Dependencies:
    numpy, matplotlib
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
def coloured_ylabel(fig, ax, fontsize=11):
    """Set a two-colour y-axis label: blue ΔDL / red ΔNLL."""
    box = ax.get_position()
    x = box.x0 - 0.05
    y_mid = 0.5 * (box.y0 + box.y1)
    fig.text(x, y_mid, r'$\Delta$DL', color='C0', fontsize=fontsize,
             rotation=90, ha='center', va='bottom')
    fig.text(x, y_mid - 0.005, '/', color='black', fontsize=fontsize,
             rotation=90, ha='center', va='top')
    fig.text(x, y_mid - 0.015, r'$\Delta$NLL', color='C3', fontsize=fontsize,
             rotation=90, ha='center', va='top')


def load_pareto_data(data_set):
    """Load DL and NLL from final_functions file."""
    source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt(
        '{}_final_functions.txt'.format(data_set), dtype=str, delimiter=';', unpack=True)
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


def add_break_marks(fig, ax_left, ax_right, dx=0.004, dy=0.007, x_shift=-0.0015):
    """Draw uniform-angle bottom break marks and keep top axis visually continuous."""
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()

    seam_x = 0.5 * (left_box.x1 + right_box.x0) + x_shift
    y_bottom = left_box.y0
    y_top = left_box.y1

    line1 = Line2D([seam_x - dx, seam_x + dx], [y_bottom - dy, y_bottom + dy],
                   transform=fig.transFigure, color='k', lw=1.0, clip_on=False)
    line2 = Line2D([seam_x + 0.006 - dx, seam_x + 0.006 + dx],
                   [y_bottom - dy, y_bottom + dy],
                   transform=fig.transFigure, color='k', lw=1.0, clip_on=False)
    top_connector = Line2D([left_box.x1, right_box.x0], [y_top, y_top],
                           transform=fig.transFigure, color='k', lw=1.0, clip_on=False)
    fig.add_artist(line1)
    fig.add_artist(line2)
    fig.add_artist(top_connector)


def plot_pareto(ax_list, segments, data_set, datasets_info):
    """Plot Pareto front for a single dataset onto a list of broken-x sub-axes."""
    source, comp, DL, NLL = load_pareto_data(data_set)

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

    # Literature functions (exclude ESR and ESR_C)
    lit_mask = ~(esr_mask | esr_c_mask)
    lit_source = source[lit_mask]
    lit_comp = comp[lit_mask]
    lit_DL_rel = DL[lit_mask] - DL_ref
    lit_NLL_rel = NLL[lit_mask] - NLL_ref

    markers = {
        'Sch.': ('x', 'Schechter'),
        'P.Sch.': ('*', 'Press-Sch.'),
        'War.': ('^', 'Warren'),
        'Tin.': ('s', 'Tinker'),
        'Ber.': ('+', 'Bernardi'),
        'Ber. Paper': ('+', 'Bernardi'),
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
        'SMF_Ser_M': 700,
        'LF_cmodel_L': 550,
        'SMF_cmodel_M': 1700,
    }
    YMIN_OVERRIDES = {
        'SMF_Ser_M': -30,
        'LF_cmodel_L': -35,
    }
    if data_set in YMAX_OVERRIDES:
        ymax = YMAX_OVERRIDES[data_set]
    if data_set in YMIN_OVERRIDES:
        ymin = YMIN_OVERRIDES[data_set]

    for ax, (xmin, xmax, ticks) in zip(ax_list, segments):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=ymin, top=ymax)

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

        ax.set_xticks(ticks)
        ax.tick_params(labelsize=11)

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


def make_single_panel_figure(data_set, output_path):
    """Create one Pareto panel (with broken x-axis if needed) for a single dataset."""
    source, comp, DL, NLL = load_pareto_data(data_set)
    segments = split_complexity_segments(comp, min_gap=3, pad=0.35)
    segments = apply_lower_x_cut(segments, LOWER_X_CUTS.get(data_set))

    fig = plt.figure(figsize=(5.6, 4.0))
    width_ratios = [max(xmax - xmin, 0.5) for xmin, xmax, _ in segments]
    inner = fig.add_gridspec(1, len(segments), wspace=0.05, width_ratios=width_ratios)

    ax_list = []
    for j in range(len(segments)):
        if j == 0:
            ax = fig.add_subplot(inner[0, j])
        else:
            ax = fig.add_subplot(inner[0, j], sharey=ax_list[0])
        ax_list.append(ax)

    break_pairs = plot_pareto(ax_list, segments, data_set, datasets_info={})

    fig.subplots_adjust(top=0.97, left=0.12, right=0.98, bottom=0.17)
    coloured_ylabel(fig, ax_list[0], fontsize=11)
    for left_ax, right_ax in break_pairs:
        add_break_marks(fig, left_ax, right_ax)

    left_box = ax_list[0].get_position()
    right_box = ax_list[-1].get_position()
    x_center = 0.5 * (left_box.x0 + right_box.x1)
    y_pos = left_box.y0 - 0.055
    fig.text(x_center, y_pos, 'Complexity', ha='center', va='top', fontsize=11)

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# All datasets
all_datasets = [
    ('LF_Ser_L', 'LF Sersic'),
    ('LF_cmodel_L', 'LF cmodel'),
    ('SMF_Ser_M', 'SMF Sersic'),
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

fig = plt.figure(figsize=(12, 11))
outer = fig.add_gridspec(3, 2, wspace=0.25, hspace=0.35)

datasets_info = {}
panel_axes = []
all_break_pairs = []

for i, (ds, title) in enumerate(all_datasets):
    source, comp, DL, NLL = load_pareto_data(ds)
    segments = split_complexity_segments(comp, min_gap=3, pad=0.35)
    segments = apply_lower_x_cut(segments, LOWER_X_CUTS.get(ds))

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

    break_pairs = plot_pareto(ax_list, segments, ds, datasets_info)
    all_break_pairs.extend(break_pairs)
    panel_axes.append(ax_list)

# Build unified legend
leg_handles = []
leg_labels = []

leg_handles.append(Line2D([], [], color='grey', marker='o', linestyle='-', ms=5))
leg_labels.append('ESR')

for key in ['Sch.', 'Ber.', 'P.Sch.', 'War.', 'Tin.']:
    if key in datasets_info:
        info = datasets_info[key]
        ds_str = ', '.join(sorted(info['datasets']))
        leg_handles.append(Line2D([], [], color='grey', marker=info['marker'],
                                  linestyle='None', ms=8, markeredgewidth=1.5))
        leg_labels.append(f"{info['name']} ({ds_str})")

leg_handles.append(Line2D([], [], color='C0', linestyle='-', lw=2))
leg_labels.append(r'$\Delta$DL')
leg_handles.append(Line2D([], [], color='C3', linestyle='-', lw=2))
leg_labels.append(r'$\Delta$NLL')

fig.legend(leg_handles, leg_labels, loc='upper center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 1.005), frameon=True)

fig.subplots_adjust(top=0.94, left=0.08, right=0.98, bottom=0.06)

# Y-axis labels
for pa in panel_axes:
    coloured_ylabel(fig, pa[0], fontsize=11)

# Panel titles centred across full broken-axis span
for ax_list, (_, title) in zip(panel_axes, all_datasets):
    left_box = ax_list[0].get_position()
    right_box = ax_list[-1].get_position()
    x_center = 0.5 * (left_box.x0 + right_box.x1)
    y_pos = left_box.y1 - 0.005
    fig.text(x_center, y_pos, title, ha='center', va='top', fontsize=13, fontweight='bold')

# Draw break marks after final layout so they sit exactly on axis seams
for left_ax, right_ax in all_break_pairs:
    add_break_marks(fig, left_ax, right_ax)

# Centered x-label under each full broken-axis panel
for ax_list in panel_axes:
    left_box = ax_list[0].get_position()
    right_box = ax_list[-1].get_position()
    x_center = 0.5 * (left_box.x0 + right_box.x1)
    y_pos = left_box.y0 - 0.03
    fig.text(x_center, y_pos, 'Complexity', ha='center', va='top', fontsize=11)

plt.savefig('Final_Plots/Pareto_all.pdf', dpi=200, bbox_inches='tight')

# Standalone panels for LaTeX subfigures
make_single_panel_figure('LF_Ser_L', 'Final_Plots/Pareto_LF_Sersic.pdf')
make_single_panel_figure('LF_cmodel_L', 'Final_Plots/Pareto_LF_cmodel.pdf')
make_single_panel_figure('SMF_Ser_M', 'Final_Plots/Pareto_SMF_Sersic.pdf')
make_single_panel_figure('SMF_cmodel_M', 'Final_Plots/Pareto_SMF_cmodel.pdf')
make_single_panel_figure('hmf_50', 'Final_Plots/Pareto_HMF.pdf')

plt.show()
plt.clf()
