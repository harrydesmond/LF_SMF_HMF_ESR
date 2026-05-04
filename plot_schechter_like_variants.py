"""Diagnostic LF/SMF plots for Schechter-like ESR functions.

Creates:
  - Plots/Pareto_LF_SMF_schechter_like_overlay.png
  - Plots/extrapolation_schechter_like.png
  - Plots/schechter_like_table_data.txt
  - Final_Plots/Pareto_LF_SMF_schechter_like_overlay.pdf
  - Final_Plots/extrapolation_schechter_like.pdf

Inputs:
  - LF/SMF ESR per-complexity files:
      LF_Ser_L_data/final_<comp>[_new].dat
      LF_cmodel_L_data/final_<comp>[_new].dat
      SMF_Ser_M_data/final_<comp>[_new].dat
      SMF_cmodel_M_data/final_<comp>[_new].dat
  - LF/SMF final-function summary files from build_final_functions.py.

The Schechter-like criterion is a true low-x power-law rise,
f(x) ~ C x^p with -1 < p < 0.  The classifier is deliberately stricter
than a finite-range slope check: the log-log slope must be stable across
widely separated low-x decades, which rejects logarithmic and super-power
divergences that only mimic a power law over a short interval.
"""

import glob
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import setup_paths  # noqa: F401
from extrapolation_plotter import (
    DATA_COLOUR,
    ESR_COLOURS,
    ESR_STYLES,
    LIT_COLOURS,
    LIT_LABELS,
    LIT_STYLES,
    eval_fcn,
    load_functions,
    load_lf_smf,
)


DATASETS = {
    "LF_Ser_L": {
        "pattern": "LF_Ser_L_data/final_*.dat",
        "title": "LF: Sersic",
        "xlabel": r"$\log(L\,/\,L_\odot)$",
        "xrange": (1e-4, 1e8),
        "plot_xlim": (5, 16),
        "inset_xlim": (5, 9),
    },
    "LF_cmodel_L": {
        "pattern": "LF_cmodel_L_data/final_*.dat",
        "title": "LF: cmodel",
        "xlabel": r"$\log(L\,/\,L_\odot)$",
        "xrange": (1e-4, 1e8),
        "plot_xlim": (5, 16),
        "inset_xlim": (5, 9),
    },
    "SMF_Ser_M": {
        "pattern": "SMF_Ser_M_data/final_*.dat",
        "title": "SMF: Sersic",
        "xlabel": r"$\log(M_\star\,/\,M_\odot)$",
        "xrange": (1e-4, 1e9),
        "plot_xlim": (5, 16),
        "inset_xlim": (5, 9),
    },
    "SMF_cmodel_M": {
        "pattern": "SMF_cmodel_M_data/final_*.dat",
        "title": "SMF: cmodel",
        "xlabel": r"$\log(M_\star\,/\,M_\odot)$",
        "xrange": (1e-4, 1e9),
        "plot_xlim": (5, 16),
        "inset_xlim": (5, 9),
    },
}

MAX_LINES_PER_COMPLEXITY = 30000
LOW_X_GRIDS = [
    [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    [1e-90, 1e-80, 1e-70, 1e-60, 1e-50],
    [1e-210, 1e-190, 1e-170, 1e-150, 1e-130],
]

LOWER_X_CUTS = {
    "LF_Ser_L": 6.8,
    "LF_cmodel_L": 6.8,
    "SMF_Ser_M": 6.8,
    "SMF_cmodel_M": 5.8,
}


def complexity_from_path(path):
    match = re.search(r"final_(\d+)(?:_new)?\.dat$", path)
    if match is None:
        raise ValueError(f"Could not parse complexity from {path}")
    return int(match.group(1))


def safe_raw_eval(expr, x, params):
    a0, a1, a2, a3 = params
    env = {
        "x": x,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "pow": np.power,
        "exp": np.exp,
        "log": np.log,
        "Abs": np.abs,
        "abs": np.abs,
        "re": np.real,
    }
    with np.errstate(all="ignore"):
        return eval(expr, {"__builtins__": {}}, env)


def plausible_power_string(expr):
    if "x" not in expr:
        return False
    tokens = ("pow(x", "/x", "*x", "x*", "log(x", "log(Abs(x", "log(1/x")
    return any(tok in expr for tok in tokens)


def loglog_slope(vals, xs):
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(vals, dtype=float))
    return np.polyfit(lx, ly, 1)[0]


def schechter_like_exponent(expr, params):
    """Return asymptotic p if the fitted row behaves as C x^p, else None."""
    slopes = []
    for xs in LOW_X_GRIDS:
        vals = []
        for x in xs:
            y = safe_raw_eval(expr, float(x), params)
            if np.ndim(y) != 0:
                y = np.asarray(y).item()
            if isinstance(y, complex) or not np.isfinite(y) or y <= 0:
                return None
            vals.append(float(y))
        slopes.append(loglog_slope(vals, xs))

    p = float(np.median(slopes))
    stability = max(abs(s - p) for s in slopes)
    tolerance = max(0.002, 0.015 * abs(p))
    if -0.999 < p < -0.0005 and stability < tolerance:
        return p
    return None


def load_schechter_like_candidates(data_set):
    """Scan the ranked ESR files and return strict Schechter-like candidates."""
    info = DATASETS[data_set]
    rows = []
    for path in sorted(glob.glob(info["pattern"]), key=complexity_from_path):
        comp = complexity_from_path(path)
        with open(path) as handle:
            for lineno, line in enumerate(handle, 1):
                if lineno > MAX_LINES_PER_COMPLEXITY:
                    break
                parts = line.rstrip("\n").split(";")
                if len(parts) < 11:
                    continue
                expr = parts[1]
                if not plausible_power_string(expr):
                    continue
                try:
                    params = [float(v) for v in parts[7:11]]
                    p = schechter_like_exponent(expr, params)
                except Exception:
                    continue
                if p is None:
                    continue
                rows.append({
                    "path": path,
                    "line": lineno,
                    "rank": int(parts[0]),
                    "expr": expr,
                    "dl": float(parts[2]),
                    "nll": float(parts[4]),
                    "params": params,
                    "complexity": comp,
                    "p": p,
                })
    rows.sort(key=lambda row: row["dl"])
    return rows


def load_all_esr_rows(data_set):
    """Load all finite ESR rows scanned for a dataset, not only Schechter-like rows."""
    info = DATASETS[data_set]
    rows = []
    for path in sorted(glob.glob(info["pattern"]), key=complexity_from_path):
        comp = complexity_from_path(path)
        with open(path) as handle:
            for lineno, line in enumerate(handle, 1):
                if lineno > MAX_LINES_PER_COMPLEXITY:
                    break
                parts = line.rstrip("\n").split(";")
                if len(parts) < 7:
                    continue
                try:
                    dl = float(parts[2])
                    nll = float(parts[4])
                except Exception:
                    continue
                if not (np.isfinite(dl) and np.isfinite(nll)):
                    continue
                if dl > 0 or nll > 0:
                    continue
                rows.append({
                    "path": path,
                    "line": lineno,
                    "rank_in_complexity": int(parts[0]),
                    "expr": parts[1],
                    "dl": dl,
                    "nll": nll,
                    "complexity": comp,
                })
    rows.sort(key=lambda row: row["dl"])
    return rows


def p_formula_and_value(row):
    """Return the analytic low-x slope formula for the selected Appendix rows."""
    expr = row["expr"]
    a0, a1, a2, _ = row["params"]
    if expr == "pow(x,a0)*exp(-pow(Abs(a1 - x),a2))":
        return "a0", a0
    if expr == "a0*pow(x,(a1 + pow(Abs(a2),x)))":
        return "a1 + 1", a1 + 1.0
    if expr == "pow(x,a0)/(a1 + exp(pow(x,a2)))":
        return "a0", a0
    if expr == "pow(Abs(a0),(x + log(x)))/(a1 + x)":
        return "log(abs(a0))", math.log(abs(a0))
    if expr == "a0*pow(x,(-exp(a1) + pow(Abs(a2),x)))":
        return "1 - exp(a1)", 1.0 - math.exp(a1)
    if expr == "a0/pow(x,(pow(Abs(a1),(pow(Abs(a2),x)))))":
        return "-abs(a1)", -abs(a1)
    if expr == "a0/pow(x,log(Abs((a1 - x)/a2)))":
        return "-log(abs(a1/a2))", -math.log(abs(a1 / a2))
    if expr == "pow((x*exp(re(pow(Abs(a0 - x),a1)))),a2)":
        return "a2", a2
    return "numeric low-x classifier", row["p"]


def format_function(expr):
    """Canonicalise function strings using the duplicate-removal rule used elsewhere."""
    expr = expr.replace("a0", "C").replace("a1", "C").replace("a2", "C").replace("a3", "C")
    expr = expr.replace("exp(C)", "C").replace("1/C", "C")
    expr = expr.replace("Abs(C + x)", "Abs(C - x)")
    expr = expr.replace("log(Abs(C))", "C")
    expr = expr.replace(",(C)", ",C")
    expr = expr.replace("Abs(1/C)", "Abs(C)")

    const = 0
    canonical = ""
    for char in expr:
        if char == "C":
            canonical += f"a{const}"
            const += 1
        else:
            canonical += char
    return canonical.replace("-a", "a")


def duplicate_canonical_map(candidates):
    """Map formatted functions with identical NLL to the first representative."""
    grouped_by_nll = defaultdict(list)
    flagged = {}
    for row in candidates:
        key = format_function(row["expr"])
        flagged.setdefault(key, None)
        grouped_by_nll[row["nll"]].append(key)

    for keys in grouped_by_nll.values():
        representative = None
        for key in keys:
            if flagged[key] is None and representative is None:
                representative = key
            elif representative is not None:
                flagged[key] = representative
    return flagged


def split_complexity_segments(comp_values, min_gap=3, pad=0.35):
    unique_comp = np.unique(np.asarray(comp_values, dtype=int))
    unique_comp.sort()
    if len(unique_comp) == 0:
        return []
    segments = []
    start = prev = unique_comp[0]
    for value in unique_comp[1:]:
        if value - prev >= min_gap:
            ticks = [c for c in unique_comp if start <= c <= prev]
            segments.append((start - pad, prev + pad, ticks))
            start = value
        prev = value
    ticks = [c for c in unique_comp if start <= c <= prev]
    segments.append((start - pad, prev + pad, ticks))
    return segments


def apply_lower_x_cut(segments, x_min_cut):
    if x_min_cut is None:
        return segments
    cut = []
    for xmin, xmax, ticks in segments:
        if xmax <= x_min_cut:
            continue
        cut.append((max(xmin, x_min_cut), xmax, [t for t in ticks if t >= x_min_cut]))
    return cut or segments


def add_break_marks(fig, ax_left, ax_right, dx=0.004, dy=0.007, sep=0.006):
    left_box = ax_left.get_position()
    right_box = ax_right.get_position()
    seam_x = 0.5 * (left_box.x1 + right_box.x0)
    y_bottom = left_box.y0
    y_top = left_box.y1
    fig.add_artist(Line2D([seam_x - dx, seam_x + dx],
                          [y_bottom - dy, y_bottom + dy],
                          transform=fig.transFigure, color="k", lw=1.0,
                          clip_on=False))
    fig.add_artist(Line2D([seam_x + sep - dx, seam_x + sep + dx],
                          [y_bottom - dy, y_bottom + dy],
                          transform=fig.transFigure, color="k", lw=1.0,
                          clip_on=False))
    fig.add_artist(Line2D([left_box.x1, right_box.x0], [y_top, y_top],
                          transform=fig.transFigure, color="k", lw=1.0,
                          clip_on=False))


def coloured_ylabel(fig, ax, fontsize=11, x_offset=0.05):
    box = ax.get_position()
    x = box.x0 - x_offset
    y_mid = 0.5 * (box.y0 + box.y1)
    fig.text(x, y_mid + 0.012, r"$\Delta$DL", color="C0", fontsize=fontsize,
             rotation=90, ha="center", va="bottom")
    fig.text(x, y_mid, "/", color="black", fontsize=fontsize,
             rotation=90, ha="center", va="center")
    fig.text(x, y_mid - 0.012, r"$\Delta$NLL", color="C3", fontsize=fontsize,
             rotation=90, ha="center", va="top")


def pareto_front_by_complexity(candidates):
    by_comp_dl = defaultdict(list)
    by_comp_nll = defaultdict(list)
    for row in candidates:
        by_comp_dl[row["complexity"]].append(row["dl"])
        by_comp_nll[row["complexity"]].append(row["nll"])
    comps = sorted(by_comp_dl)
    return {
        "complexity": np.asarray(comps, dtype=int),
        "dl": np.asarray([min(by_comp_dl[c]) for c in comps], dtype=float),
        "nll": np.asarray([min(by_comp_nll[c]) for c in comps], dtype=float),
    }


def plot_pareto_panel(fig, cell, data_set, candidates, title):
    source, comp, dl, nll, _, _ = load_functions(data_set)
    comp = comp.astype(int)
    dl = dl.astype(float)
    nll = nll.astype(float)

    front = pareto_front_by_complexity(candidates)
    all_comp = np.concatenate([comp, front["complexity"]])
    segments = split_complexity_segments(all_comp, min_gap=3, pad=0.35)
    segments = apply_lower_x_cut(segments, LOWER_X_CUTS.get(data_set))

    width_ratios = [max(xmax - xmin, 0.5) for xmin, xmax, _ in segments]
    inner = cell.subgridspec(1, len(segments), wspace=0.05,
                             width_ratios=width_ratios)
    axes = []
    for i in range(len(segments)):
        if i == 0:
            axes.append(fig.add_subplot(inner[0, i]))
        else:
            axes.append(fig.add_subplot(inner[0, i], sharey=axes[0]))

    esr_mask = source == "ESR"
    esr_t_mask = source == "ESR_T"
    esr_c_mask = source == "ESR_C"
    lit_mask = ~(esr_mask | esr_t_mask | esr_c_mask)

    dl_ref = np.min(dl)
    nll_ref = nll[np.argmin(dl)]

    esr_comp = comp[esr_mask]
    order = np.argsort(esr_comp)
    esr_comp = esr_comp[order]
    esr_dl = dl[esr_mask][order] - dl_ref
    esr_nll = nll[esr_mask][order] - nll_ref

    lit_source = source[lit_mask]
    lit_comp = comp[lit_mask]
    lit_dl = dl[lit_mask] - dl_ref
    lit_nll = nll[lit_mask] - nll_ref

    markers = {
        "Sch.": "x",
        "Ber.": "+",
        "Ber.orig": "1",
        "DblSch.": "d",
    }

    sl_comp = front["complexity"]
    sl_dl = front["dl"] - dl_ref
    sl_nll = front["nll"] - nll_ref

    all_visible = np.concatenate([esr_dl, esr_nll, sl_dl, sl_nll])
    ymax = max(50.0, np.nanmax(all_visible[np.isfinite(all_visible)]) * 1.12)
    y_overrides = {
        "LF_Ser_L": 1100,
        "LF_cmodel_L": 600,
        "SMF_Ser_M": 1100,
        "SMF_cmodel_M": 600,
    }
    ymax = y_overrides.get(data_set, ymax)
    ymin = min(-0.04 * ymax, np.nanmin(all_visible[np.isfinite(all_visible)]) * 1.15)

    break_pairs = []
    for ax, (xmin, xmax, ticks) in zip(axes, segments):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=10)
        ax.plot(esr_comp, esr_dl, "o-", color="C0", ms=5, lw=1.5, zorder=4)
        ax.plot(esr_comp, esr_nll, "o-", color="C3", ms=5, lw=1.5, zorder=3)

        in_seg = (sl_comp >= xmin) & (sl_comp <= xmax)
        if np.any(in_seg):
            # Keep off-scale values in the line so it clips at the plot boundary,
            # making clear that the front continues to lower-complexity points.
            ax.plot(sl_comp[in_seg], sl_dl[in_seg], "s-", color="purple",
                    ms=5, lw=1.7, zorder=7)
            ax.plot(sl_comp[in_seg], sl_nll[in_seg], "s-", color="darkcyan",
                    ms=5, lw=1.7, zorder=7)

        for src, c_lit, d_lit, n_lit in zip(lit_source, lit_comp, lit_dl, lit_nll):
            if src not in markers or not (xmin <= c_lit <= xmax):
                continue
            if ymin <= d_lit <= ymax:
                ax.scatter(c_lit, d_lit, marker=markers[src], color="C0",
                           s=70, linewidths=1.5, zorder=6)
            if ymin <= n_lit <= ymax:
                ax.scatter(c_lit, n_lit, marker=markers[src], color="C3",
                           s=70, linewidths=1.5, zorder=6)

    for left, right in zip(axes[:-1], axes[1:]):
        left.spines["right"].set_visible(False)
        right.spines["left"].set_visible(False)
        right.tick_params(labelleft=False, left=False)
        break_pairs.append((left, right))

    return axes, break_pairs, title


def candidate_curve(row, x_eval):
    y = safe_raw_eval(row["expr"], x_eval, row["params"])
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y), y, np.nan)


def top_unique_candidates(data_set, candidates, n_keep=4):
    _ = data_set
    chosen = []
    seen = set()
    flagged = duplicate_canonical_map(candidates)
    for row in candidates:
        key = format_function(row["expr"])
        canonical = flagged[key] if flagged[key] is not None else key
        if canonical in seen:
            continue
        chosen.append(row)
        seen.add(canonical)
        if len(chosen) == n_keep:
            break
    return chosen


def plot_extrapolation_schechter_like(data_set, ax, candidates, title, xlabel,
                                      x_extrap_range, inset_xlim, plot_xlim,
                                      lit_keys=None):
    if lit_keys is None:
        lit_keys = ["Sch.", "Ber.orig", "DblSch.", "Ber."]

    x_data, y_data, y_err, logm_data = load_lf_smf(data_set)
    x_eval = np.geomspace(x_extrap_range[0], x_extrap_range[1], 5000)
    logm_eval = np.log10(x_eval * 1e9)
    source, _, _, _, plot_fcn, _ = load_functions(data_set)
    top4 = top_unique_candidates(data_set, candidates, n_keep=4)

    ax.errorbar(logm_data, y_data, yerr=y_err, fmt="x", color=DATA_COLOUR,
                ms=5, elinewidth=0.7, capsize=0, zorder=10, label="Data")
    ax.axvspan(logm_data.min(), logm_data.max(), color="grey", alpha=0.08, zorder=0)
    low_x_y = y_data[np.argmin(x_data)]
    ax.axhline(low_x_y, color="black", linestyle=":", lw=1.2,
               label="Flat low-x", zorder=2)

    for rank, row in enumerate(top4):
        y_eval = candidate_curve(row, x_eval)
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        label = "SL rank {}".format(rank + 1)
        ax.plot(logm_eval, logy, color=ESR_COLOURS[rank],
                linestyle=ESR_STYLES[rank], lw=1.4, label=label, zorder=5)

    lit_indices = {}
    for key in lit_keys:
        idxs = np.where(np.array([s == key or s == key + "." for s in source]))[0]
        if len(idxs):
            idx = idxs[0]
            lit_indices[key] = idx
            y_eval = eval_fcn(plot_fcn[idx], x_eval)
            logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
            ax.plot(logm_eval, logy, color=LIT_COLOURS[key],
                    linestyle=LIT_STYLES[key], lw=1.6,
                    label=LIT_LABELS[key], zorder=4)

    ax.set_ylim(-30, 2)
    ax.set_xlim(*plot_xlim)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r"$\log\!\left(\phi\right)$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=13,
            va="top", ha="left", fontweight="bold")

    inset = ax.inset_axes([0.10, 0.05, 0.45, 0.45])
    mask_d = (logm_data >= inset_xlim[0]) & (logm_data <= inset_xlim[1])
    inset.errorbar(logm_data[mask_d], y_data[mask_d], yerr=y_err[mask_d],
                   fmt="x", color=DATA_COLOUR, ms=4, elinewidth=0.5,
                   capsize=0, zorder=10)
    inset.axhline(low_x_y, color="black", linestyle=":", lw=1.0, zorder=2)
    mask_e = (logm_eval >= inset_xlim[0]) & (logm_eval <= inset_xlim[1])
    for rank, row in enumerate(top4):
        y_eval = candidate_curve(row, x_eval)
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        inset.plot(logm_eval[mask_e], logy[mask_e], color=ESR_COLOURS[rank],
                   linestyle=ESR_STYLES[rank], lw=1.2, zorder=5)
    for key, idx in lit_indices.items():
        y_eval = eval_fcn(plot_fcn[idx], x_eval)
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        inset.plot(logm_eval[mask_e], logy[mask_e], color=LIT_COLOURS[key],
                   linestyle=LIT_STYLES[key], lw=1.4, zorder=4)
    inset.set_xlim(*inset_xlim)
    inset.set_ylim(-3.5, -1)
    inset.tick_params(labelsize=9)
    inset.set_xlabel("")
    inset.set_ylabel("")
    rect, connectors = ax.indicate_inset_zoom(inset, edgecolor="grey",
                                               alpha=0.5, linestyle="dotted")
    for line in connectors:
        if line is not None:
            line.set_linestyle("dotted")

    return top4


def make_pareto_overlay(candidates_by_dataset):
    fig = plt.figure(figsize=(12, 8))
    outer = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35)
    panel_info = []
    break_pairs = []
    for i, data_set in enumerate(DATASETS):
        axes, pairs, title = plot_pareto_panel(
            fig, outer[i // 2, i % 2], data_set,
            candidates_by_dataset[data_set], DATASETS[data_set]["title"].replace(":", "")
        )
        panel_info.append((axes, title))
        break_pairs.extend(pairs)

    fig.subplots_adjust(top=0.91, left=0.08, right=0.98, bottom=0.08)
    for axes, title in panel_info:
        coloured_ylabel(fig, axes[0], fontsize=11)
        left_box = axes[0].get_position()
        right_box = axes[-1].get_position()
        x_center = 0.5 * (left_box.x0 + right_box.x1)
        fig.text(x_center, left_box.y1 - 0.005, title,
                 ha="center", va="top", fontsize=13, fontweight="bold")
        fig.text(x_center, left_box.y0 - 0.035, "Complexity",
                 ha="center", va="top", fontsize=11)

    for left, right in break_pairs:
        add_break_marks(fig, left, right)

    handles = [
        Line2D([], [], color="grey", marker="o", linestyle="-", ms=5),
        Line2D([], [], color="purple", marker="s", linestyle="-", ms=5),
        Line2D([], [], color="darkcyan", marker="s", linestyle="-", ms=5),
        Line2D([], [], color="grey", marker="x", linestyle="None", ms=8),
        Line2D([], [], color="grey", marker="d", linestyle="None", ms=8),
        Line2D([], [], color="grey", marker="1", linestyle="None", ms=8),
    ]
    labels = [
        "ESR",
        r"Schechter-like $\Delta$DL",
        r"Schechter-like $\Delta$NLL",
        "Schechter",
        "Dbl. Schechter",
        "Bernardi (orig.)",
    ]
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.995), frameon=True)

    out = "Plots/Pareto_LF_SMF_schechter_like_overlay.png"
    pdf_out = "Final_Plots/Pareto_LF_SMF_schechter_like_overlay.pdf"
    os.makedirs("Final_Plots", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return out, pdf_out


def make_extrapolation_variant(candidates_by_dataset):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    selected = {}
    for ax, data_set in zip(axes.flat, DATASETS):
        info = DATASETS[data_set]
        selected[data_set] = plot_extrapolation_schechter_like(
            data_set, ax, candidates_by_dataset[data_set],
            title=info["title"],
            xlabel=info["xlabel"],
            x_extrap_range=info["xrange"],
            inset_xlim=info["inset_xlim"],
            plot_xlim=info["plot_xlim"],
        )

    axes[0, 0].set_ylabel(
        r"$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$",
        fontsize=12)
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 1].set_ylabel("")
    plt.setp(axes[0, 1].get_yticklabels(), visible=False)

    axes[1, 0].set_ylabel(
        r"$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$",
        fontsize=12)
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 1].set_ylabel("")
    plt.setp(axes[1, 1].get_yticklabels(), visible=False)

    plt.tight_layout(w_pad=0)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center",
               ncol=(len(labels) + 1) // 2, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 1.04))

    out = "Plots/extrapolation_schechter_like.png"
    pdf_out = "Final_Plots/extrapolation_schechter_like.pdf"
    os.makedirs("Final_Plots", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return out, pdf_out, selected


def write_table_data(candidates_by_dataset):
    out = "Plots/schechter_like_table_data.txt"
    with open(out, "w") as handle:
        handle.write(
            "# Table A1 source data for the Schechter-like LF/SMF appendix.\n"
            "# rank_full and deltas are relative to the scanned ESR rows across "
            "complexities up to MAX_LINES_PER_COMPLEXITY.\n"
            "# dataset;order;rank_full;comp;p_formula;p_value;delta_DL;delta_NLL;"
            "expr;params;source_file;source_line\n"
        )
        for data_set, candidates in candidates_by_dataset.items():
            all_rows = load_all_esr_rows(data_set)
            if not all_rows:
                raise RuntimeError(f"No ESR rows found for {data_set}")
            rank_map = {(row["path"], row["line"]): idx + 1
                        for idx, row in enumerate(all_rows)}
            dl_ref = all_rows[0]["dl"]
            nll_ref = all_rows[0]["nll"]
            for order, row in enumerate(top_unique_candidates(data_set, candidates, n_keep=2), 1):
                p_formula, p_value = p_formula_and_value(row)
                params = ",".join(f"{p:.12g}" for p in row["params"])
                handle.write(
                    "{dataset};{order};{rank_full};{comp};{p_formula};"
                    "{p_value:.12g};{delta_dl:.6f};{delta_nll:.6f};"
                    "{expr};{params};{path};{line}\n".format(
                        dataset=data_set,
                        order=order,
                        rank_full=rank_map[(row["path"], row["line"])],
                        comp=row["complexity"],
                        p_formula=p_formula,
                        p_value=p_value,
                        delta_dl=row["dl"] - dl_ref,
                        delta_nll=row["nll"] - nll_ref,
                        expr=row["expr"],
                        params=params,
                        path=row["path"],
                        line=row["line"],
                    )
                )
    return out


def main():
    os.makedirs("Plots", exist_ok=True)
    candidates_by_dataset = {}
    for data_set in DATASETS:
        candidates = load_schechter_like_candidates(data_set)
        if len(candidates) == 0:
            raise RuntimeError(f"No Schechter-like candidates found for {data_set}")
        candidates_by_dataset[data_set] = candidates
        print(f"{data_set}: {len(candidates)} strict Schechter-like candidates")
        all_rows = load_all_esr_rows(data_set)
        if not all_rows:
            raise RuntimeError(f"No ESR rows found for {data_set}")
        rank_map = {(row["path"], row["line"]): idx + 1
                    for idx, row in enumerate(all_rows)}
        dl_ref = all_rows[0]["dl"]
        nll_ref = all_rows[0]["nll"]
        for row in top_unique_candidates(data_set, candidates, n_keep=4):
            p_formula, p_value = p_formula_and_value(row)
            print("  rank {rank_full} comp {complexity} p={formula}={p_value:.6g} "
                  "DeltaDL={delta_dl:.1f} DeltaNLL={delta_nll:.1f} "
                  "{path}:{line} {expr}".format(
                      rank_full=rank_map[(row["path"], row["line"])],
                      formula=p_formula,
                      p_value=p_value,
                      delta_dl=row["dl"] - dl_ref,
                      delta_nll=row["nll"] - nll_ref,
                      **row))

    pareto_out, pareto_pdf = make_pareto_overlay(candidates_by_dataset)
    extrap_out, extrap_pdf, selected = make_extrapolation_variant(candidates_by_dataset)
    table_out = write_table_data(candidates_by_dataset)
    print(f"Saved {pareto_out}")
    print(f"Saved {pareto_pdf}")
    print(f"Saved {extrap_out}")
    print(f"Saved {extrap_pdf}")
    print(f"Saved {table_out}")


if __name__ == "__main__":
    main()
