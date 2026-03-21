#!/usr/bin/env python3
"""Assemble *_final_functions.txt from ESR and literature fitting outputs.

This script automates the construction of the final_functions.txt files used
by the plotting and analysis scripts. It reads the per-complexity ESR outputs,
the literature (paper) fitting outputs, and (for the HMF) the combined-DL
ranking, then writes a single file per dataset containing:

  - ESR:   best function per complexity (lowest DL at each complexity)
  - ESR_T: top N functions overall across all complexities (by DL)
  - ESR_C: top N functions from combined ranking across simulations (HMF only)
  - Literature fits: Schechter, Bernardi (LF/SMF) or P.Sch., Warren, Tinker (HMF)

Usage:
    python3 build_final_functions.py <dataset> [options]

    Datasets: LF_Ser_L, LF_cmodel_L, SMF_Ser_M, SMF_cmodel_M, hmf_<sim>

    Options:
      --top N          Number of top-ranked ESR functions to include as ESR_T
                       (default: 5)
      --esr-dir DIR    Directory containing ESR per-complexity outputs
                       (default: <dataset>_data for LF/SMF)
      --paper-file F   Path to literature fit results
                       (default: <esr-dir>/all_paper_fitting_data.txt)
      --combined FILE  Path to hmf_combined_DL.txt (HMF only)
      --hmf-sim SIM    Quijote simulation index for per-sim HMF params (HMF only)
      --top-combined N Number of combined-ranked functions (HMF, default: 5)
      --outfile FILE   Output filename (default: <dataset>_final_functions.txt)

Examples:
    # LF/SMF dataset:
    python3 build_final_functions.py LF_Ser_L

    # HMF dataset (sim 50, with combined ranking):
    python3 build_final_functions.py hmf_50 --combined hmf_combined_DL.txt

Inputs:
    - <esr-dir>/final_<comp>.dat : ESR per-complexity outputs (from fit_all.py)
      Columns: rank; blank_fcn; DL; rel_prob; NLL; param_complexity;
               func_complexity; param0; param1; ...
    - <esr-dir>/all_paper_fitting_data.txt : literature function fits
      Format: source;complexity;DL;NLL;plot_fcn;blank_fcn
    - hmf_combined_DL.txt (HMF only): combined DL ranking
    - hmf_data/hmf_<sim>_data/final_all.txt (HMF only): per-sim fit results

Outputs:
    - <dataset>_final_functions.txt : semicolon-delimited file with columns
      source;complexity;DL;NLL;plot_fcn;blank_fcn

Dependencies:
    numpy
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def substitute_params(blank_fcn, params):
    """Replace a0, a1, ... in blank_fcn with numeric parameter values.

    Replaces in reverse order (a9 before a0) to avoid partial matches
    (e.g. a10 matching a1 first).
    """
    result = blank_fcn
    for i in reversed(range(len(params))):
        result = result.replace(f'a{i}', str(params[i]))
    return result


def parse_esr_line(line):
    """Parse one line of ESR per-complexity output.

    Format: rank;blank_fcn;DL;rel_prob;NLL;param_cplx;func_cplx;p0;p1;...

    Returns dict with keys: rank, blank_fcn, DL, NLL, params, plot_fcn.
    """
    parts = line.strip().split(';')
    if len(parts) < 7:
        return None

    rank = int(parts[0])
    blank_fcn = parts[1]
    DL = float(parts[2])
    # parts[3] = relative probability (unused)
    NLL = float(parts[4])
    # parts[5] = parametric complexity, parts[6] = functional complexity

    params = [float(p) for p in parts[7:] if p.strip()]

    plot_fcn = substitute_params(blank_fcn, params)

    return {
        'rank': rank,
        'blank_fcn': blank_fcn,
        'DL': DL,
        'NLL': NLL,
        'params': params,
        'plot_fcn': plot_fcn,
    }


def parse_combined_line(line):
    """Parse one line of hmf_combined_DL.txt.

    Format: rank;function;DL_combined;delta_DL;sum_NLL;sum_DL;aifeynman;n_sims

    Returns dict with keys: rank, blank_fcn, DL_combined.
    """
    parts = line.strip().split(';')
    return {
        'rank': int(parts[0]),
        'blank_fcn': parts[1],
        'DL_combined': float(parts[2]),
    }


def parse_final_all_line(line):
    """Parse one line of hmf_<sim>_data/final_all.txt.

    Format: rank;blank_fcn;DL;NLL;[params]

    Returns dict with keys: rank, blank_fcn, DL, NLL, params, plot_fcn.
    """
    parts = line.strip().split(';')
    rank = int(parts[0])
    blank_fcn = parts[1]
    DL = float(parts[2])
    NLL = float(parts[3])

    # Parse params from numpy-style array string like "[0.65 1.69 1.63 2.78]"
    param_str = parts[4].strip().strip('[]')
    params = [float(x) for x in param_str.split()]

    plot_fcn = substitute_params(blank_fcn, params)

    return {
        'rank': rank,
        'blank_fcn': blank_fcn,
        'DL': DL,
        'NLL': NLL,
        'params': params,
        'plot_fcn': plot_fcn,
    }


def format_output_line(source, complexity, DL, NLL, plot_fcn, blank_fcn):
    """Format a single output line for *_final_functions.txt."""
    return f'{source};{complexity};{DL};{NLL};{plot_fcn};{blank_fcn}'


def normalise_fcn(fcn_str):
    """Normalise a function string for duplicate detection.

    Replaces all parameter names (a0, a1, ...) with 'C' and strips whitespace.
    """
    result = re.sub(r'a\d+', 'C', fcn_str)
    return result.replace(' ', '')


# ──────────────────────────────────────────────────────────────────────
# Main assembly
# ──────────────────────────────────────────────────────────────────────

def load_esr_per_complexity(esr_dir, dataset):
    """Load all ESR per-complexity output files for a dataset.

    Returns list of (complexity, entries) tuples, where entries is a list
    of parsed dicts sorted by DL (best first).
    """
    # Try both naming conventions: final_<comp>.dat and final_<comp>_new.dat
    pattern_new = os.path.join(esr_dir, 'final_*_new.dat')
    pattern_old = os.path.join(esr_dir, 'final_*.dat')

    files = sorted(glob.glob(pattern_new))
    if not files:
        files = sorted(glob.glob(pattern_old))

    # Exclude final_all.txt (HMF step2 output)
    files = [f for f in files if 'final_all' not in os.path.basename(f)]

    if not files:
        print(f'Warning: no ESR output files found in {esr_dir}')
        return []

    results = []
    for filepath in files:
        # Extract complexity from filename: final_<comp>.dat or final_<comp>_new.dat
        basename = os.path.basename(filepath)
        match = re.search(r'final_(\d+)', basename)
        if not match:
            continue
        comp = int(match.group(1))

        entries = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parsed = parse_esr_line(line)
                if parsed is not None:
                    entries.append(parsed)

        # Sort by DL (lowest = best)
        entries.sort(key=lambda e: e['DL'])
        results.append((comp, entries))

    results.sort(key=lambda x: x[0])
    return results


def load_paper_fits(paper_file):
    """Load literature fitting results from all_paper_fitting_data.txt.

    Each line has format: source;complexity;DL;NLL;plot_fcn;blank_fcn
    Returns list of line strings (already in output format).
    """
    if not os.path.isfile(paper_file):
        print(f'Warning: paper fits not found at {paper_file}')
        return []

    lines = []
    with open(paper_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
    return lines


def load_combined_ranking(combined_file, final_all_dir, hmf_sim, n_top):
    """Load top N functions from HMF combined DL ranking and get per-sim params.

    Args:
        combined_file: path to hmf_combined_DL.txt
        final_all_dir: directory containing hmf_<sim>_data/final_all.txt
        hmf_sim: simulation index (e.g. 50) for per-sim parameters
        n_top: number of top combined-ranked functions to include

    Returns list of output line strings with source=ESR_C.
    """
    if not os.path.isfile(combined_file):
        print(f'Warning: combined DL file not found at {combined_file}')
        return []

    # Load combined ranking
    combined = []
    with open(combined_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            combined.append(parse_combined_line(line))

    top_combined = combined[:n_top]

    # Load per-sim results for parameter values
    sim_file = os.path.join(final_all_dir,
                            f'hmf_{hmf_sim}_data', 'final_all.txt')
    sim_entries = {}
    if os.path.isfile(sim_file):
        with open(sim_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parsed = parse_final_all_line(line)
                norm_fcn = normalise_fcn(parsed['blank_fcn'])
                sim_entries[norm_fcn] = parsed

    output_lines = []
    for entry in top_combined:
        norm_fcn = normalise_fcn(entry['blank_fcn'])

        if norm_fcn in sim_entries:
            # Use per-sim DL/NLL and parameters
            sim = sim_entries[norm_fcn]
            line = format_output_line(
                'ESR_C', 10, sim['DL'], sim['NLL'],
                sim['plot_fcn'], entry['blank_fcn'])
        else:
            # No per-sim data; use combined DL (no plot_fcn available)
            print(f'Warning: combined-ranked function not found in sim {hmf_sim}: '
                  f'{entry["blank_fcn"]}')
            line = format_output_line(
                'ESR_C', 10, entry['DL_combined'], '',
                entry['blank_fcn'], entry['blank_fcn'])

        output_lines.append(line)

    return output_lines


def build_final_functions(args):
    """Main assembly routine."""
    dataset = args.dataset
    is_hmf = 'hmf' in dataset.lower() and 'SMF' not in dataset and 'LF' not in dataset

    # Determine ESR output directory
    if args.esr_dir:
        esr_dir = args.esr_dir
    elif is_hmf:
        # HMF: look in hmf_data/hmf_<sim>_data/ or hmf_<sim>_data/
        sim = dataset.replace('hmf_', '')
        candidates = [
            os.path.join('hmf_data', f'hmf_{sim}_data'),
            f'hmf_{sim}_data',
        ]
        esr_dir = next((d for d in candidates if os.path.isdir(d)), candidates[0])
    else:
        esr_dir = f'{dataset}_data'

    # Determine paper fits file
    if args.paper_file:
        paper_file = args.paper_file
    else:
        paper_file = os.path.join(esr_dir, 'all_paper_fitting_data.txt')

    # Output file
    if args.outfile:
        outfile = args.outfile
    else:
        outfile = f'{dataset}_final_functions.txt'

    n_top = args.top

    print(f'Dataset:        {dataset}')
    print(f'ESR output dir: {esr_dir}')
    print(f'Paper fits:     {paper_file}')
    print(f'Output:         {outfile}')
    print(f'Top N (ESR_T):  {n_top}')
    print()

    output_lines = []

    # ── 1. Best ESR function per complexity ──
    esr_data = load_esr_per_complexity(esr_dir, dataset)

    all_esr = []  # (DL, complexity, entry) for overall ranking
    for comp, entries in esr_data:
        if entries:
            best = entries[0]
            line = format_output_line(
                'ESR', comp, best['DL'], best['NLL'],
                best['plot_fcn'], best['blank_fcn'])
            output_lines.append(line)

            for e in entries:
                all_esr.append((e['DL'], comp, e))

    # ── 2. Top N overall ESR functions (ESR_T) ──
    all_esr.sort(key=lambda x: x[0])

    seen = set()
    n_added = 0
    for DL, comp, entry in all_esr:
        if n_added >= n_top:
            break
        norm = normalise_fcn(entry['blank_fcn'])
        if norm in seen:
            continue
        seen.add(norm)

        line = format_output_line(
            'ESR_T', comp, entry['DL'], entry['NLL'],
            entry['plot_fcn'], entry['blank_fcn'])
        output_lines.append(line)
        n_added += 1

    # ── 3. Combined ranking (HMF only) ──
    if is_hmf and args.combined:
        sim = dataset.replace('hmf_', '')
        # Determine directory containing hmf_<sim>_data/final_all.txt
        if os.path.isdir('hmf_data'):
            final_all_dir = 'hmf_data'
        else:
            final_all_dir = '.'

        combined_lines = load_combined_ranking(
            args.combined, final_all_dir, sim, args.top_combined)
        output_lines.extend(combined_lines)

    # ── 4. Literature fits ──
    paper_lines = load_paper_fits(paper_file)
    output_lines.extend(paper_lines)

    # ── Write output ──
    with open(outfile, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    # Summary
    sources = {}
    for line in output_lines:
        src = line.split(';')[0]
        sources[src] = sources.get(src, 0) + 1

    print(f'Written {len(output_lines)} entries to {outfile}:')
    for src, count in sources.items():
        print(f'  {src}: {count}')


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Assemble *_final_functions.txt from ESR and literature outputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 build_final_functions.py LF_Ser_L
  python3 build_final_functions.py hmf_50 --combined hmf_combined_DL.txt
  python3 build_final_functions.py SMF_cmodel_M --top 8 --outfile my_output.txt
""")

    parser.add_argument('dataset',
                        help='Dataset name (e.g. LF_Ser_L, SMF_cmodel_M, hmf_50)')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of top-ranked ESR functions (ESR_T entries, default: 5)')
    parser.add_argument('--esr-dir',
                        help='Directory containing ESR per-complexity outputs')
    parser.add_argument('--paper-file',
                        help='Path to literature fit results (all_paper_fitting_data.txt)')
    parser.add_argument('--combined',
                        help='Path to hmf_combined_DL.txt (HMF only)')
    parser.add_argument('--top-combined', type=int, default=5,
                        help='Number of combined-ranked functions (HMF, default: 5)')
    parser.add_argument('--outfile',
                        help='Output filename (default: <dataset>_final_functions.txt)')

    args = parser.parse_args()
    build_final_functions(args)
