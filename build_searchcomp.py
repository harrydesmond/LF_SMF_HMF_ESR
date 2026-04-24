#!/usr/bin/env python3
"""Build function -> minimum-search-complexity mapping from ESR per-complexity outputs.

The ESR search at complexity C enumerates all functions of complexity <= C,
so the minimum C at which a function appears in the search outputs is its
ESR search (a.k.a. generation) complexity. This script scans the per-sim
per-complexity output files and writes the resulting mapping.

Scans all available per-sim ESR output directories by default, since the
top combined-ranking functions may not all appear in any single sim's
top-N output at a given complexity.

The output is consumed by:
  - fiducial_checks_and_plots.py (reads hmf_fiducial_searchcomp.txt)
  - generate_extended_appendix.py, Pareto_plotter.py (read hmf_func_gencomp.txt)

Usage:
    python3 build_searchcomp.py              # fiducial -> hmf_fiducial_searchcomp.txt
    python3 build_searchcomp.py --extended   # full range         -> hmf_func_gencomp.txt
    python3 build_searchcomp.py --outfile custom_name.txt
"""

import argparse
import glob
import os
import re


def scan_complexity_outputs(fiducial):
    """Return {function_string: min_complexity} scanning all per-sim ESR outputs."""
    if fiducial:
        dir_glob = 'hmf_fiducial_*_data'
        # Accept final_<C>_fiducial.dat
        def keep(fn):
            return bool(re.fullmatch(r'final_\d+_fiducial\.dat', fn))
    else:
        # Glob with leading digit to avoid matching hmf_fiducial_*_data
        dir_glob = 'hmf_[0-9]*_data'
        # Accept final_<C>.dat or final_<C>_new.dat (either ESR-search convention).
        def keep(fn):
            return bool(re.fullmatch(r'final_\d+(_new)?\.dat', fn))

    dirs = [d for d in sorted(glob.glob(dir_glob)) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(
            f"No directories matching {dir_glob!r}. "
            "Run the ESR pipeline (sample_top_200.py step1 or run_hmf_fiducial_step1.py) first."
        )

    files = []
    for d in dirs:
        for fp in sorted(glob.glob(os.path.join(d, 'final_*.dat'))):
            if keep(os.path.basename(fp)):
                files.append(fp)
    if not files:
        raise FileNotFoundError(
            f"No per-complexity output files found in {dirs}. "
            "Run the ESR pipeline first."
        )

    mapping = {}
    for fp in files:
        m = re.search(r'final_(\d+)', os.path.basename(fp))
        if not m:
            continue
        comp = int(m.group(1))

        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                func = parts[1]
                if func not in mapping or comp < mapping[func]:
                    mapping[func] = comp

    return mapping, dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--extended', action='store_true',
                        help='Scan full-range ESR outputs. Default: fiducial.')
    parser.add_argument('--outfile', default=None,
                        help='Output filename. Default depends on --extended.')
    args = parser.parse_args()
    fiducial = not args.extended

    if args.outfile:
        outfile = args.outfile
    elif fiducial:
        outfile = 'hmf_fiducial_searchcomp.txt'
    else:
        outfile = 'hmf_func_gencomp.txt'

    mapping, dirs = scan_complexity_outputs(fiducial)

    label = 'search_complexity' if fiducial else 'generation_complexity'
    with open(outfile, 'w') as f:
        f.write(f'# function;{label}\n')
        f.write(f'# Minimum ESR complexity at which each function appears across {len(dirs)} per-sim output dirs.\n')
        for func, comp in sorted(mapping.items(), key=lambda kv: (kv[1], kv[0])):
            f.write(f'{func};{comp}\n')

    print(f"Wrote {len(mapping)} entries to {outfile} (scanned {len(dirs)} sim dirs)")


if __name__ == '__main__':
    main()
