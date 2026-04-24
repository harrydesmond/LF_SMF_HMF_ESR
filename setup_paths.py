"""Side-effect import: create standard output directories and expose REPO_ROOT.

Import once at the top of any script that writes plots or result files; the
module creates `Plots/`, `Plots/Old/`, and `Final_Plots/` under the repo root
if they don't already exist. Scripts can also reference `REPO_ROOT` to
chdir/resolve paths without hardcoding machine-specific locations.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

for _d in ("Plots", os.path.join("Plots", "Old"), "Final_Plots"):
    os.makedirs(os.path.join(REPO_ROOT, _d), exist_ok=True)
