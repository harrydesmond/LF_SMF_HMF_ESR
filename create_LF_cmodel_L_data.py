#!/usr/bin/env python3
"""
Create LF_cmodel_L.dat for ESR (4-column format matching LF_Ser_L.dat).

Source: LF_cmodel.txt (Bernardi et al. 2013 cmodel LF)
  Columns: M_r, log10(phi / Mpc^-3 mag^-1), sigma(log10 phi), V_eff [Mpc^3]

Output columns (matching cluster ESR format):
  L [L_sun], log10(phi), sigma, V_eff
"""
import numpy as np
import os

M_sun_r = 4.67  # Solar absolute magnitude in r-band

# Read the 4-column plotting file (already has V_eff)
data = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LF_cmodel.txt'))
M_abs = data[:, 0]
log_phi = data[:, 1]
sigma = data[:, 2]
V_eff = data[:, 3]

# Convert magnitude to linear luminosity in L_sun
log_L = -0.4 * (M_abs - M_sun_r)   # log10(L / L_sun)
L = 10**log_L                       # L in L_sun

print(f"LF_cmodel_L: {len(L)} bins")
print(f"  L [L_sun]: {L.min():.4e} to {L.max():.4e}")
print(f"  log10(phi): {log_phi.min():.3f} to {log_phi.max():.3f}")
print(f"  V_eff [Mpc^3]: {V_eff.min():.4e} to {V_eff.max():.4e}")

# Save 4-column ESR data file
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LF_and_SMF_data')
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'LF_cmodel_L.dat')
np.savetxt(outfile, np.column_stack([L, log_phi, sigma, V_eff]), fmt='%.18e')
print(f"Saved {outfile}")
