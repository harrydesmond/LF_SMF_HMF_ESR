"""Plot effective survey volume V_eff vs luminosity and stellar mass.

Creates figures showing V_eff for the four SDSS LF/SMF datasets (Sersic and
cmodel photometry), illustrating how brighter/more massive galaxies are
detected to larger volumes.

Figures produced:
    - Final_Plots/Veff_LF.pdf   (LF panel)
    - Final_Plots/Veff_SMF.pdf  (SMF panel)
    - Final_Plots/Veff.pdf      (combined 1x2 figure)

Inputs:
    - data/LF_Ser_L.txt, data/LF_cmodel.txt, data/SMF_Ser_M.txt, data/SMF_cmodel_M.txt
      (4-column data: x, log10(phi), sigma, Veff)

Dependencies:
    numpy, matplotlib
"""

import numpy as np
from matplotlib import pyplot as plt

# LF data: columns are L (L_sun), log10(phi), sigma, Veff
LF_Ser = np.loadtxt('data/LF_Ser_L.txt', dtype=float, delimiter=' ')
LF_cmod = np.loadtxt('data/LF_cmodel.txt', dtype=float, delimiter=' ')
SMF_Ser = np.loadtxt('data/SMF_Ser_M.txt', dtype=float, delimiter=' ')
SMF_cmod = np.loadtxt('data/SMF_cmodel_M.txt', dtype=float, delimiter=' ')

# LF panel (standalone)
fig_lf, ax_lf = plt.subplots(1, 1, figsize=(5, 4))
ax_lf.plot(np.log10(LF_Ser[:, 0]), LF_Ser[:, 3], 'o-', color='C0', ms=6, lw=2.0, label='Sersic')
lf_cmod_logL = -0.4 * (LF_cmod[:, 0] - 4.67)
ax_lf.plot(lf_cmod_logL, LF_cmod[:, 3], 's-', color='C1', ms=6, lw=2.0, label='cmodel')
ax_lf.set_xlabel(r'$\log_{10}(L / L_\odot)$', fontsize=19)
ax_lf.set_ylabel(r'$V_{\rm eff}$ [Mpc$^3$]', fontsize=19)
ax_lf.set_yscale('log')
ax_lf.legend(fontsize=15)
ax_lf.tick_params(labelsize=16)
fig_lf.tight_layout()
fig_lf.savefig('Final_Plots/Veff_LF.pdf', dpi=200, bbox_inches='tight')

# SMF panel (standalone, no legend)
fig_smf, ax_smf = plt.subplots(1, 1, figsize=(5, 4), sharey=ax_lf)
ax_smf.plot(np.log10(SMF_Ser[:, 0]), SMF_Ser[:, 3], 'o-', color='C0', ms=6, lw=2.0, label='Sersic')
ax_smf.plot(np.log10(SMF_cmod[:, 0]), SMF_cmod[:, 3], 's-', color='C1', ms=6, lw=2.0, label='cmodel')
ax_smf.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=19)
ax_smf.set_ylabel(r'$V_{\rm eff}$ [Mpc$^3$]', fontsize=19)
ax_smf.set_yscale('log')
ax_smf.tick_params(labelsize=16)
fig_smf.tight_layout()
fig_smf.savefig('Final_Plots/Veff_SMF.pdf', dpi=200, bbox_inches='tight')

# Combined version (kept for compatibility)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax1.plot(np.log10(LF_Ser[:, 0]), LF_Ser[:, 3], 'o-', color='C0', ms=6, lw=2.0, label='Sersic')
ax1.plot(lf_cmod_logL, LF_cmod[:, 3], 's-', color='C1', ms=6, lw=2.0, label='cmodel')
ax1.set_xlabel(r'$\log_{10}(L / L_\odot)$', fontsize=19)
ax1.set_ylabel(r'$V_{\rm eff}$ [Mpc$^3$]', fontsize=19)
ax1.set_yscale('log')
ax1.legend(fontsize=15)
ax1.tick_params(labelsize=16)

ax2.plot(np.log10(SMF_Ser[:, 0]), SMF_Ser[:, 3], 'o-', color='C0', ms=6, lw=2.0, label='Sersic')
ax2.plot(np.log10(SMF_cmod[:, 0]), SMF_cmod[:, 3], 's-', color='C1', ms=6, lw=2.0, label='cmodel')
ax2.set_xlabel(r'$\log_{10}(M_\star / M_\odot)$', fontsize=19)
ax2.set_yscale('log')
ax2.tick_params(labelsize=16)

fig.tight_layout()
fig.savefig('Final_Plots/Veff.pdf', dpi=200, bbox_inches='tight')
plt.show()
