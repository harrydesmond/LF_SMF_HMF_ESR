"""Plot per-bin delta-NLL contributions for literature vs ESR functions.

For each dataset, computes the Poisson NLL contribution at each data bin for
each literature function (Schechter, Bernardi, Press-Schechter, Warren, Tinker)
relative to the best ESR function. Positive values indicate bins where the
literature function fits worse than the ESR function.

Figures produced:
    - Final_Plots/LF_functions_NLL.pdf     (LF Sersic + cmodel)
    - Final_Plots/SMF_functions_NLL.pdf    (SMF Sersic + cmodel)
    - Final_Plots/HMF_50_functions_NLL.pdf (HMF realisation 50)

Inputs:
    - *_final_functions.txt : function definitions with parameters
    - *.txt data files      : binned LF/SMF/HMF data
    - mass_variance_multiplier.txt : sigma(M) conversion for HMF

Dependencies:
    numpy, matplotlib, pytexit, scipy
"""

import numpy as np
import os
from matplotlib import pyplot as plt
from pytexit import py2tex
import matplotlib.cm as cm
from scipy.special import gamma as Gamma
from matplotlib.lines import Line2D
# --- For colours on graph function lines --- #
cm = plt.get_cmap('Set1')

def main(data_set, colour, found):
    source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt('{}_final_functions.txt'.format(data_set), dtype=str, delimiter=';', unpack=True)

    DL, NLL = DL.astype(float), NLL.astype(float)

    # Find best ESR function (lowest DL): prefer ESR_T if present, else ESR/ESR_C
    esr_t_idx = [i for i, s in enumerate(source) if s == 'ESR_T']
    esr_c_idx = [i for i, s in enumerate(source) if s == 'ESR_C']
    esr_idx = [i for i, s in enumerate(source) if s == 'ESR']
    if esr_t_idx:
        best_esr_idx = esr_t_idx[np.argmin(DL[esr_t_idx])]
    elif esr_c_idx:
        best_esr_idx = esr_c_idx[np.argmin(DL[esr_c_idx])]
    else:
        best_esr_idx = esr_idx[np.argmin(DL[esr_idx])]
    
    # HMF
    if 'hmf' in data_set:
        _, counts, y_err, Veff_factor_delta, _ = np.loadtxt('data/hmf_files/{}.dat'.format(data_set), dtype=float, delimiter=' ', unpack=True)
        # M = x
        logM, sigma, factor = np.loadtxt('data/mass_variance_multiplier.txt', dtype=float, unpack=True)
        # NOTE: logM is [Msun/h]
        logM = logM[:len(counts)]
        sigma = sigma[:len(counts)]
        factor = factor[:len(counts)]
        
        x = sigma
        M = logM

        Veff = 1e9/0.6711**3 # Mpc^3
        delta_logm = 0.2
        
        y = np.log10( counts / (Veff * delta_logm) )
        y_err = 1 / ( np.log(10) * np.sqrt(counts) )
        
        labels = {
            'P.Sch.': r'$f(\sigma) = \sqrt{\frac{2}{\pi}} \left(\frac{\delta_c}{\sigma} \right) \exp\left(-\frac{\delta_c^2}{2\sigma^2}\right)$',
            'War.': r'$f(\sigma) = \theta_1 (\sigma^{\theta_2} + \theta_3) \exp\left(-\frac{\theta_4}{\sigma^2} \right)$',
            'Tin.': r'$f(\sigma) = \theta_1\left[\left(\frac{\sigma}{\theta_2}\right)^{-\theta_3}+1\right]e^{-\theta_4/\sigma^2}$'
        }

        x_label='\sigma'
        data_set_label = 'HMF'
      
    # LF or SMF
    else:
        data_file = 'data/{}.txt'.format(data_set)
        if not os.path.isfile(data_file):
            # Some LF/SMF datasets use *_L / *_M names for fit outputs but keep
            # raw binned data in the base file (e.g., LF_cmodel.txt).
            for suffix in ('_L', '_M'):
                if data_set.endswith(suffix):
                    alt = 'data/{}.txt'.format(data_set[:-2])
                    if os.path.isfile(alt):
                        data_file = alt
                    break
        M, y, y_err, Veff = np.loadtxt(data_file, dtype=float, delimiter=' ', unpack=True)

        if M[0] < 0:
            # Column 0 is absolute magnitude; convert to L in L_sun
            M_sun_r = 4.67
            M = 10**(-0.4 * (M - M_sun_r))
        x = M*1e-09
        M = np.log10(M)
        counts = 10**(y) * Veff
        
        factor = 1
        delta_logm = 1
        
        # X labels
        if 'LF' in data_set:
            x_label='L'
        else:
            x_label='M'
            
        if 'Ser' in data_set:
            data_set_label = 'Sersic'
        else:
            data_set_label = 'Cmodel'
            

    for idx, fcn in enumerate(plot_fcn):
        label = py2tex(blank_fcn[idx].replace('Abs','abs')).replace('$','').replace('a0','\Theta_0').replace('a1','\Theta_1').replace('a2','\Theta_2').replace('a3','\Theta_3').replace('x',x_label)
        label = label.replace('\\log', '\\ln')
        label = r'${}$'.format(label)
                
        y_fcn = eval(fcn.replace("log","np.log").replace("Abs","abs").replace("exp","np.exp"))
    
        # Schechter
        if source[idx] == 'Sch.':
            y_plot = np.log10(y_fcn)
            nll_contributions = y_fcn * (factor * Veff * delta_logm) - counts * np.log(y_fcn * (factor * Veff * delta_logm)) - nll_old
            if source[idx] not in found:
                label = r'$\phi_\ast \left( \left( \frac{L}{L_\ast} \right)^{\alpha} \exp \left[ -\frac{L}{L_\ast} \right] \right)$'
                label = label.replace('L',x_label)
                ax[0].plot(M, nll_contributions, label= 'Sch.: ' + label, color=colour, linestyle='--')
                found.append(source[idx])
            else:
                ax[0].plot(M, nll_contributions, color=colour, linestyle='--')
            
        # Press-Schechter
        elif source[idx] == 'P.Sch.':
            y_plot = np.log10(y_fcn * factor)
            nll_contributions = y_fcn * (factor * Veff * delta_logm) - counts * np.log(y_fcn * (factor * Veff * delta_logm)) - nll_old
            ax[0].plot(M, nll_contributions, label= 'P.Sch.: ' + labels[source[idx]], color='orange', linestyle='--')
            
        # Warren
        elif source[idx] == 'War.':
            nll_contributions = y_fcn * (factor * Veff * delta_logm) - counts * np.log(y_fcn * (factor * Veff * delta_logm)) - nll_old
            ax[0].plot(M, nll_contributions, label= 'War.: ' + labels[source[idx]], color='green', linestyle='--')
    
        # Tinker
        elif source[idx] == 'Tin.':
            y_plot = np.log10(y_fcn  * factor)
            nll_contributions = y_fcn * (factor * Veff * delta_logm) - counts * np.log(y_fcn * (factor * Veff * delta_logm)) - nll_old
            ax[0].plot(M, nll_contributions, label= 'Tin.: ' + labels[source[idx]], color='blue', linestyle='--')            
            ax[0].set_ylim(top=80, bottom=-40)
            
        # ESR's best (lowest DL)
        elif idx == best_esr_idx:
            y_plot = np.log10(y_fcn  * factor)
            nll_contributions = y_fcn * (factor * Veff * delta_logm) - counts * np.log(y_fcn * (factor * Veff * delta_logm))
            nll_old = nll_contributions
            # Skip plotting delta-NLL for best ESR: identically zero by construction (not shown)
        
        # Bernardi
        elif 'Ber.' in source[idx]:
            y_plot = np.log10(y_fcn * factor)
            nll_contributions = y_fcn * (factor * Veff * delta_logm) - counts * np.log(y_fcn * (factor * Veff * delta_logm)) - nll_old
            if source[idx] not in found:
                label = r'$\phi_{\ast} \beta \left( \frac{X}{X_{\ast}} \right)^{\alpha} \cdot \frac{ \exp \left[ -\left( {X}/{X_{\ast}} \right)^{\beta} \right] }{\operatorname{\Gamma}\left({\alpha}/{\beta}\right)} - \phi_{\gamma} \left( \frac{X}{X_{\gamma}} \right)^{\gamma} \cdot \exp \left[-\frac{X}{X_{\gamma}} \right] $'
                ax[0].plot(M, nll_contributions, label=source[idx] + ': ' + label.replace('X',x_label), color=colour, linestyle=(0, (1, 1)))#, linewidth=3)
                found.append(source[idx])
            else:
                ax[0].plot(M, nll_contributions, color=colour, linestyle=(0, (1, 1)))#, linewidth=3)

        else:
            continue
        

    ax[0].set_xlim([min(M)*0.999, max(M)*1.001])
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    
    # Exclusively handles the legend for LF or SMF
    if 'hmf' not in data_set:
        handles, labels = ax[0].get_legend_handles_labels()
    
        # --- Formats the legend --- #
        new_handles = []
        for idx, label in enumerate(labels):
            if label[:3] == 'Sch':
                new_handles.append(Line2D([], [], color='black', linestyle='--', label=labels[-2]))
            elif label[:3] == 'Ber':
                new_handles.append(Line2D([], [], color='black', linestyle=(0, (1, 1)), label=labels[-1]))
            else:
                new_handles.append(handles[idx])
                
        new_handles = [new_handles[-1]] + new_handles[:-1]
        labels = [labels[-1]] + labels[:-1]
        
    else:
        new_handles, labels = ax[0].get_legend_handles_labels()
    
    # LF axes
    if 'LF' in data_set:
        ax[0].set_xlabel(r'$\log\!\left( \frac{L}{L_\odot} \right)$')
        ax[0].set_ylabel(r'$\delta$' + ' NLL Contributions')
        ax[0].legend(new_handles, labels, fontsize=8, title='LF Model Comparisons:')

    # SMF axes
    elif 'SMF' in data_set:
        ax[0].set_xlabel(r'$\log\!\left( \frac{M_*}{h M_\odot} \right)$')
        ax[0].set_ylabel(r'$\delta$' + ' NLL Contributions')
        ax[0].legend(new_handles, labels, fontsize=8, title='SMF Model Comparisons:')
        
    # HMF axes
    else:
        ax[0].set_xlabel(r'$\log(M)\,[\mathrm{M_\odot}/h]$')
        ax[0].set_ylabel(r'$\delta$' + ' NLL Contributions')
        ax[0].legend(fontsize=8, title='HMF Model Comparisons:', loc=(0.1, 0.55))
    

# SMF
found = []
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax = [ax]
main('SMF_Ser_M', cm(0), found)
main('SMF_cmodel_M', cm(1), found)
plt.savefig('Final_Plots/SMF_functions_NLL.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()

# LF
found = []
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax = [ax]
main('LF_Ser_L', cm(0), found)
main('LF_cmodel_L', cm(1), found)
plt.savefig('Final_Plots/LF_functions_NLL.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()

# HMF
fig, ax = plt.subplots(1, 1, figsize=(7, 5))#, gridspec_kw={'height_ratios': [3, 1]})
ax = [ax]
main('hmf_50', cm(0), [])
plt.savefig('Final_Plots/HMF_50_functions_NLL.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()
