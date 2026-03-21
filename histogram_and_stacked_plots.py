"""Histogram and stacked bar chart plots for HMF function rankings.

Reads pre-computed ranking data and produces two publication figures:
  1. Histogram of delta-DL values for the top 5 HMF functions across simulations.
  2. Stacked bar chart showing how often each function ranks 1st--5th across
     100 Quijote simulations, with a legend panel listing function expressions.

This is a lightweight laptop version of the plotting code in sample_top_200.py,
designed to run without MPI or cluster access.

Inputs:
    - histogram_data.txt   : delta-DL values per function (space-delimited).
    - ordered_gold.txt     : rank tally data (!-delimited), one row per function.

Outputs:
    - Final_Plots/ranked_histogram.pdf
    - Final_Plots/stacked_rank.pdf

Dependencies:
    numpy, matplotlib, pytexit
"""

import numpy as np
from matplotlib import pyplot as plt
import string
from pytexit import py2tex

# --- Histogram Plot --- #
data_all = np.loadtxt('histogram_data.txt', unpack=True, delimiter=' ', dtype=str)
cm = plt.get_cmap('viridis_r')

fig, ax = plt.subplots(figsize=(7, 5))
for idx, data in enumerate(data_all[:5]):
    label = data[0].replace('exp','EXP').replace('x','\sigma').replace('EXP','exp')
    for i in range(4):
        label = label.replace(f"a{i}", f"\\theta_{i}")
    data = np.astype(data[1:], float)
    bin_width = 1.5
    x = np.linspace(0, 31.5 + bin_width, int((31.5 + bin_width)/bin_width))
    ax.hist(data, bins=len(x), range=(0,31.5), alpha=0.3, density=False, histtype='bar', color=cm(idx/6), zorder=6-idx)
    ax.hist(data, bins=len(x), range=(0,31.5), alpha=1.0, density=False, histtype='step', edgecolor=cm(idx/6), linewidth=2.0, label=r'${}$'.format(label), fill=None, zorder=12-idx)


ax.legend()
ax.minorticks_on()
ax.set_xlim(left=0, right=31)
ax.set_xlabel(r'$\delta$' + 'DL')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('Final_Plots/ranked_histogram.pdf', dpi=200)
plt.show()
plt.clf()




# --- Stacked Rank Plot --- #
hmf_sims=np.arange(0,100,1)
colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#6E7B8B', '#445577']
sub_categories = ['1st', '2nd', '3rd', '4th', '5th']

ordered_gold = np.loadtxt('ordered_gold.txt', unpack=True, dtype=str, delimiter='!')
ordered_gold = ordered_gold[0]

fig = plt.figure(figsize=(7, 5))
fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
ax = axs[0]
ax2 = axs[1]

# Create bottom array for stacking
categories = len(ordered_gold)

if categories > 26:
    ordered_gold = ordered_gold[:26]
    categories = len(ordered_gold)

bottom = np.zeros(categories)
x_ticks = list(string.ascii_uppercase)[:categories]
x_plot = np.arange(0,categories,1)

# Plot each segment
for i, (color, sub_category) in enumerate(zip(colors, sub_categories)):
    y_plot = np.array([])
    for j in ordered_gold:
        j = j.split(', ')
        y_plot = np.append(y_plot,int(j[i]))

    ax.bar(x_plot, y_plot, bottom=bottom, label=sub_category, color=color)
    bottom += y_plot

ax.set_xlabel('Functions')
ax.set_xticks(x_plot, x_ticks)
ax.set_ylabel('Number of Ranks Earned (in {} simulations)'.format(len(hmf_sims)))
ax.legend(sub_categories)

labels = []

# Add function list to ax2
for idx, row in enumerate(ordered_gold):
    row = row.split(', ')
    for pos in range(0,6):
        row[pos] = int(row[pos])
        
    row[6] = np.astype(np.array(row[6][1:-1].split(';')), float)
    row[7] = row[7][1:-1]
    
    text_fcn = py2tex(row[-1].replace('Abs','abs')).replace('$','').replace(' ','')

    if r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}' in text_fcn:
        text_fcn = text_fcn.replace(r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}', r'e^{a0-\left({\frac{|a1|}{x}}\right)^{\left(a2-x\right)}}')
    elif r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}' in text_fcn:
        text_fcn = text_fcn.replace(r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}', r'{|a0|}^{\eXp{\left[\frac{e^{x^{a1}}}{x}\right]}}')

    text_fcn = text_fcn.replace('a0',r'\theta_0').replace('a1',r'\theta_1').replace('a2',r'\theta_2').replace('a3',r'\theta_3').replace('x',r'\sigma').replace('eXp','exp')

    text_fcn = text_fcn.replace('e^', ' e^')

    text = '{}: '.format(x_ticks[idx]) + r'${}$'.format(text_fcn)

    labels.append(text_fcn)

    ax2.text(0.01, (categories-idx)/categories, x_ticks[idx] + ': ', fontsize=10, ha='left', va='center', wrap=True)
    ax2.text(0.2, (categories-idx)/categories, r'${}$'.format(text_fcn), fontsize=10, ha='left', va='center', wrap=True)


# Remove the axes
ax2.axis('off')

# Get functions to print, and how many times they were in top 6
for idx, row in enumerate(ordered_gold):
    print('{0}: {1}'.format(x_ticks[idx], row[:6]))

# Save
plt.tight_layout()
plt.savefig('Final_Plots/stacked_rank.pdf',bbox_inches='tight',dpi=200)
plt.show()
plt.clf()
