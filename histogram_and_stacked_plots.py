
import os
import setup_paths  # noqa: F401 — ensures Plots/ and Final_Plots/ exist
import numpy as np
from matplotlib import pyplot as plt
import string
from pytexit import py2tex

# --- Histogram Plot --- #
if os.path.exists('histogram_data.txt'):
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




# --- Stacked Rank Plot (untrimmed; formatting matches Fig 6) --- #
hmf_sims = np.arange(0, 100, 1)
colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#6E7B8B', '#445577']
sub_categories = ['1st', '2nd', '3rd', '4th', '5th']

ordered_gold = np.loadtxt('ordered_gold.txt', unpack=True, dtype=str, delimiter='!')
ordered_gold = ordered_gold[0]

# Parse items: (counts[5], template)
items = []
for row in ordered_gold:
    parts = row.split(', ')
    counts = [int(parts[k]) for k in range(5)]
    tmpl = parts[7][1:-1]
    items.append((counts, tmpl))

if len(items) > 26:
    items = items[:26]

categories = len(items)
x_ticks = list(string.ascii_uppercase)[:categories]
x_plot = np.arange(categories)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
ax = axs[0]
ax2 = axs[1]

bottom = np.zeros(categories)
for i, (color, sub_cat) in enumerate(zip(colors, sub_categories)):
    y_vals = np.array([it[0][i] for it in items])
    ax.bar(x_plot, y_vals, bottom=bottom, label=sub_cat, color=color)
    bottom += y_vals

ax.set_xlabel('Functions')
ax.set_xticks(x_plot)
ax.set_xticklabels(x_ticks)
ax.set_ylabel('Number of Ranks Earned (in {} simulations)'.format(len(hmf_sims)))
ax.legend(sub_categories)

for idx, (counts, tmpl) in enumerate(items):
    text_fcn = py2tex(tmpl.replace('Abs', 'abs')).replace('$', '').replace(' ', '')

    if r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}' in text_fcn:
        text_fcn = text_fcn.replace(r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}',
                                    r'e^{a0-\left({\frac{|a1|}{x}}\right)^{\left(a2-x\right)}}')
    elif r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}' in text_fcn:
        text_fcn = text_fcn.replace(r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}',
                                    r'{|a0|}^{\eXp{\left[\frac{e^{x^{a1}}}{x}\right]}}')

    text_fcn = (text_fcn.replace('a0', r'\theta_0').replace('a1', r'\theta_1')
                         .replace('a2', r'\theta_2').replace('a3', r'\theta_3')
                         .replace('x', r'\sigma').replace('eXp', 'exp'))
    text_fcn = text_fcn.replace('e^', ' e^')

    ax2.text(0.01, (categories - idx) / categories, x_ticks[idx] + ': ', fontsize=10,
             ha='left', va='center')
    ax2.text(0.2, (categories - idx) / categories, r'${}$'.format(text_fcn), fontsize=10,
             ha='left', va='center')

ax2.axis('off')

plt.tight_layout()
plt.savefig('Final_Plots/stacked_rank.pdf', bbox_inches='tight', dpi=200)
plt.show()
plt.clf()
