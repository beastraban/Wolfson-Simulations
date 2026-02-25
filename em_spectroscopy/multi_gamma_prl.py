"""
PRL-quality multi-γ validation figure.
Reads multi_gamma_results.npz from the 1000-realization run.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# =========================================================================
# PRL style
# =========================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})

# =========================================================================
# Load data
# =========================================================================
d = np.load('multi_gamma_results.npz')
GAMMAS = d['gammas']
T_TRUE = float(d['T_TRUE'])
THETA_CENTERS = d['theta_centers']

# Reconstruct results
results = {}
for g in GAMMAS:
    key = f'g{g:.2f}'
    beta_true = np.sqrt(1 - 1/g**2)
    beta_eb = float(d[f'{key}_beta_eb'])
    beta_eb_std = float(d[f'{key}_beta_eb_std'])
    gamma_eb = 1/np.sqrt(max(1e-10, 1 - beta_eb**2))
    T0_pix = d[f'{key}_T0pix']
    T_lab = d[f'{key}_Tlab']
    nmodes = d[f'{key}_nmodes']
    results[g] = {
        'beta_true': beta_true, 'beta_eb': beta_eb,
        'beta_eb_std': beta_eb_std, 'gamma_eb': gamma_eb,
        'T0_per_pixel': T0_pix, 'T_lab': T_lab, 'nmodes': nmodes,
        'T0_mean': np.nanmean(T0_pix), 'T0_std': np.nanstd(T0_pix),
    }

# =========================================================================
# Colors
# =========================================================================
# Colorblind-friendly palette
cmap = plt.cm.viridis
colors_gamma = [cmap(i / (len(GAMMAS) - 1)) for i in range(len(GAMMAS))]

# Selected gammas for panel C
show_gammas = [1.1, 1.5, 3.0, 10.0]
sel_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']  # Okabe-Ito

# =========================================================================
# Figure: PRL two-column width = 3.375 in; here we use full width ~7 in
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
plt.subplots_adjust(hspace=0.42, wspace=0.34, left=0.09, right=0.97,
                    top=0.88, bottom=0.08)

# Panel labels
for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
    ax.annotate(label, xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=11, fontweight='bold', va='top')

# =========================================================================
# (a) T₀ recovery vs γ
# =========================================================================
ax = axes[0, 0]

# ±1% band
ax.axhspan(T_TRUE*0.99, T_TRUE*1.01, color='#d4e6f1', alpha=0.7, zorder=0)
ax.axhline(T_TRUE, color='#c0392b', ls='-', lw=1.2, zorder=1)

for ig, gamma in enumerate(GAMMAS):
    T0_pix = results[gamma]['T0_per_pixel']
    good = ~np.isnan(T0_pix)
    x = np.full(np.sum(good), gamma)
    # Small deterministic horizontal spread
    offsets = np.linspace(-0.015, 0.015, np.sum(good))
    x_plot = gamma * (1 + offsets)
    ax.scatter(x_plot, T0_pix[good], s=12, alpha=0.7,
               color=colors_gamma[ig], edgecolors='none', zorder=5)
    # Mean ± std
    ax.errorbar(gamma, results[gamma]['T0_mean'],
                yerr=results[gamma]['T0_std'],
                fmt='_', color='k', ms=10, mew=1.5, capsize=2,
                capthick=0.8, elinewidth=0.8, zorder=10)

ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'Recovered $T_0\;(\hbar\omega_p)$')
ax.set_title(r'Rest-frame temperature recovery')
ax.set_xticks([1, 1.5, 2, 3, 5, 10])
ax.set_xticklabels(['1', '1.5', '2', '3', '5', '10'])
ax.set_xlim(0.98, 13)
ax.set_ylim(T_TRUE * 0.88, T_TRUE * 1.12)

# Custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
leg_elements = [
    Line2D([0], [0], color='#c0392b', lw=1.2, label=f'$T_0 = {T_TRUE:.0f}$'),
    Patch(facecolor='#d4e6f1', alpha=0.7, label=r'$\pm 1\%$ band'),
    Line2D([0], [0], marker='_', color='k', lw=0, ms=8, mew=1.5,
           label=r'Mean $\pm$ std'),
]
ax.legend(handles=leg_elements, loc='lower left', framealpha=0.9,
          edgecolor='0.7', fontsize=7)

# =========================================================================
# (b) γ recovery
# =========================================================================
ax = axes[0, 1]
gamma_ebs = np.array([results[g]['gamma_eb'] for g in GAMMAS])
gamma_eb_errs = np.array([
    results[g]['beta_eb_std'] * results[g]['beta_eb'] * results[g]['gamma_eb']**3
    for g in GAMMAS])

gline = np.linspace(0.95, 12, 100)
ax.plot(gline, gline, color='#c0392b', ls='-', lw=1.0, zorder=1)
ax.errorbar(GAMMAS, gamma_ebs, yerr=gamma_eb_errs,
            fmt='o', color='#2c3e50', ms=4, capsize=3,
            capthick=0.6, elinewidth=0.6, mfc='#3498db',
            mec='#2c3e50', mew=0.6, zorder=5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\gamma_{\rm true}$')
ax.set_ylabel(r'$\gamma_{\rm EB}$ (measured)')
ax.set_title(r'Lorentz factor from $\mathbf{E}$-$\mathbf{B}$ cross-correlation')
ax.set_xticks([1, 1.5, 2, 3, 5, 10])
ax.set_xticklabels(['1', '1.5', '2', '3', '5', '10'])
ax.set_yticks([1, 1.5, 2, 3, 5, 10])
ax.set_yticklabels(['1', '1.5', '2', '3', '5', '10'])
ax.set_xlim(0.98, 13)
ax.set_ylim(0.98, 13)

# =========================================================================
# (c) T_lab(θ) profiles
# =========================================================================
ax = axes[1, 0]

for gamma, col in zip(show_gammas, sel_colors):
    beta = np.sqrt(1 - 1/gamma**2)
    th_fine = np.linspace(10, 170, 200)
    D_fine = gamma * (1 - beta * np.cos(np.radians(th_fine)))
    T_theory = T_TRUE / D_fine

    T_meas = results[gamma]['T_lab']
    good = ~np.isnan(T_meas)

    ax.plot(th_fine, T_theory, '-', color=col, lw=1.0, alpha=0.6)
    ax.plot(THETA_CENTERS[good], T_meas[good], 'o', color=col, ms=4,
            mec=col, mew=0.5, label=f'$\\gamma={gamma:.1f}$')

ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
ax.set_ylabel(r'$T_{\rm lab}\;(\hbar\omega_p)$')
ax.set_title(r'Angular temperature profile $T_{\rm lab}(\theta) = T_0 / D(\theta)$')
ax.set_yscale('log')
ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9,
          edgecolor='0.7', handletextpad=0.3, columnspacing=0.8)
ax.set_xlim(8, 172)

# =========================================================================
# (d) RMS errors vs γ
# =========================================================================
ax = axes[1, 1]

T0_rms = []
beta_pct = []
for gamma in GAMMAS:
    T0_pix = results[gamma]['T0_per_pixel']
    good = ~np.isnan(T0_pix)
    errs = 100 * (T0_pix[good] - T_TRUE) / T_TRUE
    T0_rms.append(np.sqrt(np.mean(errs**2)))

    bt = np.sqrt(1 - 1/gamma**2)
    beta_pct.append(100 * abs(results[gamma]['beta_eb'] - bt) / bt)

ax.semilogy(GAMMAS, T0_rms, 'o-', color='#2c3e50', ms=4, mfc='#3498db',
            mec='#2c3e50', mew=0.6, lw=1.0, label=r'$T_0$ RMS error')
ax.semilogy(GAMMAS, beta_pct, 's--', color='#c0392b', ms=3.5, mfc='#e74c3c',
            mec='#c0392b', mew=0.6, lw=0.8, label=r'$\beta$ error')

ax.axhline(1, color='0.6', ls=':', lw=0.6)
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('Error (%)')
ax.set_title('Recovery accuracy')
ax.legend(fontsize=7, framealpha=0.9, edgecolor='0.7')
ax.set_xticks([1, 1.5, 2, 3, 5, 10])
ax.set_xticklabels(['1', '1.5', '2', '3', '5', '10'])
ax.set_xlim(0.98, 13)
ax.set_ylim(5e-3, 30)

# Annotate mode starvation
ax.annotate('mode\nstarvation', xy=(10, T0_rms[-1]),
            xytext=(5.5, 15), fontsize=7, color='0.4',
            arrowprops=dict(arrowstyle='->', color='0.4', lw=0.6),
            ha='center')

# =========================================================================
# Suptitle
# =========================================================================
fig.suptitle(
    r'Electromagnetic fluctuation spectroscopy: multi-$\gamma$ validation'
    '\n'
    rf'$T_0 = {T_TRUE:.0f}\;\hbar\omega_p$ (1 keV),  '
    r'$N_{\rm cal} = N_{\rm boost} = 10^3$ realizations,  '
    r'$4\times 10^4$ modes each',
    fontsize=10, y=0.995)

fn = 'multi_gamma_prl.png'
plt.savefig(fn, dpi=300, bbox_inches='tight')
print(f"Saved: {fn}")

# Also save PDF for journal
fn_pdf = 'multi_gamma_prl.pdf'
plt.savefig(fn_pdf, bbox_inches='tight')
print(f"Saved: {fn_pdf}")
plt.close()
