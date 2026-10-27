"""
Four-vector flatness test: Fig. 1 of the paper
================================================

Reads T_lab_results.npz written by T_lab_10k.py and plots the
four-vector transformation test (Corollary 2 of the paper):

  (a) raw angular thermometry: measured T_lab(theta) against the
      four-vector prediction T0/D(theta) and the invariant-T null;
  (b) reconstruction: T0_rec(theta) = D(theta) * T_lab(theta), flat under
      the four-vector law, against the non-four-vector null, prop. to D(theta).

D(theta) = gamma_EB * (1 - beta_EB * cos(theta)) uses the drift velocity
recovered from the E-B cross-correlation, not the injected one.

Usage:
  python T_lab_10k.py --n_ens 10000   # writes T_lab_results.npz
  python flatness_test.py             # writes flatness_test.png/.pdf

Ira Wolfson, 2026
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 10,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
})

BLUE = '#1f4e79'
RED = '#c0461a'
GREY = '#b0b0b0'

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default='T_lab_results.npz')
parser.add_argument('--gamma', type=float, default=1.5,
                    help='injected Lorentz factor (title only)')
args = parser.parse_args()

d = np.load(args.infile)
theta = d['theta']
T_lab = d['T_lab_meas']
beta_eb = float(d['beta_eb'])
n_ens = int(d['N_ENS'])
# True rest-frame temperature, recovered from the saved prediction T_lab_true = T0/D_true
gamma_true = args.gamma
beta_true = np.sqrt(1 - 1 / gamma_true**2)
D_true = gamma_true * (1 - beta_true * np.cos(np.radians(theta)))
T0 = float(np.mean(d['T_lab_true'] * D_true))

gamma_eb = 1 / np.sqrt(1 - beta_eb**2)
D_eb = gamma_eb * (1 - beta_eb * np.cos(np.radians(theta)))

good = ~np.isnan(T_lab)
T0_rec = D_eb * T_lab / T0                     # reconstructed T0(theta) / T0
rms = 100 * np.sqrt(np.mean((T0_rec[good] - 1)**2))
print(f"beta_EB = {beta_eb:.4f}, N_ens = {n_ens}, flatness RMS = {rms:.2f}%")

th_fine = np.linspace(theta.min(), theta.max(), 400)
D_fine = gamma_eb * (1 - beta_eb * np.cos(np.radians(th_fine)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# (a) raw angular thermometry
ax1.plot(th_fine, 1 / D_fine, '-', color=BLUE, lw=1.8,
         label=r'four-vector $T_0/D(\theta)$')
ax1.plot(theta[good], T_lab[good] / T0, 'o', mfc='white', mec=BLUE, mew=1.4,
         ms=7, label=f'simulation ({n_ens // 1000}k)' if n_ens >= 1000
         else f'simulation ({n_ens})')
ax1.axhline(1.0, color=GREY, ls='--', lw=1.4, label=r'invariant-$T$ null')
ax1.set_xlim(10, 170)
ax1.set_ylim(0, 2.7)
ax1.set_xlabel(r'observation angle $\theta$ (deg)')
ax1.set_ylabel(r'$T_{\rm lab}(\theta)\,/\,T_0$')
ax1.set_title('(a) raw angular thermometry')
ax1.legend(frameon=False, loc='upper right')

# (b) reconstruction: the flatness test
ax2.plot(theta[good], T0_rec[good], 'o-', color=BLUE, mfc='white', mec=BLUE,
         mew=1.4, ms=7, lw=1.4, label=r'four-vector $\beta^\mu$ (flat)')
ax2.plot(th_fine, D_fine / gamma_eb, '--', color=RED, lw=1.6,
         label=r'non-four-vector null $\propto D(\theta)$')
ax2.set_xlim(10, 170)
ax2.set_ylim(0.4, 1.9)
ax2.set_xlabel(r'observation angle $\theta$ (deg)')
ax2.set_ylabel(r'$T_0^{\rm (rec)}(\theta) = D(\theta)\,T_{\rm lab}(\theta)\,/\,T_0$')
ax2.set_title('(b) reconstruction: the flatness test')
ax2.legend(frameon=False, loc='upper left')
ax2.text(0.97, 0.06, f'RMS = {rms:.2f}%', transform=ax2.transAxes,
         ha='right', va='bottom', color=BLUE, fontsize=11)

fig.suptitle(rf'$\gamma = {gamma_true}$ ($\beta_{{\rm EB}} = {beta_eb:.4f}$), '
             rf'HIGGINS $-$ {n_ens} realizations', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ('png', 'pdf'):
    fig.savefig(f'flatness_test.{ext}', dpi=200, bbox_inches='tight')
    print(f"  Saved: flatness_test.{ext}")
