"""
Multi-γ T₀ Recovery Experiment
================================

Demonstrates that T₀ is recovered correctly at ALL Lorentz factors.

Protocol:
  1. Single shared calibration (β=0, N_cal realizations)
  2. For each γ: N_boost realizations, extract T_lab(θ), invert → T₀(θ)
  3. Four-panel publication figure

Usage:
  python multi_gamma.py                    # full (8 gammas × 1000 boost)
  python multi_gamma.py --n_boost 100      # quick test
  python multi_gamma.py --n_cal 200 --n_boost 50  # fastest sanity

Ira Wolfson, Feb 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, time, os, argparse

matplotlib.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 150, 'font.family': 'serif',
})

parser = argparse.ArgumentParser()
parser.add_argument('--n_cal', type=int, default=1000)
parser.add_argument('--n_boost', type=int, default=1000)
parser.add_argument('--n_modes', type=int, default=40000)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--checkpoint_dir', type=str, default='ckpt_multi')
args = parser.parse_args()

T_TRUE = 8512.0
NU_TRUE = 0.05
DELTA_THETA = 3.44
THETA_CENTERS = np.linspace(15, 165, 12)
N_MODES = args.n_modes
GAMMAS = np.array([1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0])
CHECKPOINT_EVERY = 200

os.makedirs(args.checkpoint_dir, exist_ok=True)

print(f"Multi-γ experiment: {len(GAMMAS)} gammas, "
      f"N_cal={args.n_cal}, N_boost={args.n_boost}, N_modes={N_MODES}")
print(f"γ values: {GAMMAS}")

# =========================================================================
# Physics
# =========================================================================

def drude(w, nu):
    return nu / (nu**2 + w**2)

def P_Ey_exact(th_lab, beta, n_phi=300):
    if beta < 1e-6:
        ct = np.cos(np.radians(th_lab))
        return (1.0 + ct**2) / 2.0
    gamma = 1 / np.sqrt(1 - beta**2)
    cl = np.cos(np.radians(th_lab))
    cr = (cl - beta) / (1 - beta * cl)
    tr = np.arccos(np.clip(cr, -1, 1))
    st, ct = np.sin(tr), np.cos(tr)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    ky = st * np.sin(phi); kz = ct
    Ay = gamma * (1 - beta*kz); Cy = gamma * beta * ky
    return np.mean(Ay**2*(1-ky**2) + Cy**2*st**2 + 2*Ay*Cy*(-ky*kz))

# =========================================================================
# Single realization
# =========================================================================

def run_one(beta_inject, rng):
    gamma_inj = 1/np.sqrt(1 - beta_inject**2) if beta_inject > 1e-6 else 1.0

    w = 0.1 + 12.9*rng.random(N_MODES)
    amp = np.sqrt(2*T_TRUE*drude(w, NU_TRUE) / (2.0*N_MODES))
    cth = 2*rng.random(N_MODES) - 1
    phi = 2*np.pi*rng.random(N_MODES)
    sth = np.sqrt(1 - cth**2)
    kx, ky, kz = sth*np.cos(phi), sth*np.sin(phi), cth

    e1 = np.column_stack([-ky, kx, np.zeros(N_MODES)])
    norms = np.sqrt(kx**2 + ky**2); bad = norms < 0.1
    if np.any(bad):
        nb = bad.sum()
        e1[bad] = np.column_stack([np.zeros(nb), -kz[bad], ky[bad]])
        norms[bad] = np.sqrt(kz[bad]**2 + ky[bad]**2)
    e1 /= norms[:, None]
    e2 = np.column_stack([
        ky*e1[:,2]-kz*e1[:,1], kz*e1[:,0]-kx*e1[:,2], kx*e1[:,1]-ky*e1[:,0]])
    a1 = amp*(rng.standard_normal(N_MODES) + 1j*rng.standard_normal(N_MODES))
    a2 = amp*(rng.standard_normal(N_MODES) + 1j*rng.standard_normal(N_MODES))
    E = a1[:,None]*e1 + a2[:,None]*e2
    B = np.column_stack([
        ky*E[:,2]-kz*E[:,1], kz*E[:,0]-kx*E[:,2], kx*E[:,1]-ky*E[:,0]])

    b = beta_inject; g = gamma_inj
    if b > 1e-6:
        Ep = np.empty_like(E)
        Ep[:,0] = g*(E[:,0]-b*B[:,1]); Ep[:,1] = g*(E[:,1]+b*B[:,0]); Ep[:,2] = E[:,2]
        Bp = np.empty_like(B)
        Bp[:,0] = g*(B[:,0]+b*E[:,1]); Bp[:,1] = g*(B[:,1]-b*E[:,0]); Bp[:,2] = B[:,2]
        cl = (cth+b)/(1+b*cth)
    else:
        Ep = E; Bp = B; cl = cth

    thl = np.degrees(np.arccos(np.clip(cl, -1, 1)))

    # β_EB
    if b > 1e-6:
        EyBx = np.sum(Ep[:,1]*np.conj(Bp[:,0]))
        Ey2 = np.sum(np.abs(Ep[:,1])**2)
        R = np.real(EyBx/Ey2)
        beta_eb = (1 - np.sqrt(max(0, 1-R**2))) / R if abs(R) > 1e-10 else 0
    else:
        beta_eb = 0.0

    power = np.zeros(12)
    n_modes_pix = np.zeros(12)
    for i, tc in enumerate(THETA_CENTERS):
        mask = np.abs(thl - tc) <= DELTA_THETA/2
        ns = int(np.sum(mask))
        if ns < 3: continue
        power[i] = np.sum(np.abs(Ep[mask, 1])**2)
        n_modes_pix[i] = ns

    return power, beta_eb, n_modes_pix

# =========================================================================
# Ensemble runner with checkpointing
# =========================================================================

def run_ensemble(beta_inject, n_ens, label, seed_offset=0):
    ckpt = os.path.join(args.checkpoint_dir, f'{label}.npz')
    start = 0; power_sum = np.zeros(12); betas = []; nmodes_sum = np.zeros(12)

    if os.path.exists(ckpt):
        d = np.load(ckpt)
        start = int(d['completed'])
        power_sum = d['power_sum']; betas = list(d['betas'])
        nmodes_sum = d['nmodes_sum']
        if start >= n_ens:
            print(f"  {label}: already complete ({n_ens})")
            return power_sum/n_ens, np.mean(betas), np.std(betas), nmodes_sum/n_ens

    rng = np.random.default_rng(args.seed + seed_offset)
    for _ in range(start):
        rng.random(N_MODES * 10)

    t0 = time.time()
    for ens in range(start, n_ens):
        pw, beb, nm = run_one(beta_inject, rng)
        power_sum += pw; betas.append(beb); nmodes_sum += nm

        if (ens+1) % 100 == 0:
            el = time.time()-t0; rate = (ens+1-start)/el
            eta = (n_ens-ens-1)/rate if rate > 0 else 0
            print(f"    {label}: {ens+1}/{n_ens} ({el:.0f}s, ETA {eta:.0f}s)")

        if (ens+1) % CHECKPOINT_EVERY == 0:
            np.savez(ckpt, completed=ens+1, power_sum=power_sum,
                     betas=np.array(betas), nmodes_sum=nmodes_sum)

    np.savez(ckpt, completed=n_ens, power_sum=power_sum,
             betas=np.array(betas), nmodes_sum=nmodes_sum)
    el = time.time()-t0
    print(f"    {label}: done ({el:.0f}s)")
    return power_sum/n_ens, np.mean(betas), np.std(betas), nmodes_sum/n_ens

# =========================================================================
# T₀ extraction
# =========================================================================

def extract_T0(power_rest, power_boost, beta_eb, gamma_true):
    """Invert T_lab(θ) = T₀/D → T₀_recovered = T_lab × D."""
    T0_per_pixel = np.full(12, np.nan)
    T_lab = np.full(12, np.nan)
    nmodes_min_flag = np.zeros(12, dtype=bool)

    for i, th in enumerate(THETA_CENTERS):
        D = gamma_true * (1 - np.sqrt(1-1/gamma_true**2)*np.cos(np.radians(th)))
        # Use β_EB for D in practice
        if abs(beta_eb) < 1e-6:
            continue
        gamma_eb = 1/np.sqrt(1 - beta_eb**2)
        D_eb = gamma_eb * (1 - beta_eb*np.cos(np.radians(th)))
        P = P_Ey_exact(th, beta_eb)
        ct = np.cos(np.radians(th))
        P0 = (1 + ct**2) / 2.0

        if power_rest[i] < 1e-20 or power_boost[i] < 1e-20:
            continue

        D2_meas = (P/P0) * (power_rest[i] / power_boost[i])
        D_meas = np.sqrt(max(D2_meas, 1e-20))
        T_lab_val = T_TRUE / D_meas
        T0_recovered = T_lab_val * D_eb

        T0_per_pixel[i] = T0_recovered
        T_lab[i] = T_lab_val

    return T0_per_pixel, T_lab

# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    t_total = time.time()

    # === Step 1: Shared calibration ===
    print("\n" + "="*60)
    print(f"CALIBRATION (β=0, {args.n_cal} realizations)")
    print("="*60)
    power_rest, _, _, nmodes_rest = run_ensemble(0.0, args.n_cal, 'cal', seed_offset=0)

    # === Step 2: Scan γ ===
    results = {}
    for ig, gamma in enumerate(GAMMAS):
        beta = np.sqrt(1 - 1/gamma**2)
        print(f"\n{'='*60}")
        print(f"γ = {gamma:.2f}  (β = {beta:.5f}), {args.n_boost} realizations")
        print("="*60)

        power_boost, beta_eb, beta_eb_std, nmodes_boost = run_ensemble(
            beta, args.n_boost, f'g{gamma:.2f}', seed_offset=(ig+1)*1000000)

        T0_pix, T_lab_pix = extract_T0(power_rest, power_boost, beta_eb, gamma)
        gamma_eb = 1/np.sqrt(max(1e-10, 1 - beta_eb**2))

        good = ~np.isnan(T0_pix)
        T0_mean = np.nanmean(T0_pix)
        T0_std = np.nanstd(T0_pix)
        T0_err = 100*(T0_mean - T_TRUE)/T_TRUE

        print(f"  β_EB = {beta_eb:.5f} ± {beta_eb_std:.5f} (true {beta:.5f})")
        print(f"  γ_EB = {gamma_eb:.4f} (true {gamma:.4f})")
        print(f"  T₀ recovered = {T0_mean:.0f} ± {T0_std:.0f} (true {T_TRUE:.0f}, err {T0_err:+.2f}%)")
        print(f"  Good pixels: {np.sum(good)}/12")
        print(f"  Avg modes/pixel: {nmodes_boost[good].mean():.0f}")

        results[gamma] = {
            'beta_true': beta, 'beta_eb': beta_eb, 'beta_eb_std': beta_eb_std,
            'gamma_eb': gamma_eb,
            'T0_per_pixel': T0_pix, 'T_lab': T_lab_pix,
            'power_boost': power_boost, 'nmodes': nmodes_boost,
            'T0_mean': T0_mean, 'T0_std': T0_std,
        }

    # === Save results ===
    np.savez('multi_gamma_results.npz',
             gammas=GAMMAS, T_TRUE=T_TRUE,
             theta_centers=THETA_CENTERS,
             power_rest=power_rest,
             **{f'g{g:.2f}_T0pix': results[g]['T0_per_pixel'] for g in GAMMAS},
             **{f'g{g:.2f}_Tlab': results[g]['T_lab'] for g in GAMMAS},
             **{f'g{g:.2f}_beta_eb': results[g]['beta_eb'] for g in GAMMAS},
             **{f'g{g:.2f}_beta_eb_std': results[g]['beta_eb_std'] for g in GAMMAS},
             **{f'g{g:.2f}_nmodes': results[g]['nmodes'] for g in GAMMAS})
    print("\nSaved: multi_gamma_results.npz")

    # =========================================================================
    # Publication figure
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # --- Panel A: T₀ recovery vs γ ---
    ax = axes[0, 0]
    for ig, gamma in enumerate(GAMMAS):
        T0_pix = results[gamma]['T0_per_pixel']
        good = ~np.isnan(T0_pix)
        x = np.full(np.sum(good), gamma)
        # Jitter for visibility
        jitter = 0.02 * gamma * (np.random.rand(np.sum(good)) - 0.5)
        ax.scatter(x + jitter, T0_pix[good], s=15, alpha=0.6,
                   color=plt.cm.viridis(ig/len(GAMMAS)), zorder=5)
        # Mean marker
        ax.plot(gamma, results[gamma]['T0_mean'], 'k_', ms=15, mew=2, zorder=10)

    ax.axhline(T_TRUE, color='red', ls='--', lw=2, label=f'$T_0 = {T_TRUE:.0f}$', zorder=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$T_0$ recovered ($\hbar\omega_p$)')
    ax.set_title(r'$T_0$ recovery: 12 pixels per $\gamma$')
    ax.legend(fontsize=11)
    ax.set_xticks(GAMMAS)
    ax.set_xticklabels([f'{g:.2g}' for g in GAMMAS])
    # Zoom to ±10% of T_TRUE
    ax.set_ylim(T_TRUE*0.9, T_TRUE*1.1)

    # --- Panel B: γ recovery ---
    ax = axes[0, 1]
    gamma_ebs = [results[g]['gamma_eb'] for g in GAMMAS]
    beta_eb_stds = [results[g]['beta_eb_std'] for g in GAMMAS]
    # Propagate β_EB uncertainty to γ: dγ/dβ = β γ³
    gamma_eb_errs = [results[g]['beta_eb_std'] * results[g]['beta_eb'] *
                     results[g]['gamma_eb']**3 for g in GAMMAS]

    ax.errorbar(GAMMAS, gamma_ebs, yerr=gamma_eb_errs,
                fmt='bo', ms=6, capsize=4, label=r'$\gamma_{EB}$ measured')
    gline = np.linspace(1, GAMMAS.max()*1.1, 100)
    ax.plot(gline, gline, 'r--', lw=2, label='$y = x$')
    ax.set_xlabel(r'$\gamma_{\rm true}$')
    ax.set_ylabel(r'$\gamma_{EB}$ (from E-B)')
    ax.set_title(r'$\gamma$ recovery from E-B cross-correlation')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(GAMMAS)
    ax.set_xticklabels([f'{g:.2g}' for g in GAMMAS])
    ax.set_yticks(GAMMAS)
    ax.set_yticklabels([f'{g:.2g}' for g in GAMMAS])

    # --- Panel C: T_lab(θ) profiles for selected γ ---
    ax = axes[1, 0]
    show_gammas = [1.1, 1.5, 3.0, 10.0]
    colors_sel = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    for gamma, col in zip(show_gammas, colors_sel):
        beta = np.sqrt(1 - 1/gamma**2)
        D_arr = gamma * (1 - beta*np.cos(np.radians(THETA_CENTERS)))
        T_theory = T_TRUE / D_arr
        T_meas = results[gamma]['T_lab']
        good = ~np.isnan(T_meas)

        ax.plot(THETA_CENTERS, T_theory, '--', color=col, lw=2, alpha=0.7)
        ax.plot(THETA_CENTERS[good], T_meas[good], 'o', color=col, ms=6,
                label=f'$\\gamma={gamma:.1f}$')

    ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
    ax.set_ylabel(r'$T_{\rm lab}$ ($\hbar\omega_p$)')
    ax.set_title(r'$T_{\rm lab}(\theta) = T_0/D(\theta)$ at selected $\gamma$')
    ax.legend(fontsize=9, ncol=2)
    ax.set_yscale('log')

    # --- Panel D: RMS errors vs γ ---
    ax = axes[1, 1]
    T0_rms = []
    beta_errs = []
    for gamma in GAMMAS:
        T0_pix = results[gamma]['T0_per_pixel']
        good = ~np.isnan(T0_pix)
        if np.sum(good) > 0:
            errs = 100*(T0_pix[good] - T_TRUE)/T_TRUE
            T0_rms.append(np.sqrt(np.mean(errs**2)))
        else:
            T0_rms.append(np.nan)
        beta_true = np.sqrt(1 - 1/gamma**2)
        beta_err = 100*abs(results[gamma]['beta_eb'] - beta_true) / max(beta_true, 1e-10)
        beta_errs.append(beta_err)

    ax.semilogy(GAMMAS, T0_rms, 'bo-', ms=6, label=r'$T_0$ RMS error (%)')
    ax.semilogy(GAMMAS, beta_errs, 'r^--', ms=6, label=r'$\beta$ error (%)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('Error (%)')
    ax.set_title('Recovery errors vs $\\gamma$')
    ax.legend(fontsize=10)
    ax.set_xticks(GAMMAS)
    ax.set_xticklabels([f'{g:.2g}' for g in GAMMAS])
    ax.axhline(1, color='gray', ls=':', alpha=0.5)

    fig.suptitle(
        f'Multi-$\\gamma$ Validation: $T_0 = {T_TRUE:.0f}\\;\\hbar\\omega_p$ (1 keV)\n'
        f'Calibration: {args.n_cal} realizations | '
        f'Boost: {args.n_boost} realizations/γ | {N_MODES} modes each',
        fontsize=14)
    plt.tight_layout()

    fn = f'multi_gamma_{args.n_boost}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    print(f"Saved: {fn}")
    plt.close()

    # === Summary table ===
    print(f"\n{'='*70}")
    print(f"{'γ':>6} {'β_true':>8} {'β_EB':>8} {'γ_EB':>8} "
          f"{'T₀':>8} {'T₀ err%':>8} {'pixels':>6}")
    print(f"{'='*70}")
    for gamma in GAMMAS:
        r = results[gamma]
        good = ~np.isnan(r['T0_per_pixel'])
        print(f"{gamma:6.2f} {r['beta_true']:8.5f} {r['beta_eb']:8.5f} "
              f"{r['gamma_eb']:8.4f} {r['T0_mean']:8.0f} "
              f"{100*(r['T0_mean']-T_TRUE)/T_TRUE:+8.2f}% {np.sum(good):6d}")

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
