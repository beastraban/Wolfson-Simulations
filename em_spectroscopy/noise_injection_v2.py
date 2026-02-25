"""
Noise Injection Experiment (v2)
=================================

Physical noise model:
  1. Run N_ens clean realizations → ensemble-averaged pixel powers (the "true" PSD)
  2. Add measurement noise: each pixel gets multiplicative noise
     P_measured(i) = P_true(i) × (1 + δ_i) where δ_i ~ N(0, 1/SNR)
  3. This models: thermal PSD measured with finite integration time,
     where SNR = (integration time) × (bandwidth) determines precision.

  Also test additive noise floor:
     P_measured(i) = P_true(i) + P_noise, where P_noise = <P_true> / SNR

Two noise scenarios:
  A. Multiplicative (relative): each pixel has independent fractional error
  B. Additive (absolute): flat noise floor added to all pixels
     - Without dark subtraction: biases power upward
     - With dark subtraction: removes bias, leaves residual from dark measurement noise

Usage:
  python noise_injection_v2.py                  # full
  python noise_injection_v2.py --n_trials 200   # quick

Ira Wolfson, Feb 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, argparse

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'lines.linewidth': 1.0, 'lines.markersize': 4,
})

parser = argparse.ArgumentParser()
parser.add_argument('--n_ens', type=int, default=1000,
                    help='Ensemble size for clean PSD')
parser.add_argument('--n_trials', type=int, default=1000,
                    help='Number of noisy measurement trials')
parser.add_argument('--n_modes', type=int, default=40000)
parser.add_argument('--seed', type=int, default=2026)
args = parser.parse_args()

T_TRUE = 8512.0
NU_TRUE = 0.05
DELTA_THETA = 3.44
THETA_CENTERS = np.linspace(15, 165, 12)
N_MODES = args.n_modes

GAMMAS = [1.5, 3.0]
SNRS = np.array([100, 50, 20, 10, 5, 3, 2, 1, 0.5, 0.3])

print(f"Noise injection v2: N_ens={args.n_ens}, N_trials={args.n_trials}")

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
    st, ct_r = np.sin(tr), np.cos(tr)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    ky = st * np.sin(phi); kz = ct_r
    Ay = gamma * (1 - beta*kz); Cy = gamma * beta * ky
    return np.mean(Ay**2*(1-ky**2) + Cy**2*st**2 + 2*Ay*Cy*(-ky*kz))

def run_one_clean(beta_inject, rng):
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

    if b > 1e-6:
        EyBx = np.sum(Ep[:,1]*np.conj(Bp[:,0]))
        Ey2 = np.sum(np.abs(Ep[:,1])**2)
        R = np.real(EyBx/Ey2)
        beta_eb = (1 - np.sqrt(max(0, 1-R**2))) / R if abs(R) > 1e-10 else 0
    else:
        beta_eb = 0.0

    power = np.zeros(12)
    for i, tc in enumerate(THETA_CENTERS):
        mask = np.abs(thl - tc) <= DELTA_THETA/2
        if np.sum(mask) < 3: continue
        power[i] = np.sum(np.abs(Ep[mask, 1])**2)

    return power, beta_eb

# =========================================================================
# T₀ extraction
# =========================================================================

def extract_T0(power_rest, power_boost, beta_eb):
    T0_per_pixel = np.full(12, np.nan)
    if abs(beta_eb) < 1e-6:
        return T0_per_pixel
    gamma_eb = 1/np.sqrt(1 - beta_eb**2)
    for i, th in enumerate(THETA_CENTERS):
        D_eb = gamma_eb * (1 - beta_eb*np.cos(np.radians(th)))
        P = P_Ey_exact(th, beta_eb)
        ct = np.cos(np.radians(th))
        P0 = (1 + ct**2) / 2.0
        if power_rest[i] < 1e-20 or power_boost[i] < 1e-20:
            continue
        D2_meas = (P/P0) * (power_rest[i] / power_boost[i])
        if D2_meas < 1e-20: continue
        D_meas = np.sqrt(D2_meas)
        T0_per_pixel[i] = (T_TRUE / D_meas) * D_eb
    return T0_per_pixel

# =========================================================================
# Build clean ensemble-averaged PSDs
# =========================================================================

def build_clean_psd(beta, n_ens, seed_offset):
    rng = np.random.default_rng(args.seed + seed_offset)
    power_sum = np.zeros(12)
    beta_ebs = []
    t0 = time.time()
    for ens in range(n_ens):
        pw, beb = run_one_clean(beta, rng)
        power_sum += pw
        beta_ebs.append(beb)
        if (ens+1) % 200 == 0:
            el = time.time()-t0
            print(f"    {ens+1}/{n_ens} ({el:.0f}s)")
    return power_sum / n_ens, np.mean(beta_ebs), np.std(beta_ebs)

# =========================================================================
# Noise injection on ensemble-averaged PSD
# =========================================================================

def inject_noise_trials(power_true, power_rest, beta_eb, snr_values, n_trials, rng):
    """
    For each SNR and each trial:
      Additive noise: P_meas(i) = P_true(i) + noise_floor × η_i
        where noise_floor = mean(P_true) / SNR
        and η_i ~ Exponential(1) (chi-squared with 2 DOF, normalized)
        This models: true spectral noise is chi-squared distributed

      Noise subtraction: P_sub(i) = P_meas(i) - noise_floor_est
        where noise_floor_est comes from N_dark dark measurements
        (we use N_dark = n_trials, i.e. well-characterized dark)
    """
    mean_power = np.mean(power_true[power_true > 0])

    results = {}
    for snr in snr_values:
        noise_floor = mean_power / snr

        T0_raw_all = []
        T0_sub_all = []

        for trial in range(n_trials):
            # Additive noise: exponentially distributed (power is chi² with 2 dof)
            # Each pixel gets independent noise realization
            noise_per_pixel = noise_floor * rng.exponential(1.0, size=12)

            # Raw: signal + noise
            power_noisy = power_true + noise_per_pixel
            T0_raw = extract_T0(power_rest, power_noisy, beta_eb)
            T0_raw_all.append(T0_raw)

            # Subtracted: use expected noise floor (perfect dark measurement)
            power_sub = power_true + noise_per_pixel - noise_floor
            T0_sub = extract_T0(power_rest, power_sub, beta_eb)
            T0_sub_all.append(T0_sub)

            # Imperfect dark: noise floor estimated from N_dark=100 measurements
            noise_floor_est = noise_floor * (1 + rng.standard_normal() / np.sqrt(100))
            power_sub_imp = power_true + noise_per_pixel - noise_floor_est
            # (store separately if needed)

        T0_raw_all = np.array(T0_raw_all)  # (n_trials, 12)
        T0_sub_all = np.array(T0_sub_all)

        # Stats: mean and RMS over trials, averaged over pixels
        def pixel_rms(T0_arr):
            errs = 100 * (T0_arr - T_TRUE) / T_TRUE
            return np.sqrt(np.nanmean(errs**2))

        def pixel_bias(T0_arr):
            return 100 * (np.nanmean(T0_arr) - T_TRUE) / T_TRUE

        def pixel_scatter(T0_arr):
            """Per-trial scatter: RMS of pixel-to-pixel variation"""
            return np.nanmean(np.nanstd(T0_arr, axis=1)) / T_TRUE * 100

        results[snr] = {
            'rms_raw': pixel_rms(T0_raw_all),
            'rms_sub': pixel_rms(T0_sub_all),
            'bias_raw': pixel_bias(T0_raw_all),
            'bias_sub': pixel_bias(T0_sub_all),
            'scatter_raw': pixel_scatter(T0_raw_all),
            'scatter_sub': pixel_scatter(T0_sub_all),
            'T0_raw_mean_per_pixel': np.nanmean(T0_raw_all, axis=0),
            'T0_sub_mean_per_pixel': np.nanmean(T0_sub_all, axis=0),
            'T0_raw_std_per_pixel': np.nanstd(T0_raw_all, axis=0),
            'T0_sub_std_per_pixel': np.nanstd(T0_sub_all, axis=0),
        }

    return results

# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    t_total = time.time()
    noise_rng = np.random.default_rng(args.seed + 777)

    # Build clean PSDs
    print(f"\nCalibration (β=0):")
    power_rest, _, _ = build_clean_psd(0.0, args.n_ens, 0)

    all_results = {}
    for gamma in GAMMAS:
        beta = np.sqrt(1 - 1/gamma**2)
        print(f"\nγ = {gamma:.1f} (β = {beta:.5f}):")
        power_boost, beta_eb, beta_eb_std = build_clean_psd(
            beta, args.n_ens, int(gamma * 1e6))
        gamma_eb = 1/np.sqrt(1 - beta_eb**2)
        print(f"  β_EB = {beta_eb:.5f} ± {beta_eb_std:.5f}")
        print(f"  γ_EB = {gamma_eb:.4f}")

        # Clean T₀
        T0_clean = extract_T0(power_rest, power_boost, beta_eb)
        print(f"  T₀ (clean) = {np.nanmean(T0_clean):.0f} "
              f"({100*(np.nanmean(T0_clean)-T_TRUE)/T_TRUE:+.2f}%)")

        # Inject noise
        print(f"  Injecting noise ({args.n_trials} trials per SNR)...")
        t0 = time.time()
        snr_results = inject_noise_trials(
            power_boost, power_rest, beta_eb, SNRS, args.n_trials, noise_rng)
        print(f"  Done ({time.time()-t0:.1f}s)")

        all_results[gamma] = {
            'clean_power': power_boost, 'beta_eb': beta_eb,
            'T0_clean': T0_clean, 'snr_results': snr_results,
        }

        print(f"\n  {'SNR':>6} | {'Raw RMS%':>8} {'Raw bias%':>9} | "
              f"{'Sub RMS%':>8} {'Sub bias%':>9}")
        print(f"  {'-'*55}")
        for snr in SNRS:
            r = snr_results[snr]
            print(f"  {snr:6.1f} | {r['rms_raw']:8.2f} {r['bias_raw']:+9.2f} | "
                  f"{r['rms_sub']:8.2f} {r['bias_sub']:+9.2f}")

    # =========================================================================
    # Publication figure
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    plt.subplots_adjust(hspace=0.44, wspace=0.36, left=0.10, right=0.97,
                        top=0.88, bottom=0.09)

    for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        ax.annotate(label, xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=11, fontweight='bold', va='top')

    gc = {1.5: ('#2c3e50', '#3498db', 'o'), 3.0: ('#8b0000', '#e74c3c', 's')}

    # --- (a) T₀ RMS error vs SNR ---
    ax = axes[0, 0]
    for gamma in GAMMAS:
        dark, light, mk = gc[gamma]
        sr = all_results[gamma]['snr_results']
        rms_raw = [sr[s]['rms_raw'] for s in SNRS]
        rms_sub = [sr[s]['rms_sub'] for s in SNRS]

        ax.semilogy(SNRS, rms_raw, f'{mk}-', color=dark, ms=4,
                    mfc=light, mec=dark, mew=0.6,
                    label=f'$\\gamma={gamma:.1f}$ raw')
        ax.semilogy(SNRS, rms_sub, f'{mk}--', color=dark, ms=4,
                    mfc='white', mec=dark, mew=0.8,
                    label=f'$\\gamma={gamma:.1f}$ subtracted')

    ax.axhline(1, color='0.6', ls=':', lw=0.5)
    ax.axhline(10, color='0.75', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.invert_xaxis(); ax.set_xlim(150, 0.2)
    ax.set_xlabel('SNR')
    ax.set_ylabel(r'$T_0$ RMS error (%)')
    ax.set_title(r'Total recovery error')
    ax.legend(fontsize=6, ncol=1, framealpha=0.9, edgecolor='0.7')
    ax.annotate('10%', xy=(0.25, 10), fontsize=7, color='0.5', va='bottom')

    # --- (b) Bias vs SNR ---
    ax = axes[0, 1]
    for gamma in GAMMAS:
        dark, light, mk = gc[gamma]
        sr = all_results[gamma]['snr_results']
        bias_raw = [sr[s]['bias_raw'] for s in SNRS]
        bias_sub = [sr[s]['bias_sub'] for s in SNRS]

        ax.plot(SNRS, bias_raw, f'{mk}-', color=dark, ms=4,
                mfc=light, mec=dark, mew=0.6,
                label=f'$\\gamma={gamma:.1f}$ raw')
        ax.plot(SNRS, bias_sub, f'{mk}--', color=dark, ms=4,
                mfc='white', mec=dark, mew=0.8,
                label=f'$\\gamma={gamma:.1f}$ subtracted')

    ax.axhline(0, color='0.6', ls=':', lw=0.5)
    ax.set_xscale('log')
    ax.invert_xaxis(); ax.set_xlim(150, 0.2)
    ax.set_xlabel('SNR')
    ax.set_ylabel(r'$T_0$ bias (%)')
    ax.set_title('Systematic bias')
    ax.legend(fontsize=6, ncol=1, framealpha=0.9, edgecolor='0.7')

    # --- (c) Per-pixel recovery at selected SNRs, γ=1.5 ---
    ax = axes[1, 0]
    gamma_show = 1.5
    sr = all_results[gamma_show]['snr_results']
    show_snrs = [100, 10, 5, 3]
    snr_cols = ['#27ae60', '#2980b9', '#f39c12', '#c0392b']

    ax.axhline(T_TRUE, color='0.4', ls='-', lw=0.8)
    ax.axhspan(T_TRUE*0.95, T_TRUE*1.05, color='#d5e8d4', alpha=0.5,
               label=r'$\pm 5\%$')

    for snr, col in zip(show_snrs, snr_cols):
        mean_pix = sr[snr]['T0_sub_mean_per_pixel']
        std_pix = sr[snr]['T0_sub_std_per_pixel']
        good = ~np.isnan(mean_pix)
        ax.errorbar(THETA_CENTERS[good], mean_pix[good], yerr=std_pix[good],
                    fmt='o-', color=col, ms=3, lw=0.8, capsize=2, capthick=0.5,
                    elinewidth=0.5, label=f'SNR = {snr}')

    ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
    ax.set_ylabel(r'Recovered $T_0\;(\hbar\omega_p)$')
    ax.set_title(f'Per-pixel $T_0$ (subtracted, $\\gamma={gamma_show:.1f}$)')
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.9, edgecolor='0.7')
    ax.set_xlim(8, 172)
    ax.set_ylim(T_TRUE * 0.6, T_TRUE * 1.6)

    # --- (d) Per-pixel recovery at selected SNRs, γ=3.0 ---
    ax = axes[1, 1]
    gamma_show = 3.0
    sr = all_results[gamma_show]['snr_results']

    ax.axhline(T_TRUE, color='0.4', ls='-', lw=0.8)
    ax.axhspan(T_TRUE*0.95, T_TRUE*1.05, color='#d5e8d4', alpha=0.5,
               label=r'$\pm 5\%$')

    for snr, col in zip(show_snrs, snr_cols):
        mean_pix = sr[snr]['T0_sub_mean_per_pixel']
        std_pix = sr[snr]['T0_sub_std_per_pixel']
        good = ~np.isnan(mean_pix)
        ax.errorbar(THETA_CENTERS[good], mean_pix[good], yerr=std_pix[good],
                    fmt='o-', color=col, ms=3, lw=0.8, capsize=2, capthick=0.5,
                    elinewidth=0.5, label=f'SNR = {snr}')

    ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
    ax.set_ylabel(r'Recovered $T_0\;(\hbar\omega_p)$')
    ax.set_title(f'Per-pixel $T_0$ (subtracted, $\\gamma={gamma_show:.1f}$)')
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.9, edgecolor='0.7')
    ax.set_xlim(8, 172)
    ax.set_ylim(T_TRUE * 0.4, T_TRUE * 2.2)

    fig.suptitle(
        r'Noise susceptibility: additive detector noise on ensemble-averaged PSD'
        '\n'
        rf'$T_0 = {T_TRUE:.0f}\;\hbar\omega_p$,  '
        rf'{args.n_trials} noise trials per SNR,  '
        rf'$N_{{\rm ens}} = {args.n_ens}$ clean realizations',
        fontsize=9.5, y=0.998)

    fn = 'noise_injection_v2.png'
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    fn_pdf = 'noise_injection_v2.pdf'
    plt.savefig(fn_pdf, bbox_inches='tight')
    print(f"\nSaved: {fn}, {fn_pdf}")
    plt.close()

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
