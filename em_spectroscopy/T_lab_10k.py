"""
Direct T_lab(θ) Extraction — Publication Run
=============================================

10,000 ensemble realizations for low-noise T_lab(θ) = T₀/D(θ).

Two-step protocol:
  Step 1 (β=0): Calibrate pixel response → pin sinθ angular pattern
  Step 2 (β≠0): Extract T_lab(θ) at each pixel from power ratio

Saves checkpoint every 500 realizations. Resumes from checkpoint if found.

Expected runtime: ~4-6 hours on a modern laptop (single core).
  Tip: set N_ENS lower for a quick test, e.g. N_ENS=100

Usage:
  python T_lab_10k.py
  python T_lab_10k.py --n_ens 100      # quick test
  python T_lab_10k.py --n_modes 80000  # more modes per realization

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

# =========================================================================
# Parameters
# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--n_ens', type=int, default=10000)
parser.add_argument('--n_modes', type=int, default=40000)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
args = parser.parse_args()

T_TRUE = 8512.0
GAMMA_TRUE = 1.5
BETA_TRUE = np.sqrt(1 - 1/GAMMA_TRUE**2)
NU_TRUE = 0.05
DELTA_THETA = 3.44
THETA_CENTERS = np.linspace(15, 165, 12)
N_MODES = args.n_modes
N_ENS = args.n_ens
CHECKPOINT_EVERY = 500

os.makedirs(args.checkpoint_dir, exist_ok=True)

print(f"T_lab extraction: {N_ENS} realizations × {N_MODES} modes")
print(f"Checkpoint every {CHECKPOINT_EVERY} realizations → {args.checkpoint_dir}/")

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
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    ky = st * np.sin(phi)
    kz = ct
    Ay = gamma * (1 - beta * kz)
    Cy = gamma * beta * ky
    return np.mean(Ay**2 * (1 - ky**2) + Cy**2 * st**2 + 2 * Ay * Cy * (-ky * kz))


# =========================================================================
# Single realization
# =========================================================================

def run_one_realization(beta_inject, rng, n_times=2048, t_max=50.0):
    """Generate modes, boost, measure PSD and pixel power. Returns (power_12,)."""
    gamma_inj = 1 / np.sqrt(1 - beta_inject**2) if beta_inject > 1e-6 else 1.0

    w = 0.1 + 12.9 * rng.random(N_MODES)
    amp = np.sqrt(2 * T_TRUE * drude(w, NU_TRUE) / (2.0 * N_MODES))
    cth = 2 * rng.random(N_MODES) - 1
    phi = 2 * np.pi * rng.random(N_MODES)
    sth = np.sqrt(1 - cth**2)
    kx, ky, kz = sth * np.cos(phi), sth * np.sin(phi), cth

    # Polarization basis
    e1 = np.column_stack([-ky, kx, np.zeros(N_MODES)])
    norms = np.sqrt(kx**2 + ky**2)
    bad = norms < 0.1
    if np.any(bad):
        nb = bad.sum()
        e1[bad] = np.column_stack([np.zeros(nb), -kz[bad], ky[bad]])
        norms[bad] = np.sqrt(kz[bad]**2 + ky[bad]**2)
    e1 /= norms[:, None]
    e2 = np.column_stack([
        ky * e1[:, 2] - kz * e1[:, 1],
        kz * e1[:, 0] - kx * e1[:, 2],
        kx * e1[:, 1] - ky * e1[:, 0]
    ])

    a1 = amp * (rng.standard_normal(N_MODES) + 1j * rng.standard_normal(N_MODES))
    a2 = amp * (rng.standard_normal(N_MODES) + 1j * rng.standard_normal(N_MODES))
    E = a1[:, None] * e1 + a2[:, None] * e2
    B = np.column_stack([
        ky * E[:, 2] - kz * E[:, 1],
        kz * E[:, 0] - kx * E[:, 2],
        kx * E[:, 1] - ky * E[:, 0]
    ])

    # Boost
    b = beta_inject
    g = gamma_inj
    if b > 1e-6:
        Ep = np.empty_like(E)
        Ep[:, 0] = g * (E[:, 0] - b * B[:, 1])
        Ep[:, 1] = g * (E[:, 1] + b * B[:, 0])
        Ep[:, 2] = E[:, 2]
        cl = (cth + b) / (1 + b * cth)
        wl = g * w * (1 + b * cth)
    else:
        Ep = E
        cl = cth
        wl = w

    thl = np.degrees(np.arccos(np.clip(cl, -1, 1)))

    # β_EB (only meaningful for boosted)
    if b > 1e-6:
        Bp = np.empty_like(B)
        Bp[:, 0] = g * (B[:, 0] + b * E[:, 1])
        Bp[:, 1] = g * (B[:, 1] - b * E[:, 0])
        Bp[:, 2] = B[:, 2]
        EyBx = np.sum(Ep[:, 1] * np.conj(Bp[:, 0]))
        Ey2 = np.sum(np.abs(Ep[:, 1])**2)
        R = np.real(EyBx / Ey2)
        beta_eb = (1 - np.sqrt(max(0, 1 - R**2))) / R if abs(R) > 1e-10 else 0
    else:
        beta_eb = 0.0

    # Pixel power
    power = np.zeros(12)
    for i, tc in enumerate(THETA_CENTERS):
        mask = np.abs(thl - tc) <= DELTA_THETA / 2
        ns = int(np.sum(mask))
        if ns < 3:
            continue
        power[i] = np.sum(np.abs(Ep[mask, 1])**2)

    return power, beta_eb


# =========================================================================
# Ensemble with checkpointing
# =========================================================================

def run_ensemble(beta_inject, label):
    ckpt_file = os.path.join(args.checkpoint_dir, f'{label}.npz')

    # Try to resume
    start_ens = 0
    power_sum = np.zeros(12)
    beta_ebs = []

    if os.path.exists(ckpt_file):
        ckpt = np.load(ckpt_file)
        start_ens = int(ckpt['completed'])
        power_sum = ckpt['power_sum']
        beta_ebs = list(ckpt['beta_ebs'])
        print(f"  Resuming {label} from realization {start_ens}/{N_ENS}")

    if start_ens >= N_ENS:
        print(f"  {label} already complete ({N_ENS} realizations)")
        power_avg = power_sum / N_ENS
        return power_avg, np.mean(beta_ebs), np.std(beta_ebs)

    rng = np.random.default_rng(args.seed + (0 if beta_inject < 1e-6 else 1000000))
    # Advance RNG past completed realizations
    for _ in range(start_ens):
        rng.random(N_MODES * 10)  # rough skip

    t0 = time.time()
    for ens in range(start_ens, N_ENS):
        power, beb = run_one_realization(beta_inject, rng)
        power_sum += power
        beta_ebs.append(beb)

        if (ens + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (ens + 1 - start_ens) / elapsed
            eta = (N_ENS - ens - 1) / rate
            rms_power = np.std(power_sum / (ens + 1)) / np.mean(power_sum / (ens + 1)) * 100
            print(f"  {label}: {ens+1}/{N_ENS}  "
                  f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta/60:.0f}min)  "
                  f"power scatter {rms_power:.1f}%")

        if (ens + 1) % CHECKPOINT_EVERY == 0:
            np.savez(ckpt_file,
                     completed=ens + 1,
                     power_sum=power_sum,
                     beta_ebs=np.array(beta_ebs))

    # Final save
    np.savez(ckpt_file,
             completed=N_ENS,
             power_sum=power_sum,
             beta_ebs=np.array(beta_ebs))

    power_avg = power_sum / N_ENS
    elapsed = time.time() - t0
    print(f"  {label} complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return power_avg, np.mean(beta_ebs), np.std(beta_ebs)


# =========================================================================
# T_lab extraction
# =========================================================================

def extract_T_lab(power_rest, power_boost, beta_eb):
    """Extract T_lab(θ) from power ratio using β_EB."""
    gamma_eb = 1 / np.sqrt(1 - beta_eb**2)
    T_lab_meas = np.zeros(12)
    T_lab_true = np.zeros(12)

    for i, th in enumerate(THETA_CENTERS):
        D = gamma_eb * (1 - beta_eb * np.cos(np.radians(th)))
        P = P_Ey_exact(th, beta_eb)
        ct_rest_0 = np.cos(np.radians(th))  # at beta=0, theta_rest = theta_lab
        P0 = (1.0 + ct_rest_0**2) / 2.0

        if power_rest[i] < 1e-20 or power_boost[i] < 1e-20:
            T_lab_meas[i] = np.nan
            T_lab_true[i] = T_TRUE / D
            continue

        D2_meas = (P / P0) * (power_rest[i] / power_boost[i])
        D_meas = np.sqrt(max(D2_meas, 1e-20))
        T_lab_meas[i] = T_TRUE / D_meas
        T_lab_true[i] = T_TRUE / D

    return T_lab_meas, T_lab_true


# =========================================================================
# Plotting
# =========================================================================

def make_figure(power_rest, power_boost, T_lab_meas, T_lab_true,
                beta_eb, beta_eb_std):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sin_th = np.sin(np.radians(THETA_CENTERS))

    good = ~np.isnan(T_lab_meas)
    err_pct = 100 * (T_lab_meas[good] - T_lab_true[good]) / T_lab_true[good]
    rms = np.sqrt(np.mean(err_pct**2))

    # Panel A: T_lab(θ)
    ax = axes[0, 0]
    ax.plot(THETA_CENTERS, T_lab_true, 'r--', lw=2.5,
            label=r'$T_0/D(\theta)$ (theory)', zorder=5)
    ax.plot(THETA_CENTERS[good], T_lab_meas[good], 'bo', ms=8,
            label=r'$T_{\rm lab}(\theta)$ (measured)', zorder=10)
    ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
    ax.set_ylabel(r'$T_{\rm lab}$ ($\hbar\omega_p$)')
    ax.set_title(r'Lab-frame temperature: $T_{\rm lab}(\theta) = T_0 / D(\theta)$')
    ax.legend(fontsize=11)
    ax.text(0.05, 0.95,
            f'$T_0 = {T_TRUE:.0f}$ $\\hbar\\omega_p$ (1 keV)\n'
            f'$\\gamma = {GAMMA_TRUE}$, $\\beta = {BETA_TRUE:.4f}$\n'
            f'$N_{{\\rm ens}} = {N_ENS}$',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: error
    ax = axes[0, 1]
    ax.bar(THETA_CENTERS[good], err_pct, width=10, color='#2166ac', alpha=0.7)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
    ax.set_ylabel('Error (%)')
    ax.set_title(f'$T_{{\\rm lab}}$ error (RMS = {rms:.2f}%)')

    # Panel C: rest-frame calibration — power vs sinθ × (1+cos²θ)/2
    ax = axes[1, 0]
    ax.plot(THETA_CENTERS, power_rest, 'ko-', ms=6, label='Measured $W_{\\rm rest}(\\theta)$')
    rest_model = sin_th * (1 + np.cos(np.radians(THETA_CENTERS))**2) / 2
    norm = np.max(power_rest) / np.max(rest_model)
    ax.plot(THETA_CENTERS, rest_model * norm, 'r--', lw=2,
            label=r'$\propto \sin\theta\,(1{+}\cos^2\!\theta)/2$')
    ax.set_xlabel(r'$\theta$ (degrees)')
    ax.set_ylabel('Pixel power')
    ax.set_title(r'Step 1: Rest-frame calibration ($\beta=0$)')
    ax.legend(fontsize=9)

    # Panel D: boosted power
    ax = axes[1, 1]
    ax.plot(THETA_CENTERS, power_boost, 'bo-', ms=6,
            label=r'$W_{\rm lab}(\theta)$')
    model_power = np.array([
        np.sin(np.radians(th)) * P_Ey_exact(th, BETA_TRUE) /
        (GAMMA_TRUE * (1 - BETA_TRUE * np.cos(np.radians(th))))**2
        for th in THETA_CENTERS
    ])
    norm2 = np.sum(power_boost * model_power) / np.sum(model_power**2)
    ax.plot(THETA_CENTERS, norm2 * model_power, 'r^--', ms=6,
            label=r'$\sin\theta \cdot P_{E_y} / D^2$')
    ax.set_xlabel(r'$\theta_{\rm lab}$ (degrees)')
    ax.set_ylabel('Pixel power')
    ax.set_title(r'Step 2: Boosted ($\gamma=1.5$)')
    ax.legend(fontsize=10)

    fig.suptitle(
        f'EM Fluctuation Spectroscopy: $T_{{\\rm lab}}(\\theta)$ Extraction\n'
        f'{N_ENS} realizations, {N_MODES} modes each | '
        f'$\\beta_{{EB}} = {beta_eb:.5f} \\pm {beta_eb_std:.5f}$',
        fontsize=14)
    plt.tight_layout()
    fn = f'T_lab_{N_ENS}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fn}")
    plt.close()


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    t_total = time.time()

    print("\n" + "=" * 60)
    print("STEP 1: Rest-frame calibration")
    print("=" * 60)
    power_rest, _, _ = run_ensemble(0.0, 'rest')

    print("\n" + "=" * 60)
    print("STEP 2: Boosted measurement")
    print("=" * 60)
    power_boost, beta_eb, beta_eb_std = run_ensemble(BETA_TRUE, 'boosted')

    print(f"\nbeta_EB = {beta_eb:.5f} +/- {beta_eb_std:.5f} (true {BETA_TRUE:.5f})")

    print("\n" + "=" * 60)
    print("T_lab EXTRACTION")
    print("=" * 60)
    T_lab_meas, T_lab_true = extract_T_lab(power_rest, power_boost, beta_eb)

    good = ~np.isnan(T_lab_meas)
    err_pct = 100 * (T_lab_meas[good] - T_lab_true[good]) / T_lab_true[good]
    print(f"\n{'theta':>5} {'T_lab meas':>12} {'T_lab true':>12} {'err%':>8}")
    for i in range(12):
        if np.isnan(T_lab_meas[i]):
            continue
        e = 100 * (T_lab_meas[i] - T_lab_true[i]) / T_lab_true[i]
        print(f"{THETA_CENTERS[i]:5.0f} {T_lab_meas[i]:12.0f} {T_lab_true[i]:12.0f} {e:+8.2f}%")
    print(f"\nRMS error: {np.sqrt(np.mean(err_pct**2)):.2f}%")

    make_figure(power_rest, power_boost, T_lab_meas, T_lab_true,
                beta_eb, beta_eb_std)

    # Save data
    np.savez('T_lab_results.npz',
             theta=THETA_CENTERS,
             T_lab_meas=T_lab_meas,
             T_lab_true=T_lab_true,
             power_rest=power_rest,
             power_boost=power_boost,
             beta_eb=beta_eb,
             beta_eb_std=beta_eb_std,
             N_ENS=N_ENS,
             N_MODES=N_MODES)
    print("  Saved: T_lab_results.npz")

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
