"""
HIGGINS-Parametrized EM Fluctuation Spectroscopy Simulation
============================================================

Physical parameters (Malka's HIGGINS at Weizmann):
  Laser:   dual 100 TW, 30 fs, 800 nm, ~1 J on target
  Target:  N₂ supersonic gas jet, ~3 mm
  Plasma:  n_e ~ 10^19 cm^-3 = 10^25 m^-3
  ω_p:     1.78 × 10^14 rad/s  (f_p ~ 28 THz, λ_p ~ 10.6 μm, mid-IR)
  T_e:     ~1 keV (after laser, cooling phase)
           k_B T / (ℏω_p) ~ 8500  →  deep classical regime

Detector array:
  R = 50 cm from source (far field: R >> L²/λ ~ 0.1 m)
  Pixel size d ~ 3 cm  →  angular acceptance Δθ = d/R = 0.06 rad ≈ 3.4°
  12 pixels in a ring from θ = 15° to 165°

Measurement protocol:
  Each pixel collects radiation from modes within its acceptance cone.
  This is physically justified: in the far field, a pixel of area d² at
  distance R integrates exp(ik·x) over its area, which for d >> λ gives
  a geometric angular window of width d/R.

  At each pixel: E'_y(t) = Σ (modes in cone) E'_{n,y} exp(-iω'_n t)
  FFT → PSD. Average over N_ens independent thermal realizations.

Simulation uses normalized units: ω in units of ω_p, T in units of ℏω_p.

Drude conductivity for the plasma:
  σ(ω) = ω_p² / (ν - iω)  →  Re σ(ω) = ω_p² ν / (ν² + ω²)
  In normalized units (ω_p = 1): σ(ω) = ν / (ν² + ω²)
  where ν = ν_ei / ω_p ~ 0.05
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'font.family': 'serif',
})

# =============================================================================
# Physical constants and HIGGINS parameters
# =============================================================================

# All in SI
e_charge = 1.602e-19
m_e = 9.109e-31
eps_0 = 8.854e-12
c_light = 2.998e8
hbar = 1.055e-34
k_B = 1.381e-23

n_e = 1e25       # m^-3 (= 10^19 cm^-3)
omega_p = np.sqrt(n_e * e_charge**2 / (m_e * eps_0))
f_p = omega_p / (2 * np.pi)
lambda_p = c_light / f_p

T_eV = 1000      # eV (1 keV)
T_K = T_eV * e_charge / k_B
T_norm = k_B * T_K / (hbar * omega_p)  # T in units of ℏω_p

print("=" * 60)
print("HIGGINS PARAMETERS")
print("=" * 60)
print(f"  n_e       = {n_e:.1e} m^-3")
print(f"  ω_p       = {omega_p:.2e} rad/s")
print(f"  f_p       = {f_p:.2e} Hz  ({f_p/1e12:.1f} THz)")
print(f"  λ_p       = {lambda_p*1e6:.1f} μm")
print(f"  T_e       = {T_eV} eV = {T_K:.1e} K")
print(f"  T_norm    = {T_norm:.0f} ℏω_p  (deep classical)")
print(f"  c/ω_p     = {c_light/omega_p*1e6:.2f} μm (skin depth)")

# Detector geometry
R_det = 0.50      # m
d_pix = 0.03      # m
delta_theta_rad = d_pix / R_det
delta_theta_deg = np.degrees(delta_theta_rad)
print(f"\n  R_det     = {R_det*100:.0f} cm")
print(f"  d_pix     = {d_pix*100:.0f} cm")
print(f"  Δθ_pixel  = {delta_theta_deg:.1f}°")
print(f"  Far-field: R/λ = {R_det/lambda_p:.0f}  (>> 1 ✓)")
print(f"  d/λ       = {d_pix/lambda_p:.0f}  (>> 1 ✓)")


# =============================================================================
# Physics in normalized units (ω_p = 1, c = 1, ℏ = 1)
# =============================================================================

def drude_re(omega, nu=0.05):
    """Re σ(ω) = ν / (ν² + ω²) in normalized units."""
    return nu / (nu**2 + omega**2)


def coth_factor(omega, T):
    """coth(ω/2T). In classical limit T >> 1: → 2T/ω."""
    x = np.clip(omega / (2.0 * T), 1e-12, 500)
    return 1.0 / np.tanh(x)


def fdt_psd(omega, T, nu=0.05):
    """
    FDT: S(ω) ∝ coth(ω/2T) × ω × Re σ(ω)
    Classical limit: S(ω) ∝ 2T × Re σ(ω)
    """
    return coth_factor(omega, T) * omega * drude_re(omega, nu)


# =============================================================================
# Generate rest-frame thermal modes
# =============================================================================

def generate_modes(n_modes, omega_min, omega_max, T, nu=0.05):
    """
    Outgoing plane waves from localized thermal source.
    Frequencies uniform in [omega_min, omega_max].
    Directions isotropic on sphere.
    Amplitude variance ∝ FDT spectral weight.
    """
    omega = omega_min + (omega_max - omega_min) * np.random.rand(n_modes)

    spectral_weight = fdt_psd(omega, T, nu)
    amp_std = np.sqrt(spectral_weight / (2.0 * n_modes))

    # Directions
    cos_theta = 2.0 * np.random.rand(n_modes) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * np.random.rand(n_modes)
    k_hat = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

    # Polarization basis
    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(k_hat, ref)
    norms = np.linalg.norm(e1, axis=1, keepdims=True)
    bad = norms.flatten() < 0.1
    if np.any(bad):
        e1[bad] = np.cross(k_hat[bad], [1.0, 0.0, 0.0])
        norms[bad] = np.linalg.norm(e1[bad], axis=1, keepdims=True)
    e1 /= norms
    e2 = np.cross(k_hat, e1)

    a1 = amp_std * (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
    a2 = amp_std * (np.random.randn(n_modes) + 1j * np.random.randn(n_modes))
    E = a1[:, None] * e1 + a2[:, None] * e2
    B = np.cross(k_hat, E)

    return E, B, k_hat, cos_theta, omega


# =============================================================================
# Lorentz transform
# =============================================================================

def boost_fields(E, B, beta):
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    Ep = np.zeros_like(E)
    Bp = np.zeros_like(B)
    Ep[:, 0] = gamma * (E[:, 0] - beta * B[:, 1])
    Ep[:, 1] = gamma * (E[:, 1] + beta * B[:, 0])
    Ep[:, 2] = E[:, 2]
    Bp[:, 0] = gamma * (B[:, 0] + beta * E[:, 1])
    Bp[:, 1] = gamma * (B[:, 1] - beta * E[:, 0])
    Bp[:, 2] = B[:, 2]
    return Ep, Bp


def aberrate(cos_theta_rest, beta):
    return (cos_theta_rest + beta) / (1.0 + beta * cos_theta_rest)


def doppler_freq(omega_rest, cos_theta_rest, beta):
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return gamma * omega_rest * (1.0 + beta * cos_theta_rest)


def extract_beta(Ep, Bp):
    EyBx = np.sum(Ep[:, 1] * np.conj(Bp[:, 0]))
    Ey2 = np.sum(np.abs(Ep[:, 1])**2)
    R = np.real(EyBx / Ey2)
    disc = max(0.0, 1.0 - R**2)
    if abs(R) < 1e-10:
        return 0.0
    return (1.0 - np.sqrt(disc)) / R


# =============================================================================
# Pixel-based measurement with ensemble averaging
# =============================================================================

def measure_pixel_psd(Ep, omega_lab, theta_lab_deg,
                      theta_center_deg, delta_theta_deg,
                      n_times=2048, t_max=50.0):
    """
    Select modes within pixel acceptance cone.
    Build time series E'_y(t), FFT → PSD.
    This is physically justified: far-field pixel does geometric angular filtering.
    """
    mask = np.abs(theta_lab_deg - theta_center_deg) <= delta_theta_deg / 2.0
    n_sel = np.sum(mask)

    if n_sel < 3:
        return None, None, 0

    Ey_sel = Ep[mask, 1]
    om_sel = omega_lab[mask]

    t = np.linspace(0, t_max, n_times)
    dt = t[1] - t[0]
    T_total = t[-1] - t[0]

    # Build time series in chunks
    signal = np.zeros(n_times, dtype=complex)
    chunk = 2000
    for s in range(0, n_sel, chunk):
        e = min(s + chunk, n_sel)
        phases = np.exp(-1j * np.outer(om_sel[s:e], t))
        signal += Ey_sel[s:e] @ phases

    ft = np.fft.rfft(np.real(signal))
    freq = np.fft.rfftfreq(n_times, d=dt) * 2 * np.pi
    psd = np.abs(ft)**2 * dt**2 / T_total

    return freq, psd, n_sel


def run_ensemble(T_true, gamma_true, n_modes_per_real, n_ensemble,
                 theta_centers_deg, delta_theta_deg,
                 nu=0.05, omega_min=0.3, omega_max=5.0,
                 n_times=2048, t_max=50.0):
    """
    Generate n_ensemble independent realizations.
    For each: generate modes, boost, measure PSD at each pixel.
    Return ensemble-averaged PSD at each pixel.
    """
    beta_true = np.sqrt(1.0 - 1.0 / gamma_true**2)
    n_det = len(theta_centers_deg)

    # First pass to get frequency array
    t_test = np.linspace(0, t_max, n_times)
    dt = t_test[1] - t_test[0]
    freq = np.fft.rfftfreq(n_times, d=dt) * 2 * np.pi
    n_freq = len(freq)

    psd_sum = np.zeros((n_det, n_freq))
    psd_count = np.zeros(n_det, dtype=int)
    beta_measurements = []
    modes_per_pixel = np.zeros(n_det)

    for ens in range(n_ensemble):
        # Generate fresh thermal realization
        E, B, k_hat, cos_th_rest, omega_rest = generate_modes(
            n_modes_per_real, omega_min, omega_max, T_true, nu=nu
        )

        # Boost
        Ep, Bp = boost_fields(E, B, beta_true)
        cos_th_lab = aberrate(cos_th_rest, beta_true)
        omega_lab = doppler_freq(omega_rest, cos_th_rest, beta_true)
        theta_lab_deg = np.degrees(np.arccos(np.clip(cos_th_lab, -1, 1)))

        # β measurement
        beta_meas = extract_beta(Ep, Bp)
        beta_measurements.append(beta_meas)

        # PSD at each pixel
        for i, th_c in enumerate(theta_centers_deg):
            f, psd, n_sel = measure_pixel_psd(
                Ep, omega_lab, theta_lab_deg,
                th_c, delta_theta_deg,
                n_times=n_times, t_max=t_max
            )
            if f is not None:
                psd_sum[i, :] += psd
                psd_count[i] += 1
                modes_per_pixel[i] += n_sel

        if (ens + 1) % 10 == 0:
            print(f"    Realization {ens+1}/{n_ensemble}")

    # Average
    psd_avg = np.zeros_like(psd_sum)
    for i in range(n_det):
        if psd_count[i] > 0:
            psd_avg[i] = psd_sum[i] / psd_count[i]

    beta_arr = np.array(beta_measurements)
    beta_mean = np.mean(beta_arr)
    gamma_mean = 1.0 / np.sqrt(max(1e-10, 1.0 - beta_mean**2))

    print(f"\n  β: {beta_mean:.5f} ± {np.std(beta_arr):.5f} (true = {beta_true:.5f})")
    print(f"  γ: {gamma_mean:.4f} (true = {gamma_true:.4f})")
    print(f"  Avg modes/pixel/realization: "
          f"{modes_per_pixel / max(1, n_ensemble)}")

    return {
        'freq': freq,
        'psd_avg': psd_avg,
        'psd_count': psd_count,
        'beta_mean': beta_mean,
        'gamma_mean': gamma_mean,
        'beta_std': np.std(beta_arr),
        'T_true': T_true,
        'gamma_true': gamma_true,
        'beta_true': beta_true,
        'theta_centers': theta_centers_deg,
        'delta_theta': delta_theta_deg,
        'nu': nu,
        'n_ensemble': n_ensemble,
        'modes_per_pixel': modes_per_pixel / max(1, n_ensemble),
    }


# =============================================================================
# Diagnostic: compare to analytic prediction at TRUE parameters
# =============================================================================

def diagnostic_plot(data):
    T0 = data['T_true']
    gamma = data['gamma_true']
    beta = data['beta_true']
    nu = data['nu']
    freq = data['freq']
    theta_deg = data['theta_centers']
    psd_avg = data['psd_avg']
    n_det = len(theta_deg)

    fmask_global = (freq > 0.3) & (freq < 5.0)  # comparison window in lab frame

    ncols = min(4, n_det)
    nrows = (n_det + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]

    residuals = []

    for i in range(n_det):
        ax = axes[i // ncols, i % ncols]
        psd_meas = psd_avg[i, :]

        # Integrate analytic prediction over pixel cone
        # S_pixel(ω') = (1/ΔΩ) ∫_cone (T₀/D(θ)) × σ(D(θ) ω') dΩ
        # For axial ring pixel: integrate over θ ∈ [θ_c - Δθ/2, θ_c + Δθ/2]
        n_sub = 20  # sub-angles within pixel
        th_lo = np.radians(theta_deg[i] - data['delta_theta'] / 2.0)
        th_hi = np.radians(theta_deg[i] + data['delta_theta'] / 2.0)
        th_sub = np.linspace(th_lo, th_hi, n_sub)
        S_ana = np.zeros_like(freq)
        for th_s in th_sub:
            D_s = gamma * (1.0 - beta * np.cos(th_s))
            S_ana += (T0 / D_s) * drude_re(D_s * freq, nu) * np.sin(th_s)
        # Normalize by solid angle weight
        S_ana /= np.sum(np.sin(th_sub))

        good = fmask_global & (psd_meas > 0) & (S_ana > 0)
        if np.sum(good) > 5:
            scale = np.sum(psd_meas[good] * S_ana[good]) / np.sum(S_ana[good]**2)
            # Normalized shape residual
            p_n = psd_meas[good] / np.mean(psd_meas[good])
            q_n = (scale * S_ana[good]) / np.mean(scale * S_ana[good])
            resid = np.std(p_n - q_n)
        else:
            scale = 1.0
            resid = np.nan
        residuals.append(resid)

        ax.loglog(freq[fmask_global], psd_meas[fmask_global], '-', color='#2166ac',
                  lw=1, alpha=0.8, label='Measured (ens avg)')
        ax.loglog(freq[fmask_global], scale * S_ana[fmask_global], '--', color='#d73027',
                  lw=2, label='Analytic (true params)')
        ax.set_title(f'θ={theta_deg[i]:.0f}° '
                     f'({data["modes_per_pixel"][i]:.0f} modes) '
                     f'resid={resid:.3f}', fontsize=9)
        if i >= n_det - ncols:
            ax.set_xlabel(r"$\omega' / \omega_p$")
        if i % ncols == 0:
            ax.set_ylabel('PSD')
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_det, nrows * ncols):
        axes[i // ncols, i % ncols].axis('off')

    fig.suptitle(
        f'HIGGINS: T={T0:.0f} ℏω_p ({T_eV} eV), '
        f'γ={gamma:.2f}, '
        f'β_meas={data["beta_mean"]:.4f}±{data["beta_std"]:.4f}, '
        f'N_ens={data["n_ensemble"]}\n'
        f'Pixel Δθ={data["delta_theta"]:.1f}° '
        f'(d={d_pix*100:.0f} cm at R={R_det*100:.0f} cm)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig('/home/claude/higgins_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Shape residuals by angle:")
    for i in range(n_det):
        print(f"    θ={theta_deg[i]:5.0f}°: residual = {residuals[i]:.4f}, "
              f"modes = {data['modes_per_pixel'][i]:.0f}")
    print(f"  Mean residual: {np.nanmean(residuals):.4f}")
    print(f"  Figure saved: higgins_diagnostic.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    np.random.seed(2026)

    print("\n" + "=" * 60)
    print("ENSEMBLE SIMULATION")
    print("=" * 60)

    # Pixel array: 12 pixels from 15° to 165°
    theta_centers = np.linspace(15, 165, 12)

    # Rest-frame ω range must cover lab-frame [0.3, 5.0] at ALL angles:
    #   ω_rest = D × ω_lab,  D_min ≈ 0.42 (θ=15°), D_max ≈ 2.58 (θ=165°)
    #   Need ω_rest_min = 0.3 × D_min ≈ 0.12
    #   Need ω_rest_max = 5.0 × D_max ≈ 12.9
    data = run_ensemble(
        T_true=T_norm,          # ~8500 in normalized units (1 keV)
        gamma_true=1.5,
        n_modes_per_real=100000,   # more modes for wider range
        n_ensemble=50,
        theta_centers_deg=theta_centers,
        delta_theta_deg=delta_theta_deg,
        nu=0.05,
        omega_min=0.1,
        omega_max=13.0,
        n_times=2048,
        t_max=50.0
    )

    diagnostic_plot(data)

    print("\nDone.")
