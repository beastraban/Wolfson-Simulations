# Electromagnetic Fluctuation Spectroscopy: Simulation Suite

## Overview

Monte Carlo simulation suite for the paper *"Operational measurement of relativistic
equilibrium from stochastic electromagnetic fields alone"* (Wolfson 2026,
[arXiv:2602.16765](https://arxiv.org/abs/2602.16765)).

Demonstrates that the angular temperature profile $T_{\rm lab}(\theta) = T_0 / D(\theta,\beta)$
can be extracted from electromagnetic fluctuation measurements using a two-step
calibration protocol:
1. **Rest-frame calibration** ($\beta = 0$): measures the product $\Theta\nu$ and
   angular response $\sin\theta \cdot P_{E_y}(\theta, 0)$
2. **Boosted measurement**: recovers $\beta$ from the E-B cross-correlation,
   then extracts $T_{\rm lab}(\theta)$ at each detector pixel

All simulations are parameterized to the **HIGGINS** laser-plasma experiment
(Malka group, Weizmann Institute): dual 100 TW laser, $n_e \sim 10^{19}$ cm$^{-3}$,
$T \sim 1$ keV, $\omega_p \sim 28$ THz (mid-IR), $\gamma \sim 1.5$.

## Files

| File | Description |
|------|-------------|
| `higgins_sim.py` | Base simulation engine. Generates thermal EM modes in rest frame, Lorentz-boosts, bins into detector pixels, computes ensemble-averaged PSD. Includes E-B cross-correlation for $\beta$ recovery. |
| `T_lab_10k.py` | Single-$\gamma$ temperature extraction. Runs the full two-step protocol with 10,000 ensemble realizations. Produces $T_{\rm lab}(\theta)$ at 12 angles with ~0.34% RMS error. **Fig. 1** |
| `multi_gamma.py` | Multi-$\gamma$ validation. Scans $\gamma \in \{1.05, 1.1, 1.2, 1.5, 2, 3, 5, 10\}$ with 1000 realizations each, shared calibration. Demonstrates $T_0$ recovery across the full relativistic range. |
| `multi_gamma_prl.py` | Publication-quality figure for multi-$\gamma$ results. Reads `multi_gamma_results.npz`. **Fig. 2** |
| `noise_injection_v2.py` | SNR susceptibility analysis. Additive detector noise on ensemble-averaged PSD, with and without dark subtraction. Tests at $\gamma = 1.5$ (HIGGINS) and $\gamma = 3.0$ (stress test). **Fig. 3** |

## Quick Start

```bash
# Single-gamma extraction (HIGGINS parameters, ~7 min for 10k realizations)
python T_lab_10k.py --n_ens 10000

# Multi-gamma validation (~3.5 min for 8 gammas × 1000 realizations)
python multi_gamma.py --n_cal 1000 --n_boost 1000

# Generate PRL figure from saved results
python multi_gamma_prl.py

# Noise susceptibility (~2 min)
python noise_injection_v2.py --n_ens 1000 --n_trials 1000
```

For quick sanity checks, reduce ensemble sizes:
```bash
python T_lab_10k.py --n_ens 200
python multi_gamma.py --n_cal 100 --n_boost 50
python noise_injection_v2.py --n_ens 200 --n_trials 200
```

## Physical Parameters (HIGGINS)

| Parameter | Value | Description |
|-----------|-------|-------------|
| $T_0$ | 8512 $\hbar\omega_p$ (1 keV) | Electron temperature |
| $\omega_p$ | $1.78 \times 10^{14}$ rad/s (28 THz) | Plasma frequency |
| $\nu/\omega_p$ | 0.05 | Drude collision rate |
| $\gamma$ | 1.5 (nominal) | Lorentz factor |
| Detector | 12 pixels, $R = 50$ cm, $d = 3$ cm | Ring array, $\Delta\theta = 3.4°$ |
| Modes | 40,000 per realization | Frequency range [0.1, 13.0] $\omega_p$ |

## Key Results

- **$T_{\rm lab}(\theta)$ extraction**: 0.34% RMS at 10,000 realizations ($\gamma = 1.5$)
- **$\beta$ recovery**: 0.1% from E-B cross-correlation
- **Multi-$\gamma$**: sub-percent $T_0$ accuracy for $\gamma \leq 2$, degrading to ~6% at $\gamma = 10$ (mode starvation)
- **Noise threshold**: SNR $\gtrsim 10$ sufficient for $< 15\%$ error with dark subtraction ($\gamma = 1.5$)

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib

## Citation

```bibtex
@article{wolfson2026em,
  title={Operational measurement of relativistic equilibrium
         from stochastic electromagnetic fields alone},
  author={Wolfson, Ira},
  year={2026},
  eprint={2602.16765},
  archivePrefix={arXiv},
  primaryClass={physics.plasm-ph}
}
```

## License

MIT
