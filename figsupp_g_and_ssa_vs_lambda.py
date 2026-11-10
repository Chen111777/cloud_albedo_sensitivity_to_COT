"""
Spectral variation of the asymmetry parameter g and single-scattering albedo
for liquid-water cloud droplets.

Optical constants (n, k) are based on Hale & Querry (1973), as tabulated in
the public-domain refractiveindex.info database. The table below covers
0.2–6.0 µm; values between tabulated wavelengths are linearly interpolated.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import miepython

from utils_fitting import format_panel_tag

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / 'figs'
OUTPUT_PATH = FIG_DIR / 'figsupp_g_and_ssa_vs_lambda.png'

diameters = [7.0, 10.0, 13.0]  # droplet diameters (µm)
wavelengths = np.arange(0.20, 6.001, 0.01)  # wavelength (µm)

# Hale & Querry (1973): wavelength (µm), real refractive index n,
# imaginary refractive index k, for liquid water at approximately 25 °C.
# Source: refractiveindex.info, H2O / Hale.
nk_table = np.array([
    [0.200, 1.396, 1.10e-7],
    [0.225, 1.373, 4.90e-8],
    [0.250, 1.362, 3.35e-8],
    [0.275, 1.354, 2.35e-8],
    [0.300, 1.349, 1.60e-8],
    [0.325, 1.346, 1.08e-8],
    [0.350, 1.343, 6.50e-9],
    [0.375, 1.341, 3.50e-9],
    [0.400, 1.339, 1.86e-9],
    [0.425, 1.338, 1.30e-9],
    [0.450, 1.337, 1.02e-9],
    [0.475, 1.336, 9.35e-10],
    [0.500, 1.335, 1.00e-9],
    [0.525, 1.334, 1.32e-9],
    [0.550, 1.333, 1.96e-9],
    [0.575, 1.333, 3.60e-9],
    [0.600, 1.332, 1.09e-8],
    [0.625, 1.332, 1.39e-8],
    [0.650, 1.331, 1.64e-8],
    [0.675, 1.331, 2.23e-8],
    [0.700, 1.331, 3.35e-8],
    [0.725, 1.330, 9.15e-8],
    [0.750, 1.330, 1.56e-7],
    [0.775, 1.330, 1.48e-7],
    [0.800, 1.329, 1.25e-7],
    [0.825, 1.329, 1.82e-7],
    [0.850, 1.329, 2.93e-7],
    [0.875, 1.328, 3.91e-7],
    [0.900, 1.328, 4.86e-7],
    [0.925, 1.328, 1.06e-6],
    [0.950, 1.327, 2.93e-6],
    [0.975, 1.327, 3.48e-6],
    [1.000, 1.327, 2.89e-6],
    [1.200, 1.324, 9.89e-6],
    [1.400, 1.321, 1.38e-4],
    [1.600, 1.317, 8.55e-5],
    [1.800, 1.312, 1.15e-4],
    [2.000, 1.306, 1.10e-3],
    [2.200, 1.296, 2.89e-4],
    [2.400, 1.279, 9.56e-4],
    [2.600, 1.242, 3.17e-3],
    [2.650, 1.219, 6.70e-3],
    [2.700, 1.188, 1.90e-2],
    [2.750, 1.157, 5.90e-2],
    [2.800, 1.142, 1.15e-1],
    [2.850, 1.149, 1.85e-1],
    [2.900, 1.201, 2.68e-1],
    [2.950, 1.292, 2.98e-1],
    [3.000, 1.371, 2.72e-1],
    [3.050, 1.426, 2.40e-1],
    [3.100, 1.467, 1.92e-1],
    [3.150, 1.483, 1.35e-1],
    [3.200, 1.478, 9.24e-2],
    [3.250, 1.467, 6.10e-2],
    [3.300, 1.450, 3.68e-2],
    [3.350, 1.432, 2.61e-2],
    [3.400, 1.420, 1.95e-2],
    [3.450, 1.410, 1.32e-2],
    [3.500, 1.400, 9.40e-3],
    [3.600, 1.385, 5.15e-3],
    [3.700, 1.374, 3.60e-3],
    [3.800, 1.364, 3.40e-3],
    [3.900, 1.357, 3.80e-3],
    [4.000, 1.351, 4.60e-3],
    [4.100, 1.346, 5.62e-3],
    [4.200, 1.342, 6.88e-3],
    [4.300, 1.338, 8.45e-3],
    [4.400, 1.334, 1.03e-2],
    [4.500, 1.332, 1.34e-2],
    [4.600, 1.330, 1.47e-2],
    [4.700, 1.330, 1.57e-2],
    [4.800, 1.330, 1.50e-2],
    [4.900, 1.328, 1.37e-2],
    [5.000, 1.325, 1.24e-2],
    [5.100, 1.322, 1.11e-2],
    [5.200, 1.317, 1.01e-2],
    [5.300, 1.312, 9.80e-3],
    [5.400, 1.305, 1.03e-2],
    [5.500, 1.298, 1.16e-2],
    [5.600, 1.289, 1.42e-2],
    [5.700, 1.277, 2.03e-2],
    [5.800, 1.262, 3.30e-2],
    [5.900, 1.248, 6.22e-2],
    [6.000, 1.265, 1.07e-1],
])

# Interpolate the complex refractive index at each wavelength.
n_real = np.interp(wavelengths, nk_table[:, 0], nk_table[:, 1])
n_imag = np.interp(wavelengths, nk_table[:, 0], nk_table[:, 2])

# ---------------------------------------------------------------------
# Mie calculations
# ---------------------------------------------------------------------
results = {}
for d in diameters:
    g_values = np.empty_like(wavelengths)
    ssa_values = np.empty_like(wavelengths)

    for i, lam in enumerate(wavelengths):
        x = np.pi * d / lam  # size parameter: x = pi * diameter / wavelength
        m = complex(n_real[i], n_imag[i])
        qext, qsca, qback, g = miepython.efficiencies_mx(m, x)

        g_values[i] = g
        ssa_values[i] = qsca / qext if qext > 0 else np.nan  # omega_0 = Qsca/Qext

    results[d] = {"g": g_values, "ssa": ssa_values}

# ---------------------------------------------------------------------
# Plot: two panels
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharex=True)
colors = ["tab:blue", "tab:orange", "tab:green"]

for color, d in zip(colors, diameters):
    axes[0].plot(wavelengths, results[d]["g"], color=color, lw=1.5,
                 label=fr"$d={d:.0f}\ \mu$m")
    axes[1].plot(wavelengths, results[d]["ssa"], color=color, lw=1.5,
                 label=fr"$d={d:.0f}\ \mu$m")

# Mark approximate spectral regions used in the original figure.
for ax in axes:
    for xline, color in [(0.3, "gray"), (0.4, "tab:red"),
                         (0.7, "tab:red"), (5.0, "gray")]:
        ax.axvline(xline, color=color, ls="--", lw=0.9, alpha=0.7)
    ax.set_xlim(0.2, 6.0)
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel(r"$g$")
axes[0].set_ylim(0, 1)
axes[0].legend(frameon=False, loc="best")

axes[1].set_ylabel(r"$\omega_0$")
axes[1].set_ylim(0, 1.02)
axes[1].legend(frameon=False, loc="lower left")

# Panel tags (A, B, ...) in the same style as the other figures of this study.
for panel_index, ax in enumerate(axes):
    ax.text(-0.03, 1.02, format_panel_tag(panel_index, 'science'),
            transform=ax.transAxes, fontsize=14, va='bottom', ha='left')

fig.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
print(f'Saved: {OUTPUT_PATH}')

