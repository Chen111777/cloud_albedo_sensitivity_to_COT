"""
utils_fitting_110723.py

Direct nonlinear COT-albedo fitting and SBDART helpers for RFOV/SBD figures.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import xarray as xr

# ============================================================
# Constants
# ============================================================

oceans = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']

season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

cot_range = np.exp(np.linspace(np.log(2.5), np.log(62.5), 15))


# ============================================================
# Transformation functions
# ============================================================

def cot_to_x(cot):
    return np.log(cot)


def albedo_to_y(albedo):
    albedo = np.asarray(albedo, dtype=float)
    albedo = np.clip(albedo, 1e-6, 1 - 1e-6)
    return np.log(albedo / (1 - albedo))


def cot_k_b_to_albedo(cot, k, b):
    """Corrected albedo: Ac = b * cot^k / (1 + b * cot^k)"""
    return b * cot ** k / (1 + b * cot ** k)


# ============================================================
# Fitting helper functions
# ============================================================

def _cot_lnb_to_albedo(cot, k, ln_b):
    """Evaluate b*COT**k/(1+b*COT**k), with b=exp(ln_b), stably."""
    eta = ln_b + k * np.log(np.asarray(cot, dtype=float))
    return np.exp(-np.logaddexp(0.0, -eta))


def _cot_lnb_jacobian(cot, k, ln_b):
    """Jacobian of untransformed albedo with respect to (k, ln_b)."""
    albedo = _cot_lnb_to_albedo(cot, k, ln_b)
    sensitivity = albedo * (1.0 - albedo)
    return np.column_stack([sensitivity * np.log(cot), sensitivity])


def fit_cot_albedo(cot, albedo):
    """Fit Ac = b*COT**k/(1+b*COT**k) by nonlinear least squares.

    Minimize squared residuals in albedo, not logit(albedo). COT and albedo
    are passed to curve_fit without coordinate transformations. Optimize
    k and ln(b) directly, with b=exp(ln(b)) positive by construction.

    Parameters
    ----------
    cot, albedo : array-like
        Binned mean COT and cloud albedo.
    Returns
    -------
    k, ln_b, k_std, ln_b_std : float
        Nonlinear-fit parameters and approximate one-sigma standard errors.
        Covariance is scaled by RSS/(n-2) in albedo space. Standard errors
        come directly from the covariance of (k, ln b). It assumes independent,
        homoscedastic albedo residuals.
    """
    cot = np.asarray(cot, dtype=float).ravel()
    albedo = np.asarray(albedo, dtype=float).ravel()

    if cot.size != albedo.size:
        raise ValueError("cot and albedo must have the same length.")

    mask = (
        np.isfinite(cot) & np.isfinite(albedo) &
        (cot > 0) & (albedo > 0) & (albedo < 1)
    )

    cot = cot[mask]
    albedo = albedo[mask]

    if cot.size < 3 or np.unique(cot).size < 2:
        return np.nan, np.nan, np.nan, np.nan
    try:
        parameters, covariance = curve_fit(
            _cot_lnb_to_albedo, cot, albedo,
            p0=(0.7, np.log(0.1)),
            method='trf', jac=_cot_lnb_jacobian,
            absolute_sigma=False, max_nfev=10000,
        )
        k, ln_b = parameters
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return k, ln_b, standard_errors[0], standard_errors[1]
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return np.nan, np.nan, np.nan, np.nan


# ============================================================
# SBDART lookup table interpolation
# ============================================================

# 3D SBDART LUT (sza x nre(=cloud effective radius) x cot) produced by
# project0904/SBDART_LUT/run_sbdart.py.
SBDART_LUT_DIR = '/home/chenyiqi/251028_albedo_cot/project0904/SBDART_LUT'
_SBDART_3D_LUT_CACHE = {}


def _sbdart_3d_interpolator(table_folder, ocean, season):
    """Cached (sza, nre, cot) interpolator for one 3D SBDART LUT netCDF file."""
    if table_folder.endswith('vis'):
        # vis folders only contain the representative TPO/MAM LUT
        ocean, season = 'TPO', 'MAM'
    base = f'{SBDART_LUT_DIR}/{table_folder}'
    path = f'{base}/albedo_cot_cer_sza_LUT_{ocean}_{season}.nc'
    if not os.path.exists(path):
        # dcp-style folders store one representative (TPO/MAM) LUT for everything
        fallback = f'{base}/albedo_cot_cer_sza_LUT_TPO_MAM.nc'
        if os.path.exists(fallback):
            path = fallback
        else:
            return None
    if path not in _SBDART_3D_LUT_CACHE:
        with xr.open_dataset(path) as ds:
            grid = ds['albedo'].values.astype(float)
            axes = (
                ds['sza'].values.astype(float),
                ds['nre'].values.astype(float),
                ds['cot'].values.astype(float),
            )
        _SBDART_3D_LUT_CACHE[path] = RegularGridInterpolator(
            axes, grid, method='linear', bounds_error=False, fill_value=np.nan
        )
    return _SBDART_3D_LUT_CACHE[path]


def cot_to_albedo(cot, method, miu=None, sza=None, cer=None, table_folder='dcp', ocean=None, season=None):
    """
    Compute cloud albedo from COT using various methods.

    Parameters
    ----------
    cot : array-like
        Cloud optical thickness.
    method : str
        'sbdart', 'analy'.
    sza : float or array-like, optional
        Solar zenith angle in degrees. Required for 'sbdart', 'quadrature', 'eddington'.
    table_folder : str, optional
        Subfolder name under build_sbdart_lookup_table/ (e.g., 'dcp', 'cp').
    ocean : str, optional
        Ocean region name (e.g., 'TPO'). If provided with season, reads region-specific LUT.
    season : str, optional
        Season name (e.g., 'MAM'). If provided with ocean, reads region-specific LUT.

    Returns
    -------
    array-like
        Cloud albedo values.
    """
    cot = np.asarray(cot, dtype=float)

    if method == 'sbdart':
        # 3D lookup by (cot, sza, cer = cloud effective radius) from the
        # netCDF LUT produced by project0904/SBDART_LUT/run_sbdart.py:
        #   SBDART_LUT/<table_folder>/albedo_cot_cer_sza_LUT_{ocean}_{season}.nc
        cot_arr = np.atleast_1d(np.asarray(cot, dtype=float))
        interpolator = _sbdart_3d_interpolator(table_folder, ocean, season)
        if interpolator is None:
            return np.full(cot_arr.shape, np.nan)
        if np.ndim(sza) == 0:
            sza_arr = np.full_like(cot_arr, float(sza))
        else:
            sza_arr = np.asarray(sza, dtype=float).ravel()
        if np.ndim(cer) == 0:
            cer_arr = np.full_like(cot_arr, float(cer))
        else:
            cer_arr = np.asarray(cer, dtype=float).ravel()
        points = np.column_stack([sza_arr, cer_arr, cot_arr])
        return interpolator(points).reshape(cot_arr.shape)

    if method == 'analy':
        g = 0.85
        b = (1 - g) / 2 / miu
        return b * cot / (1 + b * cot)

    raise ValueError(f'Unsupported method: {method}')


# ============================================================
# Panel tag formatting
# ============================================================

def format_panel_tag(panel_idx, icon_style):
    """Format panel tag: 'science' -> A, B, C...; 'nature' -> (a), (b), (c)..."""
    if icon_style == 'science':
        letter = chr(ord('A') + panel_idx)
        return rf'$\mathbf{{{letter}}}$'
    letter = chr(ord('a') + panel_idx)
    return rf'$\mathbf{{{letter}}}$'
