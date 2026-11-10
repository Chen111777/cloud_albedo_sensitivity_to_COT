# -*- coding: utf-8 -*-
"""Direct nonlinear RFOV/SBD binned fits, with annual and seasonal panels.

The transformed-coordinate figures display the same nonlinear fit; they do
not perform a separate linear regression.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_fitting import (
    albedo_to_y, cot_k_b_to_albedo, cot_to_albedo, cot_to_x,
    fit_cot_albedo, format_panel_tag, oceans,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = next(
    (p for p in (BASE_DIR, BASE_DIR.parent / 'code1107', BASE_DIR.parent)
     if (p / 'RFOV_product' / 'ocean_season').is_dir()), BASE_DIR,
)
RFOV_DIR = DATA_DIR / 'RFOV_product' / 'ocean_season'
SBDART_LUT_FOLDER = 'gascp_aodcp_sfccp_sw'
FIG_DIR = BASE_DIR / 'figs'
OUTPUT_PATH = FIG_DIR / 'fig1_fittings_ocean.png'
LINEAR_OUTPUT_PATH = FIG_DIR / 'figsupp_fittings_ocean_linear.png'
SEASONAL_OUTPUT_PATH = FIG_DIR / 'fig1_fittings_ocean_season.png'
SEASONAL_LINEAR_OUTPUT_PATH = FIG_DIR / 'figsupp_fittings_ocean_season_linear.png'
CSV_PATH = BASE_DIR / 'processed_data' / 'fig1_fitting_8oceans_k.csv'
SEASONAL_CSV_PATH = BASE_DIR / 'processed_data' / 'fig1_fitting_8oceans_seasonal_k.csv'
CSV_COLUMNS = [
    'Ocean',
    'k_fit_rfov', 'k_fit_std_rfov', 'lnb_fit_rfov', 'lnb_fit_std_rfov', 'b_fit_rfov',
    'k_fit_sbd', 'k_fit_std_sbd', 'lnb_fit_sbd', 'lnb_fit_std_sbd', 'b_fit_sbd',
]
MIN_COT = 2
COT_EDGES = np.geomspace(MIN_COT, 76, 17)
COT_FIT = np.geomspace(MIN_COT, 60, 300)
SEASONS = ('MAM', 'JJA', 'SON', 'DJF')
LAYOUT = (('Global', 'NPO', 'NAO'), ('TPO', 'TAO', 'TIO'), ('SPO', 'SAO', 'SIO'))
ANALY_COLOR = '#222222'
SBD_COLOR = '#574cff'
RFOV_COLOR = '#00bfff'


def bin_data(data, cot_column, albedo_column):
    labels = pd.cut(data[cot_column], COT_EDGES, labels=False, include_lowest=True)
    cot_values, albedo_values = [], []
    for index in range(len(COT_EDGES) - 1):
        subset = data[labels == index]
        if len(subset) < 5:
            continue
        cot_values.append(np.exp(np.log(subset[cot_column]).mean()))
        albedo_values.append(subset[albedo_column].mean())
    return np.asarray(cot_values), np.asarray(albedo_values)


def fit_line(cot, albedo):
    if len(cot) < 3:
        return np.nan, np.nan, np.nan, np.nan
    return fit_cot_albedo(cot, albedo)


def add_rfov_cot(data):
    bins = []
    pattern = re.compile(r'^cot_freq_(\d+)_(\d+)$')
    for column in data.columns:
        match = pattern.match(column)
        if match:
            lower, upper = map(int, match.groups())
            bins.append((lower, upper, column))
    if not bins:
        raise ValueError('No cot_freq_<lower>_<upper> columns found in RFOV data.')
    bins.sort()
    columns = [column for _, _, column in bins]
    midpoints = np.array([(lower + upper) / 2 for lower, upper, _ in bins])
    weights = data[columns].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).to_numpy(float)
    total = weights.sum(axis=1)
    mean_log_cot = np.divide(weights @ np.log(midpoints), total,
                             out=np.full(len(data), np.nan), where=total > 0)
    result = data.copy()
    result['cot_rfov'] = np.exp(mean_log_cot)
    return result


def load_rfov_data():
    frames = []
    for path in sorted(RFOV_DIR.glob('*.csv')):
        parts = path.stem.rsplit('_', 1)
        if len(parts) != 2 or parts[0] not in oceans or parts[1] not in SEASONS:
            continue
        ocean, season = parts
        data = add_rfov_cot(pd.read_csv(path))
        data['ret_albedo'] = pd.to_numeric(data['ret_albedo'], errors='coerce')
        data['ocean'], data['season'] = ocean, season
        frames.append(data[['cot_rfov', 'ret_albedo', 'cer_ret_mean',
                            'solar_zenith', 'ocean', 'season']])
    if not frames:
        raise FileNotFoundError(f'No ocean-season RFOV CSV files found in {RFOV_DIR}')
    data = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    return data[(data['cot_rfov'] >= MIN_COT) & data['ret_albedo'].between(0, 1)].copy()


def add_sbdart_albedo(data):
    result = data.copy()
    result['sbd_albedo'] = np.nan
    for (ocean, season), indices in result.groupby(['ocean', 'season']).groups.items():
        rows = result.loc[indices]
        result.loc[indices, 'sbd_albedo'] = cot_to_albedo(
            rows['cot_rfov'].to_numpy(), 'sbdart',
            sza=rows['solar_zenith'].to_numpy(), cer=rows['cer_ret_mean'].to_numpy(),
            table_folder=SBDART_LUT_FOLDER, ocean=ocean, season=season,
        )
    return result.dropna(subset=['sbd_albedo'])


def fit_panel(data, ocean, season=None):
    record = {'Ocean': ocean}
    if season is not None:
        record['Season'] = season
    points = {}
    for name, column in (('sbd', 'sbd_albedo'), ('rfov', 'ret_albedo')):
        cot, albedo = bin_data(data, 'cot_rfov', column)
        k, lnb, k_std, lnb_std = fit_line(cot, albedo)
        points[name] = (cot, albedo)
        record.update({f'k_fit_{name}': k, f'k_fit_std_{name}': k_std,
                       f'lnb_fit_{name}': lnb, f'lnb_fit_std_{name}': lnb_std,
                       f'b_fit_{name}': np.exp(lnb)})
    baseline = record['k_fit_rfov']
    record['err_rfov'] = record['k_fit_sbd'] - baseline
    record['err_rfov_pct'] = (
        record['err_rfov'] / baseline * 100
        if np.isfinite(baseline) and baseline != 0 else np.nan
    )
    label = ocean if season is None else f'{ocean}-{season}'
    print(f"{label}: k_sbd={record['k_fit_sbd']:.4f}, b_sbd={record['b_fit_sbd']:.5f}; "
          f"k_rfov={baseline:.4f}, b_rfov={record['b_fit_rfov']:.5f}; "
          f"(k_sbd-k_rfov)/k_rfov={record['err_rfov_pct']:+.2f}%")
    return record, points


def prepare_panels(data):
    annual = {}
    for ocean in ('Global', *oceans):
        subset = data if ocean == 'Global' else data[data['ocean'] == ocean]
        annual[ocean] = fit_panel(subset, ocean)
    seasonal = {}
    for ocean in oceans:
        for season in SEASONS:
            subset = data[(data['ocean'] == ocean) & (data['season'] == season)]
            seasonal[(ocean, season)] = fit_panel(subset, ocean, season)
    return annual, seasonal


def draw_ocean(ax, title, panel, linear=False):
    record, points = panel
    transform_x = cot_to_x if linear else np.asarray
    transform_y = albedo_to_y if linear else np.asarray
    for miu, angle, linestyle in ((3**(-0.5), '54.7', '-'), (1, '0', ':')):
        b = (1 - 0.85) / 2 / miu
        albedo = cot_to_albedo(COT_FIT, 'analy', miu=miu)
        ax.plot(transform_x(COT_FIT), transform_y(albedo), color=ANALY_COLOR,
                lw=1.5, ls=linestyle,
                label=rf'Anal-{angle}$^\circ$: $k$=1, $b$={b:.2f}')
    for name, label, color, marker, size in (
        ('sbd', 'SBD', SBD_COLOR, 's', 3.3),
        ('rfov', 'Obs', RFOV_COLOR, 'o', 2.6),
    ):
        cot, albedo = points[name]
        k, b = record[f'k_fit_{name}'], record[f'b_fit_{name}']
        fitted = cot_k_b_to_albedo(COT_FIT, k, b)
        ax.plot(transform_x(cot), transform_y(albedo), color=color,
                marker=marker, ls='none', ms=size, alpha=0.8 if name == 'rfov' else 1.0)
        ax.plot(transform_x(COT_FIT), transform_y(fitted), color=color, lw=1.3,
                alpha=0.8 if name == 'rfov' else 1.0,
                label=rf'{label}: $k$={k:.2f}, $b$={b:.2f}')
    if linear:
        ax.set(title=title)
    else:
        ax.set(xlim=(0, 60), ylim=(0.1, 0.95), title=title)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.65)


def make_figure(panels, linear=False):
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)
    for row, ocean_row in enumerate(LAYOUT):
        for column, ocean in enumerate(ocean_row):
            ax = axes[row, column]
            draw_ocean(ax, ocean, panels[ocean], linear)
            ax.text(-0.03, 1.01, format_panel_tag(row * 3 + column, 'science'),
                    transform=ax.transAxes, fontsize=12, va='bottom', ha='left')
            if row == 2:
                ax.set_xlabel(r'$\ln(\mathrm{COT})$' if linear else 'COT', fontsize=11)
            if column == 0:
                ax.set_ylabel(r'$\ln[A_c/(1-A_c)]$' if linear else r'$A_c$', fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    output = LINEAR_OUTPUT_PATH if linear else OUTPUT_PATH
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output


def make_seasonal_figure(panels, linear=False):
    # Transposed layout: one row per season, one column per ocean, so that
    # each row holds all eight oceans side by side.
    fig, axes = plt.subplots(len(SEASONS), len(oceans), figsize=(22, 12),
                             sharex=True, sharey=True, squeeze=False)
    for row, season in enumerate(SEASONS):
        for column, ocean in enumerate(oceans):
            ax = axes[row, column]
            draw_ocean(ax, ocean if row == 0 else '', panels[(ocean, season)], linear)
            if row == len(SEASONS) - 1:
                ax.set_xlabel(r'$\ln(\mathrm{COT})$' if linear else 'COT', fontsize=11)
            if column == 0:
                variable = r'$\ln[A_c/(1-A_c)]$' if linear else r'$A_c$'
                ax.set_ylabel(f'{season}\n{variable}', fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    output = SEASONAL_LINEAR_OUTPUT_PATH if linear else SEASONAL_OUTPUT_PATH
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output


def main():
    data = add_sbdart_albedo(load_rfov_data())
    annual, seasonal = prepare_panels(data)
    CSV_PATH.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([p[0] for p in annual.values()])[CSV_COLUMNS].to_csv(CSV_PATH, index=False)
    pd.DataFrame([p[0] for p in seasonal.values()])[
        ['Ocean', 'Season', *CSV_COLUMNS[1:]]
    ].to_csv(SEASONAL_CSV_PATH, index=False)
    for linear in (False, True):
        print(f'Saved: {make_figure(annual, linear)}')
        print(f'Saved: {make_seasonal_figure(seasonal, linear)}')
    print(f'Saved: {CSV_PATH}')
    print(f'Saved: {SEASONAL_CSV_PATH}')


if __name__ == '__main__':
    main()
