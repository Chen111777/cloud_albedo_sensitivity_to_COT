import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path configuration
BASE_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/processed_data/merged_msk_and_ret_csv/'
WEIGHTED_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/ocean_season_sza_weighted.csv'
HEATMAP_DATA_DIR = '/home/chenyiqi/251028_albedo_cot/build_sbdart_lookup_table/cot_sza_to_albedo_lookup_table_cp/'
KLNB_OCEAN_FILE = '/home/chenyiqi/251028_albedo_cot/processed_data/k_lnb_by_seasons_oceans.csv'

# Core parameters
OCEANS = ['NPO', 'NAO', 'TPO', 'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
SEASONS = ['MAM', 'JJA', 'SON', 'DJF']
SEASON_MONTHS = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# Calculation parameters
MIN_POINTS_FOR_FIT = 2
LN_COT_LOW = 1.5
LN_COT_HIGH = 3.0

# Output configuration
OUTPUT_PNG = 'figs/k_lnb_plot.png'
# Output CSVs
UNCOR_K1_CSV = 'processed_data/uncor_k1_values.csv'
UNCOR_K2_CSV = 'processed_data/uncor_k2_values.csv'
UNCOR_LNB2_CSV = 'processed_data/uncor_lnb2_values.csv'

SZACORR_K1_CSV = 'processed_data/szacorr_k1_values.csv'
SZACORR_K2_CSV = 'processed_data/szacorr_k2_values.csv'
SZACORR_LNB2_CSV = 'processed_data/szacorr_lnb2_values.csv'

# Plot configuration
HEATMAP_CMAP = plt.cm.GnBu
LNB_CMAP = plt.cm.pink_r
K_VMIN, K_VMAX = 0.25, 0.9
LNB_VMIN, LNB_VMAX = -2.7, -0.6
SIZE_PARAMS = {
    'large_tick': 12,
    'small_tick': 9.5,
    'xylabel': 15,
    'title': 17,
    'legend': 11,
    'cbar_tick': 10.5,
}


def load_weighted_angles(file_path):
    """
    Load weighted SZA data.
    Return dict: {(ocean, season): weighted_angle_deg}
    """
    df = pd.read_csv(file_path)
    df = df[~df['season'].isin(['Global'])]

    weight_dict = {}
    for _, row in df.iterrows():
        weight_dict[(row['ocean'], row['season'])] = row['weighted_angle_deg']
    return weight_dict


def calculate_seasonal_stats(ocean_list, data_dir):
    """
    Calculate mean SZA per ocean-season.
    Return dict: {(ocean, season): mean_sza}
    """
    seasonal_stats = {}

    for ocean in ocean_list:
        file_path = os.path.join(data_dir, f'{ocean}.csv')

        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df['month'] = df['time'].dt.month
        df['season'] = None

        for season_name, months in SEASON_MONTHS.items():
            df.loc[df['month'].isin(months), 'season'] = season_name

        df_valid = df.dropna(subset=['season'])
        seasonal_avg = df_valid.groupby('season')['sza'].mean()

        for season in SEASONS:
            mean_sza = seasonal_avg.loc[season] if season in seasonal_avg.index else np.nan
            seasonal_stats[(ocean, season)] = mean_sza

    return seasonal_stats


def compute_k_and_intercept(cot_vals, albedo_vals):
    """
    Calculate slope (k) and intercept (lnb) for
    ln(A/(1-A)) vs ln(COT) in range [1.5, 3.0].
    """
    mask = (
        (cot_vals > 0.0) &
        np.isfinite(cot_vals) &
        np.isfinite(albedo_vals) &
        (albedo_vals > 0.0) &
        (albedo_vals < 1.0)
    )
    if np.sum(mask) < MIN_POINTS_FOR_FIT:
        return np.nan, np.nan

    cot_valid = np.asarray(cot_vals[mask], dtype=float)
    a_valid = np.asarray(albedo_vals[mask], dtype=float)

    x = np.log(cot_valid)
    y = np.log(a_valid / (1.0 - a_valid))

    range_mask = (x >= LN_COT_LOW) & (x <= LN_COT_HIGH)
    if np.sum(range_mask) < MIN_POINTS_FOR_FIT:
        return np.nan, np.nan

    x_range = x[range_mask]
    y_range = y[range_mask]

    try:
        slope, intercept = np.polyfit(x_range, y_range, 1)
        return float(slope), float(intercept)
    except Exception:
        return np.nan, np.nan


def get_lookup_data(ocean, season):
    """
    Load lookup table and return cos(SZA), slope (k), and intercept (lnb).
    """
    file_path = os.path.join(
        HEATMAP_DATA_DIR,
        f'cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv'
    )

    df = pd.read_csv(file_path, index_col=0)
    sza = np.array(df.index.astype(float))
    cot = np.array(df.columns.astype(float))
    albedo_grid = df.values.astype(float)

    sort_sza_idx = np.argsort(sza)
    sza_sorted = sza[sort_sza_idx]
    cos_sza_sorted = np.cos(np.radians(sza_sorted))
    albedo_sorted = albedo_grid[sort_sza_idx, :]

    slope_list = []
    intercept_list = []
    for i in range(len(sza_sorted)):
        slope, intercept = compute_k_and_intercept(cot, albedo_sorted[i, :])
        slope_list.append(slope)
        intercept_list.append(intercept)

    return cos_sza_sorted, np.array(slope_list), np.array(intercept_list)


def get_value_at_sza(cos_sza_vals, target_cos_sza, value_array):
    """
    Get the closest value from value_array at target cos(SZA).
    """
    if cos_sza_vals is None or value_array is None:
        return np.nan

    valid_mask = np.isfinite(cos_sza_vals) & np.isfinite(value_array)
    if not np.any(valid_mask):
        return np.nan

    cos_sza_valid = cos_sza_vals[valid_mask]
    value_valid = value_array[valid_mask]

    if len(cos_sza_valid) == 0:
        return np.nan

    closest_idx = np.argmin(np.abs(cos_sza_valid - target_cos_sza))
    return value_valid[closest_idx]


def load_uncor_from_klnb(file_path, var_type='Slope', method='ret'):
    """
    Load uncorrected seasonal values directly from k_lnb_by_seasons_oceans.csv.

    Parameters
    ----------
    var_type : str
        'Slope' or 'Intercept'
    method : str
        e.g. 'msk', 'ret', 'cp', 'dcp'

    Returns
    -------
    DataFrame with columns: ['ocean', 'MAM', 'JJA', 'SON', 'DJF']
    """
    df = pd.read_csv(file_path)
    if 'Ocean' not in df.columns:
        raise ValueError("Input CSV must contain column 'Ocean'.")

    df = df[df['Ocean'] != 'Global'].copy()

    out = pd.DataFrame()
    out['ocean'] = df['Ocean']

    for season in SEASONS:
        col = f'{season}_{var_type}_{method}'
        if col in df.columns:
            out[season] = pd.to_numeric(df[col], errors='coerce')
        else:
            out[season] = np.nan

    return out


def combine_diff_with_uncor(diff_df, uncor_df):
    """
    Add SZA correction difference to original seasonal values.
    diff_df: index=Ocean, columns=SEASONS
    uncor_df: columns=['ocean', 'MAM', 'JJA', 'SON', 'DJF']
    """
    base = uncor_df.copy().set_index('ocean')

    common_oceans = diff_df.index.intersection(base.index)
    common_seasons = diff_df.columns.intersection(SEASONS)

    corrected = base.copy()
    corrected.loc[common_oceans, common_seasons] = (
        base.loc[common_oceans, common_seasons] +
        diff_df.loc[common_oceans, common_seasons]
    )

    corrected = corrected.reset_index()
    corrected = corrected.rename(columns={'ocean': 'ocean'})
    return corrected


def plot_heatmap(ax, df, title, cmap=HEATMAP_CMAP, vmin=None, vmax=None):
    """
    Plot heatmap with oceans (y-axis), seasons (x-axis).
    """
    for col in SEASONS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    heatmap_data = df[SEASONS].values.astype(np.float64)
    oceans = df['ocean'].tolist()

    heatmap_data = np.where(np.isinf(heatmap_data), np.nan, heatmap_data)

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(SEASONS)))
    ax.set_yticks(np.arange(len(oceans)))
    ax.set_xticklabels(SEASONS, fontsize=SIZE_PARAMS['large_tick'], ha='center')
    ax.set_yticklabels(oceans, fontsize=SIZE_PARAMS['large_tick'])

    for i in range(len(oceans)):
        for j in range(len(SEASONS)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f'{val:.2f}',
                    ha="center", va="center",
                    color='k', fontsize=9.5, fontweight='bold'
                )

    ax.set_title(title, fontsize=SIZE_PARAMS['title'], pad=7, loc='left')
    return im


def plot_main_ax(ax, seasonal_stats, weight_dict, is_k_plot=True, cmap=HEATMAP_CMAP):
    """
    Plot main axis with SZA range 20-75° and legend at top-left.
    """
    all_sza = []
    for ocean in OCEANS:
        for season in SEASONS:
            cos_sza, _, _ = get_lookup_data(ocean, season)
            if cos_sza is not None:
                all_sza.extend(np.degrees(np.arccos(cos_sza)))

    unique_sza = np.sort(np.unique(all_sza)) if all_sza else np.array([])
    n_y = len(unique_sza)
    n_ocean = len(OCEANS)
    n_season = len(SEASONS)
    n_x = n_ocean * n_season

    main_data = np.full((n_y, n_x), np.nan)
    x_ticks = []
    ocean_label_pos = []
    ocean_labels = []

    mean_sza_x, mean_sza_y = [], []
    weighted_sza_x, weighted_sza_y = [], []

    for o_idx, ocean in enumerate(OCEANS):
        x_start = o_idx * n_season
        x_end = x_start + n_season
        ocean_label_pos.append((x_start + x_end - 1) / 2)
        ocean_labels.append(ocean)

        for s_idx, season in enumerate(SEASONS):
            x_pos = x_start + s_idx
            x_ticks.append(season)

            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)

            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)

            if cos_sza is not None:
                sza_vals = np.degrees(np.arccos(cos_sza))
                for y_idx, target_sza in enumerate(unique_sza):
                    closest_idx = np.argmin(np.abs(sza_vals - target_sza))
                    if np.isclose(sza_vals[closest_idx], target_sza):
                        main_data[y_idx, x_pos] = (
                            slope_vals[closest_idx] if is_k_plot else intercept_vals[closest_idx]
                        )

            if not np.isnan(mean_sza_deg):
                mean_sza_x.append(x_pos)
                mean_sza_y.append(mean_sza_deg)

            if not np.isnan(weighted_sza_deg):
                weighted_sza_x.append(x_pos)
                weighted_sza_y.append(weighted_sza_deg)

    if not np.all(np.isnan(main_data)):
        main_data_transposed = main_data.T

        vmin = K_VMIN if is_k_plot else LNB_VMIN
        vmax = K_VMAX if is_k_plot else LNB_VMAX

        ax.imshow(
            main_data_transposed,
            aspect='auto',
            cmap=cmap,
            extent=[20, 75, -0.5, n_x - 0.5],
            vmin=vmin,
            vmax=vmax
        )

        mean_mask = (np.array(mean_sza_y) >= 20) & (np.array(mean_sza_y) <= 75)
        weighted_mask = (np.array(weighted_sza_y) >= 20) & (np.array(weighted_sza_y) <= 75)

        ax.scatter(
            np.array(mean_sza_y)[mean_mask],
            np.array(mean_sza_x)[mean_mask],
            color='red', s=50, marker='o',
            label='10:30', zorder=5, edgecolors='black'
        )
        ax.scatter(
            np.array(weighted_sza_y)[weighted_mask],
            np.array(weighted_sza_x)[weighted_mask],
            color='blue', s=60, marker='^',
            label='Daytime', zorder=5, edgecolors='black'
        )

        vline_positions = [n_season * (i + 1) - 0.5 for i in range(n_ocean - 1)]
        for vline_pos in vline_positions:
            ax.axhline(y=vline_pos, color='lightgray', linestyle='-', linewidth=1, zorder=4)

        ax.set_xlabel(r'SZA ($^\circ$)', fontsize=SIZE_PARAMS['xylabel'])
        ax.set_xlim(20, 75)
        ax.set_ylim(-0.5, n_x - 0.5)
        ax.set_xticks(np.arange(20, 76, 10))
        ax.set_xticklabels([f'{int(x)}' for x in np.arange(20, 76, 10)], fontsize=SIZE_PARAMS['large_tick'])

        ax.set_yticks(range(n_x))
        ax.set_yticklabels(x_ticks, fontsize=10, va='center', rotation=0)

        x_offset = 13
        for pos, label in zip(ocean_label_pos, ocean_labels):
            ax.text(x_offset, pos, label, ha='right', va='center', fontsize=SIZE_PARAMS['large_tick'])

        ax.legend(loc='upper left', fontsize=SIZE_PARAMS['legend'], frameon=True)

        if is_k_plot:
            ax.set_title(r'$\mathbf{(b)}$ $k_{\mathrm{cp}}$ vs. SZA', loc='left', fontsize=SIZE_PARAMS['title'], pad=7)
        else:
            ax.set_title(r'$\mathbf{(e)}$ ln$b_{\mathrm{cp}}$ vs. SZA', loc='left', fontsize=SIZE_PARAMS['title'], pad=7)


def create_main_plot():
    """
    Create combined plot with k_ret (row 1), lnb_ret (row 2), k_msk (row 3).
    """
    seasonal_stats = calculate_seasonal_stats(OCEANS, BASE_DATA_DIR)
    weight_dict = load_weighted_angles(WEIGHTED_FILE)

    diff_k_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)   # for k_ret
    diff_k1_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)  # for k_msk
    diff_b_data = pd.DataFrame(index=OCEANS, columns=SEASONS, dtype=float)   # for lnb_ret

    for ocean in OCEANS:
        for season in SEASONS:
            mean_sza_deg = seasonal_stats[(ocean, season)]
            weighted_sza_deg = weight_dict.get((ocean, season), np.nan)

            cos_sza, slope_vals, intercept_vals = get_lookup_data(ocean, season)

            if not np.isnan(mean_sza_deg) and not np.isnan(weighted_sza_deg):
                cos_mean_sza = np.cos(np.radians(mean_sza_deg))
                cos_weighted_sza = np.cos(np.radians(weighted_sza_deg))

                k_mean = get_value_at_sza(cos_sza, cos_mean_sza, slope_vals)
                k_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, slope_vals)
                b_mean = get_value_at_sza(cos_sza, cos_mean_sza, intercept_vals)
                b_weighted = get_value_at_sza(cos_sza, cos_weighted_sza, intercept_vals)

                if np.isfinite(k_weighted) and np.isfinite(k_mean):
                    diff_k_data.loc[ocean, season] = k_weighted - k_mean
                    diff_k1_data.loc[ocean, season] = k_weighted - k_mean

                if np.isfinite(b_weighted) and np.isfinite(b_mean):
                    diff_b_data.loc[ocean, season] = b_weighted - b_mean

    # Directly load uncorrected values from k_lnb_by_seasons_oceans.csv
    uncor_k2_df = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='Slope', method='ret').round(4)
    uncor_k1_df = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='Slope', method='msk').round(4)
    uncor_lnb2_df = load_uncor_from_klnb(KLNB_OCEAN_FILE, var_type='Intercept', method='ret').round(4)

    # Save uncorrected CSVs in the same format
    uncor_k2_df.to_csv(UNCOR_K2_CSV, index=False)
    uncor_k1_df.to_csv(UNCOR_K1_CSV, index=False)
    uncor_lnb2_df.to_csv(UNCOR_LNB2_CSV, index=False)

    # Apply SZA corrections
    szacorr_k2_df = combine_diff_with_uncor(diff_k_data, uncor_k2_df).round(4)
    szacorr_k1_df = combine_diff_with_uncor(diff_k1_data, uncor_k1_df).round(4)
    szacorr_lnb2_df = combine_diff_with_uncor(diff_b_data, uncor_lnb2_df).round(4)

    # Save corrected CSVs
    szacorr_k2_df.to_csv(SZACORR_K2_CSV, index=False)
    szacorr_k1_df.to_csv(SZACORR_K1_CSV, index=False)
    szacorr_lnb2_df.to_csv(SZACORR_LNB2_CSV, index=False)

    fig = plt.figure(figsize=(16, 20), dpi=100)

    # Row 1: k_ret
    ax_k_a = fig.add_axes([0.06, 0.69, 0.24, 0.24])
    im_k_a = plot_heatmap(
        ax_k_a, uncor_k2_df,
        r'$\mathbf{(a)}$ $k_{\mathrm{ret}}$, 10:30',
        vmin=K_VMIN, vmax=K_VMAX
    )

    ax_k_b = fig.add_axes([0.38, 0.69, 0.24, 0.24])
    plot_main_ax(ax_k_b, seasonal_stats, weight_dict, is_k_plot=True)

    ax_k_c = fig.add_axes([0.70, 0.69, 0.24, 0.24])
    plot_heatmap(
        ax_k_c, szacorr_k2_df,
        r'$\mathbf{(c)}$ $k_{\mathrm{ret}}$, Daytime Mean',
        vmin=K_VMIN, vmax=K_VMAX
    )

    cbar_k_ax = fig.add_axes([0.97, 0.69, 0.012, 0.24])
    cbar_k = fig.colorbar(im_k_a, cax=cbar_k_ax, orientation='vertical')
    cbar_k.set_label('$k$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_k.set_ticks(np.arange(0.25, 0.91, 0.1))
    cbar_k.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    # Row 2: lnb_ret
    ax_lnb_a = fig.add_axes([0.06, 0.39, 0.24, 0.24])
    im_lnb_a = plot_heatmap(
        ax_lnb_a, uncor_lnb2_df,
        r'$\mathbf{(d)}$ ln$b_{\mathrm{ret}}$, 10:30',
        cmap=LNB_CMAP, vmin=LNB_VMIN, vmax=LNB_VMAX
    )

    ax_lnb_b = fig.add_axes([0.38, 0.39, 0.24, 0.24])
    plot_main_ax(ax_lnb_b, seasonal_stats, weight_dict, is_k_plot=False, cmap=LNB_CMAP)

    ax_lnb_c = fig.add_axes([0.70, 0.39, 0.24, 0.24])
    plot_heatmap(
        ax_lnb_c, szacorr_lnb2_df,
        r'$\mathbf{(f)}$ ln$b_{\mathrm{ret}}$, Daytime Mean',
        cmap=LNB_CMAP, vmin=LNB_VMIN, vmax=LNB_VMAX
    )

    cbar_lnb_ax = fig.add_axes([0.97, 0.39, 0.012, 0.24])
    cbar_lnb = fig.colorbar(im_lnb_a, cax=cbar_lnb_ax, orientation='vertical')
    cbar_lnb.set_label('ln$b$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_lnb.set_ticks(np.arange(-2.7, -0.59, 0.3))
    cbar_lnb.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    # Row 3: k_msk
    ax_k1_g = fig.add_axes([0.06, 0.09, 0.24, 0.24])
    im_k1_g = plot_heatmap(
        ax_k1_g, uncor_k1_df,
        r'$\mathbf{(g)}$ $k_{\mathrm{msk}}$, 10:30',
        vmin=K_VMIN, vmax=K_VMAX
    )

    cbar_k1_ax = fig.add_axes([0.33, 0.09, 0.012, 0.24])
    cbar_k1 = fig.colorbar(im_k1_g, cax=cbar_k1_ax, orientation='vertical')
    cbar_k1.set_label('$k$', fontsize=SIZE_PARAMS['xylabel'])
    cbar_k1.set_ticks(np.arange(0.25, 0.91, 0.1))
    cbar_k1.ax.tick_params(labelsize=SIZE_PARAMS['cbar_tick'])

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print(f"UNCOR K1 saved to: {UNCOR_K1_CSV}")
    print(f"UNCOR K2 saved to: {UNCOR_K2_CSV}")
    print(f"UNCOR LNB2 saved to: {UNCOR_LNB2_CSV}")
    print(f"SZACORR K1 saved to: {SZACORR_K1_CSV}")
    print(f"SZACORR K2 saved to: {SZACORR_K2_CSV}")
    print(f"SZACORR LNB2 saved to: {SZACORR_LNB2_CSV}")

    return fig


if __name__ == '__main__':
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('figs', exist_ok=True)

    print("=== Generating combined k, lnb and k1 plot ===")
    create_main_plot()
