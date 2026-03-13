import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import Ac_cot_fitting_utils as acfu

def plot_8_oceans():
    all_processed_ocean_data, _ = acfu.preprocess_ocean_data()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # position projection
    position_map = {
        'NAO': 0, 'NPO': 1, 'TIO': 2, 'TAO': 3,
        'TPO': 4, 'SIO': 5, 'SAO': 6, 'SPO': 7
    }

    # to save fitting result
    all_fit_results = []
    
    # plot 8 subplots
    for ocean in acfu.oceans:
        if ocean not in position_map or all_processed_ocean_data[ocean] is None:
            continue
        
        ax_idx = position_map[ocean]
        ocean_data = all_processed_ocean_data[ocean]
        
        ocean_title = f'{ocean}'
        ocean_results = acfu.plot_axes_content(ocean_data, axes[ax_idx], title=ocean_title)
        
        # save fitting result
        ocean_result_row = {'Ocean': ocean}
        if ocean_results:
            for key in ['ret', 'cp', 'dcp', 'msk', 'LH74']:
                g_slope, g_intercept, s_slopes, s_intercepts = ocean_results[key]
                # annual
                ocean_result_row[f'Ann_Slope_{key}'] = g_slope
                ocean_result_row[f'Ann_Intercept_{key}'] = g_intercept
                # seasonal
                for s_name in acfu.season_dict.keys():
                    ocean_result_row[f'{s_name}_Slope_{key}'] = s_slopes.get(s_name, np.nan)
                    ocean_result_row[f'{s_name}_Intercept_{key}'] = s_intercepts.get(s_name, np.nan)
        else:
            for key in ['ret', 'cp', 'dcp', 'msk', 'LH74']:
                ocean_result_row[f'Ann_Slope_{key}'] = np.nan
                ocean_result_row[f'Ann_Intercept_{key}'] = np.nan
                for s_name in acfu.season_dict.keys():
                    ocean_result_row[f'{s_name}_Slope_{key}'] = np.nan
                    ocean_result_row[f'{s_name}_Intercept_{key}'] = np.nan
        all_fit_results.append(ocean_result_row)
    
    # global axis labels
    fig.text(0.5, 0.04, r'ln(COT)', ha='center', fontsize=16)
    fig.text(0.04, 0.5, r'$\ln\left[A_{\mathrm{c}}/(1-A_{\mathrm{c}})\right]$', va='center', rotation='vertical', fontsize=16)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])
    
    # save figure
    os.makedirs('figs', exist_ok=True)
    output_fig_path = 'figs/fittings_8_oceans.png'
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"8 oceans figure saved to: {output_fig_path}")
    
    # save CSV
    output_csv_path = '/home/chenyiqi/251028_albedo_cot/processed_data/slopes_intercepts_8oceans.csv'
    output_df = pd.DataFrame(all_fit_results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"8 oceans slope and intercept results saved to: {output_csv_path}")

if __name__ == "__main__":
    plot_8_oceans()
