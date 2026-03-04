import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define ocean identifiers and seasonal month mapping
oceans = ['NPO', 'NAO','TPO',  'TAO', 'TIO', 'SPO', 'SAO', 'SIO']
season_dict = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2]
}

# Color palette for 5-segment pie chart (gradient color scheme)
colors = [
    (1.0, 0.8, 0.8),      # delta1
    (1.0, 0.4, 0.4),      # delta2
    (0.8, 0.0, 0.0),      # delta3
    (0.7, 0.87, 0.98),    # delta4
    (0.0, 0.4, 0.8)       # k (slope)
]

def plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope):
    """
    Generate 5-segment pie chart for ocean-specific slope components
    Only show legend for Global ocean
    
    Parameters:
        ocean (str): Ocean identifier (e.g., 'NPO', 'Global')
        delta1-delta4 (float): Slope difference components
        slope (float): Final slope value (k_msk)
    """
    pie_sizes = [delta1, delta2, delta3, delta4, slope]
    labels = ['$1-k_{\mathrm{dcp}}$', '$k_{\mathrm{dcp}}-k_{\mathrm{cp}}$', 
              '$k_{\mathrm{cp}}-k_{\mathrm{ret}}$', '$k_{\mathrm{ret}}-k_{\mathrm{msk}}$', 
              '$k_{\mathrm{msk}}$']
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0.0)
    
    # Plot pie chart with thick white borders
    wedges, texts = ax.pie(
        pie_sizes,
        labels=[None]*5,  # Hide default labels (custom value display)
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor='w', linewidth=4)
    )
    
    # Add centered numeric values to each pie segment
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2  # Calculate central angle of segment
        x = np.cos(np.deg2rad(ang)) * 0.63  # X position for text (radius 0.63)
        y = np.sin(np.deg2rad(ang)) * 0.63  # Y position for text (radius 0.63)
        
        ax.text(
            x, y,
            f"{pie_sizes[i]:.2f}",
            ha='center', va='center',
            fontsize=32, weight='bold',
            color='k'
        )
    
    # Add legend only for Global (increased horizontal spacing)
    if ocean == 'Global':
        ax.legend(
            wedges,
            labels,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            fontsize=15,
            borderaxespad=0.,
            ncol=len(labels),
            frameon=True,
            facecolor='white',
            edgecolor='lightgray',
            framealpha=0.7,
            columnspacing=5  # Increase horizontal spacing between legend items
        )
    
    # Add ocean name label at bottom center
    plt.text(
        0, -1.25,
        f'{ocean}',
        ha='center', va='center',
        fontsize=45,
        weight='bold'
    )
    
    ax.axis('equal')  # Ensure pie chart is circular
    
    # Save plot with transparent background
    plt.savefig(
        f'figs/transparent_pie_chart_{ocean}.png',
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    plt.close()

if __name__ == "__main__":
    # Read slope/intercept data from separate files
    df_oceans = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/slopes_intercepts_8oceans.csv')
    df_global = pd.read_csv('/home/chenyiqi/251028_albedo_cot/processed_data/slopes_intercepts_global.csv')
    
    # Combine 8 oceans + Global data into single DataFrame
    df = pd.concat([df_oceans, df_global], ignore_index=True)

    # Initialize result tables (8 oceans + Global, 4 seasons + Annual)
    k_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    delta2_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    delta3_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    delta4_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    lnb1_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    lnb2_table = np.full([len(oceans)+1, len(season_dict)+1], np.nan)
    
    # Base keys for slope calculation (msk/ret/cp/dcp)
    base_keys = ['_msk', '_ret', '_cp', '_dcp']
    
    # Process 8 individual oceans
    for i, ocean in enumerate(oceans):
        ocean_data = df[df['Ocean'] == ocean].iloc[0]
        
        # Calculate seasonal values
        for j, season in enumerate(season_dict.keys()):
            keys = [f'{season}_Slope{base_key}' for base_key in base_keys]
            delta1 = 1 - ocean_data[keys[3]]
            delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
            delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
            delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
            slope = 1 - delta1 - delta2 - delta3 - delta4
            
            # Assign values to seasonal columns
            k_table[i, j] = slope
            delta2_table[i, j] = delta2
            delta3_table[i, j] = delta3
            delta4_table[i, j] = delta4
            lnb1_table[i, j] = ocean_data[f'{season}_Intercept_msk']
            lnb2_table[i, j] = ocean_data[f'{season}_Intercept_ret']

        # Calculate annual values
        keys = [f'Ann_Slope{base_key}' for base_key in base_keys]
        delta1 = 1 - ocean_data[keys[3]]
        delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
        delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
        delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
        slope = 1 - delta1 - delta2 - delta3 - delta4

        # Assign annual values (last column)
        k_table[i, -1] = slope
        delta2_table[i, -1] = delta2
        delta3_table[i, -1] = delta3
        delta4_table[i, -1] = delta4
        lnb1_table[i, -1] = ocean_data['Ann_Intercept_msk']
        lnb2_table[i, -1] = ocean_data['Ann_Intercept_ret']

        # Generate pie chart for current ocean
        plot_pie_chart(ocean, delta1, delta2, delta3, delta4, slope)
    
    # Process Global data (last row in tables)
    ocean_data = df[df['Ocean'] == 'Global'].iloc[0]
    for j, season in enumerate(season_dict.keys()):
        keys = [f'{season}_Slope{base_key}' for base_key in base_keys]
        delta1 = 1 - ocean_data[keys[3]]
        delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
        delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
        delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
        slope = 1 - delta1 - delta2 - delta3 - delta4
        
        k_table[-1, j] = slope
        delta2_table[-1, j] = delta2
        delta3_table[-1, j] = delta3
        delta4_table[-1, j] = delta4
        lnb1_table[-1, j] = ocean_data[f'{season}_Intercept_msk']
        lnb2_table[-1, j] = ocean_data[f'{season}_Intercept_ret']

    # Calculate Global annual values
    keys = [f'Ann_Slope{base_key}' for base_key in base_keys]
    delta1 = 1 - ocean_data[keys[3]]
    delta2 = ocean_data[keys[3]] - ocean_data[keys[2]]
    delta3 = ocean_data[keys[2]] - ocean_data[keys[1]]
    delta4 = ocean_data[keys[1]] - ocean_data[keys[0]]
    slope = 1 - delta1 - delta2 - delta3 - delta4

    k_table[-1, -1] = slope
    delta2_table[-1, -1] = delta2
    delta3_table[-1, -1] = delta3
    delta4_table[-1, -1] = delta4
    lnb1_table[-1, -1] = ocean_data['Ann_Intercept_msk']
    lnb2_table[-1, -1] = ocean_data['Ann_Intercept_ret']
    
    # Generate Global pie chart (with legend)
    plot_pie_chart('Global', delta1, delta2, delta3, delta4, slope)

    # Prepare table indices and column names
    complete_oceans = oceans + ['Global']
    complete_seasons = list(season_dict.keys()) + ['Annual']

    # Save slope1 table (k values)
    k1_df = pd.DataFrame(data=k_table, index=complete_oceans, columns=complete_seasons)
    k1_df.reset_index(inplace=True)
    k1_df.rename(columns={'index': 'Ocean'}, inplace=True)
    k1_df.to_csv('/home/chenyiqi/251028_albedo_cot/processed_data/table_k1.csv', 
                     index=False)

    # Save slope2 table
    k2_df = pd.DataFrame(data=k_table + delta4_table, index=complete_oceans, columns=complete_seasons)
    k2_df.reset_index(inplace=True)
    k2_df.rename(columns={'index': 'Ocean'}, inplace=True)
    k2_df.to_csv('/home/chenyiqi/251028_albedo_cot/processed_data/table_k2.csv', 
                     index=False)

    # Save slope3 table
    k3_df = pd.DataFrame(data=k_table + delta4_table + delta3_table, index=complete_oceans, columns=complete_seasons)
    k3_df.reset_index(inplace=True)
    k3_df.rename(columns={'index': 'Ocean'}, inplace=True)
    k3_df.to_csv('/home/chenyiqi/251028_albedo_cot/processed_data/table_k3.csv', 
                     index=False)

    # Save intercept table
    lnb1_df = pd.DataFrame(data=lnb1_table, index=complete_oceans, columns=complete_seasons)
    lnb1_df.reset_index(inplace=True)
    lnb1_df.rename(columns={'index': 'Ocean'}, inplace=True)
    lnb1_df.to_csv('/home/chenyiqi/251028_albedo_cot/processed_data/table_lnb1.csv', 
                        index=False)

    # Save intercept table
    lnb2_df = pd.DataFrame(data=lnb2_table, index=complete_oceans, columns=complete_seasons)
    lnb2_df.reset_index(inplace=True)
    lnb2_df.rename(columns={'index': 'Ocean'}, inplace=True)
    lnb2_df.to_csv('/home/chenyiqi/251028_albedo_cot/processed_data/table_lnb2.csv', 
                        index=False)