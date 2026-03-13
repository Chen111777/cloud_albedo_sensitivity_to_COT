from pathlib import Path
import math
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

# Suppress cartopy/pyproj warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cartopy')
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj.network')

# Path configurations
OUTPUT_DIR = Path('/home/chenyiqi/251028_albedo_cot/processed_data/')
INTERMEDIATE_DATA_PATH = OUTPUT_DIR / "IRF_distribution_data.pkl"

# Constants for area calculation
R_EARTH = 6371000  # Earth radius in meters
M2_TO_KM2 = 1e6    # Conversion factor from m² to km²

# Plot styling parameters
size_paras = {
    'xtick': 11,
    'ylabel': 14,
    'title': 14,
    'legend': 11,
}

# to unify colorbar limits
COLORBAR_VMIN = -0.25
COLORBAR_VMAX = 2.75
NORM = mcolors.Normalize(vmin=COLORBAR_VMIN, vmax=COLORBAR_VMAX)

DIVERGING_CMAP = plt.cm.rainbow

def calc_grid_cell_area(lat, lon_res=1.0, lat_res=1.0):
    """Calculate grid cell area (km2) based on latitude and resolution."""
    lat1, lat2 = math.radians(lat - lat_res/2), math.radians(lat + lat_res/2)
    dlon = math.radians(lon_res)
    area_m2 = dlon * (math.sin(lat2) - math.sin(lat1)) * (R_EARTH ** 2)
    return area_m2 / M2_TO_KM2

def plot_single_subplot(ax, lon_grid, lat_grid, data_grid, area_grid, title, cmap):
    """Plot spatial data on cartopy GeoAxes with proper normalization."""
    data_min = np.nanmin(data_grid)
    data_max = np.nanmax(data_grid)
    print(f"Subplot: {title} | Min: {data_min:.2f} | Max: {data_max:.2f}")

    if isinstance(ax, GeoAxes):
        # Add geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='k', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='#f5f5f5', alpha=0.6)
        ax.add_feature(cfeature.OCEAN, color='#eaf6fa', alpha=0.3)
        ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())

        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                          linestyle='--', alpha=0.6, color='gray')
        gl.top_labels = gl.right_labels = False

    pc = ax.pcolormesh(lon_grid, lat_grid, data_grid, 
                       cmap=cmap,
                       norm=NORM, 
                       transform=ccrs.PlateCarree(),
                       edgecolors='none', linewidth=0)
    
    # Set subplot title
    ax.set_title(title, fontsize=size_paras['title'], pad=5, loc='left')
    valid_mask = ~np.isnan(data_grid)
    
    # Calculate and display area-weighted mean value
    if np.any(valid_mask):
        mean_val = np.average(data_grid[valid_mask], weights=area_grid[valid_mask])
        ax.text(1.0, 1.02, f'{mean_val:.2f} W m$^{{-2}}$', 
                transform=ax.transAxes, ha='right', va='bottom', fontsize=size_paras['title'])
    
    return pc

def plot_maps_and_barrows(combined_df, ocean_area_order, fig_save_path=None):
    # Calculate grid cell area if not exists
    if 'grid_area_km2' not in combined_df.columns:
        combined_df['grid_area_km2'] = combined_df['lat'].apply(calc_grid_cell_area)
    
    # Aggregate data by lat/lon grid
    agg_cols = [
        'IRF_ret_orig', 'IRF_msk_orig', 
        'IRF_ret_corr1', 'IRF_msk_corr',
        'IRF_ret_corr2',
        'grid_area_km2'
    ]
    agg_cols = [col for col in agg_cols if col in combined_df.columns]
    df_grid = combined_df.groupby(['lat', 'lon']).agg({col: 'mean' for col in agg_cols}).reset_index()

    # Create grid mesh
    lats, lons = np.sort(df_grid['lat'].unique()), np.sort(df_grid['lon'].unique())
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    def make_grid(col):
        """Convert dataframe to 2D grid array"""
        pivot_grid = df_grid.pivot(index='lat', columns='lon', values=col)
        return pivot_grid.reindex(index=lats, columns=lons).values

    # Create figure with 2x3 layout (5 subplots total, 1 empty)
    fig = plt.figure(figsize=(16, 5))
    proj = ccrs.PlateCarree()
    
    # Create subplot axes (2 rows x 3 columns)
    ax1 = fig.add_subplot(2, 3, 1, projection=proj)  # Row1-Col1: IRF_ret_orig
    ax2 = fig.add_subplot(2, 3, 2, projection=proj)  # Row2-Col2: IRF_ret_corr1
    ax3 = fig.add_subplot(2, 3, 3, projection=proj)  # Row1-Col3: IRF_ret_corr2
    ax4 = fig.add_subplot(2, 3, 4, projection=proj)  # Row2-Col1: IRF_msk_orig
    ax5 = fig.add_subplot(2, 3, 5, projection=proj)  # Row2-Col2: IRF_msk_corr

    # Generate data grids
    area_grid          = make_grid('grid_area_km2')
    irf_msk_orig_grid  = make_grid('IRF_msk_orig')
    irf_ret_orig_grid  = make_grid('IRF_ret_orig')
    IRF_msk_corr_grid = make_grid('IRF_msk_corr')
    irf_ret_corr1_grid = make_grid('IRF_ret_corr1')
    irf_ret_corr2_grid = make_grid('IRF_ret_corr2')

    # Plot subplots
    pc1 = plot_single_subplot(ax1, lon_grid, lat_grid, irf_ret_orig_grid,  area_grid, r'$\mathbf{(a)}$ Original', DIVERGING_CMAP)
    pc2 = plot_single_subplot(ax2, lon_grid, lat_grid, irf_ret_corr1_grid, area_grid, r'$\mathbf{(b)}$ Corrected for 10:30',   DIVERGING_CMAP)
    pc3 = plot_single_subplot(ax3, lon_grid, lat_grid, irf_ret_corr2_grid, area_grid, r'$\mathbf{(c)}$ Corrected for daytime', DIVERGING_CMAP)
    pc4 = plot_single_subplot(ax4, lon_grid, lat_grid, irf_msk_orig_grid,  area_grid, r'$\mathbf{(d)}$ Original',       DIVERGING_CMAP)
    pc5 = plot_single_subplot(ax5, lon_grid, lat_grid, IRF_msk_corr_grid, area_grid, r'$\mathbf{(e)}$ Corrected for 10:30',     DIVERGING_CMAP)

    # Add method labels to the left of each row
    # Add "Cloud-Retrieval Method" to Row 1 (top row)
    fig.text(0.5, 0.94, 'Retrieval-Domain', fontsize=12, fontweight='bold', 
             ha='center', va='center')
    # Add "Cloud-Mask Method" to Row 2 (bottom row)
    fig.text(0.5, 0.445, 'Mask-Domain', fontsize=12, fontweight='bold', 
             ha='center', va='center')

    # Add shared horizontal colorbar
    cbar_ax_irf = fig.add_axes([0.695, 0.25, 0.26, 0.035])
    cbar_irf = fig.colorbar(
        pc1, 
        cax=cbar_ax_irf, 
        orientation='horizontal',
        extend='both',
        norm=NORM, 
        label='IRF (W m$^{-2}$)'
    )
    cbar_irf.set_label('IRF (W m$^{-2}$)', fontsize=size_paras['ylabel'])
    cbar_irf.ax.tick_params(labelsize=11)

    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.05,
        top=0.9,
        wspace=0.25,
        hspace=0.4
    )

    # Save figure if path provided
    if fig_save_path:
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_save_path}")
    plt.close(fig)

if __name__ == "__main__":
    # Check if intermediate data file exists
    if not INTERMEDIATE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Intermediate data file not found at {INTERMEDIATE_DATA_PATH}. "
            "Please run data_processing_and_plot1.py first."
        )
    
    # Load preprocessed data
    with open(INTERMEDIATE_DATA_PATH, 'rb') as f:
        plot2_data = pickle.load(f)
    
    combined_df = plot2_data["combined_df"]
    ocean_area = plot2_data["ocean_area"]

    # Create output directory and save figure
    figs_dir = Path("/data/chenyiqi/251028_albedo_cot/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    plot_maps_and_barrows(
        combined_df, 
        list(ocean_area.keys()), 
        fig_save_path=str(figs_dir / "IRF_distributions_5panels.png")
    )
