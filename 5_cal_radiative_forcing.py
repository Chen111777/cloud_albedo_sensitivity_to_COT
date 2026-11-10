"""
calc_all_forcing.py
Combine 5_ERF_CO2.py + 6_cal_radiative_forcing.py
1) compute ERF_CO2 per‑ocean basin
2) compute the uncorrected IRF_aci using CERES-derived cloud albedo Ac,
   and the Corrected IRF_aci using the fitted COT-albedo relation.
   The uncorrected Ac is derived from CERES TOA all-sky/clear-sky shortwave
   fluxes and the CERES-EBAF total cloud fraction.
The Corrected relation uses the k and ln b fitted in
fig3_sbd_seasonal_3ratios_8oceans.py; those coefficients are reported for every
ocean AND every season, so the correction is applied season by season
(each L3 season file is weighted with its own fitted k and ln b).
The aerosol-cloud interaction coefficient dlnNd/dlnAOD is the mean of
Hasekamp (2019) and Bellouin et al. (2013); H19 gives one value per ocean
(used for all four seasons), B13 is seasonal, so LNND_O_LNAOD is averaged
per ocean-season pair.
Output two csv files under processed_data/:
    ERF_CO2_ocean_regions.csv
    fig4_revision_impacts.csv
"""
import glob
import os
import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

# ========= config for ERF_CO2 part (from 5_ERF_CO2.py) =========
DATA_DIR = "/home/chenyiqi/251028_albedo_cot/cmip6"
VARIABLES = ("rsd", "rld", "rsu", "rlu")
CORRECTION_FACTOR = 2.16

OCEANS = {
    "NPO": [[-170, 20, -100, 60], [-180, 20, -170, 60], [105, 20, 180, 60]],
    "NAO": [[-100, 55, 45, 60], [-100, 40, 27, 55], [-100, 30, 45, 40], [-100, 20, 30, 30]],
    "TPO": [[-170, 16, -100, 20], [-170, 13, -89, 16], [-170, 9, -84, 13], [-170, -20, -70, 9], [100, 0, 180, 20], [130, -20, 180, 0], [-180, -20, -170, 20]],
    "TAO": [[-100, 16, -15, 20], [-84, 9, -13, 16], [-60, -20, 15, 9]],
    "TIO": [[30, 0, 100, 30], [30, -20, 130, 0]],
    "SPO": [[-170, -60, -70, -20], [130, -60, 180, -20], [-180, -60, -170, -20]],
    "SAO": [[-70, -60, 20, -20]],
    "SIO": [[20, -60, 130, -20]],
}

# ========= functions for ERF_CO2 =========
def load_variable(variable, experiment):
    pattern = os.path.join(DATA_DIR, f"{variable}_CFmon_MIROC6_piClim-{experiment}_r*.nc")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files found for {variable}, {experiment}: {pattern}")
    monthly_data = []
    lat = lon = None
    file_path = file_paths[-1]
    if os.path.getsize(file_path) == 0:
        raise OSError(f"Empty NetCDF file: {file_path}")
    with nc.Dataset(file_path, "r") as dataset:
        monthly_data.append(np.ma.filled(dataset.variables[variable][:, -1, :, :], np.nan).astype(np.float32))
        if lat is None:
            lat = dataset.variables["lat"][:].astype(np.float32)
            lon = dataset.variables["lon"][:].astype(np.float32)
    return np.concatenate(monthly_data, axis=0), lat, lon

def annual_mean(data):
    if data.shape[0] % 12 != 0:
        raise ValueError(f"Expected complete years, got {data.shape[0]} months")
    years = data.shape[0] // 12
    monthly_data = data.reshape(years, 12, data.shape[1], data.shape[2])
    return np.nanmean(monthly_data, axis=(0, 1))

def area_weighted_mean(data, lat, region_mask=None):
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis]
    valid = np.isfinite(data)
    if region_mask is not None:
        valid &= region_mask
    return np.sum(np.where(valid, data * weights, 0.0)) / np.sum(np.where(valid, weights, 0.0))

def region_mask(lat, lon, regions):
    lon_grid, lat_grid = np.meshgrid(((lon + 180) % 360) - 180, lat)
    mask = np.zeros(lat_grid.shape, dtype=bool)
    for west, south, east, north in regions:
        mask |= (lon_grid >= west) & (lon_grid <= east) & (lat_grid >= south) & (lat_grid <= north)
    return mask

def run_erf_co2(base_dir: Path):
    OUTPUT_CSV = base_dir / "processed_data" / "ERF_CO2_ocean_regions.csv"
    flux_difference = {}
    lat = lon = None
    for variable in VARIABLES:
        four_x_co2, lat, lon = load_variable(variable, "4xCO2")
        control, control_lat, control_lon = load_variable(variable, "control")
        if not (np.array_equal(lat, control_lat) and np.array_equal(lon, control_lon)):
            raise ValueError(f"Coordinate mismatch for {variable}")
        flux_difference[variable] = annual_mean(four_x_co2) - annual_mean(control)
    annual_erf = (
        flux_difference["rsd"] + flux_difference["rld"]
        - flux_difference["rsu"] - flux_difference["rlu"]
    )
    global_mean = area_weighted_mean(annual_erf, lat)
    ERF_CO2 = {}
    for name, regions in OCEANS.items():
        mask = region_mask(lat, lon, regions)
        region_mean = area_weighted_mean(annual_erf, lat, mask)
        ERF_CO2[name] = region_mean / global_mean * CORRECTION_FACTOR
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(
        [{"ocean": name, "ERF_CO2": value} for name, value in ERF_CO2.items()]
    )
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[ERF‑CO2] Saved: {OUTPUT_CSV}")
    return ERF_CO2

# ========= IRF‑ACI calculation part (from 6_cal_radiative_forcing.py) =========
# import external utilities (keep as original)
from utils_fitting import cot_k_b_to_albedo, oceans, season_dict
from utils_solar import calc_grid_cell_area, calc_monthly_swdown

# CERES-EBAF input. Cloud fraction used to derive Ac is taken from
# CERES-EBAF itself, following fig3_sbd_seasonal_k_8oceans.py.
CERES_FILE = (
    "/home/chenyiqi/251028_albedo_cot/CERES_EBAF_TOA_2020/"
    "CERES_EBAF-TOA_Ed4.2.1_Subset_202001-202012.nc"
)

def load_ebaf_ac_grid():
    """Calculate monthly CERES-EBAF Ac using CERES-EBAF cloud fraction.

    Ac = [SW_all - (1-CF)*SW_clear] / (CF*solar),
    where CF = cldarea_total_daynight_mon / 100.
    """
    with xr.open_dataset(CERES_FILE) as ds:
        cf = ds["cldarea_total_daynight_mon"] / 100.0
        ac = (
            ds["toa_sw_all_mon"] - (1.0 - cf) * ds["toa_sw_clr_c_mon"]
        ) / cf / ds["solar_mon"]
        ac = ac.where((cf > 0) & np.isfinite(ac) & (ac >= 0) & (ac <= 1))
        result = ac.load()
    return result


def match_ebaf_ac_to_l3(data, ebaf_ac_grid):
    """Match monthly CERES-EBAF Ac to L3 samples by nearest lat/lon and month.

    The Ac grid is calculated using CERES-EBAF cloud fraction, not MODIS CF.
    """
    lower = {str(c).lower(): c for c in data.columns}
    lat_col = next((lower[x] for x in ("lat", "latitude") if x in lower), None)
    lon_col = next((lower[x] for x in ("lon", "longitude") if x in lower), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"L3 data requires lat/lon columns; got {list(data.columns)}")
    month_col = next(
        (lower[x] for x in ("month", "time", "date", "year_month", "yyyymm") if x in lower),
        None,
    )
    if month_col is None:
        raise ValueError(f"L3 data requires a month/time column; got {list(data.columns)}")

    lat = pd.to_numeric(data[lat_col], errors="coerce").to_numpy(float)
    lon = ((pd.to_numeric(data[lon_col], errors="coerce").to_numpy(float) + 180.0) % 360.0) - 180.0
    grid_lat = np.asarray(ebaf_ac_grid.lat, dtype=float)
    grid_lon = ((np.asarray(ebaf_ac_grid.lon, dtype=float) + 180.0) % 360.0) - 180.0
    lat_idx = np.abs(lat[:, None] - grid_lat[None, :]).argmin(axis=1)
    lon_idx = np.abs(lon[:, None] - grid_lon[None, :]).argmin(axis=1)
    ac_values = np.asarray(ebaf_ac_grid.values)

    values = data[month_col]
    if np.issubdtype(values.dtype, np.number):
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
        months = np.where(numeric >= 10000, numeric % 100, numeric)
    else:
        months = pd.to_datetime(values, errors="coerce").dt.month.to_numpy(dtype=float)

    time_months = pd.to_datetime(ebaf_ac_grid.time.values).month
    output = np.full(len(data), np.nan, dtype=float)
    for i, month in enumerate(months):
        if not np.isfinite(month) or month < 1 or month > 12:
            continue
        month_idx = np.flatnonzero(time_months == int(month))
        if month_idx.size:
            output[i] = ac_values[month_idx[0], lat_idx[i], lon_idx[i]]
    return output


# dlnNd/dlnAOD from Jia (2021) -- kept for reference, not used.
# LNND_O_LNAOD = {
#     'NAO': 0.44,
#     'TAO': 0.24,
#     'SAO': 0.34,
#     'TIO': 0.15,
#     'SIO': 0.30,
#     'NPO': 0.57,
#     'TPO': 0.34,
#     'SPO': 0.20,
# }

# dlnNd/dlnAOD from Hasekamp (2019). H19 reports one value per ocean basin,
# so the same coefficient is used for all four seasons.
HASEKAMP2019_LNND_O_LNAOD = {
    'NAO': 0.41,
    'TAO': 0.40,
    'SAO': 0.52,
    'TIO': 0.53,
    'SIO': 0.52,
    'NPO': 0.75,
    'TPO': 0.39,
    'SPO': 0.34,
}

# dlnNd/dlnAOD from Bellouin et al. (2013), tabulated per ocean basin and
# per season (columns DJF, MAM, JJA, SON), read from processed data.
BELLOUIN2013_LNND_O_LNAOD_CSV = Path(
    "/home/chenyiqi/251028_albedo_cot/processed_data/Bellouin2013.csv"
)


def load_bellouin2013_lnnd_o_lnaod(csv_path=BELLOUIN2013_LNND_O_LNAOD_CSV):
    """Load the Bellouin et al. (2013) dlnNd/dlnAOD table.

    Returns a dict keyed by (ocean, season), e.g. {('NPO', 'MAM'): 0.266}.
    """
    df = pd.read_csv(csv_path).set_index("Ocean")
    missing = [season for season in season_dict if season not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing the season columns {missing}; "
            f"available columns: {list(df.columns)}"
        )
    table = {}
    for ocean in df.index:
        for season in season_dict:
            value = df.loc[ocean, season]
            if pd.isna(value):
                raise ValueError(
                    f"Missing dlnNd/dlnAOD for {ocean}-{season} in {csv_path}"
                )
            table[(ocean, season)] = float(value)
    return table


BELLOUIN2013_LNND_O_LNAOD = load_bellouin2013_lnnd_o_lnaod()

# Coefficient used by the calculation: the mean of Hasekamp (2019) and
# Bellouin et al. (2013). H19 has no seasonal dependence, so its value is
# repeated for every season and the average is taken per (ocean, season).
LNND_O_LNAOD = {
    (ocean, season): 0.5 * (
        HASEKAMP2019_LNND_O_LNAOD[ocean]
        + BELLOUIN2013_LNND_O_LNAOD[(ocean, season)]
    )
    for ocean in oceans
    for season in season_dict
}


def load_cmip(base_dir:Path):
    CMIP_FILE = base_dir / "processed_data" / "cmip6_AodDiff_nat1850to1860_aer2010to2020_HadGEM3.csv"
    df = pd.read_csv(CMIP_FILE)
    df.rename(
        columns={
            "longitude": "lon",
            "latitude": "lat"
        },
        inplace=True
    )
    return df

def load_coefficients(base_dir:Path):
    """k and ln b of the COT-albedo relation fitted by
    fig3_sbd_seasonal_3ratios_8oceans.py. The csv holds one row for every
    ocean-season pair, so the index is (Ocean, Season)."""
    K_CSV = base_dir / "processed_data" / "fig3_sbd_seasonal_3ratios_8oceans.csv"
    return pd.read_csv(K_CSV).set_index(["Ocean", "Season"])


def compute_irf(ocean, coeff, base_dir:Path, ebaf_ac_grid):
    L3_DIR = base_dir / "L3_product"
    cmip = load_cmip(base_dir)
    total = {"ceres": 0.0, "corrected": 0.0}
    area_total = 0.0

    for season in season_dict:
        file = L3_DIR / f"{ocean}_{season}.csv"
        if not file.exists():
            continue

        k_corrected = float(coeff.loc[(ocean, season), "k_fit"])
        b_corrected = float(np.exp(coeff.loc[(ocean, season), "lnb_fit"]))
        data = pd.read_csv(file)
        data["time"] = pd.to_datetime(data["time"])
        data["month"] = data["time"].dt.month
        data["lat_round"] = np.round(data["lat"] * 2) / 2
        data["lon_round"] = np.round(data["lon"] * 2) / 2
        data = data.merge(
            cmip,
            left_on=["month", "lat_round", "lon_round"],
            right_on=["month", "lat", "lon"],
            how="left",
            suffixes=("", "_cmip"),
        )
        data.drop(
            columns=["lat_round", "lon_round", "lat_cmip", "lon_cmip"],
            errors="ignore",
            inplace=True,
        )

        cot = data["cot"].to_numpy(float)
        # Keep the original L3 liquid-cloud fraction for forcing area weighting.
        # This is distinct from the CERES-EBAF total CF used to derive Ac.
        if "cf_liq_ceres" not in data.columns:
            raise KeyError(
                f"{file} is missing 'cf_liq_ceres'; available columns: {list(data.columns)}"
            )
        cf_liq_weight = data["cf_liq_ceres"].to_numpy(float)
        aod_diff = data["log_aod_diff"].to_numpy(float)
        lat_values = data["lat"].to_numpy(float)
        lon_values = data["lon"].to_numpy(float)
        months = data["month"].to_numpy(int)

        # Use CERES-EBAF Ac, calculated with CERES-EBAF cloud fraction.
        ac_ceres = match_ebaf_ac_to_l3(data, ebaf_ac_grid)

        swdown = np.array([
            calc_monthly_swdown(lat, month=m)
            for lat, m in zip(lat_values, months)
        ])
        area = np.array([calc_grid_cell_area(lat) for lat in lat_values])
        # dlnNd/dlnAOD for this ocean-season (mean of H19 and B13).
        lnnd = LNND_O_LNAOD[(ocean, season)] * aod_diff
        base = - swdown * lnnd / 3.0

        # Before revision: use CERES-derived Ac directly, with k=1.
        value_ceres = base * ac_ceres * (1.0 - ac_ceres) * cf_liq_weight
        valid_ceres = (
            np.isfinite(value_ceres) & np.isfinite(area) & (area > 0)
        )
        total["ceres"] += np.sum(value_ceres[valid_ceres] * area[valid_ceres])

        # After revision: use the fitted k and b for this ocean-season.
        ac_corrected = cot_k_b_to_albedo(cot, k_corrected, b_corrected)
        value_corrected = (
            base * k_corrected * ac_corrected * (1.0 - ac_corrected) * cf_liq_weight
        )
        valid_corrected = (
            np.isfinite(value_corrected) & np.isfinite(area) & (area > 0)
        )
        total["corrected"] += np.sum(
            value_corrected[valid_corrected] * area[valid_corrected]
        )

        valid_area = np.isfinite(area) & (area > 0)
        area_total += np.sum(area[valid_area])

    if area_total == 0:
        return np.nan, np.nan, area_total
    return total["ceres"] / area_total, total["corrected"] / area_total, area_total


def run_irf_calc(base_dir:Path):
    OUTPUT_CSV = base_dir / "processed_data" / "fig4_revision_impacts.csv"
    coeff = load_coefficients(base_dir)
    ebaf_ac_grid = load_ebaf_ac_grid()
    records = []

    print(
        "[IRF-ACI] lnnd_o_lnaod: mean of Hasekamp (2019) and "
        f"Bellouin et al. (2013), per ocean-season "
        f"({len(LNND_O_LNAOD)} values)",
        flush=True,
    )

    for ocean in oceans:
        if not any((ocean, season) in LNND_O_LNAOD for season in season_dict):
            continue
        print("Processing", ocean, flush=True)
        ceres_irf, corrected, area = compute_irf(ocean, coeff, base_dir, ebaf_ac_grid)
        records.append({
            "Ocean": ocean,
            "CERES": ceres_irf,
            "Corrected": corrected,
            "Area": area,
        })

    result = pd.DataFrame(records)
    weights = result["Area"]
    total_area = np.sum(weights)
    global_ceres = np.sum(result["CERES"] * weights) / total_area
    global_corrected = np.sum(result["Corrected"] * weights) / total_area
    print("\nGlobal ocean area-weighted IRF:")
    print(f"CERES-EBAF-derived Ac (before revision): {global_ceres:.4f} W m-2")
    print(f"Corrected (after revision)        : {global_corrected:.4f} W m-2")
    print(
        f"Difference (corrected - CERES-EBAF Ac): "
        f"{global_corrected - global_ceres:.4f} W m-2"
    )

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[IRF-ACI] Saved: {OUTPUT_CSV}")
    print(result)
    return result


# ========= main entry: run both calculations =========
def main():
    BASE_DIR = Path(__file__).resolve().parent
    # step 1: compute per‑ocean ERF_CO2
    _ = run_erf_co2(BASE_DIR)
    # step 2: compute the per-ocean CERES-EBAF-Ac / Corrected IRF_aci
    _ = run_irf_calc(BASE_DIR)
    print("\n==== All forcing calculation finished. ====")

if __name__ == "__main__":
    main()
