# -*- coding: utf-8 -*-
"""Global 60S-60N ratio of TOA solar insolation to clear-sky surface SW down.

Restricted to the calendar year given by ``YEAR`` (UTC). Inputs are CERES
SYN1deg Edition 4.1, 1-hourly, 1 deg x 1 deg subsets:
  * ``toa_solar_all_1h``      (TOA incoming solar insolation, all-sky) from the
    two ``/data/CERES/sun_hr`` subsets covering 2019-05-16 .. 2021-01-14;
  * ``ini_sfc_sw_down_clr_1h`` (clear-sky surface downward SW) from the 56129
    and 2020 ``/data/CERES/Flux_sur_hr`` subsets, 2019-12-07 .. 2021-01-27.

Only hours present in both products are used, so the ratio is built from
concurrent hourly fields. Each field is first reduced to an area-weighted
60S-60N hourly mean, with exact 1-degree cell areas
(``sin(lat2) - sin(lat1)``, replicated over longitude). The reported value is
the ratio of the two time sums, i.e. the flux-weighted ("ratio of means")
definition of the domain-mean ratio.

Note that ``toa_solar_all_1h`` is the *downward* TOA solar insolation, not the
reflected shortwave, so the ratio is the inverse of the clear-sky atmospheric
SW transmittance (about 1 / 0.71 for the period used here). Nighttime cells
are exactly zero in both fields, so they cancel and daylight screening is not
needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ============================================================
# Paths
# ============================================================
SUN_DIR = Path("/data/CERES/sun_hr")
SFC_DIR = Path("/data/CERES/Flux_sur_hr")
OUT_DIR = Path("./processed_data")

TOA_VAR = "toa_solar_all_1h"
SFC_VAR = "ini_sfc_sw_down_clr_1h"

YEAR = 2020               # only this UTC calendar year is used
OUT_CSV = OUT_DIR / f"global_sw_toa_over_sfc_clr_ratio_{YEAR}.csv"

TOA_FILES = [
    SUN_DIR / "36821CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20190516-20200315.nc0",
    SUN_DIR / "36822CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20200316-20210114.nc0",
]
# 56129 is requested explicitly and is also matched by the "*2020*" pattern,
# hence the set() to keep every file exactly once.
SFC_FILES = sorted(set(
    [SFC_DIR / "56129CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20191207-20200113.nc0"]
    + sorted(SFC_DIR.glob("*2020*.nc0"))
))

# ============================================================
# Settings
# ============================================================
LAT_LIMIT = 60.0          # domain: 60S-60N
TIME_CHUNK = 120          # hours read per block (keeps memory near 50 MB)
TIME_UNITS = "days since 2000-03-01"
EPOCH = pd.Timestamp("2000-03-01 00:00:00")


def hour_index(time_days):
    """Convert 'days since 2000-03-01' to the integer hour containing a step.

    The 1-hourly files store half-hour centres (e.g. 7015.0208 d = 00:30), so a
    plain round() lands exactly on the .5 boundary, where float32 noise makes
    neighbouring steps collapse onto the same hour. Snapping to the nearest
    half hour first and then flooring is stable.
    """
    half_hours = np.rint(np.asarray(time_days, dtype=np.float64) * 48.0)
    return np.floor(half_hours / 2.0).astype(np.int64)


def format_hours(hours):
    """Return the readable UTC timestamp of an hour index."""
    return (EPOCH + pd.to_timedelta(int(hours), unit="h")).strftime("%Y-%m-%d %H:%M")


def year_window(year):
    """Return the [start, stop) hour indices of one UTC calendar year."""
    start = int((pd.Timestamp(year=year, month=1, day=1) - EPOCH)
                / pd.Timedelta(hours=1))
    stop = int((pd.Timestamp(year=year + 1, month=1, day=1) - EPOCH)
               / pd.Timedelta(hours=1))
    return start, stop


def domain_setup(lat):
    """Return the 60S-60N row index and the exact cell-area weights."""
    spacing = np.diff(lat)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("Expected a regular 1-degree latitude grid.")
    dlat = float(np.median(spacing))
    index = np.abs(lat) <= LAT_LIMIT + 1e-6
    edges = np.deg2rad(lat[index] + np.array([-dlat / 2.0, dlat / 2.0])[:, None])
    weights = (np.sin(edges[1]) - np.sin(edges[0])).astype(np.float64)
    return index, weights


def hourly_domain_means(path, varname, lat_index, weights):
    """Return hour indices and area-weighted 60S-60N means for one file."""
    with Dataset(path) as nc:
        time_var = nc.variables["time"]
        if not str(time_var.units).startswith(TIME_UNITS):
            raise ValueError(f"Unexpected time units in {path.name}: {time_var.units}")
        nc.set_auto_mask(False)          # fill values are caught by the <0 test
        var = nc.variables[varname]
        n_time = time_var.shape[0]
        hours = np.empty(n_time, dtype=np.int64)
        means = np.full(n_time, np.nan, dtype=np.float64)

        for start in range(0, n_time, TIME_CHUNK):
            stop = min(start + TIME_CHUNK, n_time)
            slab = np.asarray(var[start:stop, :, :], dtype=np.float64)
            slab[slab < 0.0] = np.nan    # _FillValue = -999; fluxes are >= 0
            block = slab[:, lat_index, :]

            valid = np.isfinite(block)
            weighted = np.where(valid, block, 0.0) * weights[None, :, None]
            numerator = weighted.sum(axis=(1, 2))
            denominator = (valid * weights[None, :, None]).sum(axis=(1, 2))

            means[start:stop] = np.divide(
                numerator, denominator,
                out=np.full(stop - start, np.nan, dtype=np.float64),
                where=denominator > 0,
            )
            hours[start:stop] = hour_index(time_var[start:stop])

    return hours, means


def merge_series(series_list):
    """Concatenate hourly series, averaging hours that appear more than once."""
    hours = np.concatenate([item[0] for item in series_list])
    values = np.concatenate([item[1] for item in series_list])
    order = np.argsort(hours, kind="stable")
    hours, values = hours[order], values[order]

    unique, first, counts = np.unique(hours, return_index=True, return_counts=True)
    if np.all(counts == 1):
        return unique, values
    print(f"  note: merged {int(hours.size - unique.size)} duplicate hour indices")
    merged = np.array([
        np.nanmean(values[first[i]:first[i] + counts[i]])
        for i in range(unique.size)
    ])
    return unique, merged


def align_series(toa_hours, toa_values, sfc_hours, sfc_values):
    """Return concurrent, valid hours of both hourly series."""
    common, toa_pos, sfc_pos = np.intersect1d(
        toa_hours, sfc_hours, assume_unique=True, return_indices=True,
    )
    toa = toa_values[toa_pos]
    sfc = sfc_values[sfc_pos]
    keep = np.isfinite(toa) & np.isfinite(sfc)
    if not keep.any():
        raise ValueError("The TOA and surface subsets share no valid hours.")
    return common[keep], toa[keep], sfc[keep]


def ratio_summary(hours, toa, sfc):
    """Return one summary row for a set of concurrent hourly domain means."""
    return {
        "n_hours": int(hours.size),
        "toa_solar_all_mean_Wm2": float(toa.mean()),
        "sfc_sw_down_clr_mean_Wm2": float(sfc.mean()),
        "ratio_toa_over_sfc_clr": float(toa.sum() / sfc.sum()),
    }


def main():
    print(f"CERES SYN1deg-1H SW_TOA / SW_sur,clr, 60S-60N area-weighted, {YEAR} only")
    print(f"TOA field    : {TOA_VAR} ({len(TOA_FILES)} files)")
    print(f"Surface field: {SFC_VAR} ({len(SFC_FILES)} files)")

    with Dataset(TOA_FILES[0]) as nc:
        lat = np.asarray(nc.variables["lat"][:], dtype=np.float64)
    lat_index, weights = domain_setup(lat)
    print(f"Latitude rows kept: {int(lat_index.sum())} "
          f"({lat[lat_index][0]:.1f} to {lat[lat_index][-1]:.1f})")

    toa_series = []
    for path in TOA_FILES:
        hours, means = hourly_domain_means(path, TOA_VAR, lat_index, weights)
        print(f"  TOA {path.name}: {hours.size} h, "
              f"{format_hours(hours[0])} .. {format_hours(hours[-1])}, "
              f"mean={np.nanmean(means):.2f} W m-2")
        toa_series.append((hours, means))

    sfc_series = []
    for path in SFC_FILES:
        hours, means = hourly_domain_means(path, SFC_VAR, lat_index, weights)
        print(f"  SFC {path.name}: {hours.size} h, "
              f"{format_hours(hours[0])} .. {format_hours(hours[-1])}, "
              f"mean={np.nanmean(means):.2f} W m-2")
        sfc_series.append((hours, means))

    toa_hours, toa_means = merge_series(toa_series)
    sfc_hours, sfc_means = merge_series(sfc_series)
    hours, toa, sfc = align_series(toa_hours, toa_means, sfc_hours, sfc_means)

    year_start, year_stop = year_window(YEAR)
    in_year = (hours >= year_start) & (hours < year_stop)
    if not in_year.any():
        raise ValueError(f"No concurrent hours found in {YEAR}.")
    hours, toa, sfc = hours[in_year], toa[in_year], sfc[in_year]
    expected = year_stop - year_start
    print()
    print(f"Concurrent coverage: {format_hours(hours[0])} .. "
          f"{format_hours(hours[-1])} ({hours.size} of {expected} hours of {YEAR})")
    overall = ratio_summary(hours, toa, sfc)
    overall["period"] = "ALL"

    frame = pd.DataFrame({
        "time": EPOCH + pd.to_timedelta(hours, unit="h"),
        "toa": toa,
        "sfc": sfc,
    })
    rows = []
    for period, group in frame.groupby(frame["time"].dt.to_period("M"), sort=True):
        row = ratio_summary(
            group.index.to_numpy(), group["toa"].to_numpy(float),
            group["sfc"].to_numpy(float),
        )
        row["period"] = str(period)
        rows.append(row)
    rows.append(overall)

    table = pd.DataFrame(rows)[[
        "period", "n_hours", "toa_solar_all_mean_Wm2",
        "sfc_sw_down_clr_mean_Wm2", "ratio_toa_over_sfc_clr",
    ]]

    pd.set_option("display.width", 160)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    table.to_csv(OUT_CSV, index=False)
    print()
    print(f"60S-60N mean SW_TOA / SW_sur,clr = "
          f"{overall['ratio_toa_over_sfc_clr']:.4f} "
          f"(TOA {overall['toa_solar_all_mean_Wm2']:.2f} W m-2 / "
          f"surface clear-sky {overall['sfc_sw_down_clr_mean_Wm2']:.2f} W m-2)")
    print("Nighttime cells are exactly zero in both fields, so restricting the "
          "ratio to daylight changes nothing.")
    print(f"Saved table to: {OUT_CSV}")


if __name__ == "__main__":
    main()
