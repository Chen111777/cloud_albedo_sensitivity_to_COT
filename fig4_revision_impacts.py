# -*- coding: utf-8 -*-
"""Combined figure of intrinsic forcing (INF) before and after revision.
Susceptibility corrections are based on area-weighted k*R values, with
uncertainty propagated from the Fig. 3 bootstrap distributions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from shapely.ops import unary_union
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.cm as cm
from utils_fitting import format_panel_tag
# Panel-tag style shared with the other figures of this study:
# 'science' -> A, B, ...; 'nature' -> a, b, ...
PANEL_STYLE = "science"
# ============================================================
# Paths
# ============================================================
RATIO_CSV = Path(
    "/home/chenyiqi/251028_albedo_cot/project0904/processed_data/"
    "fig3_sbd_seasonal_3ratios_8oceans.csv"
)
MC_RATIO_CSV = Path(__file__).resolve().parent / "processed_data" / "fig3_sbd_seasonal_3ratios_8oceans_mc.csv"
OUT_DIR = Path("./figs")
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_FIG = OUT_DIR / "fig4_combined_0.png"
BAR_DIR = OUT_DIR / "ocean_bars_0"
# Shared, untruncated normal conversion factor: INF = F_INTRINSIC * IRFaci.
# The supplied 0.3 is an absolute central-90% half-width, not a relative error
# or a standard deviation. Values outside [0, 1] are allowed by this model.
F_INTRINSIC = 1.0
F_INTRINSIC_90_HALF_WIDTH = 0.59
Z90 = 1.6448536269514722
F_INTRINSIC_STD = F_INTRINSIC_90_HALF_WIDTH / Z90
# The five k*R correction distributions are loaded from the Fig. 3 bootstrap
# output. The additional SW_TOA / SW_sur,clr ratio (1.411) is applied to M17/T19.
SW_TOA_OVER_SFC_CLR = 1.411
def load_correction_draws(mc_csv_path, n_draws=1_000_000, seed=20261114):
    """Build area-weighted draws of k*R, multiplying region-season values first.

    For each Monte Carlo realization, one bootstrap draw is selected jointly
    for the five methods within each ocean-season. Seasons are averaged within
    each ocean, and the resulting ocean means are area-weighted.
    """
    columns = [f"kratio_{name}" for name in
               ("547", "0", "ebaf", "ssf", "038")]
    mc = pd.read_csv(mc_csv_path)
    required = {"Ocean", "Season", *columns}
    missing = required.difference(mc.columns)
    if missing:
        raise ValueError(f"Fig. 3 Monte Carlo CSV is missing columns: {sorted(missing)}")
    areas = pd.read_csv(RESULT_CSV, usecols=["Ocean", "Area"]).set_index("Ocean")["Area"]
    oceans_order = [o for o in areas.index if o in set(mc["Ocean"])]
    weights = areas.reindex(oceans_order).to_numpy(float)
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("Every ocean region must have a finite, positive area.")
    rng = np.random.default_rng(seed)
    ocean_draws = {o: {c: [] for c in columns} for o in oceans_order}
    for ocean in oceans_order:
        for season in ("MAM", "JJA", "SON", "DJF"):
            subset = mc[(mc["Ocean"] == ocean) & (mc["Season"] == season)]
            if subset.empty:
                continue
            values = subset[columns].to_numpy(float)
            valid = np.isfinite(values).all(axis=1)
            values = values[valid]
            if not len(values):
                continue
            selected = rng.integers(0, len(values), size=n_draws)
            sampled = values[selected, :]
            for j, col in enumerate(columns):
                ocean_draws[ocean][col].append(sampled[:, j])
    result = {}
    for col in columns:
        regional = []
        regional_weights = []
        for ocean in oceans_order:
            season_arrays = ocean_draws[ocean][col]
            if season_arrays:
                regional.append(np.mean(np.vstack(season_arrays), axis=0))
                regional_weights.append(float(areas.loc[ocean]))
        if not regional:
            raise ValueError(f"No valid k*R bootstrap draws for {col}")
        regional = np.vstack(regional)
        regional_weights = np.asarray(regional_weights, dtype=float)
        result[col.replace("kratio_", "ratio_")] = np.average(
            regional, axis=0, weights=regional_weights
        )
    print("\nArea-weighted correction coefficients (first k*R within each ocean-season):")
    for name in ("547", "0", "ebaf", "ssf", "038"):
        key = f"ratio_{name}"
        values = result[key]
        print(
            f"  {key:12s}: mean={np.mean(values):.5f}, "
            f"SD={np.std(values, ddof=1):.5f}, "
            f"90% interval=[{np.quantile(values, .05):.5f}, "
            f"{np.quantile(values, .95):.5f}]"
        )
    return result
# ============================================================
# Fig. 4a map inputs
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
RESULT_CSV = BASE_DIR / "processed_data" / "fig4_revision_impacts.csv"
ERF_CO2_FILE = BASE_DIR / "processed_data" / "ERF_CO2_ocean_regions.csv"
IRF_SCALE = F_INTRINSIC
IRF_COLUMNS = ["CERES", "Corrected"]
OCEANS_MAP = {
    "NPO": [[-170,20,-100,60], [-180,20,-170,60], [105,20,180,60]],
    "NAO": [[-100,55,45,60], [-100,40,27,55], [-100,30,45,40], [-100,20,30,30]],
    "TPO": [[-170,16,-100,20], [-170,13,-89,16], [-170,9,-84,13],
            [-170,-20,-70,9], [100,0,180,20], [130,-20,180,0], [-180,-20,-170,20]],
    "TAO": [[-100,16,-15,20], [-84,9,-13,16], [-60,-20,15,9]],
    "TIO": [[30,0,100,30], [30,-20,130,0]],
    "SPO": [[-170,-60,-70,-20], [130,-60,180,-20], [-180,-60,-170,-20]],
    "SAO": [[-70,-60,20,-20]],
    "SIO": [[20,-60,130,-20]],
}
def load_map_data():
    result = pd.read_csv(RESULT_CSV)
    missing = set(IRF_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"IRF result CSV is missing columns: {sorted(missing)}")
    result[IRF_COLUMNS] = result[IRF_COLUMNS] * IRF_SCALE
    erf_df = pd.read_csv(ERF_CO2_FILE)
    required = {"ocean", "ERF_CO2"}
    if not required.issubset(erf_df.columns):
        raise ValueError(f"ERF_CO2 file must contain {required}")
    erf_co2 = dict(zip(erf_df["ocean"], erf_df["ERF_CO2"]))
    return result, erf_co2
def region_geometry(regions):
    return unary_union([box(west, south, east, north)
                        for west, south, east, north in regions])
def draw_map(ax, result, erf_co2):
    values = {
        row["Ocean"]: (row["Corrected"] - row["CERES"]) / erf_co2[row["Ocean"]]
        for _, row in result.iterrows()
    }
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.88", edgecolor="black",
                   linewidth=0.3, zorder=4)
    ax.coastlines(linewidth=0.4, zorder=5)
    fade = 0.5
    raw_rgba = cm.viridis(np.linspace(0, 1, 256))
    light_viridis = ListedColormap([
        (r + (1-r)*fade, g + (1-g)*fade, b + (1-b)*fade, 1.0)
        for r, g, b, _ in raw_rgba
    ])
    finite_values = np.asarray([v for v in values.values() if np.isfinite(v)], dtype=float)
    if finite_values.size == 0:
        raise ValueError("No finite ocean INF differences available for the map.")
    vmin = min(0.0, float(finite_values.min()))
    vmax = max(0.0, float(finite_values.max()))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 1e-6)
        vmin -= delta
        vmax += delta
    norm = Normalize(vmin=vmin, vmax=vmax)
    for ocean, regions in OCEANS_MAP.items():
        geom = region_geometry(regions)
        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor=light_viridis(norm(values[ocean])),
                          edgecolor="black", linewidth=0.4, zorder=3)
    gl = ax.gridlines(draw_labels=True, color="none")
    gl.top_labels = False
    gl.right_labels = False
    sm = cm.ScalarMappable(norm=norm, cmap=light_viridis)
    return sm, values
def print_map_summary(result, erf_co2):
    areas = result.set_index("Ocean")["Area"]
    total_area = areas.sum()
    ceres_awm = sum(row.CERES * areas[row.Ocean] for _, row in result.iterrows()) / total_area
    corrected_awm = sum(row.Corrected * areas[row.Ocean] for _, row in result.iterrows()) / total_area
    delta_awm = sum((row.Corrected-row.CERES) * areas[row.Ocean]
                    for _, row in result.iterrows()) / total_area
    ratio_awm = sum((row.Corrected-row.CERES)/erf_co2[row.Ocean] * areas[row.Ocean]
                    for _, row in result.iterrows()) / total_area
    print(f"Global area-weighted mean of INF_original (CERES Ac): {ceres_awm:.4f} W m-2")
    print(f"Global area-weighted mean of INF_corrected: {corrected_awm:.4f} W m-2")
    print(f"Global area-weighted mean of delta_INF (corrected - original): {delta_awm:.4f} W m-2")
    print(f"Global area-weighted mean of delta_INF/ERF_CO2: {ratio_awm:.4f}")
    # Ratio of the global area-weighted means as well as the per-ocean ratios.
    global_ratio = corrected_awm / ceres_awm
    print(
        "Global area-weighted mean of INF_corrected / INF_original: "
        f"{global_ratio:.4f} ({(global_ratio - 1.0) * 100:+.2f}%)"
    )
    print("Per-ocean INF_corrected / INF_original:")
    for _, row in result.iterrows():
        regional_ratio = row["Corrected"] / row["CERES"]
        print(
            f"  {row['Ocean']:4s}: INF_original={row['CERES']:+.4f} W m-2, "
            f"INF_corrected={row['Corrected']:+.4f} W m-2, "
            f"ratio={regional_ratio:.4f} ({(regional_ratio - 1.0) * 100:+.2f}%)"
        )
    print("Map values:")
    for ocean, value in {
        row["Ocean"]: (row["Corrected"]-row["CERES"])/erf_co2[row["Ocean"]]
        for _, row in result.iterrows()
    }.items():
        print(ocean, value)
# ============================================================
# Literature values transcribed from the supplied figure
# ============================================================
# Columns are explicit: study, adopted center, lower/upper 90% endpoints,
# processing class, and susceptibility-reference column. M20 uses the interval
# midpoint; H19 retains its reported median. Missing T19 bounds are imputed below.
rows = pd.DataFrame([
    ("D20",    -0.69, -0.99, -0.44, "irf_to_intrinsic_only", None),
    ("G17",    -0.40, -1.00, -0.20, "irf_to_intrinsic", "ratio_ebaf"),
    ("H19",    -1.14, -1.72, -0.84, "irf_to_intrinsic", "ratio_ebaf"),
    ("T19",    -0.53, np.nan, np.nan, "irf_to_intrinsic", "ratio_ebaf"),
    ("M17",    -1.00, -1.40, -0.60, "irf_to_intrinsic", "ratio_547"),
    ("B13",    -0.60, -1.20,  0.00, "irf_to_intrinsic_only", None),
    ("R18",    -0.80, -1.50, -0.10, "irf_to_intrinsic_only", None),
    ("M20",    -0.90, -1.20, -0.60, "irf_to_intrinsic_only", None),
    ("Chen14", -0.50, -1.00,  0.00, "intrinsic_fixed", None),
    ("Chri16", -0.40, -0.70, -0.10, "intrinsic_fixed", None),
    ("Chri17", -0.30, -0.70,  0.10, "intrinsic_fixed", None),
], columns=["study", "raw_mean", "raw_low", "raw_high", "kind", "correction_mode"])
# ============================================================
# Construct source intervals on the central INF-conversion scale. Conversion
# uncertainty is subsequently propagated jointly across converted studies.
# All reported AR6 ranges are treated as central 90% intervals (5th to 95th
# percentiles), as requested. Endpoints are retained, not treated as 1-sigma.
# ============================================================
correction_draws = load_correction_draws(MC_RATIO_CSV)
records = []
study_correction_draws = {}
for row in rows.to_dict("records"):
    raw_mean = row["raw_mean"]
    raw_low, raw_high = row["raw_low"], row["raw_high"]
    if row["kind"] in {"irf_to_intrinsic", "irf_to_intrinsic_only"}:
        inf_factor = F_INTRINSIC
    elif row["kind"] == "intrinsic_fixed":
        inf_factor = 1.0
    else:
        raise ValueError(f"Unknown row kind: {row['kind']}")
    correction_factor = 1.0
    correction_samples = np.ones(1_000_000, dtype=float)
    if row["kind"] == "irf_to_intrinsic":
        # Use the albedo-ratio prescription assigned to each corrected study:
        # G17/H19/T19 -> ratio_ebaf; M17 -> ratio_547.
        # D20 is converted to INF but, like B13, receives no ratio correction.
        # The additional SW_TOA / SW_sur,clr factor (1.411) applies to M17 and T19.
        study_ratio = row["correction_mode"]
        draw_key = study_ratio.replace("ratio_", "ratio_")
        if draw_key not in correction_draws:
            raise ValueError(
                f"Missing area-weighted k*R draws {draw_key!r} for {row['study']}"
            )
        correction_samples = correction_draws[draw_key].copy()
        if row["study"] in {"M17", "T19"}:
            correction_samples *= SW_TOA_OVER_SFC_CLR
        correction_factor = float(np.mean(correction_samples))
    study_correction_draws[row["study"]] = correction_samples
    before_mean = raw_mean * inf_factor
    after_mean = before_mean * correction_factor
    before_low, before_high = raw_low * inf_factor, raw_high * inf_factor
    after_low = before_low * correction_factor
    after_high = before_high * correction_factor
    records.append({
        "study": row["study"],
        "before_mean": before_mean,
        "before_low": before_low,
        "before_high": before_high,
        "after_mean": after_mean,
        "after_low": after_low,
        "after_high": after_high,
        "raw_mean": raw_mean,
        "raw_low": raw_low,
        "raw_high": raw_high,
        "inf_factor": inf_factor,
        "correction_factor": correction_factor,
        "correction_factor_std": float(np.std(correction_samples, ddof=1)),
        "correction_factor_p05": float(np.quantile(correction_samples, 0.05)),
        "correction_factor_p95": float(np.quantile(correction_samples, 0.95)),
        "kind": row["kind"],
        # Mode actually applied (M17 for every converted study) and the mode
        # tabulated for that study as a reference.
        "correction_mode": row["correction_mode"],
        "reported_correction_mode": row["correction_mode"],
        "correction_scale": (SW_TOA_OVER_SFC_CLR
                             if row["study"] in {"M17", "T19"} else 1.0),
    })
df = pd.DataFrame(records)
# Impute absolute lower/upper error distances on the common baseline-INF scale,
# using only studies with reported bounds. Scale the same imputed errors for the
# revised estimate, preserving the study's paired before/after correction.
known = np.isfinite(df["before_low"]) & np.isfinite(df["before_high"])
if not known.any():
    raise ValueError("At least one study must report an uncertainty interval.")
lower_error = df.loc[known, "before_mean"] - df.loc[known, "before_low"]
upper_error = df.loc[known, "before_high"] - df.loc[known, "before_mean"]
if (lower_error < 0).any() or (upper_error < 0).any():
    raise ValueError("A study center lies outside its reported interval.")
df["uncertainty_imputed"] = ~known
df.loc[~known, "before_low"] = df.loc[~known, "before_mean"] - lower_error.mean()
df.loc[~known, "before_high"] = df.loc[~known, "before_mean"] + upper_error.mean()
df["after_low"] = df["before_low"] * df["correction_factor"]
df["after_high"] = df["before_high"] * df["correction_factor"]
# Sort from top to bottom by the final two digits in each study name,
# ascending; preserve original order when suffixes are equal.
df["_name_suffix"] = df["study"].str.extract(r"(\d{2})$").astype(int)
df["_original_order"] = np.arange(len(df))
df = (
    df.sort_values(["_name_suffix", "_original_order"], kind="stable")
      .drop(columns=["_name_suffix", "_original_order"])
      .reset_index(drop=True)
)
def propagate_f_and_mixture(data, n_draws=1_000_000, seed=20261029):
    """Propagate INF conversion, source-study and k*R correction uncertainty.

    Converted studies share one INF conversion-factor draw per realization.
    Each corrected study uses the Monte Carlo distribution of its applicable
    area-weighted k*R coefficient. Intrinsic-fixed studies are not converted.
    """
    data = data.copy()
    base_mean = data["before_mean"].to_numpy(float)
    base_low = data["before_low"].to_numpy(float)
    base_high = data["before_high"].to_numpy(float)
    converted = data["kind"].ne("intrinsic_fixed").to_numpy()
    if not (np.isfinite(base_mean).all() and np.isfinite(base_low).all()
            and np.isfinite(base_high).all()
            and (base_low <= base_mean).all() and (base_mean <= base_high).all()):
        raise ValueError("Invalid forcing values, interval bounds or corrections.")

    source_sigma = (base_high - base_low) / (2.0 * Z90)
    if not np.isfinite(source_sigma).all() or (source_sigma < 0).any():
        raise ValueError("Invalid source normal-equivalent standard deviations.")

    data["before_low_fixed_f"] = base_low
    data["before_high_fixed_f"] = base_high
    data["f_uncertainty_propagated"] = converted
    rng = np.random.default_rng(seed)
    f_samples = rng.normal(F_INTRINSIC, F_INTRINSIC_STD, size=n_draws)
    study_idx = rng.integers(0, len(base_mean), size=n_draws)
    mixture_before = np.empty(n_draws)
    mixture_after = np.empty(n_draws)

    for j in range(len(base_mean)):
        if converted[j]:
            source_samples = rng.normal(
                base_mean[j] / F_INTRINSIC,
                source_sigma[j] / F_INTRINSIC,
                size=n_draws
            )
            before_samples = source_samples * f_samples
        else:
            before_samples = rng.normal(base_mean[j], source_sigma[j], size=n_draws)

        correction_samples = study_correction_draws[data.iloc[j]["study"]]
        if len(correction_samples) != n_draws:
            correction_indices = rng.integers(0, len(correction_samples), size=n_draws)
            correction_samples = correction_samples[correction_indices]
        after_samples = before_samples * correction_samples
        data.loc[data.index[j], "before_mean"] = float(np.mean(before_samples))
        data.loc[data.index[j], ["before_low", "before_high"]] = np.quantile(
            before_samples, [0.05, 0.95]
        )
        data.loc[data.index[j], "after_mean"] = float(np.mean(after_samples))
        data.loc[data.index[j], ["after_low", "after_high"]] = np.quantile(
            after_samples, [0.05, 0.95]
        )

        selected = study_idx == j
        mixture_before[selected] = before_samples[selected]
        mixture_after[selected] = after_samples[selected]

    # The synthesis mean is the mean of the equal-weight mixture distribution,
    # not a separately averaged set of point estimates.
    before = (float(np.mean(mixture_before)),
              *map(float, np.quantile(mixture_before, [0.05, 0.95])))
    after = (float(np.mean(mixture_after)),
             *map(float, np.quantile(mixture_after, [0.05, 0.95])))
    return data, before, after

df, before_summary, after_summary = propagate_f_and_mixture(df)
before_mean_all, before_low_all, before_high_all = before_summary
after_mean_all, after_low_all, after_high_all = after_summary
display_columns = ["study", "before_mean", "before_low", "before_high",
                   "after_mean", "after_low", "after_high", "uncertainty_imputed"]
print(df[display_columns].to_string(index=False))
print(f"Mean before: {before_mean_all:.3f} [{before_low_all:.3f}, {before_high_all:.3f}] W m-2")
print(f"Mean after : {after_mean_all:.3f} [{after_low_all:.3f}, {after_high_all:.3f}] W m-2")
print(f"F_INTRINSIC: normal mean={F_INTRINSIC:.3f}, SD={F_INTRINSIC_STD:.6f}, "
      f"central-90% half-width={F_INTRINSIC_90_HALF_WIDTH:.3f}.")
print("Per-study and pooled distributions include source-study uncertainty, "
      "the shared normally distributed INF conversion factor, and the "
      "Monte Carlo uncertainty in area-weighted k*R corrections "
      "(1,000,000 draws; seed=20261029).")
# ============================================================
# Combined figure: Fig. 4a (map) above Fig. 4b (swapped dumbbell plot)
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10
})
def plot_single_ocean_bar(ocean, ceres_ac, corrected):
    """Save an individual horizontal bar chart for one ocean region."""
    fig, ax = plt.subplots(figsize=(3.2, 2.0), facecolor="none")
    ax.set_facecolor("none")
    labels = ["Corrected", "Original"]
    values = [corrected, ceres_ac]
    colors = ["orangered", "orange"]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.45,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    # Category labels sit on the right-hand axis; the left spine is not drawn.
    ax.set_yticklabels(labels, fontsize=9)
    ax.yaxis.tick_right()
    ax.set_xlabel(r"INF (W m$^{-2}$)", fontsize=10.5)
    ax.set_title(ocean, fontsize=13, fontweight="bold", loc="right")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-2.3, 0)
    # ax.set_xlim(-1.6, 0)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    fig.tight_layout()
    BAR_DIR.mkdir(exist_ok=True, parents=True)
    out = BAR_DIR / f"{ocean}_INF.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved ocean bar plot: {out}")
def interval_yerr(row, prefix):
    """Asymmetric distances for reported or imputed interval endpoints."""
    low, high = row[f"{prefix}_low"], row[f"{prefix}_high"]
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    center = row[f"{prefix}_mean"]
    if not low <= center <= high:
        raise ValueError(f"Invalid {prefix} interval for {row['study']}")
    return np.array([[center - low], [high - center]])
def main():
    result_map, erf_co2 = load_map_data()
    print_map_summary(result_map, erf_co2)
    # Keep the original per-ocean bar-chart outputs in addition to the
    # combined Fig. 4a/4b figure.
    for _, ocean_row in result_map.iterrows():
        plot_single_ocean_bar(
            ocean_row["Ocean"],
            ocean_row["CERES"],
            ocean_row["Corrected"],
        )
    before_color = "#4C78A8"
    after_color = "#E45756"
    # ---- panel (b) legend placement -------------------------------------
    LEGEND_BOX = [0.68, 0.10, 0.25, 0.34]
    # Anchor inside that box: "upper left" (current), "center", "lower right",
    # "upper center", ...
    LEGEND_LOC = "center left"
    # Optional fine offset (x, y) in axes fractions of LEGEND_BOX, e.g.
    # Right edges of the map and the external legend share the same figure x.
    fig = plt.figure(figsize=(12, 9), constrained_layout=False)
    map_ax = fig.add_axes([0.05, 0.57, 0.88, 0.36], projection=ccrs.PlateCarree())
    dumbbell_ax = fig.add_axes([0.09, 0.11, 0.57, 0.34])
    legend_ax = fig.add_axes(LEGEND_BOX)
    legend_ax.axis("off")
    # Global area-weighted means of intrinsic forcing (INF), using the same
    # ocean-area weights as the map summary. Values in result_map have already
    # been scaled by F_INTRINSIC in load_map_data().
    ocean_areas = result_map["Area"].to_numpy(dtype=float)
    total_ocean_area = np.nansum(ocean_areas)
    inf_original_global = np.nansum(
        result_map["CERES"].to_numpy(dtype=float) * ocean_areas
    ) / total_ocean_area
    inf_corrected_global = np.nansum(
        result_map["Corrected"].to_numpy(dtype=float) * ocean_areas
    ) / total_ocean_area
    # Panel (a): map.
    sm, map_values = draw_map(map_ax, result_map, erf_co2)
    map_ax.text(-0.04, 1.00, format_panel_tag(0, PANEL_STYLE),
                transform=map_ax.transAxes, fontsize=15, va="bottom", ha="left")
    # Display the two global means above panel (a).
    fig.text(
        0.31, 0.945,
        rf"$\mathrm{{INF}}_{{\mathrm{{original}}}}$ global mean = "
        rf"{inf_original_global:.2f} W m$^{{-2}}$",
        ha="center", va="center", fontsize=11,
    )
    fig.text(
        0.70, 0.945,
        rf"$\mathrm{{INF}}_{{\mathrm{{corrected}}}}$ global mean = "
        rf"{inf_corrected_global:.2f} W m$^{{-2}}$",
        ha="center", va="center", fontsize=11,
    )
    cax = fig.add_axes([0.27, 0.525, 0.48, 0.014])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$(\mathrm{INF}_{\mathrm{corrected}}-\mathrm{INF}_{\mathrm{original}}$)/ERF$_{\mathrm{CO2}}$", labelpad=4)
    # Panel (b): swap the original axes: studies on x, intrinsic forcing on y.
    x = np.arange(len(df))
    for xi, (_, row) in zip(x, df.iterrows()):
        dumbbell_ax.errorbar(
            xi, row["before_mean"], yerr=interval_yerr(row, "before"),
            fmt="o", markersize=9.0, markerfacecolor=before_color,
            markeredgecolor="white", markeredgewidth=0.5,
            ecolor=before_color, elinewidth=2.0, capsize=2.5, zorder=4,
        )
        dumbbell_ax.errorbar(
            xi, row["after_mean"], yerr=interval_yerr(row, "after"),
            fmt="o", markersize=5.5, markerfacecolor=after_color,
            markeredgecolor="white", markeredgewidth=0.5,
            ecolor=after_color, elinewidth=1.0, capsize=2.5, zorder=5,
        )
    # 5th-95th mixture percentiles, including the normal conversion uncertainty.
    dumbbell_ax.axhspan(before_low_all, before_high_all,
                       color=before_color, alpha=0.14, zorder=0)
    dumbbell_ax.axhspan(after_low_all, after_high_all,
                       color=after_color, alpha=0.14, zorder=0)
    dumbbell_ax.axhline(before_mean_all, color=before_color, lw=2.0, zorder=2)
    dumbbell_ax.axhline(after_mean_all, color=after_color, lw=2.0, zorder=2)
    dumbbell_ax.axhline(0, color="black", lw=0.9, ls="--", zorder=1)
    dumbbell_ax.set_xticks(x)
    dumbbell_ax.set_xticklabels(df["study"], rotation=45, ha="right", fontsize=9)
    dumbbell_ax.set_ylabel(r"INF (W m$^{-2}$)")
    dumbbell_ax.grid(axis="y", alpha=0.22)
    dumbbell_ax.spines["top"].set_visible(False)
    dumbbell_ax.spines["right"].set_visible(False)
    dumbbell_ax.text(-0.07, 0.97, format_panel_tag(1, PANEL_STYLE),
                     transform=dumbbell_ax.transAxes,
                     fontsize=15, va="bottom", ha="left")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=before_color, markeredgecolor="white",
               markersize=9, label=r"Original, individual studies"),
        Line2D([0], [0], color=before_color, lw=2,
               label=f"Original mean: ${before_mean_all:.2f}$ W m$^{{-2}}$"),
        Patch(facecolor=before_color, alpha=0.14, edgecolor="none",
              label=f"Original 90% interval:\n[{before_low_all:.2f}, {before_high_all:.2f}] W m$^{{-2}}$"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=after_color, markeredgecolor="white",
               markersize=5.5, label=r"Corrected, individual studies"),
        Line2D([0], [0], color=after_color, lw=2,
               label=f"Corrected mean: ${after_mean_all:.2f}$ W m$^{{-2}}$"),
        Patch(facecolor=after_color, alpha=0.14, edgecolor="none",
              label=f"Corrected 90% interval:\n[{after_low_all:.2f}, {after_high_all:.2f}] W m$^{{-2}}$"),
    ]
    legend_ax.legend(handles=handles,
        loc=LEGEND_LOC, frameon=True, framealpha=0.92, edgecolor="0.85",
        fontsize=10, borderaxespad=0.0, labelspacing=1.15, handletextpad=0.7)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(BASE_DIR / "processed_data" / "fig4_literature_conditional_intervals.csv", index=False)
    pd.DataFrame([
        ("original", before_mean_all, before_low_all, before_high_all),
        ("corrected", after_mean_all, after_low_all, after_high_all),
    ], columns=["estimate", "mean", "mixture_p05", "mixture_p95"]).to_csv(
        BASE_DIR / "processed_data" / "fig4_mixture_percentiles.csv", index=False,
    )
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined figure to: {OUT_FIG}")
if __name__ == "__main__":
    main()
