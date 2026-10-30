"""Compare monthly CWatM wetland area with the ELM-ECA wetlandfrac model output.

This is an inter-model benchmark, not validation against observations. The
two CWatM area diagnostics overlap and must not be added without de-duplication.
"""

import argparse
import calendar
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np


def wetland_union(saturated_veg, surface, veg_fraction, flood_fraction):
    """Count surface water over saturated vegetation only once."""
    potential = np.asarray(veg_fraction) + np.asarray(flood_fraction)
    vegetated_surface = np.divide(
        surface * veg_fraction, potential,
        out=np.zeros_like(np.asarray(surface), dtype=np.float64),
        where=potential > 0,
    )
    return saturated_veg + surface - vegetated_surface


def month_indices(dates, year, month):
    """Require a full Gregorian month, omitting Feb 29 for ELM's 365-day year."""
    selected = [i for i, day in enumerate(dates)
                if day.year == year and day.month == month
                and not (month == 2 and day.day == 29)]
    expected_days = 28 if month == 2 else calendar.monthrange(year, month)[1]
    if sorted(dates[i].day for i in selected) != list(range(1, expected_days + 1)):
        raise ValueError(f"CWatM needs a complete {year}-{month:02d} month")
    return selected


def coordinate_indices(global_coordinates, model_coordinates):
    indices = []
    for value in model_coordinates:
        matches = np.flatnonzero(np.isclose(global_coordinates, value, rtol=0, atol=1e-5))
        if matches.size != 1:
            raise ValueError(f"Coordinate {value} has no unique match in global grid")
        indices.append(int(matches[0]))
    return np.asarray(indices)


def read_model_daily(directory, name):
    with Dataset(directory / f"{name}_daily.nc") as data:
        return np.ma.asarray(data.variables[name][:]).filled(np.nan)


def weighted_mean(values, weights, mask):
    return float(np.average(values[mask], weights=weights[mask]))


def compare(reference_path, daily_dir, fraction_path, output_dir, year):
    reference_path = Path(reference_path)
    daily_dir = Path(daily_dir)
    fraction_path = Path(fraction_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    with Dataset(daily_dir / "wetVegSaturatedFrac_daily.nc") as model:
        model_lat = np.asarray(model.variables["lat"][:])
        model_lon = np.asarray(model.variables["lon"][:])
        time = model.variables["time"]
        model_dates = num2date(time[:], time.units, time.calendar)
    with Dataset(daily_dir / "cellArea_totalend.nc") as model:
        area = np.ma.asarray(model.variables["cellArea_totalend"][:])
        cell_area = area.filled(np.nan)
    valid_area = np.isfinite(cell_area) & (cell_area > 0)

    with Dataset(reference_path) as reference, Dataset(fraction_path) as fractions:
        if reference.variables["wetlandfrac"].units != "%":
            raise ValueError("ELM wetlandfrac must have percent units")
        if reference.variables["time"].calendar != "365_day":
            raise ValueError("ELM reference must use a 365_day calendar")
        ref_lat = np.asarray(reference.variables["lat"][:])
        ref_lon = np.asarray(reference.variables["lon"][:])
        if (not np.allclose(ref_lat, fractions.variables["lat"][:])
                or not np.allclose(ref_lon, fractions.variables["lon"][:])):
            raise ValueError("Reference and classified-fraction grids differ")
        row = coordinate_indices(ref_lat, model_lat)
        col = coordinate_indices(ref_lon, model_lon)
        veg = np.ma.asarray(fractions.variables["wetlandVegFrac"][:]).filled(np.nan)[np.ix_(row, col)]
        flood = np.ma.asarray(fractions.variables["wetlandFloodFrac"][:]).filled(np.nan)[np.ix_(row, col)]
        unknown = np.ma.asarray(fractions.variables["wetlandUnmappedFrac"][:]).filled(np.nan)[np.ix_(row, col)]
        potential = veg + flood
        if np.any(~np.isfinite(potential[valid_area])):
            raise ValueError("Classified wetland fraction is missing within the catchment")
        reference_dates = num2date(
            reference.variables["time"][:], reference.variables["time"].units,
            reference.variables["time"].calendar,
        )

        saturated = read_model_daily(daily_dir, "wetVegSaturatedFrac")
        surface = read_model_daily(daily_dir, "wetSurfaceInundFrac")
        if saturated.shape != surface.shape or saturated.shape[1:] != cell_area.shape:
            raise ValueError("Model wetland output shapes differ")
        union = wetland_union(saturated, surface, veg, flood)
        if np.any(union[:, valid_area] > potential[valid_area] + 1e-5):
            raise ValueError("Simulated wetland area exceeds its classified capacity")

        monthly_rows = []
        cell_rows = []
        for month in range(1, 13):
            days = month_indices(model_dates, year, month)
            ref_index = [i for i, date in enumerate(reference_dates)
                         if date.year == year and date.month == month]
            if len(ref_index) != 1:
                raise ValueError(f"ELM reference needs exactly one {year}-{month:02d} record")
            ref = np.ma.asarray(
                reference.variables["wetlandfrac"][ref_index[0], :, :]
            ).filled(np.nan)[np.ix_(row, col)] / 100.0
            veg_month = np.mean(saturated[days], axis=0)
            surface_month = np.mean(surface[days], axis=0)
            union_month = np.mean(union[days], axis=0)
            valid = (valid_area & np.isfinite(ref) & np.isfinite(union_month)
                     & np.isfinite(unknown))
            if not np.any(valid):
                raise ValueError(f"No matching valid cells for {year}-{month:02d}")
            difference = union_month - ref
            line = {
                "month": f"{year}-{month:02d}",
                "n_cells": int(np.sum(valid)),
                "elm_mean_pct": weighted_mean(ref, cell_area, valid) * 100,
                "classified_potential_mean_pct": weighted_mean(potential, cell_area, valid) * 100,
                "unmapped_mean_pct": weighted_mean(unknown, cell_area, valid) * 100,
                "sim_vegetated_mean_pct": weighted_mean(veg_month, cell_area, valid) * 100,
                "sim_surface_mean_pct": weighted_mean(surface_month, cell_area, valid) * 100,
                "sim_union_mean_pct": weighted_mean(union_month, cell_area, valid) * 100,
                "bias_percentage_points": weighted_mean(difference, cell_area, valid) * 100,
                "mae_percentage_points": weighted_mean(np.abs(difference), cell_area, valid) * 100,
                "rmse_percentage_points": (
                    weighted_mean(difference ** 2, cell_area, valid) ** 0.5 * 100
                ),
                "elm_above_potential_cells": int(np.sum(valid & (ref > potential + 1e-6))),
                "minimum_error_from_capacity_percentage_points": weighted_mean(
                    np.maximum(ref - potential, 0), cell_area, valid,
                ) * 100,
                "unmapped_over_90pct_cells": int(np.sum(valid & (unknown > 0.9))),
            }
            monthly_rows.append(line)
            for y, x in np.argwhere(valid):
                cell_rows.append({
                    "month": line["month"], "lat": float(model_lat[y]),
                    "lon": float(model_lon[x]), "cell_area_m2": float(cell_area[y, x]),
                    "elm_pct": float(ref[y, x] * 100),
                    "classified_potential_pct": float(potential[y, x] * 100),
                    "unmapped_pct": float(unknown[y, x] * 100),
                    "sim_vegetated_pct": float(veg_month[y, x] * 100),
                    "sim_surface_pct": float(surface_month[y, x] * 100),
                    "sim_union_pct": float(union_month[y, x] * 100),
                })

    cell_weights = np.asarray([r["cell_area_m2"] for r in cell_rows])
    cell_reference = np.asarray([r["elm_pct"] for r in cell_rows])
    cell_simulated = np.asarray([r["sim_union_pct"] for r in cell_rows])
    cell_potential = np.asarray([r["classified_potential_pct"] for r in cell_rows])
    summary = {
        "year": year,
        "reference": str(reference_path.resolve()),
        "reference_type": "ELM-ECA model output; not an observation",
        "reference_calendar": "365_day",
        "model_leap_day_excluded": bool(calendar.isleap(year)),
        "model_daily_output": str(daily_dir.resolve()),
        "classified_fractions": str(fraction_path.resolve()),
        "n_months": len(monthly_rows),
        "n_catchment_cells": monthly_rows[0]["n_cells"],
        "elm_annual_monthly_mean_pct": float(np.mean([r["elm_mean_pct"] for r in monthly_rows])),
        "sim_union_annual_monthly_mean_pct": float(np.mean([r["sim_union_mean_pct"] for r in monthly_rows])),
        "classified_potential_mean_pct": monthly_rows[0]["classified_potential_mean_pct"],
        "unmapped_mean_pct": monthly_rows[0]["unmapped_mean_pct"],
        "elm_above_potential_cell_months": int(sum(r["elm_above_potential_cells"] for r in monthly_rows)),
        "compared_cell_months": int(sum(r["n_cells"] for r in monthly_rows)),
        "area_weighted_bias_percentage_points": float(np.average(
            cell_simulated - cell_reference, weights=cell_weights,
        )),
        "area_weighted_mae_percentage_points": float(np.average(
            np.abs(cell_simulated - cell_reference), weights=cell_weights,
        )),
        "area_weighted_rmse_percentage_points": float(np.sqrt(np.average(
            (cell_simulated - cell_reference) ** 2, weights=cell_weights,
        ))),
        "minimum_mae_from_capacity_percentage_points": float(np.average(
            np.maximum(cell_reference - cell_potential, 0), weights=cell_weights,
        )),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    for filename, rows in (("monthly.csv", monthly_rows), ("cells.csv", cell_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as result:
            writer = csv.DictWriter(result, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    months = [r["month"][-2:] for r in monthly_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(months, [r["elm_mean_pct"] for r in monthly_rows], marker="o", label="ELM-ECA total")
    ax.plot(months, [r["sim_union_mean_pct"] for r in monthly_rows], marker="o", label="CWatM union")
    ax.plot(months, [r["classified_potential_mean_pct"] for r in monthly_rows],
            linestyle="--", label="Classified capacity (before lake exclusion)")
    ax.set(xlabel=f"Month of {year}", ylabel="Catchment area-weighted fraction (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_comparison.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--daily-dir", required=True)
    parser.add_argument("--fraction-file", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--output-dir", required=True, help="new directory; never overwritten")
    arguments = parser.parse_args()
    summary = compare(
        arguments.reference, arguments.daily_dir, arguments.fraction_file,
        arguments.output_dir, arguments.year,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
