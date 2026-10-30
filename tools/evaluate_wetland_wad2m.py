"""Compare CWatM monthly wetland union against WAD2M v2 on matched cells.

WAD2M combines satellite inundation dynamics with static wetland inventories;
its NetCDF ``Fw`` must not be interpreted as pure surface-water inundation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


def align_axis(target, source, tolerance=1e-5):
    """Indices in *source* for target coordinates, independent of ordering."""
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if target.ndim != 1 or source.ndim != 1 or len(np.unique(source)) != len(source):
        raise ValueError("Coordinate axes must be one-dimensional and unique")
    indices = np.searchsorted(np.sort(source), target)
    ordered = np.argsort(source)
    result = np.empty(target.size, dtype=np.int64)
    for position, index in enumerate(indices):
        if index == source.size or abs(source[ordered[index]] - target[position]) > tolerance:
            raise ValueError(f"Missing coordinate {target[position]}")
        result[position] = ordered[index]
    return result


def wetland_union(vegetated_saturated, surface_inundated, inundated_vegetated):
    """Union of two possibly overlapping simulated wetland-area fractions."""
    veg = np.ma.asarray(vegetated_saturated, dtype=np.float64)
    surface = np.ma.asarray(surface_inundated, dtype=np.float64)
    overlap = np.ma.asarray(inundated_vegetated, dtype=np.float64)
    if veg.shape != surface.shape or veg.shape != overlap.shape:
        raise ValueError("Wetland area arrays have different shapes")
    if (np.ma.any(overlap < -1e-6) or np.ma.any(overlap > np.ma.minimum(veg, surface) + 1e-6)
            or np.ma.any(veg < -1e-6) or np.ma.any(surface < -1e-6)
            or np.ma.any(veg > 1 + 1e-6) or np.ma.any(surface > 1 + 1e-6)):
        raise ValueError("Invalid overlap or wetland fractions")
    union = veg + surface - overlap
    if np.ma.any(union > 1 + 1e-6):
        raise ValueError("Wetland union exceeds one")
    return union


def area_metrics(simulated, observed, cell_area):
    """Return area-weighted statistics over valid paired cells/times."""
    sim = np.ma.asarray(simulated, dtype=np.float64)
    obs = np.ma.asarray(observed, dtype=np.float64)
    if sim.shape != obs.shape:
        raise ValueError("Simulated and observed arrays have different shapes")
    area = np.broadcast_to(np.asarray(cell_area, dtype=np.float64), sim.shape)
    sim_data = np.asarray(np.ma.filled(sim, np.nan))
    obs_data = np.asarray(np.ma.filled(obs, np.nan))
    valid = (np.isfinite(sim_data) & np.isfinite(obs_data)
             & np.isfinite(area) & (area > 0))
    if not np.any(valid):
        raise ValueError("No comparable wetland cells")
    w = area[valid]
    error = sim_data[valid] - obs_data[valid]
    total = np.sum(w)
    return {
        "n_pairs": int(np.count_nonzero(valid)),
        "weighted_bias": float(np.sum(w * error) / total),
        "weighted_mae": float(np.sum(w * np.abs(error)) / total),
        "weighted_rmse": float(np.sqrt(np.sum(w * error ** 2) / total)),
        "weighted_simulated_mean": float(np.sum(w * sim_data[valid]) / total),
        "weighted_observed_mean": float(np.sum(w * obs_data[valid]) / total),
        "area_weight_sum_m2": float(total),
    }


def _month_lookup(dataset):
    time = dataset.variables["time"]
    dates = num2date(time[:], units=time.units,
                     calendar=getattr(time, "calendar", "standard"))
    result = {}
    for index, date in enumerate(dates):
        month = (date.year, date.month)
        if month in result:
            raise ValueError(f"Duplicate month {month}")
        result[month] = index
    return result


def evaluate(wad2m_path, model_dir, cell_area_path, start_year, end_year):
    model_dir = Path(model_dir)
    names = ("wetVegSaturatedFrac", "wetSurfaceInundFrac", "wetlandInundVegFrac")
    files = [Dataset(model_dir / f"{name}_monthavg.nc") for name in names]
    try:
        with Dataset(wad2m_path) as wad, Dataset(cell_area_path) as area_file:
            reference_var = wad.variables["Fw"]
            if getattr(reference_var, "units", "") not in ("-", "1"):
                raise ValueError("Unexpected WAD2M Fw units")
            model = files[0]
            model_lat = model.variables["lat"][:]
            model_lon = model.variables["lon"][:]
            ref_lat = align_axis(model_lat, wad.variables["lat"][:])
            ref_lon = align_axis(model_lon, wad.variables["lon"][:])
            area_lat = align_axis(model_lat, area_file.variables["lat"][:])
            area_lon = align_axis(model_lon, area_file.variables["lon"][:])
            area_var = area_file.variables.get("cellarea30min")
            if area_var is None:
                raise ValueError("cellarea30min is missing from cell area file")
            area = np.ma.asarray(area_var[:])[np.ix_(area_lat, area_lon)]
            area = np.asarray(np.ma.filled(area, np.nan), dtype=np.float64)
            model_months = _month_lookup(model)
            obs_months = _month_lookup(wad)
            for other in files[1:]:
                if (not np.array_equal(model_lat, other.variables["lat"][:])
                        or not np.array_equal(model_lon, other.variables["lon"][:])
                        or _month_lookup(other) != model_months):
                    raise ValueError("Model monthly wetland files are not aligned")
            rows = []
            sums = {key: 0.0 for key in ("error", "absolute", "squared", "model", "observed", "area")}
            pairs = 0
            for month in sorted(model_months.keys() & obs_months.keys()):
                if not start_year <= month[0] <= end_year:
                    continue
                timestep = model_months[month]
                veg, surface, overlap = (
                    file.variables[f"{name}_monthavg"][timestep, :, :]
                    for file, name in zip(files, names)
                )
                union = wetland_union(veg, surface, overlap)
                observation = reference_var[obs_months[month], :, :][np.ix_(ref_lat, ref_lon)]
                try:
                    metrics = area_metrics(union, observation, area)
                except ValueError as error:
                    if str(error) == "No comparable wetland cells":
                        continue
                    raise
                rows.append({"year": month[0], "month": month[1], **metrics})
                weight = metrics["area_weight_sum_m2"]
                pairs += metrics["n_pairs"]
                for key, metric in (("error", "weighted_bias"), ("absolute", "weighted_mae"),
                                    ("squared", "weighted_rmse"), ("model", "weighted_simulated_mean"),
                                    ("observed", "weighted_observed_mean")):
                    value = metrics[metric]
                    sums[key] += weight * (value ** 2 if key == "squared" else value)
                sums["area"] += weight
            if not rows:
                raise ValueError("No comparable months in requested years")
            denominator = sums["area"]
            return rows, {
                "start_year": start_year,
                "end_year": end_year,
                "n_months": len(rows),
                "n_pairs": pairs,
                "weighted_bias": sums["error"] / denominator,
                "weighted_mae": sums["absolute"] / denominator,
                "weighted_rmse": np.sqrt(sums["squared"] / denominator),
                "weighted_simulated_mean": sums["model"] / denominator,
                "weighted_observed_mean": sums["observed"] / denominator,
                "reference": str(Path(wad2m_path).resolve()),
                "model_dir": str(model_dir.resolve()),
                "cell_area": str(Path(cell_area_path).resolve()),
            }
    finally:
        for file in files:
            file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wad2m", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cell-area", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start year must be no later than end year")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows, summary = evaluate(args.wad2m, args.model_dir, args.cell_area,
                             args.start_year, args.end_year)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "monthly.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
