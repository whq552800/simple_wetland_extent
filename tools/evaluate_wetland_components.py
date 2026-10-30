"""Compare wetland components with WAD2M numerically on a common monthly grid.

WAD2M is a reference for natural wetland area, not an independent observation
of either CWatM component. Component scores are diagnostics only.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

try:
    from tools.evaluate_wetland_wad2m import (
        _month_lookup, align_axis, area_metrics, wetland_union,
    )
except ModuleNotFoundError:
    from evaluate_wetland_wad2m import (
        _month_lookup, align_axis, area_metrics, wetland_union,
    )


NAMES = ("wetVegSaturatedFrac", "wetSurfaceInundFrac", "wetlandInundVegFrac")


def evaluate(wad2m_path, model_dir, cell_area_path, start_year, end_year):
    model_dir = Path(model_dir)
    with ExitStack() as stack:
        files = [stack.enter_context(Dataset(model_dir / f"{name}_monthavg.nc"))
                 for name in NAMES]
        wad = stack.enter_context(Dataset(wad2m_path))
        area_file = stack.enter_context(Dataset(cell_area_path))
        lat = files[0].variables["lat"][:]
        lon = files[0].variables["lon"][:]
        wad_rows = align_axis(lat, wad.variables["lat"][:])
        wad_cols = align_axis(lon, wad.variables["lon"][:])
        area_rows = align_axis(lat, area_file.variables["lat"][:])
        area_cols = align_axis(lon, area_file.variables["lon"][:])
        area = area_file.variables["cellarea30min"][:][np.ix_(area_rows, area_cols)]
        model_months = _month_lookup(files[0])
        wad_months = _month_lookup(wad)
        if any((_month_lookup(file) != model_months
                or not np.array_equal(file.variables["lat"][:], lat)
                or not np.array_equal(file.variables["lon"][:], lon))
               for file in files[1:]):
            raise ValueError("Model component grids or months differ")
        if getattr(wad.variables["Fw"], "units", None) not in ("-", "1"):
            raise ValueError("Unexpected WAD2M Fw units")
        sums = {name: {metric: 0.0 for metric in (
            "area", "bias", "mae", "squared_error", "simulated_mean",
        )} for name in ("vegetated", "surface", "overlap", "union")}
        n_months = 0
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                key = (year, month)
                if key not in model_months or key not in wad_months:
                    raise ValueError(f"Missing month {key}")
                components = [file.variables[f"{name}_monthavg"][model_months[key]]
                              for file, name in zip(files, NAMES)]
                common_valid = np.all([
                    np.isfinite(np.ma.filled(np.ma.asarray(part), np.nan))
                    for part in components
                ], axis=0)
                reference = np.ma.masked_where(
                    ~common_valid,
                    wad.variables["Fw"][wad_months[key]][np.ix_(wad_rows, wad_cols)],
                )
                values = dict(zip(("vegetated", "surface", "overlap"), components))
                values["union"] = wetland_union(*components)
                for name, simulated in values.items():
                    result = area_metrics(simulated, reference, area)
                    weight = result["area_weight_sum_m2"]
                    item = sums[name]
                    item["area"] += weight
                    item["bias"] += weight * result["weighted_bias"]
                    item["mae"] += weight * result["weighted_mae"]
                    item["squared_error"] += weight * result["weighted_rmse"] ** 2
                    item["simulated_mean"] += weight * result["weighted_simulated_mean"]
                n_months += 1
    summary = {}
    for name, item in sums.items():
        weight = item["area"]
        summary[name] = {
            "weighted_bias": item["bias"] / weight,
            "weighted_mae": item["mae"] / weight,
            "weighted_rmse": float(np.sqrt(item["squared_error"] / weight)),
            "weighted_simulated_mean": item["simulated_mean"] / weight,
        }
    return {
        "start_year": start_year, "end_year": end_year, "n_months": n_months,
        "components": summary,
        "note": "WAD2M does not separately observe the model components.",
        "model_dir": str(model_dir.resolve()),
        "reference": str(Path(wad2m_path).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wad2m", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cell-area", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start year must not exceed end year")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args.wad2m, args.model_dir, args.cell_area,
                      args.start_year, args.end_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
