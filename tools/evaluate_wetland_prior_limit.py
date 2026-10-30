"""Quantify the WAD2M area error that a fixed wetland candidate cap cannot remove."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

try:
    from tools.evaluate_wetland_wad2m import _month_lookup, align_axis
except ModuleNotFoundError:
    from evaluate_wetland_wad2m import _month_lookup, align_axis


def write_map(path, lat, lon, counts, exceeding, squared_shortfall):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fraction = np.divide(exceeding, counts,
                         out=np.full(counts.shape, np.nan), where=counts > 0)
    rmse_floor = np.sqrt(np.divide(squared_shortfall, counts,
                                   out=np.full(counts.shape, np.nan), where=counts > 0))
    with Dataset(path, "w") as output:
        output.createDimension("lat", len(lat))
        output.createDimension("lon", len(lon))
        output.createVariable("lat", "f8", ("lat",))[:] = lat
        output.createVariable("lon", "f8", ("lon",))[:] = lon
        for name, values, units in (("valid_months", counts, "months"),
                                    ("fraction_reference_above_cap", fraction, "1"),
                                    ("rmse_lower_bound", rmse_floor, "1")):
            variable = output.createVariable(name, "f4", ("lat", "lon"),
                                             zlib=True, fill_value=np.float32(-9999))
            variable.units = units
            variable[:] = np.ma.masked_invalid(values.astype(np.float32))


def evaluate(wad2m_path, model_dir, cell_area_path, start_year, end_year,
             map_output=None):
    if start_year > end_year:
        raise ValueError("Start year must not exceed end year")
    model_dir = Path(model_dir)
    with ExitStack() as stack:
        model = stack.enter_context(Dataset(model_dir / "wetlandPotentialFrac_monthavg.nc"))
        wad = stack.enter_context(Dataset(wad2m_path))
        cell_area = stack.enter_context(Dataset(cell_area_path))
        if getattr(wad.variables["Fw"], "units", None) not in ("-", "1"):
            raise ValueError("Unexpected WAD2M Fw units")
        lat = model.variables["lat"][:]
        lon = model.variables["lon"][:]
        wad_rows = align_axis(lat, wad.variables["lat"][:])
        wad_cols = align_axis(lon, wad.variables["lon"][:])
        area_rows = align_axis(lat, cell_area.variables["lat"][:])
        area_cols = align_axis(lon, cell_area.variables["lon"][:])
        area = np.ma.filled(cell_area.variables["cellarea30min"][:][
            np.ix_(area_rows, area_cols)], np.nan)
        model_months = _month_lookup(model)
        wad_months = _month_lookup(wad)
        weight_sum = 0.0
        squared_shortfall = 0.0
        positive_shortfall = 0.0
        n_pairs = 0
        n_above_cap = 0
        n_months = 0
        if map_output is not None:
            counts = np.zeros(area.shape, dtype=np.int32)
            exceeding = np.zeros(area.shape, dtype=np.int32)
            squared_by_cell = np.zeros(area.shape, dtype=np.float64)
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                key = (year, month)
                if key not in model_months or key not in wad_months:
                    raise ValueError(f"Missing month {key}")
                observed = np.ma.filled(wad.variables["Fw"][wad_months[key]][
                    np.ix_(wad_rows, wad_cols)], np.nan)
                cap = np.ma.filled(model.variables["wetlandPotentialFrac_monthavg"][
                    model_months[key]], np.nan)
                valid = (np.isfinite(observed) & np.isfinite(cap)
                         & np.isfinite(area) & (area > 0))
                shortfall = np.maximum(observed[valid] - cap[valid], 0)
                weight = area[valid]
                n_pairs += int(np.count_nonzero(valid))
                n_above_cap += int(np.count_nonzero(shortfall > 1e-6))
                weight_sum += float(np.sum(weight))
                positive_shortfall += float(np.sum(weight * shortfall))
                squared_shortfall += float(np.sum(weight * shortfall ** 2))
                if map_output is not None:
                    counts[valid] += 1
                    exceeding[valid] += shortfall > 1e-6
                    squared_by_cell[valid] += shortfall ** 2
                n_months += 1
    if weight_sum == 0:
        raise ValueError("No comparable wetland cells")
    if map_output is not None:
        write_map(map_output, lat, lon, counts, exceeding, squared_by_cell)
    return {
        "start_year": start_year, "end_year": end_year,
        "n_months": n_months, "n_pairs": n_pairs,
        "n_reference_above_cap": n_above_cap,
        "fraction_reference_above_cap": n_above_cap / n_pairs,
        "weighted_rmse_lower_bound": float(np.sqrt(squared_shortfall / weight_sum)),
        "weighted_positive_shortfall": positive_shortfall / weight_sum,
        "note": "An area error floor for this candidate cap, not a calibration score.",
        "map_output": str(Path(map_output).resolve()) if map_output is not None else None,
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
    parser.add_argument("--map-output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.map_output is not None and args.map_output.exists():
        raise FileExistsError(args.map_output)
    result = evaluate(args.wad2m, args.model_dir, args.cell_area,
                      args.start_year, args.end_year, map_output=args.map_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
