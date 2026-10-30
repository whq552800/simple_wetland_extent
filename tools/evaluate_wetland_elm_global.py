"""Compare global monthly CWatM wetland area with ELM-ECA model output.

ELM-ECA ``wetlandfrac`` is another model's percentage, not an observation.
The comparison uses the same matched-cell, area-weighted metrics as WAD2M.
"""

import argparse
import csv
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


MODEL_NAMES = (
    "wetVegSaturatedFrac", "wetSurfaceInundFrac", "wetlandInundVegFrac",
)


def evaluate(elm_path, model_dir, cell_area_path, start_year, end_year):
    """Return monthly and aggregate metrics on mutually valid grid cells."""
    model_dir = Path(model_dir)
    with ExitStack() as stack:
        files = [stack.enter_context(Dataset(model_dir / f"{name}_monthavg.nc"))
                 for name in MODEL_NAMES]
        elm = stack.enter_context(Dataset(elm_path))
        area_file = stack.enter_context(Dataset(cell_area_path))
        reference = elm.variables["wetlandfrac"]
        if getattr(reference, "units", None) != "%":
            raise ValueError("ELM wetlandfrac must use percent units")
        if getattr(elm.variables["time"], "calendar", None) != "365_day":
            raise ValueError("Expected ELM 365_day calendar")
        model = files[0]
        model_lat = model.variables["lat"][:]
        model_lon = model.variables["lon"][:]
        ref_lat = align_axis(model_lat, elm.variables["lat"][:])
        ref_lon = align_axis(model_lon, elm.variables["lon"][:])
        area_lat = align_axis(model_lat, area_file.variables["lat"][:])
        area_lon = align_axis(model_lon, area_file.variables["lon"][:])
        cell_area = np.asarray(np.ma.filled(
            np.ma.asarray(area_file.variables["cellarea30min"][:])[
                np.ix_(area_lat, area_lon)], np.nan), dtype=np.float64)
        model_months = _month_lookup(model)
        reference_months = _month_lookup(elm)
        for other in files[1:]:
            if (not np.array_equal(model_lat, other.variables["lat"][:])
                    or not np.array_equal(model_lon, other.variables["lon"][:])
                    or _month_lookup(other) != model_months):
                raise ValueError("Model monthly wetland files are not aligned")

        rows = []
        totals = {key: 0.0 for key in (
            "error", "absolute", "squared", "model", "reference", "area")}
        pairs = 0
        for month in sorted(model_months.keys() & reference_months.keys()):
            if not start_year <= month[0] <= end_year:
                continue
            index = model_months[month]
            union = wetland_union(*(
                file.variables[f"{name}_monthavg"][index]
                for file, name in zip(files, MODEL_NAMES)
            ))
            reference_pct = reference[reference_months[month]][
                np.ix_(ref_lat, ref_lon)]
            if np.ma.any((reference_pct < 0) | (reference_pct > 100)):
                raise ValueError("ELM wetlandfrac lies outside 0–100%")
            metrics = area_metrics(union, reference_pct / 100.0, cell_area)
            rows.append({"year": month[0], "month": month[1], **metrics})
            weight = metrics["area_weight_sum_m2"]
            pairs += metrics["n_pairs"]
            for key, metric in (
                ("error", "weighted_bias"), ("absolute", "weighted_mae"),
                ("squared", "weighted_rmse"),
                ("model", "weighted_simulated_mean"),
                ("reference", "weighted_observed_mean"),
            ):
                value = metrics[metric]
                totals[key] += weight * (value ** 2 if key == "squared" else value)
            totals["area"] += weight
        if not rows:
            raise ValueError("No comparable months in requested years")
        denominator = totals["area"]
        summary = {
            "reference_type": "ELM-ECA model output; inter-model comparison",
            "reference_calendar": "365_day",
            "start_year": start_year, "end_year": end_year,
            "n_months": len(rows), "n_pairs": pairs,
            "weighted_bias": totals["error"] / denominator,
            "weighted_mae": totals["absolute"] / denominator,
            "weighted_rmse": np.sqrt(totals["squared"] / denominator),
            "weighted_simulated_mean": totals["model"] / denominator,
            "weighted_reference_mean": totals["reference"] / denominator,
            "reference": str(Path(elm_path).resolve()),
            "model_dir": str(model_dir.resolve()),
            "cell_area": str(Path(cell_area_path).resolve()),
        }
        return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elm", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cell-area", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start year must not exceed end year")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows, summary = evaluate(args.elm, args.model_dir, args.cell_area,
                             args.start_year, args.end_year)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "monthly.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
