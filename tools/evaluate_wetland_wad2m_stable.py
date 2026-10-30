"""Score monthly wetland area on cells valid in every requested month.

This supplements the pooled WAD2M score: varying satellite coverage changes
the pooled score's spatial sample and the weight of each month.
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


MODEL_NAMES = (
    "wetVegSaturatedFrac", "wetSurfaceInundFrac", "wetlandInundVegFrac",
)


def stable_metrics(simulated, observed, cell_area):
    """Compute pooled metrics on the spatial intersection of monthly masks."""
    simulated = np.ma.asarray(simulated, dtype=np.float64)
    observed = np.ma.asarray(observed, dtype=np.float64)
    if simulated.ndim != 3 or simulated.shape != observed.shape:
        raise ValueError("Expected aligned time, latitude, longitude arrays")
    area = np.asarray(np.ma.filled(np.ma.asarray(cell_area), np.nan), dtype=np.float64)
    if area.shape != simulated.shape[1:]:
        raise ValueError("Cell-area grid differs from the model")
    sim = np.asarray(np.ma.filled(simulated, np.nan))
    obs = np.asarray(np.ma.filled(observed, np.nan))
    stable = (np.isfinite(area) & (area > 0)
              & np.all(np.isfinite(sim) & np.isfinite(obs), axis=0))
    if not np.any(stable):
        raise ValueError("No cells are valid in every month")
    sim = np.where(stable[None, :, :], sim, np.nan)
    obs = np.where(stable[None, :, :], obs, np.nan)
    metrics = area_metrics(sim, obs, area)
    weights = area[stable]
    simulated_monthly = [float(np.average(slice_[stable], weights=weights))
                         for slice_ in sim]
    observed_monthly = [float(np.average(slice_[stable], weights=weights))
                        for slice_ in obs]
    sim_anomaly = sim - np.mean(sim, axis=0)
    obs_anomaly = obs - np.mean(obs, axis=0)
    sim_energy = np.sum(sim_anomaly ** 2, axis=0)
    obs_energy = np.sum(obs_anomaly ** 2, axis=0)
    reference_dynamic = stable & (np.ptp(obs, axis=0) >= 0.01)
    temporal_valid = reference_dynamic & (sim_energy > 1e-12)
    temporal = np.divide(
        np.sum(sim_anomaly * obs_anomaly, axis=0),
        np.sqrt(sim_energy * obs_energy),
        out=np.full_like(area, np.nan),
        where=temporal_valid & (obs_energy > 0),
    )
    temporal_weight = area[temporal_valid]
    reference_dynamic_weight = area[reference_dynamic]
    temporal_on_reference = np.where(temporal_valid, temporal, 0.0)
    return {
        **metrics,
        "n_stable_cells": int(np.count_nonzero(stable)),
        "n_reference_dynamic_cells": int(np.count_nonzero(reference_dynamic)),
        "n_temporal_cells": int(np.count_nonzero(temporal_valid)),
        "temporal_reference_min_range": 0.01,
        "constant_model_temporal_score_on_reference_dynamic": 0.0,
        "weighted_mean_cell_temporal_correlation": (
            float(np.average(temporal[temporal_valid], weights=temporal_weight))
            if temporal_weight.size else None
        ),
        "weighted_temporal_correlation_on_reference_dynamic": (
            float(np.average(temporal_on_reference[reference_dynamic],
                             weights=reference_dynamic_weight))
            if reference_dynamic_weight.size else None
        ),
        "weighted_positive_temporal_correlation_fraction": (
            float(np.average(temporal[temporal_valid] > 0, weights=temporal_weight))
            if temporal_weight.size else None
        ),
        "simulated_monthly_mean": simulated_monthly,
        "observed_monthly_mean": observed_monthly,
    }


def cellwise_metrics(simulated, observed, cell_area):
    """Map monthly area errors and timing skill on the same complete-month cells."""
    sim = np.asarray(np.ma.filled(np.ma.asarray(simulated, dtype=np.float64), np.nan))
    obs = np.asarray(np.ma.filled(np.ma.asarray(observed, dtype=np.float64), np.nan))
    area = np.asarray(np.ma.filled(np.ma.asarray(cell_area), np.nan))
    if sim.ndim != 3 or sim.shape != obs.shape or area.shape != sim.shape[1:]:
        raise ValueError("Expected aligned time, latitude, longitude arrays")
    stable = (np.isfinite(area) & (area > 0)
              & np.all(np.isfinite(sim) & np.isfinite(obs), axis=0))
    sim = np.where(stable[None], sim, np.nan)
    obs = np.where(stable[None], obs, np.nan)
    with np.errstate(invalid="ignore"):
        error = sim - obs
        bias = np.mean(error, axis=0)
        rmse = np.sqrt(np.mean(error ** 2, axis=0))
        reference_range = np.max(obs, axis=0) - np.min(obs, axis=0)
        sim_anomaly = sim - np.mean(sim, axis=0)
        obs_anomaly = obs - np.mean(obs, axis=0)
        sim_energy = np.sum(sim_anomaly ** 2, axis=0)
        obs_energy = np.sum(obs_anomaly ** 2, axis=0)
        reference_dynamic = stable & (reference_range >= 0.01)
        temporal_valid = reference_dynamic & (sim_energy > 1e-12)
        temporal = np.divide(
            np.sum(sim_anomaly * obs_anomaly, axis=0),
            np.sqrt(sim_energy * obs_energy),
            out=np.full_like(area, np.nan, dtype=np.float64),
            where=temporal_valid & (obs_energy > 0),
        )
    return {
        "stable_cell": stable,
        "reference_dynamic_cell": reference_dynamic,
        "bias": bias,
        "rmse": rmse,
        "reference_range": reference_range,
        "temporal_correlation": temporal,
    }


def write_cellwise_map(path, lat, lon, metrics, start_year, end_year):
    """Save per-cell diagnostics for spatial review without changing source data."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as output:
        output.createDimension("lat", len(lat))
        output.createDimension("lon", len(lon))
        output.createVariable("lat", "f8", ("lat",))[:] = lat
        output.createVariable("lon", "f8", ("lon",))[:] = lon
        output.start_year = start_year
        output.end_year = end_year
        output.reference_dynamic_min_range = 0.01
        for name, values in metrics.items():
            variable = output.createVariable(name, "f4", ("lat", "lon"),
                                             zlib=True, fill_value=np.float32(-9999))
            variable.units = "1"
            variable[:] = np.ma.masked_invalid(values.astype(np.float32))


def evaluate(wad2m_path, model_dir, cell_area_path, start_year, end_year=None,
             map_output=None):
    """Evaluate complete years on one fixed spatial mask across all months."""
    if end_year is None:
        end_year = start_year
    if start_year > end_year:
        raise ValueError("Start year must not exceed end year")
    model_dir = Path(model_dir)
    with ExitStack() as stack:
        files = [stack.enter_context(Dataset(model_dir / f"{name}_monthavg.nc"))
                 for name in MODEL_NAMES]
        wad = stack.enter_context(Dataset(wad2m_path))
        area_file = stack.enter_context(Dataset(cell_area_path))
        lat = files[0].variables["lat"][:]
        lon = files[0].variables["lon"][:]
        wad_rows = align_axis(lat, wad.variables["lat"][:])
        wad_cols = align_axis(lon, wad.variables["lon"][:])
        area_rows = align_axis(lat, area_file.variables["lat"][:])
        area_cols = align_axis(lon, area_file.variables["lon"][:])
        area = np.ma.asarray(area_file.variables["cellarea30min"][:])[
            np.ix_(area_rows, area_cols)]
        model_months = _month_lookup(files[0])
        wad_months = _month_lookup(wad)
        for other in files[1:]:
            if (not np.array_equal(lat, other.variables["lat"][:])
                    or not np.array_equal(lon, other.variables["lon"][:])
                    or _month_lookup(other) != model_months):
                raise ValueError("Model monthly wetland files are not aligned")
        if getattr(wad.variables["Fw"], "units", None) not in ("-", "1"):
            raise ValueError("Unexpected WAD2M Fw units")
        simulated = []
        observed = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                key = (year, month)
                if key not in model_months or key not in wad_months:
                    raise ValueError(f"Missing month {key}")
                simulated.append(wetland_union(*(
                    file.variables[f"{name}_monthavg"][model_months[key]]
                    for file, name in zip(files, MODEL_NAMES)
                )))
                observed.append(wad.variables["Fw"][wad_months[key]][
                    np.ix_(wad_rows, wad_cols)])
        simulated = np.ma.stack(simulated)
        observed = np.ma.stack(observed)
        metrics = stable_metrics(simulated, observed, area)
        if map_output is not None:
            write_cellwise_map(map_output, lat, lon,
                               cellwise_metrics(simulated, observed, area),
                               start_year, end_year)
        return {
            "start_year": start_year, "end_year": end_year,
            "n_months": len(simulated),
            **metrics,
            "reference": str(Path(wad2m_path).resolve()),
            "model_dir": str(model_dir.resolve()),
            "cell_area": str(Path(cell_area_path).resolve()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wad2m", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cell-area", required=True)
    year = parser.add_mutually_exclusive_group(required=True)
    year.add_argument("--year", type=int)
    year.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path,
                        help="Optional NetCDF of complete-month cell errors and timing")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.map_output is not None and args.map_output.exists():
        raise FileExistsError(args.map_output)
    start = args.year if args.year is not None else args.start_year
    end = args.end_year if args.end_year is not None else start
    result = evaluate(args.wad2m, args.model_dir, args.cell_area, start, end,
                      map_output=args.map_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
