"""Plot how one wetland run changes per-cell WAD2M skill from a baseline."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


def read_map(path):
    with Dataset(path) as file:
        return {
            "lat": file.variables["lat"][:],
            "lon": file.variables["lon"][:],
            "stable": np.asarray(file.variables["stable_cell"][:]) > 0.5,
            "dynamic": np.asarray(file.variables["reference_dynamic_cell"][:]) > 0.5,
            "rmse": np.ma.filled(file.variables["rmse"][:], np.nan),
            "temporal": np.ma.filled(file.variables["temporal_correlation"][:], np.nan),
        }


def plot(baseline_path, candidate_path, output_path, candidate_label):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    baseline = read_map(baseline_path)
    candidate = read_map(candidate_path)
    for axis in ("lat", "lon"):
        if not np.array_equal(baseline[axis], candidate[axis]):
            raise ValueError(f"Mismatched {axis} grid")
    if not np.array_equal(baseline["stable"], candidate["stable"]):
        raise ValueError("Mismatched complete-month spatial masks")
    if not np.array_equal(baseline["dynamic"], candidate["dynamic"]):
        raise ValueError("Mismatched reference-dynamic spatial masks")
    stable = baseline["stable"]
    dynamic = baseline["dynamic"]
    rmse_change = np.ma.masked_where(
        ~stable, candidate["rmse"] - baseline["rmse"])
    # For a constant model series, the correlation is undefined; give it the
    # same zero score used in the fixed-reference aggregate comparison.
    base_timing = np.where(np.isfinite(baseline["temporal"]), baseline["temporal"], 0)
    cand_timing = np.where(np.isfinite(candidate["temporal"]), candidate["temporal"], 0)
    timing_change = np.ma.masked_where(~dynamic, cand_timing - base_timing)
    lon = baseline["lon"]
    lat = baseline["lat"]
    extent = (float(lon.min() - 0.25), float(lon.max() + 0.25),
              float(lat.min() - 0.25), float(lat.max() + 0.25))
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    panels = (
        (rmse_change, "Per-cell RMSE change (candidate − baseline)",
         "RdBu_r", -0.05, 0.05, "area fraction"),
        (timing_change, "Per-cell monthly timing score change (candidate − baseline)",
         "PiYG", -0.5, 0.5, "correlation score"),
    )
    for axis, (data, title, colors, low, high, units) in zip(axes, panels):
        image = axis.imshow(data, extent=extent, origin="upper", aspect="auto",
                            cmap=colors, vmin=low, vmax=high, interpolation="none")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-60, 90)
        axis.set_ylabel("Latitude (°)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        figure.colorbar(image, ax=axis, label=units, shrink=0.85, extend="both")
    axes[-1].set_xlabel("Longitude (°)")
    figure.suptitle(f"WAD2M complete-month skill: {candidate_label}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-map", required=True)
    parser.add_argument("--candidate-map", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.baseline_map, args.candidate_map, args.output,
         args.candidate_label)


if __name__ == "__main__":
    main()
