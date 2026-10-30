"""Plot per-cell WAD2M RMSE and monthly timing correlation for one wetland run."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


def plot(source, output, title):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with Dataset(source) as file:
        lat = file.variables["lat"][:]
        lon = file.variables["lon"][:]
        stable = np.ma.filled(file.variables["stable_cell"][:], 0) > 0.5
        dynamic = np.ma.filled(file.variables["reference_dynamic_cell"][:], 0) > 0.5
        rmse = np.ma.filled(file.variables["rmse"][:], np.nan)
        temporal = np.ma.filled(file.variables["temporal_correlation"][:], np.nan)
    rmse = np.ma.masked_where(~stable, rmse)
    temporal = np.ma.masked_where(~dynamic, temporal)
    extent = (float(lon.min() - 0.25), float(lon.max() + 0.25),
              float(lat.min() - 0.25), float(lat.max() + 0.25))
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    panels = ((rmse, "Per-cell area RMSE", "magma", 0, 0.2, "area fraction"),
              (temporal, "Per-cell monthly timing correlation", "BrBG", -1, 1,
               "correlation"))
    for axis, (values, label, colors, low, high, units) in zip(axes, panels):
        image = axis.imshow(values, origin="upper", extent=extent, aspect="auto",
                            cmap=colors, vmin=low, vmax=high, interpolation="none")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-60, 90)
        axis.set_ylabel("Latitude (°)")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        figure.colorbar(image, ax=axis, label=units, shrink=0.85,
                        extend="max" if low == 0 else "neither")
    axes[-1].set_xlabel("Longitude (°)")
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    plot(args.source, args.output, args.title)


if __name__ == "__main__":
    main()
