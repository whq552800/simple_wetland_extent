"""Map where the candidate wetland area cap falls below WAD2M."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


def plot(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with Dataset(source) as file:
        lat = file.variables["lat"][:]
        lon = file.variables["lon"][:]
        fraction = np.ma.asarray(file.variables["fraction_reference_above_cap"][:])
        floor = np.ma.asarray(file.variables["rmse_lower_bound"][:])
    extent = (float(lon.min() - 0.25), float(lon.max() + 0.25),
              float(lat.min() - 0.25), float(lat.max() + 0.25))
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    panels = ((fraction, "Fraction of valid months above candidate cap", 1.0),
              (floor, "Per-cell unavoidable RMSE from candidate cap", 0.25))
    for axis, (values, title, upper) in zip(axes, panels):
        image = axis.imshow(values, origin="upper", extent=extent, aspect="auto",
                            cmap="magma", vmin=0, vmax=upper, interpolation="none")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-60, 90)
        axis.set_ylabel("Latitude (°)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        figure.colorbar(image, ax=axis, label="fraction", shrink=0.85,
                        extend="max")
    axes[-1].set_xlabel("Longitude (°)")
    figure.suptitle("WAD2M area outside the wetland candidate prior")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
