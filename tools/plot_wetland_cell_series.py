"""Plot one exported grid cell's monthly wetland area and storage."""

import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with Path(source).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No cell time series")
    locations = {(row["lat"], row["lon"]) for row in rows}
    if len(locations) != 1:
        raise ValueError("Expected one grid cell")
    dates = [date(int(row["year"]), int(row["month"]), 15) for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Months are duplicated or out of order")

    def series(key):
        return np.array([float(row[key]) if row[key] else np.nan for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]},
                               layout="constrained")
    for key, label, color, style, width in (
        ("wetlandPotentialFrac", "Candidate cap", "#777777", "-", 1.0),
        ("wetSurfaceInundFrac", "Surface inundated", "#9b5fa6", "-", 1.2),
        ("wetVegSaturatedFrac", "Vegetated saturated", "#378155", "-", 1.2),
        ("wetlandUnionFrac", "CWatM union", "#dd7f22", "--", 2.0),
        ("WAD2M_Fw", "WAD2M v2", "#1565a2", "-", 1.8),
    ):
        axes[0].plot(dates, series(key), label=label, color=color,
                     linestyle=style, lw=width)
    axes[0].set_ylabel("Grid-cell area fraction")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(dates, series("wetlandStorage") / 1e6, color="#5b5896")
    axes[1].set_ylabel("Wetland storage (million m³)")
    axes[1].set_xlabel("Year")
    for axis in axes:
        axis.grid(alpha=0.2)
    lat, lon = next(iter(locations))
    figure.suptitle(f"Wetland monthly series at {lat}° latitude, {lon}° longitude")
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
