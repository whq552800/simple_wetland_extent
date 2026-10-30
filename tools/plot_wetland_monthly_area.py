"""Plot CWatM and WAD2M monthly matched-cell wetland area fractions."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(output, monthly_csv=None, stable_json=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (monthly_csv is None) == (stable_json is None):
        raise ValueError("Choose exactly one monthly score source")
    if monthly_csv is not None:
        with Path(monthly_csv).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("No monthly scores")
        dates = [date(int(row["year"]), int(row["month"]), 15) for row in rows]
        simulated = np.array([float(row["weighted_simulated_mean"]) for row in rows])
        observed = np.array([float(row["weighted_observed_mean"]) for row in rows])
        support = "matched WAD2M coverage (varies by month)"
    else:
        scores = json.loads(Path(stable_json).read_text(encoding="utf-8"))
        dates = [date(year, month, 15)
                 for year in range(scores["start_year"], scores["end_year"] + 1)
                 for month in range(1, 13)]
        simulated = np.asarray(scores["simulated_monthly_mean"], dtype=float)
        observed = np.asarray(scores["observed_monthly_mean"], dtype=float)
        support = "one complete-month fixed grid"
    if not dates or len(dates) != len(simulated) or len(dates) != len(observed):
        raise ValueError("Monthly dates and values do not match")
    if dates != sorted(set(dates)):
        raise ValueError("Months are duplicated or out of order")
    bias = simulated - observed
    figure, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]},
                               layout="constrained")
    axes[0].plot(dates, observed, color="#1565a2", label="WAD2M v2", lw=1.3)
    axes[0].plot(dates, simulated, color="#dd7f22", label="CWatM wetland union", lw=1.3)
    axes[0].set_ylabel("Wetland area fraction")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(dates, bias, color="#5b5896", lw=1.1)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("CWatM − WAD2M")
    axes[1].set_xlabel("Year")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"Global monthly wetland area on {support}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--monthly-csv")
    source.add_argument("--stable-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.output, monthly_csv=args.monthly_csv, stable_json=args.stable_json)


if __name__ == "__main__":
    main()
