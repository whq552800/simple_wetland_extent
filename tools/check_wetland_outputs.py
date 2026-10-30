"""Check monthly global wetland output bounds and area accounting."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

try:
    from tools.evaluate_wetland_wad2m import _month_lookup
except ModuleNotFoundError:
    from evaluate_wetland_wad2m import _month_lookup


NAMES = (
    "wetVegSaturatedFrac", "wetSurfaceInundFrac", "wetlandInundVegFrac",
    "wetlandPotentialFrac", "wetlandClassifiedFrac",
    "wetlandExcludedWaterbodyFrac", "wetlandStorage",
)


def check(model_dir, tolerance=2e-6):
    model_dir = Path(model_dir)
    with ExitStack() as stack:
        files = {name: stack.enter_context(Dataset(model_dir / f"{name}_monthavg.nc"))
                 for name in NAMES}
        reference = files[NAMES[0]]
        months = _month_lookup(reference)
        lat = reference.variables["lat"][:]
        lon = reference.variables["lon"][:]
        if not months:
            raise ValueError("No monthly wetland output")
        for name, file in files.items():
            if (_month_lookup(file) != months
                    or not np.array_equal(file.variables["lat"][:], lat)
                    or not np.array_equal(file.variables["lon"][:], lon)):
                raise ValueError(f"Wetland file grid or months differ: {name}")
        maxima = {key: 0.0 for key in (
            "negative_fraction", "overlap_above_veg", "overlap_above_surface",
            "union_above_potential", "classified_balance",
            "potential_above_one", "negative_storage_m3",
        )}
        for index in range(len(months)):
            arrays = {name: np.ma.asarray(
                files[name].variables[f"{name}_monthavg"][index], dtype=np.float64,
            ) for name in NAMES}
            veg = arrays["wetVegSaturatedFrac"]
            surface = arrays["wetSurfaceInundFrac"]
            overlap = arrays["wetlandInundVegFrac"]
            potential = arrays["wetlandPotentialFrac"]
            classified = arrays["wetlandClassifiedFrac"]
            excluded = arrays["wetlandExcludedWaterbodyFrac"]
            storage = arrays["wetlandStorage"]
            values = {
                "negative_fraction": max(float(np.ma.max(-arrays[name]))
                                         for name in NAMES[:-1]),
                "overlap_above_veg": float(np.ma.max(overlap - veg)),
                "overlap_above_surface": float(np.ma.max(overlap - surface)),
                "union_above_potential": float(np.ma.max(veg + surface - overlap - potential)),
                "classified_balance": float(np.ma.max(
                    np.ma.abs(classified - potential - excluded))),
                "potential_above_one": float(np.ma.max(potential - 1)),
                "negative_storage_m3": float(np.ma.max(-storage)),
            }
            for key, value in values.items():
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite wetland check {key} at month {index}")
                maxima[key] = max(maxima[key], value)
        if any(value > tolerance for value in maxima.values()):
            raise ValueError(f"Wetland monthly bound check failed: {maxima}")
        return {
            "n_months": len(months), "shape": [len(lat), len(lon)],
            "first_month": list(min(months)), "last_month": list(max(months)),
            "tolerance": tolerance, **maxima,
            "model_dir": str(model_dir.resolve()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
