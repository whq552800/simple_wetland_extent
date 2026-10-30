"""Export one 0.5° grid cell's monthly simulated wetland components and WAD2M."""

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

try:
    from tools.evaluate_wetland_wad2m import _month_lookup, align_axis
except ModuleNotFoundError:
    from evaluate_wetland_wad2m import _month_lookup, align_axis


COMPONENTS = (
    "wetVegSaturatedFrac", "wetSurfaceInundFrac", "wetlandInundVegFrac",
    "wetlandPotentialFrac", "wetlandStorage",
)


def export(model_dir, wad2m_path, latitude, longitude, start_year, end_year,
           output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if start_year > end_year:
        raise ValueError("Start year must not exceed end year")
    model_dir = Path(model_dir)
    with ExitStack() as stack:
        files = {name: stack.enter_context(Dataset(
            model_dir / f"{name}_monthavg.nc")) for name in COMPONENTS}
        wad = stack.enter_context(Dataset(wad2m_path))
        if getattr(wad.variables["Fw"], "units", None) not in ("-", "1"):
            raise ValueError("Unexpected WAD2M Fw units")
        reference = files[COMPONENTS[0]]
        lat_axis = np.asarray(reference.variables["lat"][:])
        lon_axis = np.asarray(reference.variables["lon"][:])
        row = int(np.argmin(abs(lat_axis - latitude)))
        col = int(np.argmin(abs(lon_axis - longitude)))
        if abs(lat_axis[row] - latitude) > 0.251 or abs(lon_axis[col] - longitude) > 0.251:
            raise ValueError("Requested coordinate is not on the global 0.5° grid")
        for name, file in files.items():
            if (not np.array_equal(file.variables["lat"][:], lat_axis)
                    or not np.array_equal(file.variables["lon"][:], lon_axis)
                    or _month_lookup(file) != _month_lookup(reference)):
                raise ValueError(f"Model grid or time differs: {name}")
        ref_row = int(align_axis([lat_axis[row]], wad.variables["lat"][:])[0])
        ref_col = int(align_axis([lon_axis[col]], wad.variables["lon"][:])[0])
        model_months = _month_lookup(reference)
        wad_months = _month_lookup(wad)
        records = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                key = (year, month)
                if key not in model_months or key not in wad_months:
                    raise ValueError(f"Missing month {key}")
                values = {}
                for name, file in files.items():
                    value = file.variables[f"{name}_monthavg"][model_months[key], row, col]
                    values[name] = "" if np.ma.is_masked(value) else float(value)
                if all(values[name] != "" for name in COMPONENTS[:3]):
                    values["wetlandUnionFrac"] = (values[COMPONENTS[0]]
                                                  + values[COMPONENTS[1]]
                                                  - values[COMPONENTS[2]])
                else:
                    values["wetlandUnionFrac"] = ""
                observed = wad.variables["Fw"][wad_months[key], ref_row, ref_col]
                values["WAD2M_Fw"] = "" if np.ma.is_masked(observed) else float(observed)
                records.append({"year": year, "month": month,
                                "lat": float(lat_axis[row]), "lon": float(lon_axis[col]),
                                **values})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wad2m", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.model_dir, args.wad2m, args.lat, args.lon,
           args.start_year, args.end_year, args.output)


if __name__ == "__main__":
    main()
