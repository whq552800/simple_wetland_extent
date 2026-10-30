"""Create a global historical CWatM wetland run without editing source INIs."""

import argparse
from datetime import date
from pathlib import Path
import re

import numpy as np
from netCDF4 import Dataset


FORCING = {
    "PrecipitationMaps": "pr", "TavgMaps": "tas", "TminMaps": "tasmin",
    "TmaxMaps": "tasmax", "PSurfMaps": "ps", "QAirMaps": "huss",
    "WindMaps": "sfcwind", "RSDSMaps": "rsds", "RSDLMaps": "rlds",
}
PARAMETERS = {
    "wetlandMeanDepth": 0.5,
    "wetlandResidenceDays": 10,
    "wetlandRunoffCapture": 0.5,
    "wetlandRiverCapture": 0.1,
    "wetlandSatThreshold": 0.8,
}
MONTHLY = (
    "wetVegSaturatedFrac, wetSurfaceInundFrac, wetlandInundVegFrac, "
    "wetlandStorage, wetlandPotentialFrac, wetlandClassifiedFrac, "
    "wetlandExcludedWaterbodyFrac, wetlandLocalInflowM3, "
    "wetlandRiverInflowM3, wetlandOutflowM3, wetlandEvapM"
)


def _set_option(content, key, value, required=False):
    expression = re.compile(rf"(?m)^(?!\s*[#;])\s*{re.escape(key)}\s*=.*$")
    matches = list(expression.finditer(content))
    if len(matches) > 1 or (required and not matches):
        raise ValueError(f"Expected exactly one active {key} in base INI")
    line = f"{key} = {value}"
    return expression.sub(line, content) if matches else content.rstrip() + "\n" + line + "\n"


def _check_grid(prior, clone):
    with Dataset(prior) as wet, Dataset(clone) as grid:
        for axis, size in (("lat", 360), ("lon", 720)):
            a = np.asarray(wet.variables[axis][:])
            b = np.asarray(grid.variables[axis][:])
            if a.shape != (size,) or not np.allclose(a, b, atol=1e-7):
                raise ValueError(f"Wetland prior {axis} does not match global clone")
        for name in ("wetlandVegFrac", "wetlandFloodFrac", "wetlandUnmappedFrac"):
            if name not in wet.variables or wet.variables[name].shape != (360, 720):
                raise ValueError(f"Wetland prior lacks global {name}")


def _check_forcing(meteo, start_year, end_year):
    for variable in FORCING.values():
        spans = []
        pattern = re.compile(
            rf"gswp3-w5e5_obsclim_{variable}_global_daily_(\d{{4}})_(\d{{4}})\.nc$")
        for path in meteo.glob(f"gswp3-w5e5_obsclim_{variable}_global_daily_*.nc"):
            match = pattern.fullmatch(path.name)
            if match:
                spans.append(tuple(map(int, match.groups())))
        missing = [year for year in range(start_year, end_year + 1)
                   if not any(first <= year <= last for first, last in spans)]
        if missing:
            raise FileNotFoundError(f"Missing {variable} forcing for {missing[:5]}")


def prepare_run(base_ini, path_root, path_meteo, prior, settings_file, output_dir,
                start_date, end_date, parameters=None):
    base_ini, path_root, path_meteo, prior, settings_file, output_dir = map(
        Path, (base_ini, path_root, path_meteo, prior, settings_file, output_dir))
    if settings_file.exists() or output_dir.exists():
        raise FileExistsError(settings_file if settings_file.exists() else output_dir)
    if not start_date <= end_date:
        raise ValueError("Start date must be no later than end date")
    for path in (base_ini, prior, path_root / "areamaps" / "global.map"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not any((path_root / "routing" / name).is_file()
               for name in ("cellarea.map", "cellarea.nc")):
        raise FileNotFoundError(path_root / "routing" / "cellarea.nc")
    _check_grid(prior, path_root / "landsurface" / "Global_CloneMap_30min.nc")
    _check_forcing(path_meteo, start_date.year, end_date.year)
    values = dict(PARAMETERS)
    if parameters:
        if set(parameters) - set(values):
            raise ValueError("Unknown wetland parameter")
        values.update(parameters)
    content = base_ini.read_text(encoding="utf-8")
    replacements = {
        "PathRoot": path_root.resolve().as_posix(),
        "PathOut": output_dir.resolve().as_posix(),
        "PathMeteo": path_meteo.resolve().as_posix(),
        "MaskMap": "$(FILE_PATHS:PathRoot)/areamaps/global.map",
        "StepStart": start_date.strftime("%d/%m/%Y"),
        "StepEnd": end_date.strftime("%d/%m/%Y"),
        "load_initial": "False", "save_initial": "False",
        "calc_evaporation": "True", "includeRouting": "True",
        "useHuss": "True",
        "OUT_MAP_MonthAvg": MONTHLY,
    }
    for key, value in replacements.items():
        content = _set_option(content, key, value,
                              required=key in ("PathRoot", "PathOut", "PathMeteo",
                                               "MaskMap", "StepStart", "StepEnd",
                                               "load_initial", "save_initial"))
    if re.search(r"(?m)^\s*includeWetlands\s*=", content):
        content = _set_option(content, "includeWetlands", "True")
    else:
        content = re.sub(r"(?m)^(includeRouting\s*=\s*True)$",
                         r"\1\nincludeWetlands = True", content, count=1)
    for key, variable in FORCING.items():
        pattern = (f"$(FILE_PATHS:PathMeteo)/"
                   f"gswp3-w5e5_obsclim_{variable}_global_daily_*")
        content = _set_option(content, key, pattern)
    content = content.rstrip() + "\n\n[WETLANDS]\n"
    content += f"wetlandFractionFile = {prior.resolve().as_posix()}\n"
    content += "".join(f"{key} = {value}\n" for key, value in values.items())
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    with settings_file.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    output_dir.mkdir(parents=True)
    return settings_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-ini", "path-root", "path-meteo", "prior",
                 "settings-file", "output-dir", "start-date", "end-date"):
        parser.add_argument("--" + name, required=True)
    for name in PARAMETERS:
        parser.add_argument("--" + name, type=float)
    args = parser.parse_args()
    options = vars(args)
    selected = {name: options[name] for name in PARAMETERS if options[name] is not None}
    prepare_run(args.base_ini, args.path_root, args.path_meteo, args.prior,
                args.settings_file, args.output_dir, date.fromisoformat(args.start_date),
                date.fromisoformat(args.end_date), selected)


if __name__ == "__main__":
    main()
