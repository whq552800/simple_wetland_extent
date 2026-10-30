"""Run reproducible five-parameter wetland screening against monthly WAD2M.

This is a model-run orchestrator, not a shortcut for full global calibration.
Use a short screening period first, then an independent longer validation run.
Parameter bounds are supplied explicitly by the caller.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from scipy.stats import qmc

try:
    from tools.evaluate_wetland_wad2m import evaluate
except ModuleNotFoundError:
    from evaluate_wetland_wad2m import evaluate


PARAMETER_NAMES = (
    "wetlandMeanDepth", "wetlandResidenceDays", "wetlandRunoffCapture",
    "wetlandRiverCapture", "wetlandSatThreshold",
)
CODE_PATHS = (
    "cwatm/hydrological_modules/wetlands.py",
    "patches/cwatm-a76ca0b-integration.patch",
    "tools/evaluate_wetland_wad2m.py",
    "tools/calibrate_wetlands.py",
)


def code_fingerprints(root, relative_paths=CODE_PATHS):
    """Refuse to mix a resumed search with a different model/scorer version."""
    root = Path(root)
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in relative_paths}


def candidate_sets(bounds, number, seed):
    """Deterministic Latin-hypercube candidates within explicit bounds."""
    if set(bounds) != set(PARAMETER_NAMES):
        raise ValueError("Bounds must name exactly the five wetland parameters")
    if number < 1:
        raise ValueError("Number of candidates must be positive")
    lows = []
    highs = []
    for name in PARAMETER_NAMES:
        pair = bounds[name]
        if len(pair) != 2:
            raise ValueError(f"Expected [low, high] for {name}")
        low, high = map(float, pair)
        if not low < high:
            raise ValueError(f"Empty bounds for {name}")
        if name in ("wetlandMeanDepth", "wetlandResidenceDays") and low <= 0:
            raise ValueError(f"{name} must be positive")
        if name in ("wetlandRunoffCapture", "wetlandRiverCapture") and (low < 0 or high > 1):
            raise ValueError(f"{name} must lie in [0,1]")
        if name == "wetlandSatThreshold" and (low < 0 or high >= 1):
            raise ValueError("wetlandSatThreshold must lie in [0,1)")
        lows.append(low)
        highs.append(high)
    samples = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=seed).random(number)
    scaled = qmc.scale(samples, lows, highs)
    return [dict(zip(PARAMETER_NAMES, map(float, row))) for row in scaled]


def render_candidate_ini(source, parameters, output_directory):
    """Replace only the five active wetland settings and PathOut."""
    if set(parameters) != set(PARAMETER_NAMES):
        raise ValueError("Candidate must set exactly the five wetland parameters")
    updates = {**parameters, "PathOut": output_directory}
    result = source
    for name, value in updates.items():
        pattern = r"(?m)^([ \t]*" + re.escape(name) + r"[ \t]*=[ \t]*).*$"
        result, count = re.subn(pattern, lambda match: match.group(1) + str(value), result)
        if count != 1:
            raise ValueError(f"Missing setting or duplicate setting: {name}")
    if not re.search(r"(?mi)^\s*includeWetlands\s*=\s*True\s*$", result):
        raise ValueError("Template must enable includeWetlands")
    return result


def _write_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def calibrate(template, bounds_file, wad2m, cell_area, run_cwatm,
              start_year, end_year, number, seed, work_dir):
    """Run each candidate in its own directory and rank by weighted RMSE."""
    root = Path(work_dir)
    bounds = json.loads(Path(bounds_file).read_text(encoding="utf-8"))
    candidates = candidate_sets(bounds, number, seed)
    source = Path(template).read_text(encoding="utf-8")
    definition = {
        "bounds": bounds, "seed": seed, "number": number,
        "start_year": start_year, "end_year": end_year,
        "template": str(Path(template).resolve()),
        "template_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "reference": str(Path(wad2m).resolve()),
        "cell_area": str(Path(cell_area).resolve()),
        "run_cwatm": str(Path(run_cwatm).resolve()),
        "code_sha256": code_fingerprints(Path(__file__).resolve().parent.parent),
    }
    definition_path = root / "search_definition.json"
    if root.exists():
        if not definition_path.is_file() or json.loads(
                definition_path.read_text(encoding="utf-8")) != definition:
            raise ValueError("Existing search has a different definition")
    else:
        root.mkdir(parents=True)
        definition_path.write_text(json.dumps(definition, indent=2),
                                   encoding="utf-8")
    rows = []
    for index, parameters in enumerate(candidates):
        prefix = f"candidate_{index:03d}"
        prior_attempts = [root / prefix, *sorted(root.glob(prefix + "_attempt_*"))]
        completed = None
        for attempt_dir in prior_attempts:
            result_path = attempt_dir / "result.json"
            if result_path.is_file():
                recorded = json.loads((attempt_dir / "parameters.json").read_text(
                    encoding="utf-8"))
                if recorded != parameters:
                    raise ValueError(f"Candidate parameters changed: {attempt_dir}")
                row = json.loads(result_path.read_text(encoding="utf-8"))
                if "weighted_rmse" in row:
                    completed = row
                    break
        if completed is not None:
            rows.append(completed)
            _write_table(root / "candidates.csv", rows)
            continue
        candidate_dir = root / prefix
        attempt = 0
        while candidate_dir.exists():
            attempt += 1
            candidate_dir = root / f"{prefix}_attempt_{attempt:03d}"
        output_dir = candidate_dir / "out"
        output_dir.mkdir(parents=True)
        settings = candidate_dir / "settings.ini"
        settings.write_text(render_candidate_ini(source, parameters,
                                                output_dir.resolve().as_posix()),
                            encoding="utf-8")
        (candidate_dir / "parameters.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8",
        )
        with (candidate_dir / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(Path(run_cwatm).resolve()),
                                     str(settings.resolve()), "-l"],
                                    cwd=Path(run_cwatm).resolve().parent,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    check=False)
        row = {"candidate": index, **parameters, "returncode": result.returncode}
        if result.returncode == 0:
            try:
                monthly, summary = evaluate(wad2m, output_dir, cell_area,
                                            start_year, end_year)
                _write_table(candidate_dir / "monthly.csv", monthly)
                (candidate_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8",
                )
                row.update({key: summary[key] for key in (
                    "n_months", "n_pairs", "weighted_bias", "weighted_mae",
                    "weighted_rmse")})
            except (ValueError, FileNotFoundError, KeyError) as error:
                row["evaluation_error"] = str(error)
        rows.append(row)
        (candidate_dir / "result.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8",
        )
        _write_table(root / "candidates.csv", rows)
        print(f"Candidate {index + 1}/{number}: "
              f"{row.get('weighted_rmse', row.get('evaluation_error', result.returncode))}",
              flush=True)
    successful = [row for row in rows if "weighted_rmse" in row]
    if not successful:
        raise RuntimeError(f"No successful candidate; inspect {root}")
    best = min(successful, key=lambda row: row["weighted_rmse"])
    (root / "best_screening_candidate.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8",
    )
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--bounds", required=True, help="JSON: each parameter -> [low, high]")
    parser.add_argument("--wad2m", required=True)
    parser.add_argument("--cell-area", required=True)
    parser.add_argument("--run-cwatm", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start year must not exceed end year")
    calibrate(args.template, args.bounds, args.wad2m, args.cell_area,
              args.run_cwatm, args.start_year, args.end_year,
              args.number, args.seed, args.work_dir)


if __name__ == "__main__":
    main()
