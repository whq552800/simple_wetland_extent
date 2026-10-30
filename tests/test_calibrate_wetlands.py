import pytest
import json
import csv
from types import SimpleNamespace

import tools.calibrate_wetlands as calibration
from tools.calibrate_wetlands import candidate_sets, code_fingerprints, render_candidate_ini


PARAMETERS = {
    "wetlandMeanDepth": (0.1, 1.0),
    "wetlandResidenceDays": (1.0, 30.0),
    "wetlandRunoffCapture": (0.0, 1.0),
    "wetlandRiverCapture": (0.0, 1.0),
    "wetlandSatThreshold": (0.5, 0.95),
}


def test_candidate_sets_are_reproducible_and_bounded():
    first = candidate_sets(PARAMETERS, 4, seed=13)
    assert first == candidate_sets(PARAMETERS, 4, seed=13)
    assert len(first) == 4
    for candidate in first:
        assert set(candidate) == set(PARAMETERS)
        for name, value in candidate.items():
            low, high = PARAMETERS[name]
            assert low <= value <= high


def test_code_fingerprint_changes_with_source(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("version = 1", encoding="utf-8")
    first = code_fingerprints(tmp_path, ("model.py",))
    source.write_text("version = 2", encoding="utf-8")
    assert code_fingerprints(tmp_path, ("model.py",)) != first


def test_render_candidate_ini_sets_parameters_and_new_output():
    source = """[OPTIONS]
includeWetlands = True
[FILE_PATHS]
PathOut = old
[WETLANDS]
wetlandMeanDepth = 0.5
wetlandResidenceDays = 10
wetlandRunoffCapture = 0.5
wetlandRiverCapture = 0.1
wetlandSatThreshold = 0.8
"""
    text = render_candidate_ini(source, {key: 0.25 for key in PARAMETERS},
                                "C:/new-output")
    assert "PathOut = C:/new-output" in text
    for name in PARAMETERS:
        assert f"{name} = 0.25" in text
    assert "PathOut = old" not in text


def test_render_candidate_ini_rejects_missing_parameter():
    with pytest.raises(ValueError, match="Missing setting"):
        render_candidate_ini("[FILE_PATHS]\nPathOut = old\n",
                             {key: 0.25 for key in PARAMETERS}, "C:/new")


def test_calibration_resumes_completed_candidates_without_rerunning(tmp_path, monkeypatch):
    template = tmp_path / "template.ini"
    template.write_text("includeWetlands = True\nPathOut = old\n" +
                        "".join(f"{name} = 0.5\n" for name in PARAMETERS),
                        encoding="utf-8")
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps(PARAMETERS), encoding="utf-8")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    def evaluate(*args):
        return ([{"year": 2000}], {"n_months": 1, "n_pairs": 10,
                "weighted_bias": 0.1, "weighted_mae": 0.1,
                "weighted_rmse": 0.1})

    monkeypatch.setattr(calibration.subprocess, "run", run)
    monkeypatch.setattr(calibration, "evaluate", evaluate)
    arguments = (template, bounds, tmp_path / "obs.nc", tmp_path / "area.nc",
                 tmp_path / "run_cwatm.py", 2000, 2000, 2, 13,
                 tmp_path / "search")
    first = calibration.calibrate(*arguments)
    second = calibration.calibrate(*arguments)
    assert first == second
    assert len(calls) == 2


def test_calibration_preserves_interrupted_attempt_on_resume(tmp_path, monkeypatch):
    template = tmp_path / "template.ini"
    template.write_text("includeWetlands = True\nPathOut = old\n" +
                        "".join(f"{name} = 0.5\n" for name in PARAMETERS),
                        encoding="utf-8")
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps(PARAMETERS), encoding="utf-8")
    arguments = (template, bounds, tmp_path / "obs.nc", tmp_path / "area.nc",
                 tmp_path / "run_cwatm.py", 2000, 2000, 1, 13,
                 tmp_path / "search")
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(calibration.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        calibration.calibrate(*arguments)
    partial = tmp_path / "search" / "candidate_000" / "settings.ini"
    original = partial.read_text(encoding="utf-8")
    monkeypatch.setattr(calibration.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(calibration, "evaluate",
                        lambda *args: ([{"year": 2000}], {"n_months": 1,
                                       "n_pairs": 10, "weighted_bias": 0.1,
                                       "weighted_mae": 0.1, "weighted_rmse": 0.1}))
    calibration.calibrate(*arguments)
    assert partial.read_text(encoding="utf-8") == original
    assert (tmp_path / "search" / "candidate_000_attempt_001" /
            "summary.json").is_file()


def test_candidate_table_keeps_failure_and_success_columns(tmp_path):
    path = tmp_path / "candidates.csv"
    calibration._write_table(path, [
        {"candidate": 0, "returncode": 1},
        {"candidate": 1, "returncode": 0, "weighted_rmse": 0.1},
    ])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["weighted_rmse"] == "0.1"
