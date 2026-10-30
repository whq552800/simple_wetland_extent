from datetime import date

import numpy as np
import pytest
from netCDF4 import Dataset

from tools.prepare_global_wetland_run import prepare_run


def _grid(path):
    with Dataset(path, "w") as ds:
        ds.createDimension("lat", 360)
        ds.createDimension("lon", 720)
        ds.createVariable("lat", "f8", ("lat",))[:] = 89.75 - np.arange(360) * .5
        ds.createVariable("lon", "f8", ("lon",))[:] = -179.75 + np.arange(720) * .5


def test_generates_isolated_global_settings_from_read_only_inputs(tmp_path):
    root = tmp_path / "static"
    (root / "areamaps").mkdir(parents=True)
    (root / "routing").mkdir()
    (root / "landsurface").mkdir()
    (root / "areamaps" / "global.map").touch()
    (root / "routing" / "cellarea.nc").touch()
    _grid(root / "landsurface" / "Global_CloneMap_30min.nc")
    prior = tmp_path / "prior.nc"
    _grid(prior)
    with Dataset(prior, "a") as ds:
        for name in ("wetlandVegFrac", "wetlandFloodFrac", "wetlandUnmappedFrac"):
            ds.createVariable(name, "f4", ("lat", "lon"))[:] = 0
    meteo = tmp_path / "meteo"
    meteo.mkdir()
    for var in ("pr", "tas", "tasmin", "tasmax", "ps", "huss",
                "sfcwind", "rsds", "rlds"):
        (meteo / f"gswp3-w5e5_obsclim_{var}_global_daily_1991_2000.nc").touch()
    base = tmp_path / "base.ini"
    source = "[OPTIONS]\ncalc_evaporation = False\nincludeRouting = True\n"
    base.write_text(source + "[FILE_PATHS]\nPathRoot = original\nPathMeteo = original\n"
                    "PathOut = original\nMaskMap = old\n"
                    "[TIME]\nStepStart = old\nStepEnd = old\n"
                    "[INITIAL]\nload_initial = True\nsave_initial = True\n"
                    "[OUTPUT]\n", encoding="utf-8")
    settings = tmp_path / "runs" / "trial.ini"
    output = tmp_path / "runs" / "trial-out"
    prepare_run(base, root, meteo, prior, settings, output,
                date(1999, 1, 1), date(2000, 12, 31))
    assert output.is_dir()
    generated = settings.read_text(encoding="utf-8")
    assert base.read_text(encoding="utf-8") == source + (
        "[FILE_PATHS]\nPathRoot = original\nPathMeteo = original\n"
        "PathOut = original\nMaskMap = old\n"
        "[TIME]\nStepStart = old\nStepEnd = old\n"
        "[INITIAL]\nload_initial = True\nsave_initial = True\n"
        "[OUTPUT]\n")
    assert "includeWetlands = True" in generated
    assert generated.index("includeWetlands = True") < generated.index("[FILE_PATHS]")
    assert "calc_evaporation = True" in generated
    assert "load_initial = False" in generated
    assert "StepStart = 01/01/1999" in generated
    assert "StepEnd = 31/12/2000" in generated
    assert "wetlandFractionFile = " in generated
    assert "wetVegSaturatedFrac" in generated
    assert "wetlandRiverInflowM3" in generated
    assert "wetlandLocalInflowM3" in generated
    assert "QAirMaps = " in generated
    with pytest.raises(FileExistsError):
        prepare_run(base, root, meteo, prior, settings, output,
                    date(1999, 1, 1), date(2000, 12, 31))
