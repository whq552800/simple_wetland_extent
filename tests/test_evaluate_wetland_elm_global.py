"""ELM's percent units and 365-day months must survive global alignment."""

import numpy as np
import pytest
from netCDF4 import Dataset

from tools.evaluate_wetland_elm_global import evaluate


def test_elm_global_comparison_converts_percent_and_reverses_latitude(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name, value in (("wetVegSaturatedFrac", 0.2),
                        ("wetSurfaceInundFrac", 0.1),
                        ("wetlandInundVegFrac", 0.05)):
        with Dataset(model_dir / f"{name}_monthavg.nc", "w") as file:
            file.createDimension("time", 1)
            file.createDimension("lat", 2)
            file.createDimension("lon", 1)
            file.createVariable("lat", "f8", ("lat",))[:] = [1.25, 0.75]
            file.createVariable("lon", "f8", ("lon",))[:] = [0.25]
            time = file.createVariable("time", "f8", ("time",))
            time.units = "days since 2000-01-01"
            time[:] = [30]
            file.createVariable(f"{name}_monthavg", "f4", ("time", "lat", "lon"))[:] = value
    reference = tmp_path / "elm.nc"
    with Dataset(reference, "w") as file:
        file.createDimension("time", 1)
        file.createDimension("lat", 2)
        file.createDimension("lon", 1)
        file.createVariable("lat", "f8", ("lat",))[:] = [0.75, 1.25]
        file.createVariable("lon", "f8", ("lon",))[:] = [0.25]
        time = file.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.calendar = "365_day"
        time[:] = [0]
        variable = file.createVariable("wetlandfrac", "f4", ("time", "lat", "lon"))
        variable.units = "%"
        variable[0] = [[25.0], [50.0]]
    area_path = tmp_path / "area.nc"
    with Dataset(area_path, "w") as file:
        file.createDimension("lat", 2)
        file.createDimension("lon", 1)
        file.createVariable("lat", "f8", ("lat",))[:] = [1.25, 0.75]
        file.createVariable("lon", "f8", ("lon",))[:] = [0.25]
        file.createVariable("cellarea30min", "f4", ("lat", "lon"))[:] = [[3], [1]]
    rows, summary = evaluate(reference, model_dir, area_path, 2000, 2000)
    assert len(rows) == 1
    assert summary["n_pairs"] == 2
    assert summary["weighted_simulated_mean"] == pytest.approx(0.25)
    assert summary["weighted_reference_mean"] == pytest.approx(0.4375)
    assert summary["weighted_bias"] == pytest.approx(-0.1875)
    assert summary["weighted_rmse"] == pytest.approx(np.sqrt((3 * 0.25**2) / 4))
