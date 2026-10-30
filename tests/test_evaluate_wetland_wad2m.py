import numpy as np
import pytest
from netCDF4 import Dataset

from tools.evaluate_wetland_wad2m import align_axis, area_metrics, evaluate, wetland_union


def test_align_axis_matches_coordinates_in_reverse_order():
    np.testing.assert_array_equal(
        align_axis(np.array([51.75, 51.25]), np.array([51.25, 51.75])),
        [1, 0],
    )
    with pytest.raises(ValueError, match="Missing coordinate"):
        align_axis(np.array([51.75, 50.75]), np.array([51.25, 51.75]))


def test_union_removes_only_known_overlap_and_validates_bounds():
    np.testing.assert_allclose(
        wetland_union(np.array([0.3, 0.1]), np.array([0.2, 0.4]),
                      np.array([0.1, 0.0])),
        [0.4, 0.5],
    )
    with pytest.raises(ValueError, match="overlap"):
        wetland_union(np.array([0.1]), np.array([0.1]), np.array([0.2]))


def test_area_metrics_ignore_missing_and_weight_by_cell_area():
    metrics = area_metrics(
        simulated=np.array([[0.2, 0.5], [0.4, 0.1]]),
        observed=np.ma.array([[0.1, 0.4], [0.2, 0.9]],
                             mask=[[0, 0], [0, 1]]),
        cell_area=np.array([1.0, 3.0]),
    )
    assert metrics["n_pairs"] == 3
    assert metrics["weighted_bias"] == pytest.approx(0.12)
    assert metrics["weighted_mae"] == pytest.approx(0.12)
    assert metrics["weighted_rmse"] == pytest.approx(np.sqrt(0.016))


def test_area_metrics_reject_no_overlap():
    with pytest.raises(ValueError, match="No comparable"):
        area_metrics(np.array([[0.2]]), np.ma.masked_all((1, 1)),
                     np.array([1.0]))


def test_evaluate_matches_month_and_opposite_latitude_order(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name, value in (("wetVegSaturatedFrac", 0.2),
                        ("wetSurfaceInundFrac", 0.1),
                        ("wetlandInundVegFrac", 0.05)):
        with Dataset(model_dir / f"{name}_monthavg.nc", "w") as file:
            file.createDimension("time", 1)
            file.createDimension("lat", 2)
            file.createDimension("lon", 2)
            file.createVariable("lat", "f8", ("lat",))[:] = [89.75, 89.25]
            file.createVariable("lon", "f8", ("lon",))[:] = [-179.75, -179.25]
            time = file.createVariable("time", "f8", ("time",))
            time.units = "days since 2000-01-01"
            time[:] = [30]
            file.createVariable(f"{name}_monthavg", "f4", ("time", "lat", "lon"))[:] = value
    reference = tmp_path / "wad.nc"
    with Dataset(reference, "w") as file:
        file.createDimension("time", 1)
        file.createDimension("lat", 2)
        file.createDimension("lon", 2)
        file.createVariable("lat", "f8", ("lat",))[:] = [89.25, 89.75]
        file.createVariable("lon", "f8", ("lon",))[:] = [-179.75, -179.25]
        time = file.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time[:] = [15]
        variable = file.createVariable("Fw", "f4", ("time", "lat", "lon"))
        variable.units = "-"
        variable[0, :, :] = [[0.1, 0.2], [0.3, 0.4]]
    cell_area = tmp_path / "area.nc"
    with Dataset(cell_area, "w") as file:
        file.createDimension("lat", 2)
        file.createDimension("lon", 2)
        file.createVariable("lat", "f8", ("lat",))[:] = [89.75, 89.25]
        file.createVariable("lon", "f8", ("lon",))[:] = [-179.75, -179.25]
        file.createVariable("cellarea30min", "f4", ("lat", "lon"))[:] = 1
    rows, summary = evaluate(reference, model_dir, cell_area, 2000, 2000)
    assert len(rows) == 1
    assert summary["n_pairs"] == 4
    assert summary["weighted_bias"] == pytest.approx(0, abs=1e-7)
    assert summary["weighted_mae"] == pytest.approx(0.1, abs=1e-7)
