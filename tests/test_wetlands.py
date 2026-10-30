import numpy as np
import pytest
from datetime import datetime
from types import SimpleNamespace

import cwatm.hydrological_modules.wetlands as wetland_module
from cwatm.hydrological_modules.wetlands import (
    inundated_fraction,
    local_step,
    river_eligible_cells,
    river_step,
    soil_saturated_fraction,
)


def test_local_transfer_conserves_water_and_limits_evaporation_and_release():
    result = local_step(
        storage_m3=np.array([0.0, 20.0]),
        runoff_m3=np.array([100.0, 0.0]),
        potential_frac=np.array([0.5, 0.0]),
        cell_area_m2=np.array([100.0, 100.0]),
        mean_depth_m=1.0,
        capture_factor=1.0,
        residence_days=2.0,
        ewref_m=np.array([0.1, 0.1]),
    )
    np.testing.assert_allclose(result["captured_m3"], [50.0, 0.0])
    np.testing.assert_allclose(result["storage_m3"], [22.5, 0.0])
    np.testing.assert_allclose(result["evap_m3"], [5.0, 0.0])
    np.testing.assert_allclose(result["released_m3"], [22.5, 0.0])
    np.testing.assert_allclose(result["spill_m3"], [0.0, 20.0])
    np.testing.assert_allclose(
        result["storage_m3"] + result["runoff_to_channel_m3"] + result["evap_m3"],
        [100.0, 20.0],
    )


def test_area_diagnostics_are_bounded_and_keep_soil_and_surface_distinct():
    np.testing.assert_allclose(
        inundated_fraction(np.array([0.0, 25.0, 100.0]), np.array([100.0] * 3),
                           np.array([0.5] * 3), 1.0),
        [0.0, 0.25, 0.5],
    )
    np.testing.assert_allclose(
        soil_saturated_fraction(
            np.array([0.4, 0.4, 0.4]), np.array([0.7, 0.9, 1.0]),
            np.array([1.0, 1.0, 1.0]), 0.8,
        ),
        [0.0, 0.2, 0.4],
    )


def test_river_capture_only_uses_above_bankfull_and_available_wetland_capacity():
    result = river_step(
        storage_m3=np.array([0.0, 49.0, 0.0]),
        channel_storage_m3=np.array([80.0, 80.0, 40.0]),
        bankfull_storage_m3=np.array([50.0] * 3),
        max_wetland_storage_m3=np.array([50.0] * 3),
        capture_fraction=1.0,
        step_fraction=0.5,
    )
    np.testing.assert_allclose(result["captured_m3"], [15.0, 1.0, 0.0])
    np.testing.assert_allclose(result["storage_m3"], [15.0, 50.0, 0.0])
    assert np.all(result["captured_m3"] <= np.array([30.0, 30.0, 0.0]))


def test_river_capture_excludes_mapped_waterbodies():
    result = river_step(
        storage_m3=np.zeros(3),
        channel_storage_m3=np.full(3, 80.0),
        bankfull_storage_m3=np.full(3, 50.0),
        max_wetland_storage_m3=np.full(3, 50.0),
        capture_fraction=1.0,
        step_fraction=0.5,
        eligible=np.array([True, False, False]),
    )
    np.testing.assert_allclose(result["captured_m3"], [15.0, 0.0, 0.0])


def test_future_reservoir_is_river_until_construction_year():
    types = np.array([0, 1, 2, 2, 3])
    years = np.array([0, 0, 2005, 1995, 2005])
    np.testing.assert_array_equal(
        river_eligible_cells(types, years, 2000),
        [True, False, True, False, False],
    )
    np.testing.assert_array_equal(
        river_eligible_cells(types, years, 2005),
        [True, False, False, False, False],
    )
    np.testing.assert_array_equal(
        river_eligible_cells(types, years, 2000, use_reservoirs=False),
        [True, False, True, True, False],
    )
    with pytest.raises(ValueError, match="year grid"):
        river_eligible_cells(types, np.array([2005]), 2000)


def test_adapter_loads_fraction_and_transfers_local_runoff(monkeypatch):
    config = {
        "wetlandFractionFile": "fractions.nc", "wetlandMeanDepth": "1",
        "wetlandResidenceDays": "2", "wetlandRunoffCapture": "1",
        "wetlandRiverCapture": "0.5", "wetlandSatThreshold": "0.8",
    }
    maps = {
        "wetlandVegFrac": np.array([0.3]),
        "wetlandFloodFrac": np.array([0.2]),
        "wetlandUnmappedFrac": np.array([0.1]),
    }
    monkeypatch.setattr(wetland_module, "checkOption", lambda key, checkfirst=False: True, raising=False)
    monkeypatch.setattr(wetland_module, "cbinding", config.__getitem__, raising=False)
    monkeypatch.setattr(
        wetland_module, "readnetcdfWithoutTime", lambda path, name: maps[name], raising=False,
    )
    var = SimpleNamespace(
        cellArea=np.array([100.0]), InvCellArea=np.array([0.01]),
        load_initial=lambda name, default: default,
        EWRef=np.array([0.1]), runoff_m3=np.array([100.0]), totalET=np.array([0.0]),
        fracVegCover=[np.array([0.5]), np.array([0.5])],
        w1=[np.array([0.9]), np.array([0.9])],
        w2=[np.array([0.0]), np.array([0.0])],
        ws1=[np.array([1.0]), np.array([1.0])],
        ws2=[np.array([0.0]), np.array([0.0])],
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.initial()
    module.dynamic_local()
    module.finalize()
    np.testing.assert_allclose(var.wetlandStorage, [22.5])
    np.testing.assert_allclose(var.runoff_m3, [72.5])
    np.testing.assert_allclose(var.wetlandLocalInflowM3, [50.0])
    np.testing.assert_allclose(var.wetlandEvapM, [0.05])
    np.testing.assert_allclose(var.wetSurfaceInundFrac, [0.225])
    np.testing.assert_allclose(var.wetVegSaturatedFrac, [0.15])


def test_adapter_rejects_overlapping_class_fractions(monkeypatch):
    config = {
        "wetlandFractionFile": "fractions.nc", "wetlandMeanDepth": "1",
        "wetlandResidenceDays": "2", "wetlandRunoffCapture": "1",
        "wetlandRiverCapture": "0.5", "wetlandSatThreshold": "0.8",
    }
    maps = {
        "wetlandVegFrac": np.array([0.8]),
        "wetlandFloodFrac": np.array([0.5]),
        "wetlandUnmappedFrac": np.array([0.0]),
    }
    monkeypatch.setattr(wetland_module, "checkOption", lambda key, checkfirst=False: True, raising=False)
    monkeypatch.setattr(wetland_module, "cbinding", config.__getitem__, raising=False)
    monkeypatch.setattr(
        wetland_module, "readnetcdfWithoutTime", lambda path, name: maps[name], raising=False,
    )
    var = SimpleNamespace(
        cellArea=np.array([100.0]), InvCellArea=np.array([0.01]),
        load_initial=lambda name, default: default,
    )
    with pytest.raises(ValueError, match="fractions"):
        wetland_module.wetlands(SimpleNamespace(var=var)).initial()


def test_open_water_bound_applies_without_active_waterbody_id():
    var = SimpleNamespace(
        includeWetlands=True,
        cellArea=np.array([100.0, 100.0]),
        wetlandVegFrac=np.array([0.3, 0.2]),
        wetlandFloodFrac=np.array([0.1, 0.1]),
        wetlandClassifiedFrac=np.array([0.4, 0.3]),
        wetlandStorage=np.array([0.0, 0.0]),
        fracVegCover=[np.zeros(2) for _ in range(5)] +
                     [np.array([0.8, 0.0])],
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.mean_depth = 1.0
    module.constrain_open_water()
    np.testing.assert_allclose(var.wetlandExcludedWaterbodyFrac, [0.2, 0.0])
    np.testing.assert_allclose(var.wetlandPotentialFrac, [0.2, 0.3])
    np.testing.assert_allclose(var.wetlandVegFrac, [0.15, 0.2])
    np.testing.assert_allclose(module.max_storage, [20.0, 30.0])


def test_open_water_fraction_change_releases_storage_to_runoff():
    var = SimpleNamespace(
        includeWetlands=True,
        cellArea=np.array([100.0]), wetlandVegFrac=np.array([0.3]),
        wetlandFloodFrac=np.array([0.1]), wetlandClassifiedFrac=np.array([0.4]),
        wetlandStorage=np.array([30.0]),
        fracVegCover=[np.zeros(1) for _ in range(5)] + [np.array([0.0])],
        EWRef=np.zeros(1), runoff_m3=np.zeros(1), totalET=np.zeros(1),
        InvCellArea=np.array([0.01]),
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.mean_depth = 1.0
    module.residence_days = 2.0
    module.runoff_capture = 0.5
    module.constrain_open_water()
    var.fracVegCover[5][:] = 0.8
    module.dynamic_local()
    np.testing.assert_allclose(var.wetlandPotentialFrac, [0.2])
    np.testing.assert_allclose(var.wetlandStorage, [10.0])
    np.testing.assert_allclose(var.wetlandOutflowM3, [20.0])
    np.testing.assert_allclose(var.runoff_m3, [20.0])


def test_routing_adapter_captures_only_above_bankfull():
    var = SimpleNamespace(
        includeWetlands=True,
        cellArea=np.array([100.0, 100.0]),
        wetlandStorage=np.array([0.0, 0.0]),
        channelAlpha=np.ones(2), chanLength=np.ones(2), beta=1.0,
        discharge=np.array([80.0, 40.0]),
        totalCrossSectionAreaBankFull=np.array([50.0, 50.0]),
        dtRouting=0.5, DtSec=1.0,
        wetlandRiverInflowM3=np.zeros(2),
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.max_storage = np.array([20.0, 20.0])
    module.river_capture = 1.0
    np.testing.assert_allclose(module.dynamic_routing(), [15.0, 0.0])
    np.testing.assert_allclose(var.wetlandStorage, [15.0, 0.0])


def test_routing_adapter_does_not_capture_lake_or_reservoir_outflow():
    var = SimpleNamespace(
        includeWetlands=True,
        cellArea=np.ones(3), wetlandStorage=np.zeros(3),
        channelAlpha=np.ones(3), chanLength=np.ones(3), beta=1.0,
        discharge=np.full(3, 80.0),
        totalCrossSectionAreaBankFull=np.full(3, 50.0),
        dtRouting=0.5, DtSec=1.0,
        waterBodyTyp=np.array([0, 1, 2]),
        wetlandRiverInflowM3=np.zeros(3),
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.max_storage = np.full(3, 20.0)
    module.river_capture = 1.0
    np.testing.assert_allclose(module.dynamic_routing(), [15.0, 0.0, 0.0])
    np.testing.assert_allclose(var.wetlandStorage, [15.0, 0.0, 0.0])


def test_routing_adapter_updates_reservoir_mask_when_year_changes(monkeypatch):
    monkeypatch.setattr(wetland_module, "returnBool", lambda key: True)
    monkeypatch.setitem(wetland_module.dateVar, "currDate", datetime(2000, 1, 1))
    var = SimpleNamespace(
        includeWetlands=True,
        cellArea=np.ones(2), wetlandStorage=np.zeros(2),
        channelAlpha=np.ones(2), chanLength=np.ones(2), beta=1.0,
        discharge=np.full(2, 80.0),
        totalCrossSectionAreaBankFull=np.full(2, 50.0),
        dtRouting=0.5, DtSec=1.0,
        waterBodyTyp=np.array([0, 2]), resYear=np.array([0, 2005]),
        wetlandRiverInflowM3=np.zeros(2),
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.max_storage = np.full(2, 20.0)
    module.river_capture = 1.0
    np.testing.assert_allclose(module.dynamic_routing(), [15.0, 15.0])
    var.wetlandStorage[:] = 0.0
    monkeypatch.setitem(wetland_module.dateVar, "currDate", datetime(2005, 1, 1))
    np.testing.assert_allclose(module.dynamic_routing(), [15.0, 0.0])


def test_routing_adapter_uses_fixed_reservoir_year_when_requested(monkeypatch):
    monkeypatch.setattr(wetland_module, "returnBool",
                        lambda key: key == "useResAndLakes")
    monkeypatch.setattr(wetland_module, "cbinding", lambda key: "2000")
    monkeypatch.setitem(wetland_module.dateVar, "currDate", datetime(1900, 1, 1))
    var = SimpleNamespace(
        includeWetlands=True,
        cellArea=np.ones(1), wetlandStorage=np.zeros(1),
        channelAlpha=np.ones(1), chanLength=np.ones(1), beta=1.0,
        discharge=np.array([80.0]),
        totalCrossSectionAreaBankFull=np.array([50.0]),
        dtRouting=0.5, DtSec=1.0,
        waterBodyTyp=np.array([2]), resYear=np.array([1995]),
        wetlandRiverInflowM3=np.zeros(1),
    )
    module = wetland_module.wetlands(SimpleNamespace(var=var))
    module.max_storage = np.array([20.0])
    module.river_capture = 1.0
    np.testing.assert_allclose(module.dynamic_routing(), [0.0])
