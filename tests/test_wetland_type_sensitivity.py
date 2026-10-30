import numpy as np
import pytest
from netCDF4 import Dataset

from tools.wetland_type_sensitivity import (
    create_scenario, reassign_other_to_flood, reassign_other_with_glwd,
)


def test_other_wetland_reassignment_keeps_candidate_total():
    layers = {
        "wetlandFusedVegFrac": np.array([[0.3, 0.0]]),
        "wetlandFusedFloodFrac": np.array([[0.1, 0.2]]),
        "wetlandFusedUnspecifiedFrac": np.array([[0.2, 0.0]]),
        "wetlandFusedExcludedFrac": np.array([[0.0, 0.1]]),
        "wetlandGlwdVegFrac": np.array([[0.0, 0.0]]),
        "wetlandGlwdFloodFrac": np.array([[0.0, 0.5]]),
    }
    result = reassign_other_to_flood(layers)
    np.testing.assert_allclose(result["total"], [[0.4, 0.5]])
    np.testing.assert_allclose(result["veg"], [[0.1, 0.0]])
    np.testing.assert_allclose(result["flood"], [[0.3, 0.5]])


def test_glwd_type_split_uses_local_evidence_and_neutral_fallback():
    layers = {
        "wetlandFusedVegFrac": np.array([[0.4, 0.4]]),
        "wetlandFusedFloodFrac": np.array([[0.0, 0.0]]),
        "wetlandFusedUnspecifiedFrac": np.array([[0.4, 0.4]]),
        "wetlandFusedExcludedFrac": np.array([[0.0, 0.0]]),
        "wetlandGlwdVegFrac": np.array([[0.1, 0.0]]),
        "wetlandGlwdFloodFrac": np.array([[0.3, 0.0]]),
    }
    result, share = reassign_other_with_glwd(layers)
    np.testing.assert_allclose(share, [[0.25, 0.5]])
    np.testing.assert_allclose(result["veg"], [[0.1, 0.2]])
    np.testing.assert_allclose(result["flood"], [[0.3, 0.2]])
    np.testing.assert_allclose(result["total"], [[0.4, 0.4]])


def test_scenario_preserves_source_and_refuses_overwrite(tmp_path):
    source = tmp_path / "vegetated.nc"
    destination = tmp_path / "floodplain.nc"
    source_layers = {
        "wetlandFusedVegFrac": 0.3,
        "wetlandFusedFloodFrac": 0.1,
        "wetlandFusedUnspecifiedFrac": 0.2,
        "wetlandFusedExcludedFrac": 0.0,
        "wetlandGlwdVegFrac": 0.0,
        "wetlandGlwdFloodFrac": 0.0,
        "wetlandVegFrac": 0.3,
        "wetlandFloodFrac": 0.1,
        "wetlandUnmappedFrac": 0.6,
    }
    with Dataset(source, "w") as dataset:
        dataset.createDimension("lat", 1)
        dataset.createDimension("lon", 1)
        for name, value in source_layers.items():
            dataset.createVariable(name, "f4", ("lat", "lon"))[:] = value
    create_scenario(source, destination)
    with Dataset(source) as original, Dataset(destination) as scenario:
        np.testing.assert_allclose(original.variables["wetlandVegFrac"][:], [[0.3]])
        np.testing.assert_allclose(scenario.variables["wetlandVegFrac"][:], [[0.1]])
        np.testing.assert_allclose(scenario.variables["wetlandFloodFrac"][:], [[0.3]])
        np.testing.assert_allclose(scenario.variables["wetlandFusedUnspecifiedFrac"][:], [[0.2]])
        assert scenario.class17_assignment == "floodplain sensitivity scenario"
    with pytest.raises(FileExistsError):
        create_scenario(source, destination)
