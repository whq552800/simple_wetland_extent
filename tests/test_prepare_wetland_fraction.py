import numpy as np
import pytest
import rasterio
from netCDF4 import Dataset
from rasterio.transform import from_origin

from tools.prepare_wetland_fraction import aggregate_classes, prepare_file


def test_area_fractions_keep_unmapped_separate_from_open_water():
    classes = np.array(
        [
            [1, 2, 13, 13],
            [0, 5, 13, 13],
            [3, 4, 0, 0],
            [5, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    fractions = aggregate_classes(
        classes,
        from_origin(0, 1, 0.25, 0.25),
        from_origin(0, 1, 0.5, 0.5),
        (2, 2),
    )

    np.testing.assert_allclose(fractions["wetlandVegFrac"], [[0.5, 0], [0.5, 0]])
    np.testing.assert_allclose(fractions["wetlandFloodFrac"], [[0.25, 0], [0.25, 0]])
    np.testing.assert_allclose(fractions["wetlandUnmappedFrac"], [[0.25, 0], [0.25, 1]])
    np.testing.assert_allclose(fractions["wetlandKnownExcludedFrac"], [[0, 1], [0, 0]])


def test_source_confirmed_mixed_wetland_is_included_as_provisional_veg():
    fractions = aggregate_classes(
        np.array([[17, 0], [5, 13]], dtype=np.uint8),
        from_origin(0, 0.5, 0.25, 0.25),
        from_origin(0, 0.5, 0.5, 0.5),
        (1, 1),
    )
    np.testing.assert_allclose(fractions["wetlandVegFrac"], [[0.25]])
    np.testing.assert_allclose(fractions["wetlandFloodFrac"], [[0.25]])
    np.testing.assert_allclose(fractions["wetlandKnownExcludedFrac"], [[0.25]])


def test_prepare_file_writes_clone_grid_and_refuses_overwrite(tmp_path):
    raster_path = tmp_path / "classes.tif"
    clone_path = tmp_path / "clone.nc"
    output_path = tmp_path / "fractions.nc"
    classes = np.array(
        [[1, 2, 13, 13], [0, 5, 13, 13], [3, 4, 0, 0], [5, 0, 0, 0]],
        dtype=np.uint8,
    )
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 1, 0.25, 0.25),
        nodata=0,
    ) as dataset:
        dataset.write(classes, 1)
    with Dataset(clone_path, "w") as dataset:
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 2)
        dataset.createVariable("lat", "f4", ("lat",))[:] = [0.75, 0.25]
        dataset.createVariable("lon", "f4", ("lon",))[:] = [0.25, 0.75]

    prepare_file(raster_path, clone_path, output_path, (0, 0, 1, 1))
    with Dataset(output_path) as dataset:
        np.testing.assert_allclose(dataset.variables["wetlandVegFrac"][:], [[0.5, 0], [0.5, 0]])
        np.testing.assert_allclose(dataset.variables["wetlandFloodFrac"][:], [[0.25, 0], [0.25, 0]])
        np.testing.assert_allclose(dataset.variables["wetlandUnmappedFrac"][:], [[0.25, 0], [0.25, 1]])
        assert dataset.variables["lat"].shape == (2,)
        assert dataset.variables["lon"].shape == (2,)
    with pytest.raises(FileExistsError):
        prepare_file(raster_path, clone_path, output_path, (0, 0, 1, 1))


def test_invalid_source_does_not_leave_partial_fraction_file(tmp_path):
    raster_path = tmp_path / "wrong_crs.tif"
    clone_path = tmp_path / "clone.nc"
    output_path = tmp_path / "fractions.nc"
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=2, width=2, count=1,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 1, 0.5, 0.5),
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype=np.uint8), 1)
    with Dataset(clone_path, "w") as dataset:
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 2)
        dataset.createVariable("lat", "f4", ("lat",))[:] = [0.75, 0.25]
        dataset.createVariable("lon", "f4", ("lon",))[:] = [0.25, 0.75]
    with pytest.raises(ValueError, match="EPSG:4326"):
        prepare_file(raster_path, clone_path, output_path, (0, 0, 1, 1))
    assert not output_path.exists()
