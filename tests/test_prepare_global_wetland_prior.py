import numpy as np
import rasterio
from rasterio.transform import from_origin

from tools.prepare_global_wetland_prior import (
    GLWD_FLOOD_CLASSES, GLWD_VEG_CLASSES, aggregate_fusion, aggregate_glwd,
    merge_prior,
)


def test_aggregate_fusion_keeps_unmapped_distinct_from_dryland(tmp_path):
    path = tmp_path / "classes.tif"
    classes = np.array([[1, 2, 0, 5], [3, 4, 6, 0]], dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=4, height=2,
                       count=1, dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=from_origin(-180, 90, 0.25, 0.25),
                       tiled=True, blockxsize=16, blockysize=16) as raster:
        raster.write(classes, 1)
    with rasterio.open(path) as source:
        maps = aggregate_fusion(source, np.array([-179.75, -179.25]),
                                np.array([89.75]))
    np.testing.assert_allclose(maps["veg"], [[1.0, 0.0]])
    np.testing.assert_allclose(maps["flood"], [[0.0, 0.25]])
    np.testing.assert_allclose(maps["excluded"], [[0.0, 0.25]])
    np.testing.assert_allclose(maps["unmapped"], [[0.0, 0.5]])


def test_aggregate_fusion_preserves_source_confirmed_other_wetlands(tmp_path):
    path = tmp_path / "other_wetlands.tif"
    classes = np.array([[17, 0], [5, 13]], dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2,
                       count=1, dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=from_origin(-180, 90, 0.25, 0.25)) as raster:
        raster.write(classes, 1)
    with rasterio.open(path) as source:
        maps = aggregate_fusion(source, np.array([-179.75]), np.array([89.75]))
    np.testing.assert_allclose(maps["veg"], [[0.25]])
    np.testing.assert_allclose(maps["unspecified"], [[0.25]])
    np.testing.assert_allclose(maps["flood"], [[0.25]])
    np.testing.assert_allclose(maps["excluded"], [[0.25]])
    np.testing.assert_allclose(maps["unmapped"], [[0.25]])


def test_merge_prior_uses_larger_coverage_without_double_counting():
    maps = merge_prior(
        fusion_veg=np.array([[0.10, 0.30]]),
        fusion_flood=np.array([[0.10, 0.10]]),
        fusion_excluded=np.array([[0.0, 0.20]]),
        glwd_veg=np.array([[0.20, 0.10]]),
        glwd_flood=np.array([[0.20, 0.10]]),
    )
    np.testing.assert_allclose(maps["veg"], [[0.20, 0.30]])
    np.testing.assert_allclose(maps["flood"], [[0.20, 0.10]])
    np.testing.assert_allclose(maps["total"], [[0.40, 0.40]])


def test_glwd_nodata_is_zero_coverage_in_coarse_cell(tmp_path):
    for class_id in GLWD_VEG_CLASSES + GLWD_FLOOD_CLASSES:
        path = tmp_path / f"GLWD_v2_0_class_{class_id:02d}_pct.tif"
        data = (np.array([[20, 255], [0, 0]], dtype="uint8") if class_id == 8
                else np.zeros((2, 2), dtype="uint8"))
        with rasterio.open(path, "w", driver="GTiff", width=2, height=2,
                           count=1, dtype="uint8", crs="EPSG:4326", nodata=255,
                           transform=from_origin(-180, 90, 0.25, 0.25)) as raster:
            raster.write(data, 1)
    layers = aggregate_glwd(tmp_path, np.array([-179.75]), np.array([89.75]))
    np.testing.assert_allclose(layers["veg"], [[0.05]])
    np.testing.assert_allclose(layers["flood"], [[0]])
