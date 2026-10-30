"""Prepare an aligned global wetland candidate map from fusion and GLWD v2.

The retained fusion classification is read by TIFF blocks to avoid a huge
in-memory raster. GLWD v2 natural inland classes supplement its unmapped
coverage. The output is a spatial prior, not simulated wetland area.
"""

import argparse
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.transform import from_origin
from rasterio.windows import Window

try:
    import tifffile
except ImportError:  # Optional optimization for the sparse fused GeoTIFF.
    tifffile = None


GLWD_VEG_CLASSES = (8, 9, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
GLWD_FLOOD_CLASSES = (10, 11, 12, 13, 16, 17)
FUSION_TYPED_VEG_CLASSES = (1, 2, 3, 4)
FUSION_FLOOD_CLASSES = (5,)
FUSION_UNSPECIFIED_CLASSES = (17,)


def _grid(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.size < 1 or lat.size < 1:
        raise ValueError("Empty clone coordinates")
    dx = 0.5 if lon.size == 1 else lon[1] - lon[0]
    dy = 0.5 if lat.size == 1 else lat[0] - lat[1]
    if (dx <= 0 or dy <= 0 or not np.allclose(np.diff(lon), dx)
            or not np.allclose(np.diff(lat), -dy)):
        raise ValueError("Clone must be regular with descending latitude")
    return from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy), dx, dy


def _runs(indices):
    starts = np.r_[0, np.flatnonzero(np.diff(indices)) + 1]
    stops = np.r_[starts[1:], len(indices)]
    return [(int(indices[a]), slice(int(a), int(b))) for a, b in zip(starts, stops)]


def aggregate_fusion(source, lon, lat, progress=None):
    """Aggregate mapped inland wetland, excluded, and unknown pixels.

    Source-confirmed mixed class 17 is provisionally assigned to the
    vegetated candidate and also retained in a separate provenance layer.
    """
    if source.crs != rasterio.crs.CRS.from_epsg(4326) or source.count != 1:
        raise ValueError("Fusion source must be one-band EPSG:4326")
    transform, dx, dy = _grid(lon, lat)
    height, width = len(lat), len(lon)
    source_rows = np.arange(source.height)
    source_cols = np.arange(source.width)
    row_centers = source.transform.f + (source_rows + 0.5) * source.transform.e
    col_centers = source.transform.c + (source_cols + 0.5) * source.transform.a
    target_rows = np.floor((transform.f - row_centers) / dy).astype(np.int32)
    target_cols = np.floor((col_centers - transform.c) / dx).astype(np.int32)
    valid_rows = (target_rows >= 0) & (target_rows < height)
    valid_cols = (target_cols >= 0) & (target_cols < width)
    row_counts = np.bincount(target_rows[valid_rows], minlength=height)
    col_counts = np.bincount(target_cols[valid_cols], minlength=width)
    denominator = np.outer(row_counts, col_counts)
    counts = {key: np.zeros((height, width), dtype=np.float64)
              for key in ("veg", "flood", "unspecified", "excluded")}
    # The retained global TIFF is sparse: absent tiles represent all-zero
    # unmapped pixels. Skipping them avoids decoding millions of empty tiles.
    nonempty_tiles = None
    if tifffile is not None and source.driver == "GTiff" and source.profile.get("tiled"):
        with tifffile.TiffFile(source.name) as tif:
            bytecounts = np.asarray(tif.pages[0].databytecounts)
        block_height, block_width = source.block_shapes[0]
        tile_columns = math.ceil(source.width / block_width)
        expected = tile_columns * math.ceil(source.height / block_height)
        if bytecounts.size == expected:
            nonempty_tiles = bytecounts > 0
    for n, (window_index, window) in enumerate(source.block_windows(1), start=1):
        if nonempty_tiles is not None and not nonempty_tiles[
                window_index[0] * tile_columns + window_index[1]]:
            if progress is not None and n % 100000 == 0:
                progress(n)
            continue
        block = source.read(1, window=window)
        if source.nodata is not None and source.nodata != 0:
            block[block == source.nodata] = 0
        if np.any(block):
            rows = target_rows[window.row_off:window.row_off + window.height]
            cols = target_cols[window.col_off:window.col_off + window.width]
            for row, row_slice in _runs(rows):
                if not 0 <= row < height:
                    continue
                for col, col_slice in _runs(cols):
                    if not 0 <= col < width:
                        continue
                    tile = block[row_slice, col_slice]
                    if not np.any(tile):
                        continue
                    known = np.count_nonzero(tile)
                    unspecified = np.count_nonzero(
                        np.isin(tile, FUSION_UNSPECIFIED_CLASSES))
                    veg = np.count_nonzero(
                        np.isin(tile, FUSION_TYPED_VEG_CLASSES)) + unspecified
                    flood = np.count_nonzero(
                        np.isin(tile, FUSION_FLOOD_CLASSES))
                    counts["veg"][row, col] += veg
                    counts["flood"][row, col] += flood
                    counts["unspecified"][row, col] += unspecified
                    counts["excluded"][row, col] += known - veg - flood
        if progress is not None and n % 100000 == 0:
            progress(n)
    result = {}
    for name, values in counts.items():
        result[name] = np.divide(values, denominator,
                                 out=np.zeros_like(values), where=denominator > 0)
    result["unmapped"] = np.clip(
        1 - result["veg"] - result["flood"] - result["excluded"], 0, 1,
    )
    return result


def aggregate_glwd(glwd_directory, lon, lat, progress=None):
    """Average natural inland GLWD v2 percentages, counting nodata as zero area."""
    transform, dx, dy = _grid(lon, lat)
    groups = {"veg": GLWD_VEG_CLASSES, "flood": GLWD_FLOOD_CLASSES}
    result = {name: np.zeros((len(lat), len(lon)), dtype=np.float64) for name in groups}
    for group, classes in groups.items():
        for class_id in classes:
            path = (Path(glwd_directory)
                    / f"GLWD_v2_0_class_{class_id:02d}_pct.tif")
            if not path.is_file():
                raise FileNotFoundError(path)
            with rasterio.open(path) as source:
                if source.crs != rasterio.crs.CRS.from_epsg(4326):
                    raise ValueError(f"GLWD raster has unexpected CRS: {path}")
                if source.count != 1 or source.nodata is None:
                    raise ValueError(f"GLWD raster lacks a nodata value: {path}")
                fine_x = round(dx / source.transform.a)
                fine_y = round(dy / -source.transform.e)
                start_col = round((source.transform.c - transform.c) / dx)
                start_row = round((transform.f - source.transform.f) / dy)
                cols, rem_x = divmod(source.width, fine_x)
                rows, rem_y = divmod(source.height, fine_y)
                if (fine_x < 1 or fine_y < 1 or rem_x or rem_y
                        or not np.isclose(source.transform.a * fine_x, dx)
                        or not np.isclose(-source.transform.e * fine_y, dy)
                        or not np.isclose(transform.c + start_col * dx,
                                          source.transform.c)
                        or not np.isclose(transform.f - start_row * dy,
                                          source.transform.f)
                        or start_row < 0 or start_col < 0
                        or start_row + rows > len(lat)
                        or start_col + cols > len(lon)):
                    raise ValueError(f"GLWD raster does not align with clone: {path}")
                for row in range(rows):
                    block = source.read(1, window=Window(
                        0, row * fine_y, source.width, fine_y))
                    block[block == source.nodata] = 0
                    if np.any(block > 100):
                        raise ValueError(f"GLWD percentage exceeds 100: {path}")
                    result[group][start_row + row,
                                  start_col:start_col + cols] += (
                        block.reshape(fine_y, cols, fine_x).sum(
                            axis=(0, 2), dtype=np.float64) /
                        (fine_x * fine_y * 100.0))
            if progress is not None:
                progress(class_id)
    if np.any(result["veg"] + result["flood"] > 1.02):
        raise ValueError("GLWD natural-class fractions exceed one")
    return result


def merge_prior(fusion_veg, fusion_flood, fusion_excluded,
                glwd_veg, glwd_flood):
    """Keep the larger total extent and divide it by best-supported type."""
    arrays = [np.asarray(value, dtype=np.float64) for value in (
        fusion_veg, fusion_flood, fusion_excluded, glwd_veg, glwd_flood)]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Prior layers have different shapes")
    if any(np.any(~np.isfinite(array) | (array < -1e-6)) for array in arrays):
        raise ValueError("Prior layers must be finite and nonnegative")
    fv, ff, fx, gv, gf = arrays
    if np.any(fv + ff + fx > 1 + 1e-5) or np.any(gv + gf > 1.02):
        raise ValueError("Source fractions exceed one")
    total = np.minimum(1 - fx, np.maximum(fv + ff, gv + gf))
    preferred_veg = np.maximum(fv, gv)
    preferred_flood = np.maximum(ff, gf)
    preferred_total = preferred_veg + preferred_flood
    veg = np.divide(total * preferred_veg, preferred_total,
                    out=np.zeros_like(total), where=preferred_total > 0)
    flood = total - veg
    return {"veg": veg, "flood": flood, "total": total,
            "unmapped": np.clip(np.minimum(1 - fv - ff - fx, 1 - total), 0, 1)}


def prepare_file(fusion_path, glwd_directory, clone_path, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    with Dataset(clone_path) as clone:
        lon = np.asarray(clone.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(clone.variables["lat"][:], dtype=np.float64)
    with rasterio.open(fusion_path) as source:
        fused = aggregate_fusion(source, lon, lat,
                                 progress=lambda n: print(f"fusion blocks: {n}", flush=True))
    glwd = aggregate_glwd(glwd_directory, lon, lat,
                          progress=lambda n: print(f"GLWD class: {n}", flush=True))
    prior = merge_prior(fused["veg"], fused["flood"], fused["excluded"],
                        glwd["veg"], glwd["flood"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.stem + "-", suffix=".partial.nc",
                                          dir=output.parent)
    os.close(fd)
    try:
        with Dataset(temporary_name, "w") as target:
            target.createDimension("lat", lat.size)
            target.createDimension("lon", lon.size)
            target.createVariable("lat", "f8", ("lat",))[:] = lat
            target.createVariable("lon", "f8", ("lon",))[:] = lon
            layers = {
                "wetlandVegFrac": prior["veg"],
                "wetlandFloodFrac": prior["flood"],
                "wetlandUnmappedFrac": prior["unmapped"],
                "wetlandFusedVegFrac": fused["veg"],
                "wetlandFusedUnspecifiedFrac": fused["unspecified"],
                "wetlandFusedFloodFrac": fused["flood"],
                "wetlandFusedUnmappedFrac": fused["unmapped"],
                "wetlandFusedExcludedFrac": fused["excluded"],
                "wetlandGlwdVegFrac": glwd["veg"],
                "wetlandGlwdFloodFrac": glwd["flood"],
            }
            for name, values in layers.items():
                variable = target.createVariable(name, "f4", ("lat", "lon"),
                                                 zlib=True, complevel=4)
                variable.units = "1"
                variable[:] = values
            target.fusion_source = str(Path(fusion_path).resolve())
            target.glwd_v2_source = str(Path(glwd_directory).resolve())
            target.note = ("Candidate area is a static prior, not simulated area. "
                           "Fusion class 17 is provisionally assigned to the vegetated "
                           "candidate and preserved as wetlandFusedUnspecifiedFrac; "
                           "this does not establish its hydrological subtype. "
                           "GLWD classes 8-27 only, grouped by hydrological type.")
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary_name, output)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--glwd-directory", required=True)
    parser.add_argument("--clone", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_file(args.fusion, args.glwd_directory, args.clone, args.output)


if __name__ == "__main__":
    main()
