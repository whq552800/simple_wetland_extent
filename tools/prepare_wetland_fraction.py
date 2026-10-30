"""Prepare regional wetland fractions from the retained categorical fusion TIFF.

Code zero is unmapped, not confirmed upland. Unprocessed target cells are
marked unmapped. This is deliberately regional and does not upsample coarse
source products or infer wetland absence outside the requested bounds.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds


def aggregate_classes(classes, source_transform, target_transform, target_shape):
    """Average mutually exclusive source pixels onto a coarser grid."""
    groups = {
        # Class 17 is confirmed wetland with unresolved hydrological subtype.
        # The vegetated assignment is provisional, matching the global prior.
        "wetlandVegFrac": np.isin(classes, (1, 2, 3, 4, 17)),
        "wetlandFloodFrac": classes == 5,
        "wetlandUnmappedFrac": classes == 0,
        "wetlandKnownExcludedFrac": ~np.isin(classes, (0, 1, 2, 3, 4, 5, 17)),
    }
    fractions = {}
    for name, membership in groups.items():
        destination = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=membership.astype(np.float32),
            destination=destination,
            src_transform=source_transform,
            src_crs="EPSG:4326",
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
        fractions[name] = destination
    return fractions


def prepare_file(raster_path, clone_path, output_path, bounds):
    """Write four fraction maps on the clone grid, processing one cell at a time."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)

    with Dataset(clone_path) as clone:
        lon = np.asarray(clone.variables["lon"][:], dtype=np.float64)
        lat = np.asarray(clone.variables["lat"][:], dtype=np.float64)
    if lon.size < 2 or lat.size < 2:
        raise ValueError("Clone must contain at least two longitudes and latitudes")
    dx, dy = lon[1] - lon[0], lat[0] - lat[1]
    if dx <= 0 or dy <= 0 or not np.allclose(np.diff(lon), dx) or not np.allclose(np.diff(lat), -dy):
        raise ValueError("Clone coordinates must be regular with descending latitude")

    model_transform = from_origin(lon[0] - dx / 2, lat[0] + dy / 2, dx, dy)
    west, south, east, north = bounds
    raw_indices = (
        (west - model_transform.c) / dx,
        (east - model_transform.c) / dx,
        (model_transform.f - north) / dy,
        (model_transform.f - south) / dy,
    )
    indices = tuple(round(value) for value in raw_indices)
    if not all(abs(value - index) < 1e-5 for value, index in zip(raw_indices, indices)):
        raise ValueError("Bounds must follow clone cell edges")
    col_start, col_stop, row_start, row_stop = indices
    if not (0 <= col_start < col_stop <= lon.size and 0 <= row_start < row_stop <= lat.size):
        raise ValueError("Bounds lie outside the clone grid")

    names = (
        "wetlandVegFrac", "wetlandFloodFrac", "wetlandUnmappedFrac",
        "wetlandKnownExcludedFrac",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as source:
        if source.crs != rasterio.crs.CRS.from_epsg(4326) or source.count != 1:
            raise ValueError("Source must be a single-band EPSG:4326 category raster")
        _write_fractions(
            source, output_path, lat, lon, bounds, model_transform,
            (col_start, col_stop, row_start, row_stop), names, raster_path, dx, dy,
        )


def _write_fractions(source, output_path, lat, lon, bounds, model_transform,
                     indices, names, raster_path, dx, dy):
    col_start, col_stop, row_start, row_stop = indices
    with Dataset(output_path, "w") as target:
        target.createDimension("lat", lat.size)
        target.createDimension("lon", lon.size)
        target.createVariable("lat", "f8", ("lat",))[:] = lat
        target.createVariable("lon", "f8", ("lon",))[:] = lon
        variables = {}
        for name in names:
            variable = target.createVariable(name, "f4", ("lat", "lon"), zlib=True)
            variable.units = "1"
            variable[:] = 1.0 if name == "wetlandUnmappedFrac" else 0.0
            variables[name] = variable

        for row in range(row_start, row_stop):
            for col in range(col_start, col_stop):
                cell_west = model_transform.c + col * dx
                cell_east = cell_west + dx
                cell_north = model_transform.f - row * dy
                cell_south = cell_north - dy
                floating_window = from_bounds(
                    cell_west, cell_south, cell_east, cell_north,
                    transform=source.transform,
                )
                left = math.floor(floating_window.col_off)
                top = math.floor(floating_window.row_off)
                right = math.ceil(floating_window.col_off + floating_window.width)
                bottom = math.ceil(floating_window.row_off + floating_window.height)
                window = Window(left, top, right - left, bottom - top)
                classes = source.read(1, window=window, boundless=True, fill_value=0)
                cell_transform = from_origin(cell_west, cell_north, dx, dy)
                fractions = aggregate_classes(
                    classes, source.window_transform(window), cell_transform, (1, 1)
                )
                if sum(float(fractions[name][0, 0]) for name in names) > 1.00001:
                    raise ValueError(f"Class fractions exceed one at row {row}, col {col}")
                for name in names:
                    variables[name][row, col] = fractions[name][0, 0]
        target.source_classification = str(Path(raster_path).resolve())
        target.processed_bounds = ", ".join(map(str, bounds))
        target.note = "Outside processed bounds: unknown fraction=1; class 0: unmapped"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="retained categorical fusion TIFF")
    parser.add_argument("--clone", required=True, help="CWatM 30-arcmin clone NetCDF")
    parser.add_argument("--output", required=True, help="new fraction NetCDF; never overwritten")
    parser.add_argument(
        "--bounds", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True, help="regional bounds aligned to clone cell edges",
    )
    args = parser.parse_args()
    prepare_file(args.source, args.clone, args.output, args.bounds)


if __name__ == "__main__":
    main()
