"""Create a class-17 floodplain scenario from an existing global wetland prior.

The main prior provisionally assigns source-confirmed but untyped fusion class
17 to vegetation. This alternative keeps its extent and changes only the type
allocation before merging with GLWD. It never overwrites an existing prior.
"""

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from netCDF4 import Dataset

try:
    from tools.prepare_global_wetland_prior import merge_prior
except ModuleNotFoundError:
    from prepare_global_wetland_prior import merge_prior


LAYER_NAMES = (
    "wetlandFusedVegFrac", "wetlandFusedFloodFrac",
    "wetlandFusedUnspecifiedFrac", "wetlandFusedExcludedFrac",
    "wetlandGlwdVegFrac", "wetlandGlwdFloodFrac",
)


def reassign_other_to_flood(layers):
    """Return candidate layers with fused class 17 allocated to floodplain."""
    fv, ff, other, fx, gv, gf = (
        np.asarray(layers[name], dtype=np.float64) for name in LAYER_NAMES
    )
    if np.any(other < 0) or np.any(other > fv + 1e-6):
        raise ValueError("Untyped wetland fraction exceeds fused vegetation")
    return merge_prior(fv - other, ff + other, fx, gv, gf)


def reassign_other_with_glwd(layers):
    """Split class 17 using local GLWD type fractions, or half if absent."""
    fv, ff, other, fx, gv, gf = (
        np.asarray(layers[name], dtype=np.float64) for name in LAYER_NAMES
    )
    if np.any(other < 0) or np.any(other > fv + 1e-6):
        raise ValueError("Untyped wetland fraction exceeds fused vegetation")
    share = np.divide(gv, gv + gf, out=np.full_like(gv, 0.5),
                      where=gv + gf > 0)
    return merge_prior(fv - other + share * other,
                       ff + (1 - share) * other, fx, gv, gf), share


def create_scenario(source_path, output_path, allocation="floodplain"):
    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    with Dataset(source_path) as source:
        layers = {name: np.ma.filled(source.variables[name][:], np.nan)
                  for name in LAYER_NAMES}
    if allocation == "floodplain":
        scenario = reassign_other_to_flood(layers)
        share = np.zeros_like(layers["wetlandFusedUnspecifiedFrac"])
    elif allocation == "glwd":
        scenario, share = reassign_other_with_glwd(layers)
    else:
        raise ValueError("Unknown class-17 allocation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.stem + "-", suffix=".partial.nc",
        dir=output_path.parent,
    )
    os.close(fd)
    try:
        shutil.copyfile(source_path, temporary_name)
        with Dataset(temporary_name, "r+") as target:
            for name, key in (("wetlandVegFrac", "veg"),
                              ("wetlandFloodFrac", "flood"),
                              ("wetlandUnmappedFrac", "unmapped")):
                target.variables[name][:] = scenario[key]
            fraction = target.createVariable(
                "wetlandClass17VegShare", "f4", ("lat", "lon"),
                zlib=True, complevel=4,
            )
            fraction.units = "1"
            fraction.long_name = "Share of fusion class 17 allocated to vegetation"
            fraction[:] = share
            target.class17_assignment = allocation + " sensitivity scenario"
            target.note = (getattr(target, "note", "")
                           + " Class 17 reallocated in this sensitivity scenario; "
                           "source-provenance layers are unchanged.")
        if output_path.exists():
            raise FileExistsError(output_path)
        os.replace(temporary_name, output_path)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allocation", choices=("floodplain", "glwd"),
                        default="floodplain")
    args = parser.parse_args()
    create_scenario(args.source, args.output, args.allocation)


if __name__ == "__main__":
    main()
