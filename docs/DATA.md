# Data files

| Input | Use |
| --- | --- |
| CWatM-Earth 30-minute maps | Model grid, soil, routing, and land-cover inputs. |
| GSWP3-W5E5 daily forcing | Meteorological input for the historical experiment. |
| Fused 30 m wetland-class TIFF | Spatial wetland classification used by `tools/prepare_global_wetland_prior.py`; mosaic of European Wetland Map, African wetland classification, MapBiomas Brazil 2020, and GWL_FCS30D 2020. |
| GLWD v2 area-by-class-percent rasters | Natural inland wetland classes 8–27, used to supplement the candidate map; [source](https://doi.org/10.5194/essd-17-2277-2025). |
| `wetlandFractionFile` NetCDF | Grid-aligned candidate fractions `wetlandVegFrac`, `wetlandFloodFrac`, `wetlandUnmappedFrac`. It is an input spatial prior, not simulated monthly extent. |
| WAD2M v2 monthly 0.5° `Fw` | Union-area screening/comparison, 2000–2019; [source](https://doi.org/10.5281/zenodo.5553187). |
| ELM-ECA monthly `wetlandfrac` | Cross-model comparison; model output, not observation. |
| GWL_FCS30D annual classification | Additional spatial/annual comparison after 0.5° aggregation; 2020 source is also in the fused prior, so the comparison is partly dependent; [source](https://doi.org/10.1038/s41597-024-03143-0). |

Raw input and output NetCDF/GeoTIFF files are not included.
