# CWatM wetland module

Wetland extension for CWatM (working-tree commit `55cbb4ac06c06066557f4d8a2135712c6f15b41d`). The previous `wetlands_extent.py` version is saved in tag `legacy-dwes-before-replacement-20261030`.

| File | Contents |
| --- | --- |
| `cwatm/hydrological_modules/wetlands.py` | Wetland storage, runoff/river exchange, evaporation, and vegetated-saturation/surface-inundation diagnostics. |
| `patches/cwatm-a76ca0b-integration.patch` | CWatM integration changes for upstream commit `a76ca0b02a99e4a7dadfe1a14fe2ad0b9a282d6c`. |
| `tools/*.py` | Candidate-map preparation, run setup, parameter screening, output checks, evaluation, and plots. |
| `tests/*.py` | Unit tests for the module and tools. |
| `docs/INSTALL.md` | Installation and test commands. |
| `docs/DATA.md` | Required input files and comparison datasets. |
| `docs/RESULTS.md` | Key global 2000–2019 results and figures. |
| `figures/*.png` | Spatial, scatter, and monthly comparison figures. |
| `requirements-analysis.txt` | Python analysis/test dependencies. |
| `CITATION.cff`, `LICENSE`, `NOTICE` | Citation, license, and CWatM attribution. |

Cite this repository and CWatM (Burek et al., 2020, DOI: 10.5194/gmd-13-3267-2020).
