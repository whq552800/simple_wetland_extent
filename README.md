# CWatM wetland module sample 20261030

| File | Purpose |
| --- | --- |
| `cwatm/hydrological_modules/wetlands.py` | Simulates wetland storage, runoff and river inflow, evaporation, outflow, and two wetland-area states. |
| `patches/cwatm-a76ca0b-integration.patch` | Integrates the module into CWatM commit `a76ca0b02a99e4a7dadfe1a14fe2ad0b9a282d6c`. |
| `tools/prepare_global_wetland_prior.py` | Builds the grid-aligned candidate wetland map from the fused classification and GLWD v2. |
| `tools/wetland_type_sensitivity.py` | Generates alternative allocations of fused wetland class 17. |
| `tools/prepare_global_wetland_run.py` | Creates global historical-run settings and an output directory. |
| `tools/calibrate_wetlands.py` | Screens five wetland parameters and saves each candidate's settings, run log and WAD2M score. |
| `tools/evaluate_wetland_wad2m.py` | Scores monthly simulated wetland area against WAD2M v2. |
| `tools/evaluate_wetland_wad2m_stable.py` | Computes fixed-cell scores and within-cell monthly correlations. |
| `tools/evaluate_wetland_elm_global.py` | Compares simulated wetland area with ELM-ECA model output. |
| `tools/check_wetland_outputs.py` | Checks wetland storage and area constraints in model outputs. |
| `LICENSE`, `NOTICE` | GPL license and CWatM attribution. |

The earlier full repository version is available at tag `v0.6.0`.
