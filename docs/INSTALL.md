# Install

```powershell
git clone https://github.com/iiasa/CWatM.git CWatM-wetland
cd CWatM-wetland
git checkout a76ca0b02a99e4a7dadfe1a14fe2ad0b9a282d6c
$wetlandRepo = 'PATH/TO/simple_wetland_extent'
git apply --check "$wetlandRepo/patches/cwatm-a76ca0b-integration.patch"
git apply "$wetlandRepo/patches/cwatm-a76ca0b-integration.patch"
Copy-Item "$wetlandRepo/cwatm/hydrological_modules/wetlands.py" 'cwatm/hydrological_modules/wetlands.py'
$env:PYTHONPATH = "$PWD;$wetlandRepo"
python -m pytest -q "$wetlandRepo/tests"
```

```ini
includeWetlands = True
wetlandFractionFile = /path/to/wetland-prior.nc
wetlandMeanDepth = 0.39536107896821243
wetlandResidenceDays = 8.558144077190583
wetlandRunoffCapture = 0.5493425577585285
wetlandRiverCapture = 0.03905598412314429
wetlandSatThreshold = 0.7880805992333092
OUT_MAP_MonthAvg = wetVegSaturatedFrac, wetSurfaceInundFrac, wetlandInundVegFrac, wetlandStorage, wetlandPotentialFrac
```

`wetlandFractionFile` needs `wetlandVegFrac`, `wetlandFloodFrac`, and `wetlandUnmappedFrac` on the CWatM grid. The parameter values above are the exploratory candidate 005.
