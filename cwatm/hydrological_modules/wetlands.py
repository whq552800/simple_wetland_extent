"""Wetland surface storage and separate soil-saturation/inundation diagnostics."""

import numpy as np

from cwatm.management_modules.data_handling import (
    cbinding, checkOption, readnetcdfWithoutTime, returnBool,
)
from cwatm.management_modules.globals import dateVar, metaNetcdfVar


WETLAND_METADATA = {
    "wetlandVegFrac": ("Input candidate vegetated wetland fraction", "1"),
    "wetlandFloodFrac": ("Input candidate floodplain fraction", "1"),
    "wetlandUnmappedFrac": ("Unmapped candidate prior fraction", "1"),
    "wetlandClassifiedFrac": ("Input candidate wetland fraction before water-body exclusion", "1"),
    "wetlandExcludedWaterbodyFrac": ("Candidate wetland fraction excluded by open-water land cover", "1"),
    "wetlandPotentialFrac": ("Available candidate wetland fraction after water-body exclusion", "1"),
    "wetlandStorage": ("Wetland surface water storage", "m3"),
    "wetlandLocalInflowM3": ("Local runoff captured by wetlands per time step", "m3"),
    "wetlandRiverInflowM3": ("River water captured by wetlands per time step", "m3"),
    "wetlandOutflowM3": ("Wetland spill and release per time step", "m3"),
    "wetlandEvapM": ("Wetland surface evaporation over grid cell per time step", "m"),
    "wetSurfaceInundFrac": ("Surface inundation fraction in grid cell", "1"),
    "wetVegSaturatedFrac": ("Saturated vegetated wetland fraction in grid cell", "1"),
    "wetlandInundVegFrac": ("Inundated vegetated wetland fraction in grid cell", "1"),
}


def inundated_fraction(storage_m3, cell_area_m2, potential_frac, mean_depth_m):
    potential_frac = np.asarray(potential_frac)
    flooded = np.divide(
        np.maximum(storage_m3, 0.0), np.asarray(cell_area_m2) * mean_depth_m,
        out=np.zeros_like(potential_frac, dtype=np.float64),
        where=np.asarray(cell_area_m2) > 0,
    )
    return np.minimum(potential_frac, flooded)


def soil_saturated_fraction(veg_frac, soil_storage, soil_capacity, threshold):
    saturation = np.divide(
        soil_storage, soil_capacity,
        out=np.zeros_like(np.asarray(veg_frac), dtype=np.float64),
        where=np.asarray(soil_capacity) > 0,
    )
    return np.asarray(veg_frac) * np.clip((saturation - threshold) / (1 - threshold), 0, 1)


def local_step(storage_m3, runoff_m3, potential_frac, cell_area_m2,
               mean_depth_m, capture_factor, residence_days, ewref_m):
    storage = np.asarray(storage_m3, dtype=np.float64).copy()
    runoff = np.asarray(runoff_m3, dtype=np.float64)
    potential = np.asarray(potential_frac, dtype=np.float64)
    area = np.asarray(cell_area_m2, dtype=np.float64)
    capacity = potential * area * mean_depth_m
    captured = np.maximum(runoff, 0.0) * np.clip(capture_factor * potential, 0, 1)
    storage += captured
    spill = np.maximum(storage - capacity, 0.0)
    storage -= spill
    flooded = inundated_fraction(storage, area, potential, mean_depth_m)
    evaporation = np.minimum(storage, np.maximum(ewref_m, 0.0) * flooded * area)
    storage -= evaporation
    released = np.minimum(storage, storage / residence_days)
    storage -= released
    return {
        "storage_m3": storage,
        "runoff_to_channel_m3": runoff - captured + spill + released,
        "captured_m3": captured,
        "released_m3": released,
        "evap_m3": evaporation,
        "spill_m3": spill,
    }


def river_step(storage_m3, channel_storage_m3, bankfull_storage_m3,
               max_wetland_storage_m3, capture_fraction, step_fraction,
               eligible=None):
    excess = np.maximum(0.0, np.asarray(channel_storage_m3) - bankfull_storage_m3)
    room = np.maximum(0.0, max_wetland_storage_m3 - storage_m3)
    captured = np.minimum(room, excess * capture_fraction * step_fraction)
    if eligible is not None:
        captured = np.where(eligible, captured, 0.0)
    return {"storage_m3": storage_m3 + captured, "captured_m3": captured}


def river_eligible_cells(waterbody_type, reservoir_year=None, model_year=None,
                         use_reservoirs=True):
    """Exclude active lake/reservoir footprints from the river overflow rule.

    CWatM treats future type-2 reservoirs as ordinary river cells until their
    construction year. Type-3 future reservoirs behave as lakes before then.
    """
    body_type = np.asarray(waterbody_type)
    eligible = body_type == 0
    if not use_reservoirs:
        return eligible | (body_type == 2)
    if reservoir_year is not None and model_year is not None:
        built = np.asarray(reservoir_year)
        if built.shape != body_type.shape:
            raise ValueError("Reservoir year grid differs from water-body type grid")
        eligible = eligible | ((body_type == 2) & (built > model_year))
    return eligible


class wetlands:
    def __init__(self, model):
        self.model = model
        self.var = model.var

    def initial(self):
        self.var.includeWetlands = checkOption("includeWetlands", True)
        if not self.var.includeWetlands:
            return

        for name, (long_name, unit) in WETLAND_METADATA.items():
            metaNetcdfVar[name] = {"long_name": long_name, "unit": unit}

        fraction_file = cbinding("wetlandFractionFile")
        fractions = {}
        for name in ("wetlandVegFrac", "wetlandFloodFrac", "wetlandUnmappedFrac"):
            fractions[name] = np.asarray(
                np.ma.filled(readnetcdfWithoutTime(fraction_file, name), np.nan),
                dtype=np.float64,
            )
        veg = fractions["wetlandVegFrac"]
        flood = fractions["wetlandFloodFrac"]
        unmapped = fractions["wetlandUnmappedFrac"]
        if (any(values.shape != self.var.cellArea.shape for values in fractions.values())
                or any(not np.all(np.isfinite(values)) for values in fractions.values())
                or any(np.any((values < -1e-6) | (values > 1 + 1e-6))
                       for values in fractions.values())
                or np.any(veg + flood + unmapped > 1 + 1e-5)):
            raise ValueError("Wetland fractions must be finite, within 0–1, and disjoint")

        self.var.wetlandClassifiedFrac = veg + flood
        self.var.wetlandExcludedWaterbodyFrac = np.zeros_like(veg)
        self.var.wetlandVegFrac = veg
        self.var.wetlandFloodFrac = flood
        self.var.wetlandUnmappedFrac = unmapped
        self.var.wetlandPotentialFrac = veg + flood
        self._candidate_veg = veg.copy()
        self._candidate_flood = flood.copy()
        self._open_water_active = False
        self._pending_capacity_spill = np.zeros_like(veg)

        self.mean_depth = float(cbinding("wetlandMeanDepth"))
        self.residence_days = float(cbinding("wetlandResidenceDays"))
        self.runoff_capture = float(cbinding("wetlandRunoffCapture"))
        self.river_capture = float(cbinding("wetlandRiverCapture"))
        self.sat_threshold = float(cbinding("wetlandSatThreshold"))
        if not (self.mean_depth > 0 and self.residence_days > 0
                and 0 <= self.runoff_capture <= 1 and 0 <= self.river_capture <= 1
                and 0 <= self.sat_threshold < 1):
            raise ValueError("Invalid wetland hydrological parameters")
        if self.river_capture > 0 and not checkOption("includeRouting", True):
            raise ValueError("River wetland exchange requires includeRouting=True")

        self.max_storage = self.var.wetlandPotentialFrac * self.var.cellArea * self.mean_depth
        self.var.wetlandStorage = np.asarray(
            self.var.load_initial("wetlandStorage", default=np.zeros_like(veg)),
            dtype=np.float64,
        )
        if (self.var.wetlandStorage.shape != veg.shape
                or np.any(~np.isfinite(self.var.wetlandStorage))
                or np.any(self.var.wetlandStorage < 0)
                or np.any(self.var.wetlandStorage > self.max_storage + 1e-5)):
            raise ValueError("Wetland warm-start storage lies outside candidate capacity")
        zeros = np.zeros_like(veg)
        self.var.wetlandLocalInflowM3 = zeros.copy()
        self.var.wetlandRiverInflowM3 = zeros.copy()
        self.var.wetlandOutflowM3 = zeros.copy()
        self.var.wetlandEvapM = zeros.copy()
        self.var.wetSurfaceInundFrac = zeros.copy()
        self.var.wetVegSaturatedFrac = zeros.copy()
        self.var.wetlandInundVegFrac = zeros.copy()

    def constrain_open_water(self):
        """Keep candidate wetlands and mapped open water disjoint in each cell."""
        if not self.var.includeWetlands:
            return
        if not hasattr(self, "_candidate_veg"):
            self._candidate_veg = np.asarray(self.var.wetlandVegFrac).copy()
            self._candidate_flood = np.asarray(self.var.wetlandFloodFrac).copy()
            self._pending_capacity_spill = np.zeros_like(self._candidate_veg)
        self._open_water_active = True
        open_water = np.asarray(self.var.fracVegCover[5])
        if (open_water.shape != self.var.wetlandClassifiedFrac.shape or
                np.any(~np.isfinite(open_water)) or
                np.any((open_water < -1e-6) | (open_water > 1 + 1e-6))):
            raise ValueError("Open-water land-cover fraction must be within 0–1")
        candidate = self._candidate_veg + self._candidate_flood
        available = 1 - np.clip(open_water, 0, 1)
        potential = np.minimum(candidate, available)
        scale = np.divide(potential, candidate, out=np.zeros_like(candidate),
                          where=candidate > 0)
        self.var.wetlandVegFrac = self._candidate_veg * scale
        self.var.wetlandFloodFrac = self._candidate_flood * scale
        self.var.wetlandPotentialFrac = potential
        self.var.wetlandExcludedWaterbodyFrac = candidate - potential
        self.max_storage = self.var.wetlandPotentialFrac * self.var.cellArea * self.mean_depth
        spill = np.maximum(self.var.wetlandStorage - self.max_storage, 0)
        self.var.wetlandStorage -= spill
        self._pending_capacity_spill += spill

    def dynamic_local(self):
        if not self.var.includeWetlands:
            return
        if getattr(self, "_open_water_active", False):
            self.constrain_open_water()
        result = local_step(
            self.var.wetlandStorage, self.var.runoff_m3, self.var.wetlandPotentialFrac,
            self.var.cellArea, self.mean_depth, self.runoff_capture,
            self.residence_days, self.var.EWRef,
        )
        self.var.wetlandStorage = result["storage_m3"]
        self.var.runoff_m3 = result["runoff_to_channel_m3"] + self._pending_capacity_spill
        self.var.wetlandLocalInflowM3 = result["captured_m3"]
        self.var.wetlandOutflowM3 = (result["released_m3"] + result["spill_m3"]
                                    + self._pending_capacity_spill)
        self._pending_capacity_spill = np.zeros_like(self.var.wetlandStorage)
        self.var.wetlandRiverInflowM3 = np.zeros_like(self.var.wetlandStorage)
        self.var.wetlandEvapM = result["evap_m3"] * self.var.InvCellArea
        self.var.totalET = self.var.totalET + self.var.wetlandEvapM

    def dynamic_routing(self):
        if not self.var.includeWetlands or self.river_capture == 0:
            return np.zeros_like(self.var.cellArea)
        # The bankfull channel criterion is undefined inside mapped lake and
        # reservoir cells. CWatM routes their outflow through these cells, so
        # applying the river rule there can mislabel lake outflow as overflow.
        waterbody_type = getattr(self.var, "waterBodyTyp", None)
        eligible = None
        if waterbody_type is not None:
            reservoir_year = getattr(self.var, "resYear", None)
            if reservoir_year is None:
                eligible = river_eligible_cells(waterbody_type)
            else:
                use_reservoirs = returnBool("useResAndLakes")
                year = (dateVar["currDate"].year if returnBool("dynamicLakesRes")
                        else int(float(cbinding("fixLakesResYear"))))
                cache_key = (year, use_reservoirs)
                if getattr(self, "_river_mask_key", None) != cache_key:
                    self._river_mask = river_eligible_cells(
                        waterbody_type, reservoir_year, year, use_reservoirs)
                    self._river_mask_key = cache_key
                eligible = self._river_mask
        channel_storage = (
            self.var.channelAlpha * self.var.chanLength
            * np.maximum(self.var.discharge, 0.0) ** self.var.beta
        )
        bankfull_storage = self.var.totalCrossSectionAreaBankFull * self.var.chanLength
        result = river_step(
            self.var.wetlandStorage, channel_storage, bankfull_storage,
            self.max_storage, self.river_capture,
            self.var.dtRouting / self.var.DtSec,
            eligible=eligible,
        )
        self.var.wetlandStorage = result["storage_m3"]
        self.var.wetlandRiverInflowM3 += result["captured_m3"]
        return result["captured_m3"]

    def finalize(self):
        if not self.var.includeWetlands:
            return
        flooded = inundated_fraction(
            self.var.wetlandStorage, self.var.cellArea,
            self.var.wetlandPotentialFrac, self.mean_depth,
        )
        self.var.wetSurfaceInundFrac = flooded
        veg_inundated = np.divide(
            flooded * self.var.wetlandVegFrac, self.var.wetlandPotentialFrac,
            out=np.zeros_like(flooded), where=self.var.wetlandPotentialFrac > 0,
        )
        self.var.wetlandInundVegFrac = veg_inundated
        soil_storage = sum(
            self.var.fracVegCover[cover] * (self.var.w1[cover] + self.var.w2[cover])
            for cover in (0, 1)
        )
        soil_capacity = sum(
            self.var.fracVegCover[cover] * (self.var.ws1[cover] + self.var.ws2[cover])
            for cover in (0, 1)
        )
        self.var.wetVegSaturatedFrac = np.maximum(
            veg_inundated,
            soil_saturated_fraction(
                self.var.wetlandVegFrac, soil_storage, soil_capacity, self.sat_threshold,
            ),
        )
