from cwatm.management_modules.data_handling import *
import numpy as np


class wetlands_extent(object):

    """
    Dynamic wetland extent module (DWES-like) at DAILY time step.

    Unit conventions (STRICT, as requested):
    - All fluxes provided by CWatM to this module are in [m/day]:
        Precipitation, ETRef, runoff (and any future I_gc)
    - Wetland volume V is [m3]
    - cellArea (A_gc) is [m2]
    - Wetland fraction f_w is [-] in [0,1]
    - Internal hydraulic velocity uses [m/s], with DtSec = 86400 s/day.

    Key equations:
    - Sub-grid slope: s(f) = (1 - (1-f)^(1/b)) * (smax-smin) + smin
    - Wetland depth: h = V / (f_w*A_gc)
    - Inflow: I_w = I_gc * f_w^z      (I_gc in m/day)
    - Velocity: v = c * h^(2/3) * s^(1/2)   (v in m/s)
    - Timescale: k = Δx / v           (k in s)
    - Outflow (depth form): O_w = (V/A_gc) * (DtSec/k)   (m/day)
    - Volume update: ΔV = (P_w - ET_w - D_w + I_w - O_w) * A_gc
    - Extent update: Δf = (ΔV/V) * τ / f_w,   τ = 1/(1 + κ*s(f))
    """

    # ---- numerical stability ----
    _EPS_F = 1e-8       # avoid division by zero in f_w
    _EPS_V = 1e-12      # avoid division by zero in V
    _EPS_S = 1e-12      # min slope
    _EPS_H = 1e-8       # min depth [m]
    _DF_MAX = 0.01      # max daily change in f_w (stability limiter)

    def __init__(self, model):
        self.var = model.var
        self.model = model
    
    def initial(self):

        if not checkOption("wetland_extent"):
            return

        # ---- static maps ----
        topo_fp = cbinding('dwes_topo_params') 
        sy_fp = cbinding('hwsd_sy_sub_integrate')  

        self.var.b = readnetcdfWithoutTime(topo_fp, "b").astype(np.float32)
        self.var.s_min = readnetcdfWithoutTime(topo_fp, "smin").astype(np.float32)
        self.var.s_max = readnetcdfWithoutTime(topo_fp, "smax").astype(np.float32)
        self.var.sy = readnetcdfWithoutTime(sy_fp, "sy").astype(np.float32)

        soilVars = ['f_w','V','h','s_fw','dS_depth']
        for variable in soilVars:
                vars(self.var)[variable] = globals.inZero.copy()

        # ---- parameters (ideally read from ini/settings) ----
        # If you have getOption available, you can replace these three lines.
        self.var.kappa   = float(cbinding('kappa'))
        self.var.z_local = float(cbinding('z_local'))
        self.var.z_river = float(cbinding('z_river'))
        self.var.c    = float(cbinding('cwet'))
        
        # ---- geometry ----
        self.var.A_gc = self.var.cellArea.astype(np.float64)   # [m2]
        self.var.s_range = (self.var.s_max - self.var.s_min).astype(np.float32)

        # validity mask
        topo_ok = (
            np.isfinite(self.var.b) &
            np.isfinite(self.var.s_min) &
            np.isfinite(self.var.s_max) &
            np.isfinite(self.var.s_range) &
            (self.var.s_range > 0)
        )
        self.var._wet_ok = topo_ok.astype(np.bool_)

        # ---- initial states ----
        f0 = readnetcdfWithoutTime(cbinding('glwd_frac'), "glwd_fraction")
        V0 = f0*self.var.A_gc

        self.var.f_w = np.clip(f0.astype(np.float32), 0.0, 1.0)
        self.var.V   = np.zeros_like(self.var.A_gc, dtype=np.float64)

        # disable invalid cells
        self.var.f_w = np.where(self.var._wet_ok, self.var.f_w, 0.0).astype(np.float32)
        self.var.V   = np.where(self.var._wet_ok, self.var.V,   0.0).astype(np.float64)

        # Optional: exclude permanent open water (HydroLakes) to avoid double counting
        # If waterBodyArea exists and is [m2]:
        if hasattr(self.var, "waterBodyArea"):
            f_open = np.clip(self.var.waterBodyArea.astype(np.float64) / self.var.A_gc, 0.0, 1.0)
            self.var._f_openwater = f_open
        else:
            self.var._f_openwater = None

    # ---------------- helpers ----------------
    def _s_of_f(self, f):
        """s(f): consistent with your DEM-fit convention."""
        f = np.clip(f, 0.0, 1.0).astype(np.float64)
        b = np.maximum(self.var.b.astype(np.float64), 1e-6)
        s = (1.0 - np.power(1.0 - f, 1.0 / b)) * self.var.s_range.astype(np.float64) + self.var.s_min.astype(np.float64)
        return np.maximum(s, self._EPS_S)

    def _wet_area(self, f):
        return np.maximum(f, self._EPS_F) * self.var.A_gc  # [m2]

    def _height(self, V, f):
        return np.maximum(V / self._wet_area(f), 0.0)      # [m]

    # ---------------- main loop ----------------
    def dynamic(self):
        if not checkOption("wetland_extent"):
            return

        ok = self.var._wet_ok
        if not np.any(ok):
            return

        # Daily step: use seconds per day ONLY for converting k(s) to daily depth
        DtSec = 86400.0

        # states
        f = self.var.f_w.astype(np.float64)
        V = self.var.V.astype(np.float64)

        # apply topo mask
        f = np.where(ok, f, 0.0)
        V = np.where(ok, V, 0.0)

        # optional open-water exclusion (HydroLakes)
        if self.var._f_openwater is not None:
            fmax = 1.0 - self.var._f_openwater
            f = np.minimum(f, fmax)

        # effective fraction for computations
        f_eff = np.maximum(f, self._EPS_F)

        # slope and depth
        s_fw = self._s_of_f(f_eff)
        h = np.maximum(self._height(V, f_eff), self._EPS_H)

        # ---------------- fluxes (ALL in m/day) ----------------
        # Precipitation and ETRef are assumed [m/day] as you stated
        P = np.asarray(getattr(self.var, "Precipitation", 0.0), dtype=np.float64)
        ETref = np.asarray(getattr(self.var, "ETRef", 0.0), dtype=np.float64)

        # Apply only on wetland area -> expressed as cell-mean depth [m/day]
        P_w  = P * f_eff
        # P_w=0
        ET_w = ETref * f_eff

        # Lateral inflow: strictly in m/day (cell-mean depth)
        I_gc = np.asarray(getattr(self.var, "I_gc", 0.0), dtype=np.float64)  # [m/day], MUST be provided externally
        I_local = np.asarray(getattr(self.var, "I_local", 0.0), dtype=np.float64)  # [m/day], MUST be provided externally
        I_river = np.asarray(getattr(self.var, "I_river", 0.0), dtype=np.float64)  # [m/day], MUST be provided externally
        # I_w = I_gc * np.power(f_eff, float(self.var.z))  # [m/day]
        I_w = I_local * f_eff**self.var.z_local  +  I_river * f_eff**self.var.z_river

        # Lateral outflow:
        # v [m/s] using Manning-like form
        c = float(self.var.c)
        v = c * np.power(h, 2.0/3.0) * np.power(s_fw, 0.5)  # [m/s]

        # characteristic length Δx: prefer chanLength if available, else sqrt(area)
        delta_x = getattr(self.var, "chanLength", None)
        if delta_x is None:
            delta_x = np.sqrt(self.var.A_gc)   # [m]
        delta_x = np.asarray(delta_x, dtype=np.float64)

        k = np.where(v > 0, delta_x / v, np.inf)          # [s]
        S_depth = V / self.var.A_gc                       # [m] (cell-mean storage depth)
        O_w = np.where(np.isfinite(k), S_depth * (DtSec / np.maximum(k, 1e-6)), 0.0)  # [m/day]

        self.var.wet_in_m  = I_w.astype(np.float32)
        self.var.wet_out_m = O_w.astype(np.float32)
        # Drainage (D_w): daily depth loss over cell
        # This is the only term not uniquely defined by the DWES paper in your current CWatM coupling.
        # Keep it minimal & explicit:
        #   D_w = alpha * h * f_w, where alpha is per-day drainage coefficient.
        alpha_d = 0.005  # [1/day] placeholder; can be sensitivity-tested
        D_w = alpha_d * h * f_eff  # [m/day]

        # ---------------- volume update (m3/day) ----------------
        dS = (P_w - ET_w - D_w + I_w - O_w)  # [m/day] cell-mean depth change
        dV = dS * self.var.A_gc              # [m3/day]

        V_new = V + dV
        V_new = np.maximum(V_new, 0.0)

        # ---------------- extent update: STRICT paper-like form ----------------
        # τ = 1/(1 + κ*s)
        kappa = float(self.var.kappa)
        tau = 1.0 / (1.0 + kappa * s_fw)

        V_safe = np.maximum(V, self._EPS_V)
        df = (dV / V_safe) * tau / f_eff

        # stability limiter
        df = np.clip(df, -self._DF_MAX, self._DF_MAX)

        f_new = np.clip(f + df, 0.0, 1.0)

        # re-apply constraints
        if self.var._f_openwater is not None:
            f_new = np.minimum(f_new, 1.0 - self.var._f_openwater)

        f_new = np.where(ok, f_new, 0.0)
        V_new = np.where(ok, V_new, 0.0)

        # write back
        self.var.f_w = f_new.astype(np.float32)
        self.var.V   = V_new.astype(np.float64)

        self.var.f_w = np.maximum(self.var.f_w, 0.0)
        
        # self.var.runoff_conc = self.var.runoff_conc + O_w.astype(self.var.runoff_conc.dtype)
   
