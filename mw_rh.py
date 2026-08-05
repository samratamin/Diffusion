"""
mw_rh.py — Hydrodynamic radius and molecular-weight calculations from
diffusion coefficients (DOSY / PGSE NMR).

Three families of methods, all derived from the Stokes–Einstein relation:

    D = k_B * T / (6 * pi * eta * R_H)
    R_H ~ M^alpha           (Mark–Houwink / Rouse–Zimm scaling)
    =>  D = K * M^(-alpha)
    =>  log D = log K - alpha * log M

References (canonical):
  * Ruzicka, Pellechia, Benicewicz, Anal. Chem. 95, 7849 (2023).
    DOI: 10.1021/acs.analchem.2c05531
  * Hou, Pearce, Anal. Chem. 93, 7958 (2021).
    DOI: 10.1021/acs.analchem.1c00793
  * Li, Chung, Daeffler, Johnson, Grubbs, Macromolecules 45, 9595 (2012).
    DOI: 10.1021/ma301666x
  * Gong, Hansen, Chen, Macromol. Chem. Phys. 212, 1007 (2011).
    DOI: 10.1002/macp.201000706
  * Voorter et al., Angew. Chem. Int. Ed. 61, e202114536 (2022)
    — universal-calibration viscosity correction (D_corr = D * (T/eta) / (T_ref/eta_ref))

Conventions:
  * D in m^2/s (the unit used everywhere in app.py).
  * eta in Pa*s   (1 cP = 1 mPa*s = 1e-3 Pa*s).
  * T in Kelvin.
  * M in g/mol    (Da).
  * R_H in metres  (caller divides by 1e-9 for nm).
  * T = 298.15 K by default for "room temperature" -- but the user must
    always supply their own experimental T for a publishable result.

The module exposes a single high-level entry point:
    calculate(payload) -> dict
which is what the Flask endpoint in app.py calls.

Pylint/typing: keep public functions small and annotated; everything that is
meant to be displayed in the UI is built as plain JSON-serialisable values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from scipy import constants
from scipy.stats import linregress

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (CODATA 2018 inside scipy.constants)
# ─────────────────────────────────────────────────────────────────────────────
K_B   = constants.k * 1.0             # Boltzmann constant in J/K
N_A   = constants.N_A * 1.0           # Avogadro number in 1/mol
PI    = math.pi

# Sticky bookend for a numerically safe log when callers pass in D ≈ 0
_FLOOR_D = 1e-20   # m^2/s


# ─────────────────────────────────────────────────────────────────────────────
# Solvent viscosity library (numerical values from CRC Handbook, vol. 90+,
# and from Ruzicka 2023 SI Table S3 where relevant).  All values are at
# T = 25 °C unless otherwise noted; caller can apply a temperature correction.
#
# Viscosity units in this table: cP (= mPa·s = 1e-3 Pa·s).
# ─────────────────────────────────────────────────────────────────────────────
SOLVENT_VISCOSITY_CP = {
    # name               : viscosity (cP), source
    "CDCl3 (chloroform-d)": 0.539,    # CRC 99th, cP at 25 °C
    "benzene-d6":           0.604,    # Ruzicka 2023 Table S3
    "DMSO-d6":              1.987,    # CRC 99th
    "acetone-d6":           0.306,    # CRC 99th
    "THF-d8":               0.456,    # CRC 99th
    "D2O":                  1.095,    # NIST
    "toluene-d8":           0.554,    # CRC 99th
    "DMF-d7":               0.802,    # CRC 99th
    "pyridine-d5":          0.879,    # CRC 99th
    "methanol-d4":          0.544,    # CRC 99th
    "acetonitrile-d3":      0.343,    # CRC 99th
    "custom":              None,      # sentinel: user enters their own
}


# ─────────────────────────────────────────────────────────────────────────────
# Polymer calibration library  (D = K * M^(-alpha)  -> logD = logK - alpha*logM)
# Coefficients are taken directly from the cited works; the table is what the
# user sees in the dropdown.  We never silently invent K or α.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PolymerCalibration:
    name: str            # human label
    polymer: str         # short chemical name
    solvent: str         # calibrated against
    K: float             # coefficient (so that D = K * M^(-alpha) with M in g/mol, D in m^2/s)
    alpha: float         # Mark–Houwink exponent
    T_C: float           # temperature the calibration is valid at
    source: str          # short citation

    # Pre-computed convenience: log10(K), so the user sees a stable form.
    @property
    def log10_K(self) -> float:
        return math.log10(self.K)


POLYMER_CALIBRATIONS = [
    # Ruzicka 2023 (PMMA / PS / PB in CDCl3 + 0.5 mg/mL = 25 °C)
    PolymerCalibration(
        name="PMMA · CDCl3 · 25 °C",
        polymer="poly(methyl methacrylate)",
        solvent="CDCl3",
        K=2.5e-9, alpha=0.55,
        T_C=25.0,
        source="Ruzicka 2023, Anal. Chem. 95, 7849",
    ),
    PolymerCalibration(
        name="PS · CDCl3 · 25 °C",
        polymer="polystyrene",
        solvent="CDCl3",
        K=2.6e-9, alpha=0.57,
        T_C=25.0,
        source="Ruzicka 2023, SI",
    ),
    PolymerCalibration(
        name="PB · CDCl3 · 25 °C",
        polymer="polybutadiene",
        solvent="CDCl3",
        K=2.0e-9, alpha=0.55,
        T_C=25.0,
        source="Ruzicka 2023, SI",
    ),

    # Li 2012 -- PS in benzene-d6 at 25 °C
    # Reported equation y = -0.537 x - 7.697  (x = log Mw, y = log D),
    # i.e. log10 K = -7.697  =>  K = 10^-7.697  (with D in m^2/s and M in g/mol).
    PolymerCalibration(
        name="PS · benzene-d6 · 25 °C",
        polymer="polystyrene",
        solvent="benzene-d6",
        K=10 ** -7.697, alpha=0.537,
        T_C=25.0,
        source="Li 2012, Macromolecules 45, 9595",
    ),

    # Hou 2021 -- PMMA · CDCl3 · 25 °C, near "sufficiently dilute" condition
    PolymerCalibration(
        name="PMMA · CDCl3 · dilute · 25 °C",
        polymer="poly(methyl methacrylate)",
        solvent="CDCl3",
        K=10 ** -7.66, alpha=0.56,
        T_C=25.0,
        source="Hou 2021, Anal. Chem. 93, 7958",
    ),

    # Chen 1995 -- PEO in D2O  (pioneering work, 1–500 kDa)
    PolymerCalibration(
        name="PEO · D2O · 25 °C",
        polymer="poly(ethylene oxide)",
        solvent="D2O",
        K=2.40e-8, alpha=0.62,
        T_C=25.0,
        source="Chen 1995, JACS 117, 7965",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core physics
# ─────────────────────────────────────────────────────────────────────────────
def compute_rh(D_m2s: float, T_K: float, eta_Pa_s: float) -> float:
    """Stokes–Einstein hydrodynamic radius in metres.

    R_H = k_B T / (6 pi eta D)

    Args:
        D_m2s:     diffusion coefficient in m^2/s.
        T_K:       temperature in Kelvin (>0).
        eta_Pa_s:  solvent dynamic viscosity in Pa·s (>0).

    Returns:
        R_H in metres.
    """
    D = max(float(D_m2s), _FLOOR_D)
    if T_K <= 0:
        raise ValueError(f"T_K must be > 0 K (got {T_K})")
    if eta_Pa_s <= 0:
        raise ValueError(f"eta must be > 0 Pa·s (got {eta_Pa_s})")
    return K_B * T_K / (6.0 * PI * eta_Pa_s * D)


def rh_error(D_m2s: float, dD_frac: float,
             T_K: float = 298.15, eta_Pa_s: float = 0.539e-3) -> float:
    """1-sigma absolute error on R_H, assuming a fractional error on D.

    R_H = (k_B T) / (6 pi eta D)
    dR_H / dD = -R_H / D
    =>  sigma_R = (R_H / D) * sigma_D  =  R_H * (sigma_D / D)
    """
    if D_m2s <= 0 or dD_frac < 0:
        return float("nan")
    rh = compute_rh(D_m2s, T_K, eta_Pa_s)
    return rh * dD_frac


def compute_mw_from_kalpha(D_m2s: float, K: float, alpha: float) -> float:
    """Inverse Mark–Houwink.

    D = K · M^(-alpha)
    => M = (D / K)^(-1/alpha)

    Args:
        D_m2s: diffusion coefficient in m^2/s.
        K:     coefficient (D and M both in base SI units: m^2/s and g/mol).
        alpha: Mark–Houwink exponent (typically 0.4–0.7 for linear polymers).

    Returns:
        M in g/mol.
    """
    if alpha == 0:
        raise ValueError("alpha cannot be 0")
    D = max(float(D_m2s), _FLOOR_D)
    return (D / K) ** (-1.0 / alpha)


def fit_kalpha(standards: Iterable[dict]) -> dict:
    """Fit K and α from a set of narrow-standard (D, M) pairs.

    Linearised form:  log D = log K - alpha * log M.
    Returns K, alpha, slope (intercept is log K), R² and the per-point
    residuals so the caller can show goodness-of-fit.

    Args:
        standards: iterable of {'d', 'm'} dicts (or tuples), all > 0.

    Returns:
        dict with K, alpha, slope, intercept, r_squared, n, residuals.
    """
    pts = []
    for s in standards:
        if isinstance(s, dict):
            d, m = s.get("d"), s.get("m")
        else:
            d, m = s[0], s[1]
        if d is None or m is None:
            continue
        d = float(d); m = float(m)
        if d > 0 and m > 0:
            pts.append((d, m))

    if len(pts) < 2:
        raise ValueError("Need at least 2 (D, M) standards to fit K and α")

    d_arr = np.array([p[0] for p in pts], dtype=float)
    m_arr = np.array([p[1] for p in pts], dtype=float)
    x = np.log10(m_arr)
    y = np.log10(d_arr)

    res = linregress(x, y)
    slope      = float(res.slope)            # this is "−α"
    intercept  = float(res.intercept)        # this is "log10 K"
    r2         = float(res.rvalue ** 2)

    # Convention: alpha > 0  (so D decreases with M)
    alpha  = -slope
    logK   = intercept
    K      = 10 ** logK

    # Residuals: predicted log D minus observed, per standard
    predicted_logD = slope * x + intercept
    residuals      = (y - predicted_logD).tolist()

    return {
        "K": K,
        "alpha": alpha,
        "log10_K": logK,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r2,
        "n": len(pts),
        "residuals": residuals,
        "points_used": pts,
    }


def compute_mw_from_fit(D_m2s: float, slope: float, intercept: float) -> float:
    """Apply a calibration fit (slope, intercept) to a new D value.

    logD  = slope * logM + intercept
    =>    logM = (logD - intercept) / slope
    """
    if slope == 0:
        raise ValueError("Calibration slope is 0 (constant log D) -- cannot invert")
    D = max(float(D_m2s), _FLOOR_D)
    logM = (math.log10(D) - intercept) / slope
    return 10 ** logM


def viscosity_correct(D_m2s: float, T_K: float, eta_Pa_s: float,
                      T_ref_K: float, eta_ref_Pa_s: float) -> float:
    """Apply the universal-calibration viscosity/T correction from
    Ruzicka 2023 (eq. 7) and Voorter 2022:

        D_corr = D * (T/eta) / (T_ref/eta_ref)

    The corrected D is used in place of the raw D when applying a
    calibration that was obtained in a different solvent / temperature.
    """
    if eta_ref_Pa_s <= 0 or eta_Pa_s <= 0:
        raise ValueError("viscosities must be > 0")
    if T_ref_K <= 0 or T_K <= 0:
        raise ValueError("temperatures must be > 0 K")
    return float(D_m2s) * (T_K / eta_Pa_s) / (T_ref_K / eta_ref_Pa_s)


# ─────────────────────────────────────────────────────────────────────────────
# Payloads for the /api/mw_rh_calc endpoint
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CalcRequest:
    """Validated, deserialised client payload."""

    mode: str                            # 'library' | 'custom_kalpha' | 'standards'
    T_K: float
    eta_Pa_s: float
    D_list: list[float]                  # one per analysis peak
    D_err_frac_list: list[float] | None  # optional fractional 1-sigma

    # Mode-specific (exactly one of the three blocks is populated):
    polymer_calibration: PolymerCalibration | None = None
    custom_K: float | None = None
    custom_alpha: float | None = None
    custom_T_ref_K: float | None = None
    custom_eta_ref_Pa_s: float | None = None  # for viscosity-corrected universal cal
    standards: list[dict] | None = None      # [{'d':..., 'm':...}, ...]

    # Optional peak labels for display only
    peak_labels: list[str] | None = None


def _validate_request(payload: dict) -> CalcRequest:
    """Parse and validate the JSON body sent by the front-end.

    Raises ValueError with a clean message for any user-input problem so the
    Flask layer can return a 400 with a useful error string.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    mode = payload.get("mode")
    if mode not in ("library", "custom_kalpha", "standards"):
        raise ValueError("mode must be one of 'library', 'custom_kalpha', 'standards'")

    # Temperature
    T_K = payload.get("T_K")
    try:
        T_K = float(T_K)
    except (TypeError, ValueError):
        raise ValueError("T_K must be a number (Kelvin)")
    if not (200.0 <= T_K <= 400.0):
        raise ValueError("T_K out of plausible range (200–400 K)")

    # Solvent viscosity
    eta_cP = payload.get("eta_cP")
    try:
        eta_cP = float(eta_cP)
    except (TypeError, ValueError):
        raise ValueError("eta_cP must be a number (cP / mPa·s)")
    if eta_cP <= 0:
        raise ValueError("eta_cP must be positive")
    eta_Pa_s = eta_cP * 1e-3

    # D values
    D_list = payload.get("D_list") or []
    if not isinstance(D_list, list) or len(D_list) == 0:
        raise ValueError("D_list must be a non-empty list of D values (m²/s)")
    cleaned_D = []
    for i, d in enumerate(D_list):
        try:
            d = float(d)
        except (TypeError, ValueError):
            raise ValueError(f"D_list[{i}] is not a number")
        if d <= 0 or not math.isfinite(d):
            raise ValueError(f"D_list[{i}] must be a finite positive number")
        cleaned_D.append(d)

    D_err = payload.get("D_err_frac_list")
    cleaned_D_err = None
    if D_err is not None:
        if not isinstance(D_err, list):
            raise ValueError("D_err_frac_list must be a list")
        if len(D_err) != len(cleaned_D):
            raise ValueError("D_err_frac_list must have the same length as D_list")
        cleaned_D_err = []
        for i, e in enumerate(D_err):
            try:
                e = float(e)
            except (TypeError, ValueError):
                raise ValueError(f"D_err_frac_list[{i}] is not a number")
            if e < 0 or not math.isfinite(e):
                raise ValueError(f"D_err_frac_list[{i}] must be a finite non-negative number")
            cleaned_D_err.append(e)

    labels = payload.get("peak_labels")
    if labels is not None:
        if not isinstance(labels, list) or len(labels) != len(cleaned_D):
            raise ValueError("peak_labels, if provided, must have the same length as D_list")

    req = CalcRequest(
        mode=mode,
        T_K=T_K,
        eta_Pa_s=eta_Pa_s,
        D_list=cleaned_D,
        D_err_frac_list=cleaned_D_err,
        peak_labels=labels,
    )

    # Mode-specific
    if mode == "library":
        idx = payload.get("polymer_calibration_index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            raise ValueError("polymer_calibration_index must be an integer")
        if not (0 <= idx < len(POLYMER_CALIBRATIONS)):
            raise ValueError(f"polymer_calibration_index out of range (0..{len(POLYMER_CALIBRATIONS)-1})")
        cal = POLYMER_CALIBRATIONS[idx]
        req.polymer_calibration = cal

        # Optional universal-calibration viscosity correction when the
        # user's solvent doesn't match the calibration solvent.
        if payload.get("apply_viscosity_correction"):
            req.custom_T_ref_K    = cal.T_C + 273.15
            # eta_ref_pa_s from the same library that the calibration was run in
            ref_key = next((k for k in SOLVENT_VISCOSITY_CP
                            if cal.solvent.lower() in k.lower()), None)
            if ref_key is None or SOLVENT_VISCOSITY_CP[ref_key] is None:
                raise ValueError(
                    f"Reference viscosity for '{cal.solvent}' is not in the "
                    "built-in library; please use 'custom K, α' mode instead."
                )
            req.custom_eta_ref_Pa_s = SOLVENT_VISCOSITY_CP[ref_key] * 1e-3

    elif mode == "custom_kalpha":
        try:
            req.custom_K     = float(payload.get("K"))
            req.custom_alpha = float(payload.get("alpha"))
        except (TypeError, ValueError):
            raise ValueError("K and α must both be numbers")
        if req.custom_K <= 0:
            raise ValueError("K must be > 0")
        if req.custom_alpha <= 0 or req.custom_alpha >= 2:
            raise ValueError("α must be in (0, 2) -- typical 0.3–0.7")

        # Optional reference-solvent viscosity/T for universal calibration
        if payload.get("apply_viscosity_correction"):
            try:
                req.custom_T_ref_K     = float(payload.get("T_ref_K"))
                req.custom_eta_ref_Pa_s = float(payload.get("eta_ref_cP")) * 1e-3
            except (TypeError, ValueError):
                raise ValueError(
                    "T_ref_K and eta_ref_cP must be numbers when "
                    "applying the universal-calibration viscosity correction"
                )

    elif mode == "standards":
        stds = payload.get("standards") or []
        if not isinstance(stds, list) or len(stds) < 2:
            raise ValueError("Need at least 2 (D, M) standards to fit a calibration")
        # Basic validation done inside fit_kalpha; just pass through.
        req.standards = [
            {"d": float(s["d"]), "m": float(s["m"])}
            for s in stds
        ]

    return req


def calculate(payload: dict) -> dict:
    """Top-level entry point used by the Flask endpoint.

    Returns a JSON-serialisable dict with per-peak results + the
    calibration that was actually used (so the UI can display it).
    """
    req = _validate_request(payload)

    out = {
        "ok": True,
        "inputs": {
            "mode": req.mode,
            "T_K": req.T_K,
            "eta_cP": req.eta_Pa_s / 1e-3,
        },
        "per_peak": [],
        "warnings": [],
    }

    if req.mode == "library":
        cal = req.polymer_calibration
        K, alpha = cal.K, cal.alpha
        T_ref, eta_ref = req.custom_T_ref_K, req.custom_eta_ref_Pa_s
        cal_desc = cal.name + " · " + cal.source
        using_universal = bool(req.custom_eta_ref_Pa_s)
        if using_universal:
            cal_desc += (
                f" · universal-cal. viscosity correction "
                f"(T_ref/η_ref = {T_ref:.2f} K / {eta_ref*1e3:.3f} cP)"
            )
    elif req.mode == "custom_kalpha":
        K, alpha = req.custom_K, req.custom_alpha
        T_ref, eta_ref = req.custom_T_ref_K, req.custom_eta_ref_Pa_s
        cal_desc = f"Custom K = {K:.3e}, α = {alpha:.3f}"
        if T_ref is not None and eta_ref is not None:
            cal_desc += (
                f" · universal-cal. viscosity correction "
                f"(T_ref/η_ref = {T_ref:.2f} K / {eta_ref*1e3:.3f} cP)"
            )
            using_universal = True
        else:
            using_universal = False
    else:                                       # 'standards'
        fit = fit_kalpha(req.standards)
        K, alpha = fit["K"], fit["alpha"]
        T_ref, eta_ref = None, None
        using_universal = False
        cal_desc = (
            f"Internal fit on {fit['n']} standards: "
            f"log₁₀ D = {fit['slope']:.4f}·log₁₀ M + {fit['intercept']:.4f} "
            f"(K = {fit['K']:.3e}, α = {fit['alpha']:.4f}, R² = {fit['r_squared']:.4f})"
        )
        out["fit"] = {
            "K": fit["K"],
            "alpha": fit["alpha"],
            "log10_K": fit["log10_K"],
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r_squared": fit["r_squared"],
            "n": fit["n"],
            "residuals": fit["residuals"],
        }

    out["calibration"] = {
        "K": K, "alpha": alpha, "description": cal_desc,
        "T_ref_K": T_ref, "eta_ref_cP": eta_ref / 1e-3 if eta_ref else None,
        "using_universal": using_universal,
    }

    # ── per-peak results
    for i, D in enumerate(req.D_list):
        d_err_frac = (req.D_err_frac_list[i] if req.D_err_frac_list else None)

        # Apply viscosity correction if requested
        D_eff = D
        if using_universal:
            try:
                D_eff = viscosity_correct(D, req.T_K, req.eta_Pa_s, T_ref, eta_ref)
            except ValueError as e:
                out["warnings"].append(str(e))

        # Hydrodynamic radius
        rh       = compute_rh(D, req.T_K, req.eta_Pa_s)
        rh_nm    = rh * 1e9
        rh_err_nm = None
        if d_err_frac is not None:
            rh_err_nm  = rh_error(D, d_err_frac, req.T_K, req.eta_Pa_s) * 1e9

        # Molecular weight
        mw = compute_mw_from_kalpha(D_eff, K, alpha)

        # Conservative 1-sigma error on Mw from linearisation of log10 M
        # (sigma_log10 M ~= sigma_log10 D / alpha)
        mw_err = None
        if d_err_frac is not None and alpha:
            # d(D)/D = d_err_frac; since D = K M^-α, d(ln D) = -α d(ln M)
            #   => d(ln M) = -d(ln D) / α  (fractional)
            sigma_logM = math.log(1 + d_err_frac) / alpha    # ln
            mw_lo  = mw * math.exp(-sigma_logM)
            mw_hi  = mw * math.exp(+sigma_logM)
            mw_err = (mw - mw_lo, mw_hi - mw)

        # Sanity warnings
        peak_warnings = []
        if d_err_frac is not None and d_err_frac > 0.20:
            peak_warnings.append(
                f"D uncertainty is {d_err_frac*100:.0f}% (>20 %) -- "
                "R_H and M_W are order-of-magnitude estimates only."
            )
        # R_H much smaller than typical solvent molecule -> physically suspect
        if rh_nm < 0.10:
            peak_warnings.append(
                f"R_H = {rh_nm:.3f} nm is smaller than typical solvent molecules. "
                "Check solvent viscosity / temperature."
            )
        if rh_nm > 100:
            peak_warnings.append(
                f"R_H = {rh_nm:.1f} nm is very large; verify that this signal "
                "is a polymer rather than a small-molecule impurity."
            )

        out["per_peak"].append({
            "index":  i,
            "label":  (req.peak_labels[i] if req.peak_labels else str(i + 1)),
            "D":      D,
            "D_effective": D_eff if using_universal else D,
            "D_err_frac": d_err_frac,
            "R_H_nm": rh_nm,
            "R_H_nm_err": rh_err_nm,
            "M_W_gmol":   mw,
            "M_W_gmol_err_minus": mw_err[0] if mw_err else None,
            "M_W_gmol_err_plus":  mw_err[1] if mw_err else None,
            "warnings": peak_warnings,
        })

    # Global warnings
    if req.mode == "standards" and out.get("fit", {}).get("r_squared", 1) < 0.95:
        out["warnings"].append(
            "Calibration fit R² < 0.95 -- the standards may not be on a single "
            "log-linear D(M) trend, or the chosen M range is too narrow."
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Self-test  (run via:  python3 -m mw_rh  or  python3 mw_rh.py)
# ─────────────────────────────────────────────────────────────────────────────
def _selftest():
    """A few canned checks; not exhaustive but enough to catch regressions."""
    # 1. Stokes–Einstein: at 298.15 K in CDCl3 (η=0.539 cP), D = 1e-10 m²/s
    #    yields R_H = kT/(6πηD) ≈ 4.05 nm.  Hand calc:
    #        kT ≈ 4.116e-21 J ; 6π·η·D ≈ 6π·5.39e-4·1e-10 ≈ 1.016e-12
    #        R_H  ≈ 4.05e-9 m  ≈  4.05 nm.
    rh = compute_rh(1.0e-10, 298.15, 0.539e-3)
    assert 3.9e-9 < rh < 4.2e-9, f"expected ~4.05 nm, got {rh*1e9:.3f} nm"

    # 2. Mark–Houwink round trip for PS-in-CDCl3 (Ruzicka 2023)
    K, alpha = 2.6e-9, 0.57
    for M in (5_000, 50_000, 500_000):
        D = K * M ** (-alpha)
        M2 = compute_mw_from_kalpha(D, K, alpha)
        assert abs(M2 - M) / M < 1e-9, f"round-trip failed: {M2} vs {M}"

    # 3. Fit on synthetic standards: known K=1e-8, α=0.5  (round numbers)
    true_K, true_alpha = 1.0e-8, 0.5
    Ms = [1_000, 10_000, 100_000, 1_000_000]
    Ds = [true_K * M ** (-true_alpha) for M in Ms]
    fit = fit_kalpha([{"d": d, "m": m} for d, m in zip(Ds, Ms)])
    assert abs(fit["alpha"] - true_alpha) < 1e-6, fit["alpha"]
    assert abs(fit["K"]      - true_K)     < 1e-16, fit["K"]
    assert fit["r_squared"] > 0.9999

    # 4. Viscosity correction round-trip (Voorter 2022): if D is already
    #    measured in the calibration solvent, correction should be identity.
    D = 2.5e-10
    eta = 0.539e-3
    T = 298.15
    D_corr = viscosity_correct(D, T, eta, T, eta)
    assert abs(D_corr - D) < 1e-20

    # 5. Calibration library is well-formed
    for c in POLYMER_CALIBRATIONS:
        assert c.K > 0 and 0 < c.alpha < 2, c

    print("mw_rh.self-test: ALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
