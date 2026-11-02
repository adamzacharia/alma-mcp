"""Imaging sanity checks: beam areas, Jy/beam <-> K, line-free rms, maximum recoverable scale, flux recovery.

Conventions follow the corpus skills imaging-choices and array-combination.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import j1

from ._util import ARCSEC_RAD, C_CGS, H_CGS, JY_CGS, K_CGS, as_array, load_visibilities, q, result

IMG_PRACTICE = ["imaging-choices"]
MRS_PRACTICE = ["array-combination", "imaging-choices"]
GF = math.pi**2 / (4.0 * math.log(2.0))


def beam_solid_angle_sr(bmaj_arcsec: float, bmin_arcsec: float) -> float:
    """Gaussian beam solid angle Omega = pi bmaj bmin / (4 ln 2), FWHMs in arcsec, result in sr."""
    return math.pi * bmaj_arcsec * bmin_arcsec * ARCSEC_RAD**2 / (4.0 * math.log(2.0))


def beam_and_brightness(
    value: float,
    freq_ghz: float,
    bmaj_arcsec: float,
    bmin_arcsec: float,
    direction: str = "jy_to_k",
    planck: bool = False,
    pixel_arcsec: float | None = None,
) -> dict:
    """Convert between Jy/beam and brightness temperature (K) for a Gaussian beam, and report beam areas.

    Rayleigh-Jeans: T = S c^2 / (2 k nu^2 Omega) = 1.222e6 S(Jy/beam) / (nu_GHz^2 bmaj bmin) with FWHMs in arcsec.
    planck=True uses the full Planck inversion T = (h nu / k) / ln(1 + 2 h nu^3 / (c^2 I_nu)) (no CMB term).
    direction: 'jy_to_k' (value in Jy/beam) or 'k_to_jy' (value in K).
    Limits: T_B equals the gas or dust temperature only for optically thick emission that fills the beam.
    """
    warnings: list[str] = []
    om = beam_solid_angle_sr(bmaj_arcsec, bmin_arcsec)
    nu = freq_ghz * 1e9
    rj = C_CGS**2 / (2 * K_CGS * nu**2 * om) * JY_CGS  # K per Jy/beam
    if direction == "jy_to_k":
        if planck:
            inu = value * JY_CGS / om
            if inu <= 0:
                raise ValueError("Planck inversion needs a positive intensity")
            t = (H_CGS * nu / K_CGS) / math.log1p(2 * H_CGS * nu**3 / (C_CGS**2 * inu))
        else:
            t = value * rj
        res = {"brightness_temperature": q(t, "K"), "rayleigh_jeans_K_per_Jy_beam": rj}
    elif direction == "k_to_jy":
        if planck:
            inu = 2 * H_CGS * nu**3 / C_CGS**2 / math.expm1(H_CGS * nu / (K_CGS * value))
            s = inu * om / JY_CGS
        else:
            s = value / rj
        res = {"flux_density_per_beam": q(s, "Jy/beam"), "rayleigh_jeans_K_per_Jy_beam": rj}
    else:
        raise ValueError("direction must be 'jy_to_k' or 'k_to_jy'")
    res["beam_solid_angle"] = q(om, "sr")
    res["beam_area_arcsec2"] = om / ARCSEC_RAD**2
    if pixel_arcsec:
        res["pixels_per_beam"] = om / ARCSEC_RAD**2 / pixel_arcsec**2
        res["pixels_across_minor_axis"] = bmin_arcsec / pixel_arcsec
        if bmin_arcsec / pixel_arcsec < 3:
            warnings.append("Fewer than 3 pixels across the beam minor axis (papers use about 5).")
    t_ref = res["brightness_temperature"]["value"] if direction == "jy_to_k" else value
    if not planck and t_ref > 0 and H_CGS * nu / K_CGS > 0.1 * t_ref:
        warnings.append("h nu / k is not small compared with T: the Rayleigh-Jeans temperature differs from the Planck one; use planck=True.")
    conv = ["Gaussian beam Omega = pi bmaj bmin / (4 ln 2).",
            "Planck inversion" if planck else "Rayleigh-Jeans: T = 1.222e6 S / (nu_GHz^2 bmaj bmin)"]
    return result(res, conv, warnings, IMG_PRACTICE)


def rms_from_line_free(
    fits_path: str | None = None,
    spectrum: list[float] | None = None,
    line_free_channels: list[list[int]] | None = None,
    rest_freq_ghz: float | None = None,
) -> dict:
    """Noise from line-free channels of a FITS cube (per-channel and median rms, robust MAD and plain std) or of
    a 1D spectrum. line_free_channels: [[start, end], ...] inclusive channel indices (cube channels are ordered
    by increasing velocity). Without it, all channels are used (biased high if the line is extended).
    Limits: primary-beam corrected cubes have noise rising toward the field edge; measure rms before PB
    correction or in the field centre.
    """
    warnings: list[str] = []
    if fits_path:
        from .kinematics import read_cube
        cube = read_cube(fits_path, rest_freq_ghz)
        data = cube["data"]
        unit = cube["bunit"]
    elif spectrum is not None:
        data = as_array(spectrum, "spectrum")[:, None, None]
        unit = "input units"
    else:
        raise ValueError("give fits_path or spectrum")
    idx = np.zeros(data.shape[0], bool)
    if line_free_channels:
        for a, b in line_free_channels:
            idx[int(a):int(b) + 1] = True
    else:
        idx[:] = True
        warnings.append("No line-free channels given: all channels used.")
    sub = data[idx]
    finite = sub[np.isfinite(sub)]
    mad = float(1.4826 * np.median(np.abs(finite - np.median(finite))))
    std = float(np.std(finite))
    per_chan = [float(1.4826 * np.median(np.abs(c[np.isfinite(c)] - np.median(c[np.isfinite(c)])))) for c in sub
                if np.isfinite(c).any()]
    out = {"rms_robust_mad": q(mad, unit), "rms_std": q(std, unit), "n_channels": int(idx.sum()),
           "per_channel_rms_median": float(np.median(per_chan)) if per_chan else None,
           "per_channel_rms_spread": float(np.std(per_chan)) if per_chan else None}
    if std > 1.5 * mad:
        warnings.append("std >> MAD: line emission, artefacts or PB-corrected edges contaminate the noise sample.")
    return result(out, "rms = 1.4826 x median absolute deviation (robust); plain std for comparison", warnings, IMG_PRACTICE)


def largest_angular_scale(
    freq_ghz: float,
    baselines_m: list[float] | None = None,
    b_min_m: float | None = None,
    vis_path: str | None = None,
    uv_unit: str = "lambda",
) -> dict:
    """Maximum recoverable scale (MRS) and angular resolution from baseline lengths.

    Gives both conventions: the older ALMA Technical Handbook / Wilner and Welch (1994) form
    theta_MRS ~ 0.6 lambda / B_min, and the current Handbook form theta_MRS ~ 0.983 lambda / L05 with L05 the 5th
    percentile baseline; resolution ~ 0.574 lambda / L80 (80th percentile). The Handbook percentiles are over
    unique antenna pairs. Input: baselines_m (one projected length in metres per antenna pair), or a visibility
    file (u, v; with antenna id columns ant1/ant2 the percentiles use each pair's mean length, otherwise they are
    over visibility rows and say so), or just b_min_m (then only the 0.6 form).
    Limits: these are rules of thumb for smooth structure; flux on scales near the MRS is already partly lost
    (use flux_recovery for a model estimate) and the actual MRS depends on uv coverage and imaging.
    """
    warnings: list[str] = []
    lam_m = C_CGS / 1e2 / (freq_ghz * 1e9)
    if vis_path:
        vis = load_visibilities(vis_path, uv_unit, freq_ghz if uv_unit == "m" else None)
        b_rows = np.hypot(vis["u"], vis["v"]) * lam_m
        if "ant1" in vis:
            a1, a2 = vis["ant1"].astype(str), vis["ant2"].astype(str)
            pair = np.array([f"{min(x, y)}|{max(x, y)}" for x, y in zip(a1, a2)])
            cross = a1 != a2
            keys, inv = np.unique(pair[cross], return_inverse=True)
            b = np.bincount(inv, weights=b_rows[cross]) / np.bincount(inv)
            pnote = f"percentiles over {keys.size} unique antenna pairs (mean projected length per pair)"
        else:
            b = b_rows
            pnote = (f"percentiles over {b_rows.size} visibility rows, not unique antenna pairs (no ant1/ant2 columns); "
                     "pairs with more rows weigh more")
            warnings.append("L05/L80 computed over visibility rows: add ant1/ant2 columns or pass baselines_m (one per "
                            "antenna pair) for the Handbook per-baseline percentiles.")
    elif baselines_m is not None:
        b = as_array(baselines_m, "baselines_m")
        pnote = f"percentiles over the {b.size} given baselines_m values (assumed one per antenna pair)"
    elif b_min_m:
        b = None
    else:
        raise ValueError("give baselines_m, vis_path or b_min_m")
    out = {"wavelength_mm": lam_m * 1e3}
    rad2as = 1.0 / ARCSEC_RAD
    if b is not None:
        b = b[b > 0]
        bmin = float(b.min())
        l05 = float(np.percentile(b, 5))
        l80 = float(np.percentile(b, 80))
        out.update({"B_min_m": bmin, "L05_m": l05, "L80_m": l80, "B_max_m": float(b.max()),
                    "mrs_0p983_lambda_over_L05_arcsec": 0.983 * lam_m / l05 * rad2as,
                    "resolution_0p574_lambda_over_L80_arcsec": 0.574 * lam_m / l80 * rad2as,
                    "percentile_basis": pnote})
    else:
        bmin = float(b_min_m)
        out["B_min_m"] = bmin
        warnings.append("Only B_min given: the L05-based Handbook MRS needs the baseline distribution.")
    out["mrs_0p6_lambda_over_Bmin_arcsec"] = 0.6 * lam_m / bmin * rad2as
    out["fringe_spacing_lambda_over_Bmin_arcsec"] = lam_m / bmin * rad2as
    conv = ["theta_MRS ~ 0.6 lambda / B_min (older ALMA Technical Handbook, after Wilner and Welch 1994)",
            "theta_MRS ~ 0.983 lambda / L05 and theta_res ~ 0.574 lambda / L80 (current ALMA Technical Handbook)"]
    return result(out, conv, warnings, MRS_PRACTICE)


def flux_recovery(
    source_size_arcsec: float,
    freq_ghz: float,
    b_min_m: float,
    shape: str = "gaussian",
) -> dict:
    """Rough fraction of a smooth source's total flux still seen on the shortest baseline.

    gaussian (size = FWHM): V(q)/V(0) = exp(-pi^2 theta^2 q^2 / (4 ln 2)); uniform_disk (size = diameter):
    2 J1(pi D q)/(pi D q); q = B_min / lambda. Also gives the source sizes at which this fraction falls to 0.5
    and 0.1. This is a note, not a measurement: actual recovery depends on the full uv coverage and on
    deconvolution, and is usually worse for extended emission. Remedy: add ACA 7 m and total power data.
    """
    lam_m = C_CGS / 1e2 / (freq_ghz * 1e9)
    qmin = b_min_m / lam_m

    def frac(size_as):
        th = size_as * ARCSEC_RAD
        if shape == "gaussian":
            return math.exp(-GF * th**2 * qmin**2)
        if shape == "uniform_disk":
            x = math.pi * th * qmin
            return 1.0 if x < 1e-9 else 2 * j1(x) / x
        raise ValueError("shape must be 'gaussian' or 'uniform_disk'")

    f = frac(source_size_arcsec)
    sizes = {}
    for level in (0.5, 0.1):
        try:
            hi = 1.0
            while frac(hi) > level and hi < 1e5:
                hi *= 2
            sizes[f"size_for_{level}_arcsec"] = brentq(lambda s: frac(s) - level, 1e-6, hi)
        except ValueError:
            sizes[f"size_for_{level}_arcsec"] = None
    out = {"visible_fraction_at_shortest_baseline": f, "q_min_klambda": qmin / 1e3, **sizes}
    warnings = ["Heuristic only: true flux recovery depends on uv coverage and imaging; compare with single-dish or ACA/TP fluxes."]
    if f < 0.5:
        warnings.append("Less than half of the flux is sampled at the shortest baseline: expect substantial missing flux.")
    return result(out, f"{shape} visibility evaluated at q = B_min / lambda", warnings, MRS_PRACTICE)
