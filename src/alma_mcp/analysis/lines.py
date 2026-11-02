"""Spectral lines: Gaussian fits, line luminosities, CO gas masses, rotational diagrams and LTE columns.

Conventions follow the corpus skills line-identification, gas-mass-co-conversion and rotational-diagram-lte.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit, least_squares

from ._util import (
    C_CGS, FWHM_TO_SIGMA, H_CGS, K_CGS, M_H_G, M_SUN_G, PC_CM, T_CMB0, as_array, jnu, q, result,
)
from .dust import luminosity_distance_cm

LINEFIT_PRACTICE = ["line-identification", "moment-maps", "redshift-line-search"]
LUM_PRACTICE = ["gas-mass-co-conversion", "star-formation-laws"]
CO_PRACTICE = ["gas-mass-co-conversion", "gmc-decomposition-virial"]
ROT_PRACTICE = ["rotational-diagram-lte", "chemical-modelling"]
SQRT2PI = math.sqrt(2.0 * math.pi)
HELIUM = 1.36


# ------------------------------------------------------------------------------------------ Gaussian lines


def _multi_gauss(x, *p):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(p), 3):
        a, c, s = p[i:i + 3]
        y = y + a * np.exp(-0.5 * ((x - c) / s) ** 2)
    return y


def fit_gaussian_lines(
    x: list[float],
    flux: list[float],
    n_components: int = 1,
    initial_guesses: list[list[float]] | None = None,
    rms: float | None = None,
    x_unit: str = "km/s",
    flux_unit: str = "mJy",
    window_nsigma: float = 3.0,
) -> dict:
    """Fit 1..N Gaussians to a continuum-subtracted spectrum.

    x: channel centres (km/s, or GHz/MHz; integrated fluxes are then in flux_unit x x_unit). flux: per channel.
    rms: per-channel noise; if None it is estimated from the fit residuals (MAD). initial_guesses: optional
    [[amplitude, centre, sigma], ...]; otherwise peaks are found greedily on the residual spectrum.
    For each component: amplitude, centre, sigma, FWHM = 2.3548 sigma, integrated flux = sqrt(2 pi) A sigma with its
    covariance error, and a direct channel sum over centre +/- window_nsigma sigma with error rms dv sqrt(N_chan).
    Limits: assumes independent channels (Hanning-smoothed or oversampled data have correlated noise, so the
    errors are too small by up to sqrt(2)); a Gaussian may be a poor shape for double-peaked disk lines.
    """
    warnings: list[str] = []
    xv = as_array(x, "x")
    yv = as_array(flux, "flux")
    if xv.size != yv.size:
        raise ValueError("x and flux must have the same length")
    order = np.argsort(xv)
    xv, yv = xv[order], yv[order]
    dx = float(np.median(np.abs(np.diff(xv))))
    n = int(n_components)
    if initial_guesses:
        p0 = [float(v) for g in initial_guesses for v in g]
        n = len(p0) // 3
    else:
        p0 = []
        resid = yv.copy()
        for _ in range(n):
            k = int(np.argmax(resid))
            a = resid[k]
            half = a / 2.0
            lo = k
            while lo > 0 and resid[lo] > half:
                lo -= 1
            hi = k
            while hi < len(resid) - 1 and resid[hi] > half:
                hi += 1
            sg = max((xv[hi] - xv[lo]) * FWHM_TO_SIGMA, dx)
            p0 += [a, xv[k], sg]
            resid = resid - _multi_gauss(xv, a, xv[k], sg)
    span = xv[-1] - xv[0]
    lo_b = []
    hi_b = []
    for i in range(n):
        lo_b += [-np.inf, xv[0], dx / 4.0]
        hi_b += [np.inf, xv[-1], span]
    sigma_arr = None if rms is None else np.full_like(yv, float(rms))
    popt, pcov = curve_fit(_multi_gauss, xv, yv, p0=p0, sigma=sigma_arr, absolute_sigma=rms is not None,
                           bounds=(lo_b, hi_b), maxfev=20000)
    model = _multi_gauss(xv, *popt)
    res = yv - model
    rms_resid = float(1.4826 * np.median(np.abs(res - np.median(res))))
    rms_used = float(rms) if rms is not None else rms_resid
    if rms is None:
        warnings.append("rms estimated from fit residuals (MAD); pass the line-free rms for a cleaner error.")
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    comps = []
    for i in range(n):
        a, c, s = popt[3 * i:3 * i + 3]
        ea, ec, es = perr[3 * i:3 * i + 3]
        cov_as = pcov[3 * i, 3 * i + 2]
        integ = SQRT2PI * a * s
        integ_err = SQRT2PI * math.sqrt(max((s * ea) ** 2 + (a * es) ** 2 + 2 * a * s * cov_as, 0.0))
        win = np.abs(xv - c) <= window_nsigma * abs(s)
        nch = int(win.sum())
        direct = float(np.sum(yv[win]) * dx)
        direct_err = rms_used * dx * math.sqrt(max(nch, 1))
        comps.append({
            "amplitude": q(a, flux_unit, ea),
            "centre": q(c, x_unit, ec),
            "sigma": q(abs(s), x_unit, es),
            "fwhm": q(abs(s) / FWHM_TO_SIGMA, x_unit, es / FWHM_TO_SIGMA),
            "integrated_flux_gaussian": q(integ, f"{flux_unit} {x_unit}", integ_err),
            "integrated_flux_channel_sum": q(direct, f"{flux_unit} {x_unit}", direct_err, n_channels=nch,
                                             window=f"centre +/- {window_nsigma} sigma"),
            "peak_snr": a / rms_used if rms_used > 0 else None,
        })
        if rms_used > 0 and integ / max(direct_err, 1e-300) < 3:
            warnings.append(f"Component {i + 1}: integrated S/N < 3; report a 3 sigma upper limit instead.")
        if abs(s) < dx:
            warnings.append(f"Component {i + 1}: sigma below one channel width; the width is not resolved.")
    comps.sort(key=lambda d: d["centre"]["value"])
    dof = max(len(yv) - 3 * n, 1)
    chi2 = float(np.sum((res / rms_used) ** 2)) if rms_used > 0 else None
    out = {"components": comps, "rms_per_channel": q(rms_used, flux_unit), "channel_width": q(dx, x_unit),
           "chi2": chi2, "dof": dof, "reduced_chi2": chi2 / dof if chi2 is not None else None,
           "residual_rms_mad": rms_resid}
    if chi2 is not None and chi2 / dof > 2 and rms is not None:
        warnings.append("Reduced chi2 > 2: consider another component or a non-Gaussian profile.")
    warnings.append("Channel errors assume independent channels; for Hanning-smoothed data inflate them.")
    conv = ["Gaussian: A exp(-(x - x0)^2 / (2 sigma^2)); FWHM = 2 sqrt(2 ln 2) sigma; integral = sqrt(2 pi) A sigma.",
            "Channel-sum error = rms x channel width x sqrt(N_channels)."]
    return result(out, conv, warnings, LINEFIT_PRACTICE)


# ------------------------------------------------------------------------------------------ line luminosity


def line_luminosity(
    integrated_flux_jy_kms: float,
    rest_freq_ghz: float,
    redshift: float = 0.0,
    distance_mpc: float | None = None,
    integrated_flux_err_jy_kms: float | None = None,
    flux_cal_frac: float = 0.0,
    magnification: float = 1.0,
) -> dict:
    """Line luminosity in the Solomon et al. (1997) / Solomon and Vanden Bout (2005) forms.

    L'_line = 3.25e7 S dv nu_obs^-2 D_L^2 (1+z)^-3   [K km/s pc^2]  (S dv in Jy km/s, nu_obs in GHz, D_L in Mpc)
    L_line  = 1.04e-3 S dv nu_obs D_L^2 = 1.04e-3 S dv nu_rest (1+z)^-1 D_L^2   [Lsun]
    and L_line = 3.18e-11 nu_rest^3 L'. For a local source give distance_mpc (z may stay 0); for z > 0 leave
    distance_mpc empty and D_L comes from Planck18 (a given distance wins, with a warning if it disagrees). Lensed sources are
    divided by magnification. Limits: interferometric fluxes can miss extended emission.
    """
    warnings: list[str] = []
    z = float(redshift or 0.0)
    if distance_mpc:
        dl = float(distance_mpc)
        dnote = f"D_L = {dl} Mpc (given)"
        if z > 0:  # a client model may pass a rounded or invented distance next to the redshift
            d_cm, _ = luminosity_distance_cm(z, None)
            d_cosmo = d_cm / (1e6 * PC_CM)
            if abs(dl - d_cosmo) > 0.02 * d_cosmo:
                warnings.append(f"distance_mpc = {dl:g} Mpc was used, but Planck18 gives D_L = {d_cosmo:.1f} Mpc at "
                                f"z = {z}; L' scales as D_L^2, so it is off by {(dl / d_cosmo) ** 2 - 1:+.1%}. "
                                "Omit distance_mpc for a cosmological source.")
    else:
        d_cm, dnote = luminosity_distance_cm(z, None)
        dl = d_cm / (1e6 * PC_CM)
    nu_obs = rest_freq_ghz / (1.0 + z)
    sdv = integrated_flux_jy_kms / magnification
    lp = 3.25e7 * sdv * nu_obs**-2 * dl**2 * (1.0 + z) ** -3
    ll = 1.04e-3 * sdv * nu_obs * dl**2
    err = None
    if integrated_flux_err_jy_kms is not None or flux_cal_frac:
        rel = math.hypot((integrated_flux_err_jy_kms or 0.0) / integrated_flux_jy_kms, flux_cal_frac)
        err = rel
    out = {
        "L_prime": q(lp, "K km/s pc^2", lp * err if err is not None else None),
        "L_line": q(ll, "Lsun", ll * err if err is not None else None),
        "log10_L_prime": math.log10(lp) if lp > 0 else None,
        "observed_frequency_ghz": nu_obs,
        "luminosity_distance_mpc": dl,
    }
    if magnification != 1.0:
        warnings.append(f"Divided by magnification mu = {magnification}; differential lensing between tracers is not handled.")
    conv = ["Solomon et al. 1997: L' = 3.25e7 S dv nu_obs^-2 D_L^2 (1+z)^-3 K km/s pc^2; L = 1.04e-3 S dv nu_obs D_L^2 Lsun (Solomon and Vanden Bout 2005 eq. 1).",
            dnote]
    return result(out, conv, warnings, LUM_PRACTICE)


# ------------------------------------------------------------------------------------------ CO gas mass


def co_gas_mass(
    l_prime_k_kms_pc2: float,
    j_upper: int = 1,
    r_j1: float | None = None,
    ratio_is_flux_density: bool = False,
    alpha_co: float | None = 4.35,
    alpha_includes_helium: bool = True,
    x_co: float | None = None,
    l_prime_err: float | None = None,
    alpha_co_err: float | None = None,
) -> dict:
    """Molecular gas mass from a CO line luminosity.

    L'_CO(1-0) = L'_CO(J-[J-1]) / r_J1, where r_J1 is the brightness-temperature (L') ratio. If you only have a
    flux-density ratio S_J/S_1, set ratio_is_flux_density=True: r_J1 = (S_J/S_1) / J^2 (CO).
    M_mol = alpha_CO L'_CO(1-0). alpha_CO in Msun (K km/s pc^2)^-1: 4.35 Galactic INCLUDING helium (x1.36;
    3.2 without), 0.8 ULIRG/SMG convention (stated with helium in most papers; check).
    Or give x_co (cm^-2 (K km/s)^-1, e.g. 2e20, H2 only): alpha_H2 = X_CO x 2 m_H / (Msun pc^-2), about 3.2 for 2e20,
    and x1.36 for M_mol with helium. Returns both M_mol (with He) and M_H2 (without) with the factor used.
    alpha_co_err is an absolute error on the factor in use: on alpha_co, or on the derived alpha_mol when x_co is
    given (the default alpha_co is ignored then).
    Limits: alpha_CO depends on metallicity and environment (factors of several); r_J1 for mid-J lines is a large
    assumed term; a flux ratio used as r_J1 underestimates the mass by J^2.
    """
    warnings: list[str] = []
    j = int(j_upper)
    if j < 1:
        raise ValueError("j_upper must be >= 1")
    if j == 1:
        r = 1.0
    else:
        if r_j1 is None:
            raise ValueError("j_upper > 1 needs r_j1 (typical r21 0.7-0.8, r31 0.5-0.8; see corpus skill gas-mass-co-conversion)")
        r = float(r_j1) / j**2 if ratio_is_flux_density else float(r_j1)
        if ratio_is_flux_density:
            warnings.append(f"Flux-density ratio {r_j1} converted to brightness ratio r_J1 = {r:.3f} (divided by J^2 = {j**2}).")
        if r > 1.0:
            warnings.append("r_J1 > 1: check the ratio direction, flux vs brightness, or optically thin/super-thermal gas.")
    l10 = l_prime_k_kms_pc2 / r
    if x_co is not None:
        alpha_h2 = x_co * 2.0 * M_H_G / M_SUN_G * PC_CM**2
        alpha_mol = alpha_h2 * HELIUM
        aconv = f"X_CO = {x_co:.3g} cm^-2 (K km/s)^-1 (H2 only) -> alpha_H2 = {alpha_h2:.3f}, alpha_mol (x1.36 He) = {alpha_mol:.3f} Msun (K km/s pc^2)^-1"
    else:
        if alpha_co is None:
            raise ValueError("give alpha_co or x_co")
        if alpha_includes_helium:
            alpha_mol, alpha_h2 = alpha_co, alpha_co / HELIUM
        else:
            alpha_h2, alpha_mol = alpha_co, alpha_co * HELIUM
        aconv = f"alpha_CO = {alpha_co} Msun (K km/s pc^2)^-1, {'includes' if alpha_includes_helium else 'excludes'} helium (factor 1.36)"
    m_mol = alpha_mol * l10
    m_h2 = alpha_h2 * l10
    # alpha_co_err is in the same units as the conversion factor in use: alpha_co as given, or alpha_mol when x_co is set
    alpha_ref = alpha_mol if x_co is not None else alpha_co
    rel = math.hypot((l_prime_err or 0.0) / l_prime_k_kms_pc2,
                     (alpha_co_err or 0.0) / alpha_ref if (alpha_co_err and alpha_ref) else 0.0)
    out = {
        "L_prime_CO10": q(l10, "K km/s pc^2"),
        "r_J1_used": r,
        "M_mol_with_helium": q(m_mol, "Msun", m_mol * rel if rel else None),
        "M_H2_without_helium": q(m_h2, "Msun", m_h2 * rel if rel else None),
        "alpha_mol_with_helium": alpha_mol,
        "alpha_H2_without_helium": alpha_h2,
    }
    if j > 1:
        out["alpha_for_observed_transition_with_helium"] = alpha_mol / r
    if not alpha_co_err:
        warnings.append("No alpha_CO uncertainty given; the conversion factor usually dominates (factor 2 or more).")
    conv = [aconv, "M = alpha_CO x L'_CO(1-0); L'_CO(1-0) = L'_J / r_J1 with r_J1 a brightness-temperature ratio; He factor 1.36."]
    return result(out, conv, warnings, CO_PRACTICE)


# ------------------------------------------------------------------------------------------ LTE helpers


def _partition(t, q_table=None, linear_rotor_b_ghz=None):
    t = np.asarray(t, float)
    if q_table:
        tab = np.asarray(q_table, float)
        tab = tab[np.argsort(tab[:, 0])]
        lt, lq = np.log(tab[:, 0]), np.log(tab[:, 1])
        val = np.exp(np.interp(np.log(t), lt, lq, left=np.nan, right=np.nan))
        if np.any(~np.isfinite(val)):
            # linear extrapolation in log-log outside the table
            slope_lo = (lq[1] - lq[0]) / (lt[1] - lt[0])
            slope_hi = (lq[-1] - lq[-2]) / (lt[-1] - lt[-2])
            ltt = np.log(t)
            val = np.where(ltt < lt[0], np.exp(lq[0] + slope_lo * (ltt - lt[0])), val)
            val = np.where(ltt > lt[-1], np.exp(lq[-1] + slope_hi * (ltt - lt[-1])), val)
        return val
    if linear_rotor_b_ghz:
        return K_CGS * t / (H_CGS * linear_rotor_b_ghz * 1e9) + 1.0 / 3.0
    raise ValueError("give partition_table [[T_K, Q], ...] (e.g. from CDMS/JPL) or linear_rotor_b_ghz")


def upper_level_column(w_k_kms, freq_ghz, a_ul):
    """Optically thin, Rayleigh-Jeans N_u = 8 pi k nu^2 W / (h c^3 A_ul), W in K km/s, result in cm^-2."""
    nu = np.asarray(freq_ghz, float) * 1e9
    return 8.0 * math.pi * K_CGS * nu**2 * (np.asarray(w_k_kms, float) * 1e5) / (H_CGS * C_CGS**3 * np.asarray(a_ul, float))


def jy_kms_to_k_kms(s_jy_kms, freq_ghz, bmaj_arcsec, bmin_arcsec):
    """Rayleigh-Jeans: W(K km/s) = 1.222e6 S / (nu_GHz^2 bmaj bmin), S in Jy/beam km/s, beam FWHMs in arcsec.

    S must be a per-beam integrated intensity (moment 0 value in Jy/beam km/s). A total flux in Jy km/s gives the
    beam-averaged W only for a source smaller than the beam.
    """
    return 1.222e6 * np.asarray(s_jy_kms, float) / (np.asarray(freq_ghz, float) ** 2 * bmaj_arcsec * bmin_arcsec)


def _tau_from_nu(n_u, freq_ghz, a_ul, t, fwhm_kms):
    nu = np.asarray(freq_ghz, float) * 1e9
    phi_peak = 2.0 * math.sqrt(math.log(2.0) / math.pi) / (np.asarray(fwhm_kms, float) * 1e5)  # s/cm
    return C_CGS**3 * np.asarray(a_ul, float) * n_u / (8.0 * math.pi * nu**3) * np.expm1(H_CGS * nu / (K_CGS * t)) * phi_peak


def rotational_diagram(
    lines: list[dict],
    partition_table: list[list[float]] | None = None,
    linear_rotor_b_ghz: float | None = None,
    beam_filling: float = 1.0,
    tau_correction: bool = False,
    flux_cal_frac: float = 0.10,
) -> dict:
    """LTE rotational (population) diagram: fit ln(N_u/g_u) = ln(N_tot/Q(T_rot)) - E_u/T_rot.

    Each line dict: freq_ghz, a_ul (s^-1), e_u_k (K), g_u, w_k_kms (integrated intensity, K km/s) and w_err_k_kms.
    Instead of w_k_kms you can give flux_jy_kms with bmaj_arcsec, bmin_arcsec (converted with the RJ
    1.222e6 factor). flux_jy_kms must be the per-beam integrated intensity in Jy/beam km/s (moment 0 at the
    position), not a spatially integrated flux in Jy km/s, unless the source is smaller than the beam. For tau_correction also give fwhm_kms per line.
    N_u = 8 pi k nu^2 W / (h c^3 A_ul) / beam_filling (thin, RJ, background neglected). Errors on W get
    flux_cal_frac in quadrature. Q(T) from partition_table (log-log interpolation; include vibrational states) or a
    linear rotor approximation Q = kT/(hB) + 1/3. tau_correction fits ln N_tot and T_rot jointly with each line's
    thin-assumption column modelled as N_u (1 - exp(-tau))/tau, tau the Gaussian peak opacity from N_u, T_rot and
    fwhm_kms (Goldsmith and Langer 1999 C_tau).
    Limits: needs >= 3 lines over a range of E_u; T_rot is a lower limit to T_kin for subthermal gas; beam
    filling < 1 and opacity make N a lower limit; blends bias it.
    """
    warnings: list[str] = []
    if len(lines) < 2:
        raise ValueError("need at least two lines (three or more for a meaningful T_rot)")
    if len(lines) < 3:
        warnings.append("Only two lines: T_rot has no degrees of freedom check.")
    nu = np.array([float(l["freq_ghz"]) for l in lines])
    a = np.array([float(l["a_ul"]) for l in lines])
    eu = np.array([float(l["e_u_k"]) for l in lines])
    gu = np.array([float(l["g_u"]) for l in lines])
    w = []
    werr = []
    for l in lines:
        if "w_k_kms" in l:
            w.append(float(l["w_k_kms"]))
            werr.append(float(l.get("w_err_k_kms", 0.0)))
        elif "flux_jy_kms" in l:
            f = jy_kms_to_k_kms(1.0, float(l["freq_ghz"]), float(l["bmaj_arcsec"]), float(l["bmin_arcsec"]))
            w.append(float(l["flux_jy_kms"]) * f)
            werr.append(float(l.get("flux_err_jy_kms", 0.0)) * f)
        else:
            raise ValueError("each line needs w_k_kms (K km/s) or flux_jy_kms + bmaj_arcsec + bmin_arcsec")
    w = np.array(w)
    werr = np.hypot(np.array(werr), flux_cal_frac * w)
    if np.any(w <= 0):
        raise ValueError("integrated intensities must be positive (drop non-detections or use them as limits)")
    if np.ptp(eu) < 20:
        warnings.append(f"E_u spans only {np.ptp(eu):.0f} K: T_rot is poorly constrained.")
    nu_thin = upper_level_column(w, nu, a) / beam_filling
    ctau = np.ones_like(w)
    fwhm = np.array([float(l.get("fwhm_kms", np.nan)) for l in lines])
    if tau_correction and np.any(~np.isfinite(fwhm)):
        raise ValueError("tau_correction needs fwhm_kms for every line")

    def fit(nu_cols):
        y = np.log(nu_cols / gu)
        sy = werr / w
        wt = 1.0 / sy**2
        A = np.vstack([np.ones_like(eu), -eu]).T
        cov = np.linalg.inv(A.T @ (A * wt[:, None]))
        c0, inv_t = cov @ (A.T @ (wt * y))
        chi2 = float(np.sum(wt * (y - A @ np.array([c0, inv_t])) ** 2))
        return c0, inv_t, cov, chi2

    tau = np.zeros_like(w)
    c0, inv_t, cov, chi2 = fit(nu_thin)
    if inv_t <= 0:
        raise ValueError("fitted slope is non-negative (T_rot undefined or negative): check line assignments, blends or opacity")
    t_rot = 1.0 / inv_t
    t_err = math.sqrt(cov[1, 1]) / inv_t**2
    qv = float(_partition(t_rot, partition_table, linear_rotor_b_ghz))
    n_tot = math.exp(c0) * qv
    # error on ln N_tot = c0 + ln Q(1/s): propagate the full intercept-slope covariance
    dt = max(1e-3 * t_rot, 1e-3)
    dlnq_dt = (math.log(float(_partition(t_rot + dt, partition_table, linear_rotor_b_ghz)))
               - math.log(float(_partition(t_rot - dt, partition_table, linear_rotor_b_ghz)))) / (2 * dt)
    g = dlnq_dt * (-t_rot**2)  # d lnQ / d(1/T)
    dlnn = math.sqrt(max(cov[0, 0] + g**2 * cov[1, 1] + 2 * g * cov[0, 1], 0.0))
    method = "weighted linear least squares on the diagram"
    if tau_correction:
        # Joint nonlinear fit of (ln N_tot, T_rot): the observed thin-assumption column of each line is
        # N_u (1 - exp(-tau)) / tau, with N_u = N_tot g_u exp(-E_u/T) / Q(T) and tau = peak tau of a Gaussian line.
        y = np.log(nu_thin)
        sy = werr / w

        def model_cols(p):
            lnn, t = p
            n_u = np.exp(lnn) * gu * np.exp(-eu / t) / _partition(t, partition_table, linear_rotor_b_ghz)
            tt = _tau_from_nu(n_u, nu, a, t, fwhm)
            return n_u, tt, n_u * np.where(tt > 1e-8, -np.expm1(-tt) / np.maximum(tt, 1e-300), 1.0)

        def resid(p):
            return (np.log(model_cols(p)[2]) - y) / sy

        best = None
        for t0 in (t_rot, 0.5 * t_rot, 2.0 * t_rot, 4.0 * t_rot):
            for dn in (0.0, 1.0, 3.0):
                p0 = [math.log(n_tot) + dn, min(max(t0, 3.0), 4000.0)]
                rr = least_squares(resid, p0, bounds=([-np.inf, 2.0], [np.inf, 5000.0]))
                if best is None or rr.cost < best.cost:
                    best = rr
        lnn, t_rot = best.x
        try:
            pc = np.linalg.inv(best.jac.T @ best.jac)
        except np.linalg.LinAlgError:
            pc = np.full((2, 2), np.nan)
        n_tot = math.exp(lnn)
        dlnn = float(math.sqrt(max(pc[0, 0], 0.0)))
        t_err = float(math.sqrt(max(pc[1, 1], 0.0)))
        n_u, tau, _ = model_cols(best.x)
        ctau = np.where(tau > 1e-8, tau / -np.expm1(-tau), 1.0)
        chi2 = float(2 * best.cost)
        qv = float(_partition(t_rot, partition_table, linear_rotor_b_ghz))
        c0 = math.log(n_tot / qv)
        cov = pc
        method = "joint nonlinear least squares in (ln N_tot, T_rot) with the opacity term"
    dof = len(lines) - 2
    out = {
        "T_rot": q(t_rot, "K", t_err),
        "N_tot": q(n_tot, "cm^-2", n_tot * dlnn),
        "log10_N_tot": math.log10(n_tot),
        "Q_at_T_rot": qv,
        "intercept_ln_Ntot_over_Q": c0,
        "covariance": cov.tolist(),
        "covariance_parameters": ["ln_N_tot", "T_rot"] if tau_correction else ["intercept", "1/T_rot"],
        "fit_method": method,
        "chi2": chi2, "dof": dof, "reduced_chi2": chi2 / dof if dof > 0 else None,
        "points": [{"e_u_k": float(eu[i]), "ln_Nu_over_gu": float(math.log(nu_thin[i] * ctau[i] / gu[i])),
                    "error": float(werr[i] / w[i]), "tau_peak": float(tau[i]), "c_tau": float(ctau[i])}
                   for i in range(len(lines))],
    }
    if tau_correction and np.max(tau) > 1:
        warnings.append(f"Max line tau = {np.max(tau):.2f}: lines this thick are often excluded (papers cap tau at 0.2-1).")
    if dof > 0 and chi2 / dof > 3:
        warnings.append("Reduced chi2 > 3: curvature (two temperature components, opacity or non-LTE) or underestimated errors.")
    if beam_filling == 1.0:
        warnings.append("Beam filling 1 assumed: N_tot is beam averaged (a lower limit for compact emission).")
    warnings.append("Fit errors exclude partition-function and blend systematics; T_rot is a lower limit to T_kin for subthermal gas.")
    conv = ["N_u = 8 pi k nu^2 W / (h c^3 A_ul) (optically thin, Rayleigh-Jeans, T_bg neglected), divided by beam filling.",
            "ln(N_u/g_u) = ln(N_tot/Q(T_rot)) - E_u/T_rot; weighted least squares with sigma_ln = sigma_W/W.",
            f"W errors: given (+) {flux_cal_frac:.0%} calibration in quadrature.",
            "Q(T): " + ("user table, log-log interpolation" if partition_table else "linear rotor kT/(hB) + 1/3"),
            "tau correction: " + ("C_tau = tau/(1 - exp(-tau)) (Goldsmith and Langer 1999) fitted jointly with T_rot and N_tot" if tau_correction else "none")]
    return result(out, conv, warnings, ROT_PRACTICE)


def lte_column_density(
    w_k_kms: float,
    freq_ghz: float,
    a_ul: float,
    e_u_k: float,
    g_u: float,
    t_ex_k: float,
    partition_q: float | None = None,
    partition_table: list[list[float]] | None = None,
    linear_rotor_b_ghz: float | None = None,
    t_bg_k: float = T_CMB0,
    tau: float | None = None,
    peak_tb_k: float | None = None,
    beam_filling: float = 1.0,
    w_err_k_kms: float | None = None,
) -> dict:
    """Total column density from one line in LTE (Mangum and Shirley 2015 form).

    N_u = 8 pi k nu^2 W / (h c^3 A_ul) x J(T_ex) / (J(T_ex) - J(T_bg)) / beam_filling, with
    J(T) = (h nu/k) / (exp(h nu / k T) - 1); N_tot = N_u Q(T_ex) / g_u exp(E_u / T_ex).
    Optical depth: give tau directly, or peak_tb_k so tau = -ln(1 - T_peak / (f (J(T_ex) - J(T_bg)))); then
    N is multiplied by C_tau = tau / (1 - exp(-tau)). Q from partition_q, a [[T, Q]] table or a linear rotor B.
    Limits: T_ex must be assumed (N can change by factors of several over 10-100 K); one line cannot test LTE.
    """
    warnings: list[str] = []
    if partition_q is None:
        partition_q = float(_partition(t_ex_k, partition_table, linear_rotor_b_ghz))
    nu = freq_ghz * 1e9
    jex, jbg = float(jnu(nu, t_ex_k)), float(jnu(nu, t_bg_k))
    if jex <= jbg:
        raise ValueError("T_ex must exceed T_bg for an emission line")
    n_u_thin = float(upper_level_column(w_k_kms, freq_ghz, a_ul)) * jex / (jex - jbg) / beam_filling
    ctau = 1.0
    if tau is None and peak_tb_k is not None:
        ratio = peak_tb_k / (beam_filling * (jex - jbg))
        if ratio >= 1:
            warnings.append("Peak T_B >= f (J(T_ex) - J(T_bg)): line saturated or T_ex too low; tau undefined, N is a lower limit.")
            tau = None
        else:
            tau = -math.log(1.0 - ratio)
    if tau is not None and tau > 0:
        ctau = tau / -math.expm1(-tau)
    n_u = n_u_thin * ctau
    n_tot = n_u * partition_q / g_u * math.exp(e_u_k / t_ex_k)
    n_thin_tot = n_u_thin * partition_q / g_u * math.exp(e_u_k / t_ex_k)
    out = {
        "N_tot": q(n_tot, "cm^-2", n_tot * (w_err_k_kms / w_k_kms) if w_err_k_kms else None),
        "N_tot_optically_thin": q(n_thin_tot, "cm^-2"),
        "N_upper": q(n_u, "cm^-2"),
        "tau": tau, "C_tau": ctau, "Q_at_T_ex": partition_q,
        "J_Tex_K": jex, "J_Tbg_K": jbg,
    }
    if tau is None:
        warnings.append("Optically thin assumed; N is a lower limit if the line is thick.")
    warnings.append("T_ex is an assumption; quote N for a range of T_ex.")
    conv = ["Mangum and Shirley 2015: N_u from W with the J(T_ex)/(J(T_ex)-J(T_bg)) background term; N_tot = N_u Q/g_u exp(E_u/T_ex).",
            "Optical depth correction C_tau = tau/(1 - exp(-tau))."]
    return result(out, conv, warnings, ROT_PRACTICE)
