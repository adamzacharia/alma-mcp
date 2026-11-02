"""Dust continuum: masses, modified blackbody SED fits and spectral indices.

Pure functions. Conventions follow the corpus skills dust-continuum-mass, sed-fitting and spectral-index.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from ._util import (
    C_CGS, H_CGS, K_CGS, JY_CGS, L_SUN_CGS, M_EARTH_G, M_JUP_G, M_SUN_G, MPC_CM, PC_CM, T_CMB0,
    as_array, optional_import, planck_nu, q, result,
)

# Opacity presets: kappa per gram of DUST, kappa_nu = kappa0 (nu / nu0)^beta
KAPPA_PRESETS = {
    "beckwith1990": {
        "kappa0_cm2_g": 10.0, "nu0_ghz": 1000.0, "beta": 1.0,
        "note": "Beckwith et al. 1990 disk law: 10 cm2/g at 1000 GHz (2.3 cm2/g at 230 GHz) with beta = 1, per gram of dust. Common for protoplanetary and debris disks.",
    },
    "oh94": {
        "kappa0_cm2_g": 0.899, "nu0_ghz": 230.6, "beta": 1.8,
        "note": "Ossenkopf and Henning 1994, thin ice mantles at 1e6 cm^-3: 0.899 cm2/g at 1.3 mm (1.89 cm2/g at 870 um), beta about 1.8, per gram of dust. Common for cores and clumps.",
    },
    "galaxy_850um": {
        "kappa0_cm2_g": 0.77, "nu0_ghz": 352.7, "beta": 1.8,
        "note": "0.077 m2/kg = 0.77 cm2/g at 850 um (Dunne et al. 2000 / James et al. 2002 family), per gram of dust, beta 1.8 adopted here. Common for galaxies; 0.15 m2/kg is also used (a factor 2 change).",
    },
}

_trapz = getattr(np, "trapezoid", None) or np.trapz

DUST_PRACTICE = ["dust-continuum-mass"]
SED_PRACTICE = ["sed-fitting", "dust-continuum-mass"]
ALPHA_PRACTICE = ["spectral-index", "free-free-synchrotron"]


def _cosmo():
    from astropy.cosmology import Planck18
    return Planck18


def luminosity_distance_cm(redshift: float | None, distance_pc: float | None) -> tuple[float, str]:
    if distance_pc:
        return float(distance_pc) * PC_CM, f"distance {distance_pc} pc (given)"
    if redshift is not None and redshift > 0:
        d = _cosmo().luminosity_distance(redshift).to("Mpc").value
        return d * MPC_CM, f"luminosity distance {d:.2f} Mpc from Planck18 at z = {redshift}"
    raise ValueError("give distance_pc (local) or redshift > 0")


def _kappa(preset: str, kappa0_cm2_g, nu0_ghz, beta):
    if preset == "custom" or kappa0_cm2_g is not None:
        if kappa0_cm2_g is None or nu0_ghz is None or beta is None:
            raise ValueError("custom opacity needs kappa0_cm2_g, nu0_ghz and beta")
        return float(kappa0_cm2_g), float(nu0_ghz), float(beta), "custom opacity law (user supplied)"
    if preset not in KAPPA_PRESETS:
        raise ValueError(f"kappa_preset must be one of {list(KAPPA_PRESETS)} or 'custom'")
    p = KAPPA_PRESETS[preset]
    b = p["beta"] if beta is None else float(beta)
    return p["kappa0_cm2_g"], p["nu0_ghz"], b, p["note"]


def cmb_heated_temperature(t_dust_z0: float, redshift: float, beta: float) -> float:
    """da Cunha et al. 2013 eq. 12: dust temperature at z including CMB heating."""
    e = 4.0 + beta
    return (t_dust_z0**e + T_CMB0**e * ((1.0 + redshift) ** e - 1.0)) ** (1.0 / e)


def dust_mass_from_flux(
    flux_mjy: float,
    freq_ghz: float,
    distance_pc: float | None = None,
    redshift: float | None = None,
    t_dust_k: float = 20.0,
    kappa_preset: str = "beckwith1990",
    kappa0_cm2_g: float | None = None,
    nu0_ghz: float | None = None,
    beta: float | None = None,
    kappa_includes_gas: bool = False,
    gas_to_dust: float | None = None,
    flux_err_mjy: float | None = None,
    flux_cal_frac: float = 0.10,
    t_dust_err_k: float | None = None,
    cmb: str = "none",
) -> dict:
    """Optically thin dust mass (Hildebrand 1983): M = S_nu D^2 / (kappa_nu B_nu(T_d)).

    At redshift z (freq_ghz is the observed frequency): M = S_obs D_L^2 / ((1 + z) kappa(nu_rest) B(nu_rest, T_d)).
    The full Planck function is used, never the Rayleigh-Jeans 1/T form.
    kappa_nu = kappa0 (nu / nu0)^beta per gram of dust unless kappa_includes_gas=True (then the result is a gas+dust
    mass and the dust mass is that divided by gas_to_dust).
    cmb: 'none', 'contrast' (divide by 1 - B(T_CMB(z))/B(T_d), da Cunha 2013 eq. 18) or 'heating_and_contrast'
    (t_dust_k is then the z = 0 temperature and is heated by the CMB first, da Cunha 2013 eq. 12). Never apply heating
    twice. Limits: optically thick emission makes the mass a lower limit; kappa is uncertain by about 25% to a factor
    3-5 and is NOT propagated; interferometers can resolve out extended flux.
    """
    warnings: list[str] = []
    k0, n0, b, knote = _kappa(kappa_preset, kappa0_cm2_g, nu0_ghz, beta)
    z = float(redshift or 0.0)
    d_cm, dnote = luminosity_distance_cm(redshift, distance_pc)
    nu_obs = freq_ghz * 1e9
    nu_rest = nu_obs * (1.0 + z)
    kappa = k0 * (nu_rest / (n0 * 1e9)) ** b
    t_used = float(t_dust_k)
    cmb = cmb.lower()
    f_cmb = 1.0
    if cmb not in ("none", "contrast", "heating_and_contrast"):
        raise ValueError("cmb must be 'none', 'contrast' or 'heating_and_contrast'")
    if cmb == "heating_and_contrast":
        t_used = cmb_heated_temperature(t_dust_k, z, b)
    if cmb != "none":
        t_cmb_z = T_CMB0 * (1.0 + z)
        if t_used <= t_cmb_z:
            raise ValueError(f"T_d = {t_used:.2f} K is not above T_CMB(z) = {t_cmb_z:.2f} K")
        f_cmb = 1.0 - planck_nu(nu_rest, t_cmb_z) / planck_nu(nu_rest, t_used)
    elif z > 2:
        warnings.append(f"z = {z}: the CMB lowers the observed contrast (da Cunha 2013); consider cmb='contrast' or 'heating_and_contrast'.")

    def mass_g(s_mjy, t):
        return s_mjy * 1e-3 * JY_CGS * d_cm**2 / ((1.0 + z) * kappa * planck_nu(nu_rest, t) * f_cmb)

    m = float(mass_g(flux_mjy, t_used))
    stat = float(flux_err_mjy or 0.0)
    s_err = math.hypot(stat, flux_cal_frac * flux_mjy)
    m_err_flux = m * s_err / flux_mjy
    m_err_t = None
    if t_dust_err_k:
        lo = float(mass_g(flux_mjy, max(t_used - t_dust_err_k, 1.0)))
        hi = float(mass_g(flux_mjy, t_used + t_dust_err_k))
        m_err_t = 0.5 * abs(lo - hi)
    m_err = math.hypot(m_err_flux, m_err_t or 0.0)
    x = H_CGS * nu_rest / (K_CGS * t_used)
    if x > 0.3:
        warnings.append(f"h nu / k T = {x:.2f}: outside the Rayleigh-Jeans limit, so the mass does not scale as 1/T; full Planck used.")
    warnings.append("Optically thin assumed: for bright compact sources the mass is a lower limit.")
    warnings.append("Opacity uncertainty (about 25% to a factor 3-5) is not included in the error.")
    if flux_err_mjy is None:
        warnings.append("No statistical flux error given; error is the flux calibration term only.")
    out = {
        "frequency_rest_ghz": q(nu_rest / 1e9, "GHz"),
        "kappa_nu": q(kappa, "cm2 g^-1", basis="per gas+dust" if kappa_includes_gas else "per dust"),
        "t_dust_used": q(t_used, "K"),
        "h_nu_over_kT": x,
        "cmb_contrast_factor": f_cmb,
        "error_from_flux": q(m_err_flux / M_SUN_G, "Msun"),
    }
    if m_err_t is not None:
        out["error_from_temperature"] = q(m_err_t / M_SUN_G, "Msun")
    if kappa_includes_gas:
        out["gas_plus_dust_mass"] = q(m / M_SUN_G, "Msun", m_err / M_SUN_G)
        if gas_to_dust:
            md = m / gas_to_dust
            out["dust_mass"] = q(md / M_SUN_G, "Msun", m_err / gas_to_dust / M_SUN_G, earth_masses=md / M_EARTH_G)
        warnings.append("kappa includes the gas (per gram of gas+dust): do NOT multiply by the gas-to-dust ratio again.")
    else:
        out["dust_mass"] = q(m / M_SUN_G, "Msun", m_err / M_SUN_G, earth_masses=m / M_EARTH_G,
                             earth_masses_error=m_err / M_EARTH_G)
        if gas_to_dust:
            mg = m * gas_to_dust
            out["gas_mass"] = q(mg / M_SUN_G, "Msun", m_err * gas_to_dust / M_SUN_G,
                                jupiter_masses=mg / M_JUP_G, gas_to_dust=gas_to_dust)
            warnings.append("Gas-to-dust is not universal (higher at low metallicity, different in AGB envelopes and disks).")
    conv = [
        "Hildebrand 1983 optically thin: M = S_nu D^2 / (kappa_nu B_nu(T_d)); at z: M = S_obs D_L^2 / ((1+z) kappa(nu_rest) B(nu_rest,T_d)).",
        f"kappa_nu = {k0} cm2/g (nu / {n0} GHz)^{b}. {knote}",
        dnote,
        f"Flux error = statistical (+) {flux_cal_frac:.0%} calibration in quadrature.",
        f"CMB treatment: {cmb} (da Cunha et al. 2013).",
    ]
    return result(out, conv, warnings, DUST_PRACTICE)


# --------------------------------------------------------------------------------------------- SED fitting


def _mbb_model(nu_obs_hz, logm_msun, t, beta, z, d_cm, kappa0, nu_k_hz, form, lambda0_um, cmb):
    nu_rest = nu_obs_hz * (1.0 + z)
    m = 10.0**logm_msun * M_SUN_G
    bnu = planck_nu(nu_rest, t)
    if cmb:
        bnu = bnu - planck_nu(nu_rest, T_CMB0 * (1.0 + z))
    kappa = kappa0 * (nu_rest / nu_k_hz) ** beta
    if form == "thin":
        return (1.0 + z) * m * kappa * bnu / d_cm**2 / JY_CGS * 1e3  # mJy
    nu0 = C_CGS / (lambda0_um * 1e-4)
    area = m * kappa0 * (nu0 / nu_k_hz) ** beta  # tau(nu0) = 1 defines the emitting area
    tau = (nu_rest / nu0) ** beta
    return (1.0 + z) * area * (-np.expm1(-tau)) * bnu / d_cm**2 / JY_CGS * 1e3


def fit_modified_blackbody(
    freq_ghz: list[float],
    flux_mjy: list[float],
    flux_err_mjy: list[float],
    redshift: float = 0.0,
    distance_pc: float | None = None,
    form: str = "thin",
    lambda0_um: float = 100.0,
    fix_beta: float | None = None,
    fix_t_dust_k: float | None = None,
    kappa_preset: str = "galaxy_850um",
    kappa0_cm2_g: float | None = None,
    nu0_ghz: float | None = None,
    flux_cal_frac: float = 0.10,
    cmb_contrast: bool = False,
    mcmc: bool = False,
    mcmc_steps: int = 3000,
    mcmc_walkers: int = 32,
    seed: int = 1,
) -> dict:
    """Fit a single-temperature modified blackbody to an SED (observed frequencies in GHz, fluxes in mJy).

    form='thin': S = (1+z) M kappa(nu_rest) B(nu_rest,T) / D_L^2.
    form='general': S = (1+z) A (1 - exp(-tau)) B / D_L^2 with tau = (nu_rest/nu_0)^beta, nu_0 = c/lambda0_um, and
    A = M kappa(nu_0) (tau = 1 at lambda0 defines the area). The mass in this form depends on that area assumption.
    Free parameters: log10 M_dust (Msun), T_d (K), beta (unless fixed). Errors in quadrature with flux_cal_frac.
    L_IR is integrated over rest 8-1000 um (and L_FIR over 42.5-122.5 um) from the intrinsic B(T) model.
    cmb_contrast subtracts B(T_CMB(z)) (da Cunha 2013 contrast term only; T_d is then the heated temperature).
    Returns least-squares values with covariance errors; mcmc=True adds an emcee posterior (needs [fit] extra).
    Limits: one temperature misses cold dust mass; T and beta are degenerate; upper limits are not handled.
    """
    warnings: list[str] = []
    nu = as_array(freq_ghz, "freq_ghz") * 1e9
    s = as_array(flux_mjy, "flux_mjy")
    e = as_array(flux_err_mjy, "flux_err_mjy")
    if not (nu.size == s.size == e.size):
        raise ValueError("freq_ghz, flux_mjy and flux_err_mjy must have the same length")
    if form not in ("thin", "general"):
        raise ValueError("form must be 'thin' or 'general'")
    sig = np.hypot(e, flux_cal_frac * s)
    z = float(redshift or 0.0)
    have_d = bool(distance_pc) or z > 0
    if have_d:
        d_cm, dnote = luminosity_distance_cm(redshift if z > 0 else None, distance_pc)
    else:
        d_cm, dnote = PC_CM, "no distance given: normalisation is the mass at 1 pc (no physical mass or L_IR)"
        warnings.append("No distance or redshift: mass and L_IR are not physical; only T_d and beta are meaningful.")
    k0, n0, _, knote = _kappa(kappa_preset, kappa0_cm2_g, nu0_ghz, 1.8)
    nu_k = n0 * 1e9
    t_floor = T_CMB0 * (1 + z) + 0.5

    free = []
    if fix_t_dust_k is None:
        free.append("t")
    if fix_beta is None:
        free.append("beta")
    n_free = 1 + len(free)
    if nu.size < n_free:
        raise ValueError(f"{nu.size} points cannot constrain {n_free} free parameters; fix beta and/or T_d")
    if nu.size == n_free:
        warnings.append("As many points as free parameters: no degrees of freedom, errors are not tested by the data.")

    def unpack(p):
        logm = p[0]
        i = 1
        t = fix_t_dust_k
        b = fix_beta
        if "t" in free:
            t = p[i]; i += 1
        if "beta" in free:
            b = p[i]
        return logm, t, b

    def model(p, nu_hz=nu, cmb=cmb_contrast):
        logm, t, b = unpack(p)
        return _mbb_model(nu_hz, logm, t, b, z, d_cm, k0, nu_k, form, lambda0_um, cmb)

    def resid(p):
        return (model(p) - s) / sig

    lo, hi, p0 = [-12.0], [16.0], [0.0]
    if "t" in free:
        lo.append(t_floor); hi.append(500.0); p0.append(max(35.0, t_floor + 5))
    if "beta" in free:
        lo.append(0.0); hi.append(4.0); p0.append(1.8)
    # starting mass from the lowest-frequency point
    p0 = np.array(p0)
    trial = model(p0)
    j = int(np.argmin(nu))
    if trial[j] > 0 and s[j] > 0:
        p0[0] = np.clip(math.log10(s[j] / trial[j]), -11, 15)
    best = None
    for tstart in ([p0[1]] if "t" in free else [None]) + ([15.0, 60.0, 120.0] if "t" in free else []):
        pp = p0.copy()
        if tstart is not None:
            pp[1] = max(tstart, t_floor + 1)
            tr = model(pp)
            if tr[j] > 0 and s[j] > 0:
                pp[0] = np.clip(pp[0] + math.log10(s[j] / tr[j]), -11, 15)
        r = least_squares(resid, pp, bounds=(lo, hi), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    r = best
    chi2 = float(2 * r.cost)
    dof = max(nu.size - n_free, 0)
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac)
    except np.linalg.LinAlgError:
        cov = np.full((n_free, n_free), np.nan)
        warnings.append("Singular covariance: parameters are degenerate with this sampling; fix beta or T_d.")
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    names = ["log10_mass_msun"] + (["t_dust_k"] if "t" in free else []) + (["beta"] if "beta" in free else [])

    # derived quantities with Monte Carlo from the covariance
    rng = np.random.default_rng(seed)
    nu_rest_grid = np.logspace(np.log10(C_CGS / 1000e-4), np.log10(C_CGS / 8e-4), 600)
    nu_fir = np.logspace(np.log10(C_CGS / 122.5e-4), np.log10(C_CGS / 42.5e-4), 300)

    def lum(p, grid):
        sv = _mbb_model(grid / (1 + z), *unpack(p), z, d_cm, k0, nu_k, form, lambda0_um, False)
        flux_int = _trapz(sv * 1e-3 * JY_CGS, grid / (1 + z))
        return 4 * math.pi * d_cm**2 * flux_int / L_SUN_CGS

    def peak_um(p):
        sv = _mbb_model(nu_rest_grid / (1 + z), *unpack(p), z, d_cm, k0, nu_k, form, lambda0_um, False)
        return C_CGS / nu_rest_grid[int(np.argmax(sv))] * 1e4

    samples = None
    sampler_note = "scipy least_squares (trust region reflective); errors from the Jacobian covariance, not scaled by chi2."
    if mcmc:
        emcee, msg = optional_import("emcee", "fit")
        if emcee is None:
            warnings.append(msg)
        else:
            ndim = n_free
            nw = max(mcmc_walkers, 2 * ndim + 2)
            lo_a, hi_a = np.array(lo), np.array(hi)

            def lnp(p):
                if np.any(p < lo_a) or np.any(p > hi_a):
                    return -np.inf
                return -0.5 * float(np.sum(resid(p) ** 2))

            start = r.x + 1e-3 * rng.standard_normal((nw, ndim)) * np.where(np.isfinite(perr) & (perr > 0), perr, 0.01)
            start = np.clip(start, lo_a + 1e-6, hi_a - 1e-6)
            np.random.seed(seed)
            sampler = emcee.EnsembleSampler(nw, ndim, lnp)
            sampler.run_mcmc(start, mcmc_steps, progress=False)
            burn = mcmc_steps // 3
            samples = sampler.get_chain(discard=burn, flat=True)
            try:
                tau = sampler.get_autocorr_time(discard=burn, tol=0)
                if np.any(mcmc_steps - burn < 50 * tau):
                    warnings.append(f"Chain shorter than 50 autocorrelation times (tau = {np.round(tau, 1).tolist()}); run longer.")
            except Exception:
                pass
            sampler_note = f"emcee {nw} walkers x {mcmc_steps} steps, first {burn} discarded; flat priors within the bounds; median and 16/84 percentiles."
    if samples is None:
        draws = rng.multivariate_normal(r.x, np.nan_to_num(cov), size=400) if np.all(np.isfinite(cov)) else np.tile(r.x, (2, 1))
        draws = np.clip(draws, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    else:
        draws = samples[rng.choice(len(samples), size=min(400, len(samples)), replace=False)]

    def summarize(vals):
        v = np.asarray(vals, float)
        return float(np.median(v)), float(np.percentile(v, 16)), float(np.percentile(v, 84))

    params = {}
    for k, name in enumerate(names):
        if samples is not None:
            med, p16, p84 = summarize(samples[:, k])
            params[name] = {"value": med, "lower_16": p16, "upper_84": p84}
        else:
            params[name] = {"value": float(r.x[k]), "error": float(perr[k])}
    units = {"log10_mass_msun": "log10(Msun)", "t_dust_k": "K", "beta": ""}
    for n_, d_ in params.items():
        d_["unit"] = units[n_]
    if fix_t_dust_k is not None:
        params["t_dust_k"] = {"value": fix_t_dust_k, "unit": "K", "fixed": True}
    if fix_beta is not None:
        params["beta"] = {"value": fix_beta, "unit": "", "fixed": True}
    best_p = r.x if samples is None else np.median(samples, axis=0)
    out = {"parameters": params, "chi2": chi2, "dof": dof, "reduced_chi2": chi2 / dof if dof else None,
           "model_mjy": model(best_p).tolist(), "peak_wavelength_rest_um": peak_um(best_p)}
    if have_d:
        l_ir = [lum(p, nu_rest_grid) for p in draws]
        l_fir = [lum(p, nu_fir) for p in draws]
        m_lin = 10 ** np.array([p[0] for p in draws])
        out["dust_mass"] = q(float(10 ** best_p[0]), "Msun", float(np.std(m_lin)))
        out["L_IR_8_1000um"] = q(lum(best_p, nu_rest_grid), "Lsun", float(np.std(l_ir)))
        out["L_FIR_42.5_122.5um"] = q(lum(best_p, nu_fir), "Lsun", float(np.std(l_fir)))
    if dof and chi2 / dof > 3:
        warnings.append(f"Reduced chi2 = {chi2 / dof:.1f}: one modified blackbody does not describe these data (multiple components, contamination, or underestimated errors).")
    if form == "thin":
        warnings.append("Optically thin form: if the dust is thick near the peak, T_d is biased low and M_d high; try form='general'.")
    if fix_beta is None and nu.size < 4:
        warnings.append("Free beta with fewer than 4 points: T-beta degeneracy is strong.")
    conv = [
        f"form = {form}" + (f", tau = (nu_rest / nu(lambda0 = {lambda0_um} um))^beta" if form == "general" else ""),
        f"kappa_nu = {k0} cm2/g (nu / {n0} GHz)^beta per gram of dust. {knote}",
        dnote,
        f"Errors: given errors (+) {flux_cal_frac:.0%} calibration in quadrature.",
        "L_IR: rest 8-1000 um integral of the intrinsic model (no CMB contrast).",
        sampler_note,
    ]
    return result(out, conv, warnings, SED_PRACTICE)


# --------------------------------------------------------------------------------------------- spectral index


def spectral_index(
    freq_ghz: list[float],
    flux_mjy: list[float],
    flux_err_mjy: list[float] | None = None,
    flux_cal_frac: float | list[float] = 0.10,
    t_dust_k: float | None = None,
) -> dict:
    """Spectral index alpha (S_nu proportional to nu^alpha) from two or more frequencies.

    Two points: alpha = ln(S2/S1) / ln(nu2/nu1), sigma_alpha = sqrt((s1/S1)^2 + (s2/S2)^2) / |ln(nu2/nu1)|.
    More points: weighted least squares of ln S on ln nu. Each flux error is the statistical error (+) the per-band
    flux calibration fraction (default 10%) in quadrature, so the absolute alpha error is calibration dominated
    (about 0.4 for 10% per band on nearby bands). Also returns beta = alpha - 2 (optically thin Rayleigh-Jeans) and,
    if t_dust_k is given, beta corrected for the Planck curvature: beta = alpha - dlnB/dlnnu.
    Limits: needs matched uv ranges / beams across bands; optical depth and scattering lower alpha (mimicking grain
    growth); non-simultaneous data on variable sources bias it.
    """
    warnings: list[str] = []
    nu = as_array(freq_ghz, "freq_ghz")
    s = as_array(flux_mjy, "flux_mjy")
    if nu.size < 2 or nu.size != s.size:
        raise ValueError("need at least two frequencies with matching fluxes")
    if np.any(s <= 0):
        raise ValueError("fluxes must be positive (treat non-detections as limits separately)")
    e = np.zeros_like(s) if flux_err_mjy is None else as_array(flux_err_mjy, "flux_err_mjy")
    cal = np.broadcast_to(np.asarray(flux_cal_frac, float), s.shape)
    sig = np.hypot(e, cal * s)
    rel = sig / s
    if nu.size == 2:
        lr = math.log(nu[1] / nu[0])
        alpha = math.log(s[1] / s[0]) / lr
        alpha_err = math.sqrt(rel[0] ** 2 + rel[1] ** 2) / abs(lr)
        stat_err = math.sqrt((e[0] / s[0]) ** 2 + (e[1] / s[1]) ** 2) / abs(lr)
        method = "two-point"
        chi2 = None
    else:
        x = np.log(nu)
        y = np.log(s)
        w = 1.0 / rel**2
        A = np.vstack([np.ones_like(x), x]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        coef = cov @ (A.T @ (w * y))
        alpha = float(coef[1])
        alpha_err = float(math.sqrt(cov[1, 1]))
        chi2 = float(np.sum(w * (y - A @ coef) ** 2))
        stat_err = None
        method = f"weighted least squares in log-log over {nu.size} points"
        if chi2 / (nu.size - 2) > 3:
            warnings.append("Poor power-law fit (reduced chi2 > 3): curvature or several emission components.")
    out = {"alpha": q(alpha, "", alpha_err), "method": method, "beta_rayleigh_jeans_thin": q(alpha - 2.0, "", alpha_err)}
    if stat_err is not None:
        out["alpha_error_statistical_only"] = stat_err
    if chi2 is not None:
        out["chi2"] = chi2
    if t_dust_k:
        lo, hi = float(nu.min()) * 1e9, float(nu.max()) * 1e9
        a_planck = math.log(planck_nu(hi, t_dust_k) / planck_nu(lo, t_dust_k)) / math.log(hi / lo)
        out["alpha_planck"] = a_planck
        out["beta_planck_corrected"] = q(alpha - a_planck, "", alpha_err)
    if np.all(cal > 0) and nu.size == 2:
        warnings.append("Relative alpha changes across one source are more robust than the absolute value (common calibration offset).")
    warnings.append("beta = alpha - 2 assumes optically thin, Rayleigh-Jeans dust; thick or cold emission biases it low.")
    conv = ["S_nu proportional to nu^alpha.",
            "Flux errors: statistical (+) per-band calibration fraction in quadrature (default 10%)."]
    return result(out, conv, warnings, ALPHA_PRACTICE)
