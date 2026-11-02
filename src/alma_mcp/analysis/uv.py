"""Visibility-plane analysis: parametric model fits, deprojected radial profiles, frank wrapper.

Conventions follow the corpus skills uv-plane-fitting and disk-substructure-profiles.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import j0, j1

from ._util import ARCSEC_RAD, load_visibilities, optional_import, q, result

UV_PRACTICE = ["uv-plane-fitting", "disk-substructure-profiles"]
GF = math.pi**2 / (4.0 * math.log(2.0))  # Gaussian FWHM factor

MODELS = {
    "point": ["flux_jy", "dra_arcsec", "ddec_arcsec"],
    "gaussian": ["flux_jy", "dra_arcsec", "ddec_arcsec", "fwhm_arcsec"],
    "elliptical_gaussian": ["flux_jy", "dra_arcsec", "ddec_arcsec", "fwhm_major_arcsec", "axis_ratio", "pa_deg"],
    "uniform_disk": ["flux_jy", "dra_arcsec", "ddec_arcsec", "diameter_arcsec", "inclination_deg", "pa_deg"],
    "ring": ["flux_jy", "dra_arcsec", "ddec_arcsec", "radius_arcsec", "width_fwhm_arcsec", "inclination_deg", "pa_deg"],
}


def _proj(u, v, inc_deg, pa_deg):
    """Return (q_major, q_minor_deprojected) for a source whose major axis has PA east of north."""
    pa = math.radians(pa_deg)
    qmaj = u * math.sin(pa) + v * math.cos(pa)
    qmin = u * math.cos(pa) - v * math.sin(pa)
    return qmaj, qmin * math.cos(math.radians(inc_deg))


def model_visibilities(model: str, params: dict, u, v) -> np.ndarray:
    """Complex model visibilities in Jy at (u, v) in wavelengths.

    Convention: V(u,v) = integral I(l,m) exp(-2 pi i (u l + v m)) dl dm, l = east offset, m = north offset (the
    sign used by CASA and most uv-fitting codes). A source offset (dRA, dDec) multiplies V by
    exp(-2 pi i (u dRA + v dDec)). PA is east of north of the major axis.
    """
    p = params
    f = p["flux_jy"]
    if model == "point":
        amp = np.ones_like(u)
    elif model == "gaussian":
        th = p["fwhm_arcsec"] * ARCSEC_RAD
        amp = np.exp(-GF * th**2 * (u**2 + v**2))
    elif model == "elliptical_gaussian":
        thmaj = p["fwhm_major_arcsec"] * ARCSEC_RAD
        thmin = thmaj * p["axis_ratio"]
        pa = math.radians(p["pa_deg"])
        qmaj = u * math.sin(pa) + v * math.cos(pa)
        qmin = u * math.cos(pa) - v * math.sin(pa)
        amp = np.exp(-GF * (thmaj**2 * qmaj**2 + thmin**2 * qmin**2))
    elif model == "uniform_disk":
        qmaj, qmin = _proj(u, v, p["inclination_deg"], p["pa_deg"])
        qd = np.hypot(qmaj, qmin)
        x = math.pi * p["diameter_arcsec"] * ARCSEC_RAD * qd
        with np.errstate(invalid="ignore", divide="ignore"):
            amp = np.where(x > 1e-8, 2.0 * j1(x) / x, 1.0)
    elif model == "ring":
        qmaj, qmin = _proj(u, v, p["inclination_deg"], p["pa_deg"])
        qd = np.hypot(qmaj, qmin)
        r = p["radius_arcsec"] * ARCSEC_RAD
        w = p["width_fwhm_arcsec"] * ARCSEC_RAD
        amp = j0(2 * math.pi * r * qd) * np.exp(-GF * w**2 * qd**2)
    else:
        raise ValueError(f"model must be one of {list(MODELS)}")
    phase = -2.0 * math.pi * (u * p["dra_arcsec"] + v * p["ddec_arcsec"]) * ARCSEC_RAD
    return f * amp * np.exp(1j * phase)


def _bounds(model, guess_size):
    lo = {"flux_jy": -np.inf, "dra_arcsec": -60, "ddec_arcsec": -60, "fwhm_arcsec": 0.0, "fwhm_major_arcsec": 0.0,
          "axis_ratio": 0.0, "pa_deg": -360, "diameter_arcsec": 0.0, "inclination_deg": 0.0, "radius_arcsec": 0.0,
          "width_fwhm_arcsec": 0.0}
    hi = {"flux_jy": np.inf, "dra_arcsec": 60, "ddec_arcsec": 60, "fwhm_arcsec": 60, "fwhm_major_arcsec": 60,
          "axis_ratio": 1.0, "pa_deg": 360, "diameter_arcsec": 60, "inclination_deg": 89.9, "radius_arcsec": 60,
          "width_fwhm_arcsec": 60}
    return lo, hi


def _initial(model, vis, given):
    u, v, re, im, w = vis["u"], vis["v"], vis["re"], vis["im"], vis["weight"]
    quv = np.hypot(u, v)
    short = quv <= np.percentile(quv, 10)
    f0 = float(np.sum(w[short] * re[short]) / np.sum(w[short]))
    amp = np.hypot(re, im)
    # size guess: uv distance where binned amplitude falls to half of the short-spacing value
    order = np.argsort(quv)
    nb = max(min(len(quv) // 50, 40), 5)
    edges = np.array_split(order, nb)
    qb = np.array([quv[e].mean() for e in edges])
    ab = np.array([np.abs(np.sum(w[e] * (re[e] + 1j * im[e])) / np.sum(w[e])) for e in edges])
    below = np.where(ab < 0.5 * abs(f0))[0]
    if below.size:
        qhalf = qb[below[0]]
        size = 0.9 / qhalf / ARCSEC_RAD * 0.5
    else:
        size = 0.25 / qb[-1] / ARCSEC_RAD
    init = {"flux_jy": f0 if f0 != 0 else float(np.median(amp)), "dra_arcsec": 0.0, "ddec_arcsec": 0.0,
            "fwhm_arcsec": size, "fwhm_major_arcsec": size, "axis_ratio": 0.8, "pa_deg": 45.0,
            "diameter_arcsec": 1.6 * size, "inclination_deg": 30.0, "radius_arcsec": size,
            "width_fwhm_arcsec": 0.3 * size}
    init.update({k: float(val) for k, val in (given or {}).items()})
    return init


def dirty_peak(u, v, re, im, w, search_radius_arcsec=2.0, max_vis=20000, seed=0):
    """Starting offset (dRA, dDec) in arcsec: flux-weighted centroid of the pixels above 30% of the peak of a
    coarse dirty map (direct Fourier sum). The centroid, unlike the peak, lands at the centre of a ring."""
    if u.size > max_vis:
        idx = np.random.default_rng(seed).choice(u.size, max_vis, replace=False)
        u, v, re, im, w = u[idx], v[idx], re[idx], im[idx], w[idx]
    qmax = float(np.hypot(u, v).max())
    step = 0.25 / qmax / ARCSEC_RAD
    n = int(min(2 * search_radius_arcsec / step, 300)) + 1
    xs = np.linspace(-search_radius_arcsec, search_radius_arcsec, n)
    vis = w * (re + 1j * im)
    ex = np.exp(2j * math.pi * np.outer(u, xs) * ARCSEC_RAD)
    ey = np.exp(2j * math.pi * np.outer(v, xs) * ARCSEC_RAD)
    img = np.real((ex * vis[:, None]).T @ ey)  # [x, y]
    sel = img > 0.3 * img.max()
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    wt = np.where(sel, img, 0.0)
    return float(np.sum(wt * gx) / np.sum(wt)), float(np.sum(wt * gy) / np.sum(wt))


def fit_uv_model(
    vis_path: str,
    model: str = "gaussian",
    uv_unit: str = "lambda",
    freq_ghz: float | None = None,
    initial: dict | None = None,
    fixed: dict | None = None,
    rescale_weights: str = "none",
    uv_range_klambda: list[float] | None = None,
    search_radius_arcsec: float = 2.0,
    mcmc: bool = False,
    mcmc_steps: int = 2000,
    mcmc_walkers: int | None = None,
    seed: int = 1,
) -> dict:
    """Fit a parametric source model directly to visibilities (u, v, Re, Im, weight) from .npz or .csv.

    Models: point, gaussian (circular, FWHM), elliptical_gaussian (FWHM major, axis ratio, PA), uniform_disk
    (diameter, inclination, PA), ring (radius, Gaussian FWHM width via the thin-ring x Gaussian approximation,
    valid for width << radius; inclination, PA). All carry flux_jy and offsets dra_arcsec (east), ddec_arcsec.
    chi^2 = sum w [(Re - Re_m)^2 + (Im - Im_m)^2] with CASA WEIGHT = 1 / sigma^2 per real and per imaginary part.
    rescale_weights: 'none', or 'residual_scatter' (refit after scaling weights so the residual scatter of
    sqrt(w) (Re, Im) has unit robust std; pipeline weights are often off by factors 2-4).
    fixed: parameters to hold, e.g. {"dra_arcsec": 0}. uv_range_klambda: [min, max] selection.
    Starting offsets come from the peak of a coarse dirty map within search_radius_arcsec of the phase centre
    (unless given in initial); sizes, PA and inclination are grid-searched before the least-squares fit.
    Returns least-squares values with covariance errors (statistical only), reduced chi^2 and the weight scale;
    mcmc=True adds an emcee posterior (uniform priors in bounds; needs the [fit] extra).
    Limits: flux errors exclude the absolute flux scale (about 5-10%, add in quadrature); sizes near the
    resolution limit need S/N >= 8-10; the model sign convention matters for offsets (see model_visibilities).
    """
    warnings: list[str] = []
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    vis = load_visibilities(vis_path, uv_unit, freq_ghz)
    if vis["n_dropped"]:
        warnings.append(f"Dropped {vis['n_dropped']} rows with non-finite values or weight <= 0 (flagged).")
    if uv_range_klambda:
        quv = np.hypot(vis["u"], vis["v"]) / 1e3
        keep = (quv >= uv_range_klambda[0]) & (quv <= uv_range_klambda[1])
        vis = {k: (val[keep] if isinstance(val, np.ndarray) else val) for k, val in vis.items()}
    u, v, re, im, w = vis["u"], vis["v"], vis["re"], vis["im"], vis["weight"]
    nvis = u.size
    names = MODELS[model]
    fixed = {k: float(val) for k, val in (fixed or {}).items()}
    for k in fixed:
        if k not in names:
            raise ValueError(f"cannot fix '{k}': not a parameter of {model} ({names})")
    free = [n for n in names if n not in fixed]
    init = _initial(model, vis, initial)
    if not ({"dra_arcsec", "ddec_arcsec"} & set((initial or {}).keys())) and ("dra_arcsec" in free or "ddec_arcsec" in free):
        x_pk, y_pk = dirty_peak(u, v, re, im, w, search_radius_arcsec)
        init["dra_arcsec"], init["ddec_arcsec"] = x_pk, y_pk
    lo_d, hi_d = _bounds(model, init.get("fwhm_arcsec"))
    lo = np.array([lo_d[n] for n in free])
    hi = np.array([hi_d[n] for n in free])

    def params_of(x):
        p = dict(fixed)
        p.update(dict(zip(free, x)))
        return p

    def make_resid(weights):
        sw = np.sqrt(weights)

        def resid(x):
            m = model_visibilities(model, params_of(x), u, v)
            return np.concatenate([sw * (re - m.real), sw * (im - m.imag)])
        return resid

    def run(weights, x0s):
        best = None
        f = make_resid(weights)
        for x0 in x0s:
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            r = least_squares(f, x0, bounds=(lo, hi), x_scale="jac", method="trf")
            if best is None or r.cost < best.cost:
                best = r
        return best

    base = np.array([init[n] for n in free])
    # Multi-start: grid over sizes, PA and inclination; the flux is linear, so each start gets its best flux
    # analytically and only the best few starts are optimised (ring and disk chi2 surfaces are multimodal).
    grids = []
    size_keys = [n for n in free if n in ("fwhm_arcsec", "fwhm_major_arcsec", "diameter_arcsec", "radius_arcsec")]
    user = set((initial or {}).keys())
    size_f = [1.0] if all(k in user for k in size_keys) else [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]
    pa_opts = [None] if ("pa_deg" not in free or "pa_deg" in user) else [0.0, 45.0, 90.0, 135.0]
    inc_opts = [None] if ("inclination_deg" not in free or "inclination_deg" in user) else [10.0, 40.0, 65.0]
    ar_opts = [None] if ("axis_ratio" not in free or "axis_ratio" in user) else [0.9, 0.6, 0.3]
    for fs in size_f:
        for pa in pa_opts:
            for ic in inc_opts:
                for ar in ar_opts:
                    x = base.copy()
                    for sk in size_keys:
                        x[free.index(sk)] *= fs
                    if pa is not None:
                        x[free.index("pa_deg")] = pa
                    if ic is not None:
                        x[free.index("inclination_deg")] = ic
                    if ar is not None:
                        x[free.index("axis_ratio")] = ar
                    grids.append(x)
    scored = []
    for x in grids:
        p = params_of(x)
        if "flux_jy" in free:
            p["flux_jy"] = 1.0
            m = model_visibilities(model, p, u, v)
            den = float(np.sum(w * (m.real**2 + m.imag**2)))
            if den > 0:
                x = x.copy()
                x[free.index("flux_jy")] = float(np.sum(w * (re * m.real + im * m.imag)) / den)
                p["flux_jy"] = x[free.index("flux_jy")]
        m = model_visibilities(model, p, u, v)
        scored.append((float(np.sum(w * ((re - m.real) ** 2 + (im - m.imag) ** 2))), x))
    scored.sort(key=lambda t: t[0])
    starts = [base] + [x for _, x in scored[:4]]
    r = run(w, starts)
    scale = 1.0
    if rescale_weights == "residual_scatter":
        res = r.fun
        s = 1.4826 * np.median(np.abs(res - np.median(res)))
        scale = 1.0 / s**2
        w = w * scale
        r = run(w, [r.x])
        warnings.append(f"Weights multiplied by {scale:.4g} from the residual scatter.")
    elif rescale_weights != "none":
        raise ValueError("rescale_weights must be 'none' or 'residual_scatter'")
    chi2 = float(2 * r.cost)
    dof = 2 * nvis - len(free)
    red = chi2 / dof
    if rescale_weights == "none" and (red > 1.5 or red < 0.67):
        warnings.append(f"Reduced chi2 = {red:.3f}: weights look mis-scaled (or the model is poor); consider rescale_weights='residual_scatter' or CASA statwt.")
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac)
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = None
        perr = np.full(len(free), np.nan)
        warnings.append("Singular covariance: some parameters are unconstrained (e.g. PA of a near-circular source).")
    best = params_of(r.x)
    params = {}
    for i, n in enumerate(free):
        params[n] = {"value": float(r.x[i]), "error": float(perr[i])}
    for n, val in fixed.items():
        params[n] = {"value": val, "fixed": True}
    # normalise PA into [0, 180)
    if "pa_deg" in params and not params["pa_deg"].get("fixed"):
        params["pa_deg"]["value"] = params["pa_deg"]["value"] % 180.0
    sampler_note = "scipy least_squares (trf, multi-start); errors from the Jacobian covariance (statistical only)."
    if mcmc:
        emcee, msg = optional_import("emcee", "fit")
        if emcee is None:
            warnings.append(msg)
        else:
            f = make_resid(w)
            ndim = len(free)
            nw = mcmc_walkers or max(10 * ndim, 2 * ndim + 2)

            def lnp(x):
                if np.any(x <= lo) or np.any(x >= hi):
                    return -np.inf
                return -0.5 * float(np.sum(f(x) ** 2))

            rng = np.random.default_rng(seed)
            spread = np.where(np.isfinite(perr) & (perr > 0), perr, 1e-3)
            start = np.clip(r.x + 0.1 * spread * rng.standard_normal((nw, ndim)), lo + 1e-9, hi - 1e-9)
            np.random.seed(seed)
            sampler = emcee.EnsembleSampler(nw, ndim, lnp)
            sampler.run_mcmc(start, mcmc_steps, progress=False)
            burn = mcmc_steps // 3
            chain = sampler.get_chain(discard=burn, flat=True)
            for i, n in enumerate(free):
                p16, p50, p84 = np.percentile(chain[:, i], [16, 50, 84])
                params[n].update({"mcmc_median": float(p50), "mcmc_lower_16": float(p16), "mcmc_upper_84": float(p84)})
            try:
                tau = sampler.get_autocorr_time(discard=burn, tol=0)
                if np.any((mcmc_steps - burn) < 50 * tau):
                    warnings.append(f"Chain shorter than 50 autocorrelation times (tau = {np.round(tau, 1).tolist()}); run longer before quoting.")
            except Exception:
                pass
            sampler_note += f" emcee: {nw} walkers x {mcmc_steps} steps, {burn} burn-in, uniform priors within bounds."
    units = {"flux_jy": "Jy", "dra_arcsec": "arcsec", "ddec_arcsec": "arcsec", "fwhm_arcsec": "arcsec",
             "fwhm_major_arcsec": "arcsec", "axis_ratio": "", "pa_deg": "deg", "diameter_arcsec": "arcsec",
             "inclination_deg": "deg", "radius_arcsec": "arcsec", "width_fwhm_arcsec": "arcsec"}
    for n in params:
        params[n]["unit"] = units[n]
    quv = np.hypot(u, v)
    out = {"model": model, "parameters": params, "chi2": chi2, "dof": dof, "reduced_chi2": red,
           "weight_scale_applied": scale, "n_visibilities": nvis,
           "uv_distance_range_klambda": [float(quv.min() / 1e3), float(quv.max() / 1e3)],
           "resolution_scale_arcsec": float(1.0 / quv.max() / ARCSEC_RAD)}
    if "flux_jy" in params and np.isfinite(params["flux_jy"].get("error", np.nan)) and params["flux_jy"]["error"] > 0:
        out["snr_flux"] = params["flux_jy"]["value"] / params["flux_jy"]["error"]
        if out["snr_flux"] < 10 and model != "point":
            warnings.append("S/N < 10: size and shape parameters are weakly constrained; compare with a point-source fit.")
    if model == "ring":
        warnings.append("Ring visibility uses J0(2 pi R q) x Gaussian taper, accurate only for width << radius.")
    warnings.append("Flux errors are statistical; add the absolute flux-scale uncertainty (about 5-10% for ALMA) in quadrature.")
    conv = ["V = integral I exp(-2 pi i (u l + v m)), l east, m north; offset shifts multiply by exp(-2 pi i (u dRA + v dDec)).",
            "chi2 = sum w [(Re - Re_m)^2 + (Im - Im_m)^2]; CASA WEIGHT = 1/sigma^2 per real and per imaginary part.",
            "PA east of north of the major axis; Gaussian sizes are FWHM.", sampler_note]
    return result(out, conv, warnings, UV_PRACTICE)


def deproject_and_bin(
    vis_path: str,
    inclination_deg: float = 0.0,
    pa_deg: float = 0.0,
    dra_arcsec: float = 0.0,
    ddec_arcsec: float = 0.0,
    uv_unit: str = "lambda",
    freq_ghz: float | None = None,
    bin_width_klambda: float | None = None,
    n_bins: int = 30,
    max_klambda: float | None = None,
) -> dict:
    """Phase-centre shift, deprojection and radial binning of visibilities.

    Shift: V' = V exp(+2 pi i (u dRA + v dDec)) moves a source at (dRA, dDec) to the phase centre.
    Deprojection: q = sqrt(q_maj^2 + (q_min cos i)^2) with q_maj along the PA (east of north) of the major axis.
    Each bin gives weighted means of Re and Im (weights = CASA WEIGHT) and errors 1/sqrt(sum w).
    Returns uv distance in klambda. Limits: assumes axisymmetry; Im is near zero only for a centred
    symmetric source, so residual Im flags offsets or asymmetry.
    """
    warnings: list[str] = []
    vis = load_visibilities(vis_path, uv_unit, freq_ghz)
    u, v, re, im, w = vis["u"], vis["v"], vis["re"], vis["im"], vis["weight"]
    V = (re + 1j * im) * np.exp(2j * math.pi * (u * dra_arcsec + v * ddec_arcsec) * ARCSEC_RAD)
    qmaj, qmin = _proj(u, v, inclination_deg, pa_deg)
    qd = np.hypot(qmaj, qmin) / 1e3
    qmax = max_klambda or float(qd.max())
    if bin_width_klambda:
        edges = np.arange(0.0, qmax + bin_width_klambda, bin_width_klambda)
    else:
        edges = np.linspace(0.0, qmax, n_bins + 1)
    idx = np.digitize(qd, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        sw = float(w[sel].sum())
        rows.append({"uv_klambda": float(np.sum(w[sel] * qd[sel]) / sw), "re_jy": float(np.sum(w[sel] * V.real[sel]) / sw),
                     "im_jy": float(np.sum(w[sel] * V.imag[sel]) / sw), "err_jy": float(1.0 / math.sqrt(sw)),
                     "n_vis": int(sel.sum())})
    if rows:
        im_sig = np.array([abs(r_["im_jy"]) / r_["err_jy"] for r_ in rows])
        if np.mean(im_sig > 3) > 0.2:
            warnings.append("Imaginary part significant in >20% of bins: phase centre offset or non-axisymmetric structure.")
    out = {"bins": rows, "bin_edges_klambda": edges.tolist(), "geometry": {"inclination_deg": inclination_deg, "pa_deg": pa_deg,
                                                                           "dra_arcsec": dra_arcsec, "ddec_arcsec": ddec_arcsec}}
    conv = ["Shift: V exp(+2 pi i (u dRA + v dDec)); deprojected q = sqrt(q_maj^2 + (q_min cos i)^2).",
            "Bin error = 1/sqrt(sum w), CASA WEIGHT = 1/sigma^2 per real/imag part (check that weights are calibrated)."]
    return result(out, conv, warnings, UV_PRACTICE)


def frank_profile(
    vis_path: str,
    inclination_deg: float,
    pa_deg: float,
    dra_arcsec: float = 0.0,
    ddec_arcsec: float = 0.0,
    r_max_arcsec: float = 2.0,
    n_points: int = 300,
    alpha: float = 1.05,
    weights_smooth: float = 1e-4,
    uv_unit: str = "lambda",
    freq_ghz: float | None = None,
) -> dict:
    """Non-parametric 1D radial brightness profile with frank (Jennings et al. 2020), if installed.

    Wraps frank.radial_fitters.FrankFitter with a FixedGeometry (inclination, PA east of north, offsets).
    alpha (1.05 default, papers use 1.01-1.3) and weights_smooth (1e-4 default, 1e-1 to 1e-4 used) set the
    Gaussian-process prior; report both. Returns r (arcsec), I (Jy/sr) and 1-sigma errors.
    Limits: frank assumes axisymmetry; its uncertainties are likely underestimated; the innermost 1-2 beams are
    least reliable; visibility weights must be correctly scaled first. Without frank installed this returns the
    install command.
    """
    frank, msg = optional_import("frank", "uv")
    if frank is None:
        return result({"available": False}, "frank (Jennings et al. 2020)", [msg], UV_PRACTICE)
    from frank.geometry import FixedGeometry
    from frank.radial_fitters import FrankFitter

    vis = load_visibilities(vis_path, uv_unit, freq_ghz)
    geom = FixedGeometry(inclination_deg, pa_deg, dRA=dra_arcsec, dDec=ddec_arcsec)
    ff = FrankFitter(Rmax=r_max_arcsec, N=n_points, geometry=geom, alpha=alpha, weights_smooth=weights_smooth)
    sol = ff.fit(vis["u"], vis["v"], vis["re"] + 1j * vis["im"], vis["weight"])
    intensity = getattr(sol, "I", None)
    if intensity is None:
        intensity = sol.mean
    err = None
    cov = getattr(sol, "covariance", None)
    if cov is not None:
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = {"available": True, "r_arcsec": np.asarray(sol.r), "intensity_jy_sr": np.asarray(intensity),
           "intensity_err_jy_sr": err, "settings": {"alpha": alpha, "weights_smooth": weights_smooth, "N": n_points,
                                                    "Rmax_arcsec": r_max_arcsec}}
    warnings = ["frank uncertainties are likely underestimated; the innermost 1-2 beams are least certain."]
    return result(out, "frank FrankFitter with FixedGeometry (Jennings et al. 2020)", warnings, UV_PRACTICE)
