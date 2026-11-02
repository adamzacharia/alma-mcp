"""Kinematics: moment maps, Keplerian masks, dynamical and virial masses, Toomre Q.

Conventions follow the corpus skills moment-maps, keplerian-disk-kinematics, gmc-decomposition-virial and
toomre-stability.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ._util import (
    AU_CM, C_CGS, FWHM_TO_SIGMA, G_CGS, G_PC_KMS2_MSUN, K_CGS, KPC_CM, M_H_G, M_SUN_G, PC_CM,
    check_path, optional_import, output_path, q, result, resolve_in_workdir,
)

MOM_PRACTICE = ["moment-maps", "galaxy-3d-kinematics"]
KEP_PRACTICE = ["keplerian-disk-kinematics", "moment-maps"]
DYN_PRACTICE = ["keplerian-disk-kinematics", "smbh-gas-dynamics", "galaxy-3d-kinematics", "gmc-decomposition-virial"]
VIR_PRACTICE = ["gmc-decomposition-virial", "core-extraction-cmf"]
TOOMRE_PRACTICE = ["toomre-stability"]
C_KMS = C_CGS / 1e5

_SPEC_TYPES = ("FREQ", "VRAD", "VELO", "VOPT", "FELO", "VELOCITY")


# ------------------------------------------------------------------------------------------ cube I/O


def read_cube(path: str, rest_freq_ghz: float | None = None) -> dict:
    """Read a FITS cube, drop degenerate (Stokes) axes, return data[nchan, ny, nx] and radio velocities in km/s."""
    from astropy.io import fits

    p = check_path(path)
    with fits.open(p) as hdul:
        hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 3)
        hdr = hdu.header.copy()
        data = np.asarray(hdu.data, dtype=float)
    naxis = hdr["NAXIS"]
    spec_ax = None
    for i in range(1, naxis + 1):
        ct = str(hdr.get(f"CTYPE{i}", "")).upper()
        if ct.startswith(_SPEC_TYPES):
            spec_ax = i
    if spec_ax is None:
        raise ValueError("no spectral axis (FREQ/VRAD/VELO/VOPT) found in the FITS header")
    # numpy axes are reversed relative to FITS
    np_spec = naxis - spec_ax
    keep = [naxis - 1, naxis - 2, np_spec]  # x (FITS 1), y (FITS 2), spectral
    for ax in range(data.ndim):
        if ax not in keep and data.shape[ax] != 1:
            raise ValueError("cube has a non-degenerate extra axis (e.g. several Stokes planes); select one first")
    # move to (spec, y, x)
    data = np.moveaxis(data, [np_spec, naxis - 2, naxis - 1], [0, 1, 2])
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2]) if data.ndim == 3 else data.reshape(
        data.shape[0], data.shape[1], data.shape[2], -1)[..., 0]
    n = data.shape[0]
    crval, cdelt, crpix = hdr[f"CRVAL{spec_ax}"], hdr[f"CDELT{spec_ax}"], hdr[f"CRPIX{spec_ax}"]
    world = crval + (np.arange(n) + 1 - crpix) * cdelt
    ct = str(hdr[f"CTYPE{spec_ax}"]).upper()
    unit = str(hdr.get(f"CUNIT{spec_ax}", "")).strip().lower()
    warnings = []
    if ct.startswith("FREQ"):
        rf = rest_freq_ghz * 1e9 if rest_freq_ghz else hdr.get("RESTFRQ", hdr.get("RESTFREQ"))
        if not rf:
            raise ValueError("frequency axis without RESTFRQ in the header: pass rest_freq_ghz")
        scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit, 1.0)
        vel = C_KMS * (1.0 - world * scale / float(rf))
        vnote = f"radio velocity v = c (1 - nu/nu0), nu0 = {float(rf) / 1e9:.6f} GHz, frame as in the header (SPECSYS={hdr.get('SPECSYS', '?')})"
    else:
        scale = {"m/s": 1e-3, "km/s": 1.0, "": 1e-3}.get(unit, 1e-3)
        vel = world * scale
        vnote = f"velocity axis {ct} taken as is (SPECSYS={hdr.get('SPECSYS', '?')})"
        if ct.startswith(("VOPT", "FELO")):
            warnings.append("Optical-convention velocity axis: not the radio convention used by most ALMA products.")
    if vel.size > 1 and vel[1] < vel[0]:
        data = data[::-1]
        vel = vel[::-1]
    pix = abs(float(hdr.get("CDELT2", hdr.get("CD2_2", 0.0)))) * 3600.0
    bmaj = hdr.get("BMAJ")
    bmin = hdr.get("BMIN")
    beam = None
    if bmaj and bmin:
        beam = {"bmaj_arcsec": float(bmaj) * 3600, "bmin_arcsec": float(bmin) * 3600, "bpa_deg": float(hdr.get("BPA", 0.0))}
    # 2D celestial header for outputs
    from astropy.wcs import WCS
    w2 = WCS(hdr).celestial
    h2 = w2.to_header()
    for k in ("BMAJ", "BMIN", "BPA", "OBJECT", "TELESCOP"):
        if k in hdr:
            h2[k] = hdr[k]
    return {"data": data, "velocity_kms": vel, "header2d": h2, "wcs2d": w2, "pix_arcsec": pix, "beam": beam,
            "bunit": str(hdr.get("BUNIT", "")).strip(), "velocity_note": vnote, "warnings": warnings}


def _write2d(path, arr, h2, bunit):
    from astropy.io import fits
    hh = h2.copy()
    hh["BUNIT"] = bunit
    fits.PrimaryHDU(np.asarray(arr, dtype=np.float32), header=hh).writeto(path, overwrite=True)


def _robust_rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def beam_area_pixels(beam: dict | None, pix_arcsec: float) -> float | None:
    if not beam or not pix_arcsec:
        return None
    return math.pi * beam["bmaj_arcsec"] * beam["bmin_arcsec"] / (4 * math.log(2)) / pix_arcsec**2


def moment_maps(
    fits_path: str,
    v_min_kms: float | None = None,
    v_max_kms: float | None = None,
    rms: float | None = None,
    line_free_channels: list[list[int]] | None = None,
    clip_sigma_m0: float | None = None,
    clip_sigma_m12: float = 3.0,
    mask_path: str | None = None,
    out_dir: str | None = None,
    rest_freq_ghz: float | None = None,
    overwrite: bool = False,
) -> dict:
    """Moment 0/1/2, peak intensity and a quadratic peak velocity from a FITS cube.

    Velocity range v_min_kms..v_max_kms selects channels. rms per channel comes from `rms`, or the robust (MAD)
    rms of line_free_channels ([[start, end], ...], inclusive indices after the cube is sorted by increasing
    velocity), or of channels outside the velocity range, or of the whole cube (warned). Masking: an optional
    mask cube (non-zero = keep, same shape), else clip_sigma_m0 (None = no clip, the unbiased default for
    moment 0) and clip_sigma_m12 (default 3 sigma) for moments 1 and 2.
    M0 = sum I dv (BUNIT km/s); noise map = rms dv sqrt(N_chan in mask); M1 = sum I v / sum I;
    M2 = sqrt(sum I (v - M1)^2 / sum I); peak = max I; v_peak from a parabola through the peak channel and its
    neighbours (Teague and Foreman-Mackey 2018, bettermoments-style). Writes FITS maps <stem>.<map>.fits into
    out_dir (default: the cube's folder). All paths must lie inside the allowed folder (env ALMA_MCP_WORKDIR,
    default the server's working directory); existing map files are refused unless overwrite=true.
    Limits: hard clips bias M0 low and M1 toward bright gas; M2 includes beam smearing and channel width, not
    only turbulence; one centroid averages multi-component lines; interferometer-only maps miss extended flux.
    """
    warnings: list[str] = []
    fits_path = check_path(fits_path)
    out_dir = resolve_in_workdir(out_dir) if out_dir else os.path.dirname(fits_path)
    stem = os.path.splitext(os.path.basename(fits_path))[0]
    map_names = ("mom0", "mom0_noise", "mom1", "mom2", "peak", "vpeak_quadratic", "fpeak_quadratic")
    targets = {n: output_path(os.path.join(out_dir, f"{stem}.{n}.fits"), overwrite) for n in map_names}
    cube = read_cube(fits_path, rest_freq_ghz)
    warnings += cube["warnings"]
    data, vel = cube["data"], cube["velocity_kms"]
    dv = float(np.median(np.abs(np.diff(vel)))) if vel.size > 1 else 1.0
    sel = np.ones(vel.size, bool)
    if v_min_kms is not None:
        sel &= vel >= v_min_kms
    if v_max_kms is not None:
        sel &= vel <= v_max_kms
    if sel.sum() < 1:
        raise ValueError("no channels in the requested velocity range")
    if rms is None:
        if line_free_channels:
            idx = np.zeros(vel.size, bool)
            for a, b in line_free_channels:
                idx[int(a):int(b) + 1] = True
            rms = _robust_rms(data[idx])
            rnote = f"robust rms of {int(idx.sum())} line-free channels"
        elif (~sel).sum() >= 3:
            rms = _robust_rms(data[~sel])
            rnote = f"robust rms of {int((~sel).sum())} channels outside the velocity range"
        else:
            rms = _robust_rms(data)
            rnote = "robust rms of the whole cube (no line-free channels given; biased high if line emission is extended)"
            warnings.append("rms estimated from the whole cube; pass line_free_channels or rms.")
    else:
        rnote = "rms given"
    d = data[sel]
    v = vel[sel]
    if mask_path:
        from astropy.io import fits
        m = np.squeeze(fits.getdata(check_path(mask_path))).astype(bool)
        if m.shape != data.shape:
            raise ValueError(f"mask shape {m.shape} does not match cube {data.shape} (after dropping Stokes)")
        m = m[sel]
        mask0 = m12 = m
        mnote = "user mask"
    else:
        mask0 = np.ones_like(d, bool) if clip_sigma_m0 is None else d > clip_sigma_m0 * rms
        m12 = d > clip_sigma_m12 * rms
        mnote = f"moment 0 clip: {'none' if clip_sigma_m0 is None else f'{clip_sigma_m0} sigma'}; moments 1/2 clip: {clip_sigma_m12} sigma"
    finite = np.isfinite(d)
    d0 = np.where(mask0 & finite, d, 0.0)
    m0 = d0.sum(axis=0) * dv
    nchan = (mask0 & finite).sum(axis=0)
    m0_noise = rms * dv * np.sqrt(np.maximum(nchan, 1))
    d12 = np.where(m12 & finite, d, 0.0)
    s = d12.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (d12 * v[:, None, None]).sum(axis=0) / s
        m2 = np.sqrt(np.clip((d12 * (v[:, None, None] - m1) ** 2).sum(axis=0) / s, 0, None))
    good12 = (s > 0) & ((m12 & finite).sum(axis=0) >= 2)
    m1 = np.where(good12, m1, np.nan)
    m2 = np.where(good12, m2, np.nan)
    dd = np.where(finite, d, -np.inf)
    k = np.argmax(dd, axis=0)
    peak = np.take_along_axis(dd, k[None], 0)[0]
    # quadratic peak velocity
    bm, _ = optional_import("bettermoments", "cube")
    vq_method = "parabola through the peak channel and its two neighbours (bettermoments-style, own implementation)"
    km = np.clip(k, 1, d.shape[0] - 2) if d.shape[0] >= 3 else k
    if d.shape[0] >= 3:
        ym = np.take_along_axis(d, (km - 1)[None], 0)[0]
        y0 = np.take_along_axis(d, km[None], 0)[0]
        yp = np.take_along_axis(d, (km + 1)[None], 0)[0]
        den = ym - 2 * y0 + yp
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = np.where(den != 0, 0.5 * (ym - yp) / den, 0.0)
        delta = np.clip(delta, -1, 1)
        vstep = v[1] - v[0] if v.size > 1 else dv
        vpeak = v[km] + delta * vstep
        fpeak = y0 - 0.25 * (ym - yp) * delta
    else:
        vpeak = v[k]
        fpeak = peak
    detected = peak > clip_sigma_m12 * rms
    vpeak = np.where(detected, vpeak, np.nan)
    unit = cube["bunit"] or "unknown"
    out = {"rms_per_channel": q(rms, unit), "rms_method": rnote, "channel_width_kms": dv,
           "n_channels_used": int(sel.sum()), "velocity_range_kms": [float(v.min()), float(v.max())],
           "masking": mnote, "velocity_axis": cube["velocity_note"], "beam": cube["beam"]}
    bpix = beam_area_pixels(cube["beam"], cube["pix_arcsec"])
    det0 = m0 > 3 * m0_noise
    if bpix and unit.lower().replace(" ", "") in ("jy/beam", "jybeam-1", "jy/beam."):
        tot = float(m0[det0].sum() / bpix)
        err = float(math.sqrt(np.sum(m0_noise[det0] ** 2) / bpix)) if det0.any() else None
        out["integrated_flux_m0_above_3sigma"] = q(tot, "Jy km/s", err, n_pixels=int(det0.sum()),
                                                    note="sum of moment 0 pixels > 3 sigma_M0 divided by beam area in pixels; error is rms x sqrt(N_beams), i.e. noise correlated within a beam")
    else:
        warnings.append("BUNIT is not Jy/beam or the beam is missing: no integrated flux computed.")
    out["peak_max"] = q(float(np.nanmax(np.where(np.isfinite(peak), peak, np.nan))), unit)
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for name, arr, bu in (("mom0", m0, f"{unit}.km/s"), ("mom0_noise", m0_noise, f"{unit}.km/s"),
                          ("mom1", m1, "km/s"), ("mom2", m2, "km/s"), ("peak", peak, unit),
                          ("vpeak_quadratic", vpeak, "km/s"), ("fpeak_quadratic", fpeak, unit)):
        pth = targets[name]
        _write2d(pth, np.where(np.isfinite(arr), arr, np.nan), cube["header2d"], bu)
        files[name] = pth
    out["files"] = files
    out["quadratic_method"] = vq_method + ("" if bm is None else "; bettermoments is installed and can be used for its full uncertainty maps")
    if clip_sigma_m0 is not None:
        warnings.append("Moment 0 with a hard clip is biased low; prefer a smoothed-cube or Keplerian mask.")
    warnings.append("Moment 2 includes beam smearing and channel broadening; it is not turbulence by itself.")
    conv = ["M0 = sum(I) dv; M1 = sum(I v)/sum(I); M2 = sqrt(sum(I (v-M1)^2)/sum(I)); noise(M0) = rms dv sqrt(N).",
            "Quadratic peak: Teague and Foreman-Mackey 2018 (bettermoments)."]
    return result(out, conv, warnings, MOM_PRACTICE)


# ------------------------------------------------------------------------------------------ Keplerian mask


def keplerian_mask(
    fits_path: str,
    m_star_msun: float,
    inclination_deg: float,
    pa_deg: float,
    distance_pc: float,
    v_sys_kms: float,
    r_max_au: float,
    r_min_au: float = 0.0,
    line_width_kms: float = 0.5,
    center_ra_deg: float | None = None,
    center_dec_deg: float | None = None,
    dilate_beams: float = 1.0,
    out_path: str | None = None,
    rest_freq_ghz: float | None = None,
    overwrite: bool = False,
) -> dict:
    """Build a Keplerian mask cube for a thin disk on the grid of a template FITS cube.

    Disk coordinates: x_d = dRA sin(PA) + dDec cos(PA), y_d = (-dRA cos(PA) + dDec sin(PA)) / cos(i), r = sqrt(x_d^2 + y_d^2)
    with dRA (east) and dDec offsets in arcsec from the disk centre (default: the reference pixel of the cube).
    PA is east of north to the REDSHIFTED major axis. v_los = v_sys + sqrt(G M / r) sin(i) x_d / r.
    A voxel is in the mask if r_min <= r <= r_max and |v_chan - v_los| <= line_width_kms + dv/2. Each channel is
    then dilated by an elliptical beam (dilate_beams x beam FWHM). Writes a 0/1 FITS cube (spectral axis increasing)
    to out_path (default <cube stem>.keplerian_mask.fits next to the cube). Paths must lie inside the allowed folder
    (env ALMA_MCP_WORKDIR, default the server's working directory); an existing file is refused unless overwrite=true.
    Limits: thin disk only (no elevated emission surface, so 12CO back sides can fall outside); a mask built from
    assumed M*, i, PA biases weak-line maps toward that model.
    """
    from astropy.io import fits
    from scipy.ndimage import binary_dilation

    warnings: list[str] = []
    fits_path = check_path(fits_path)
    if out_path is None:
        out_path = os.path.splitext(fits_path)[0] + ".keplerian_mask.fits"
    out_path = output_path(out_path, overwrite)
    cube = read_cube(fits_path, rest_freq_ghz)
    data, vel = cube["data"], cube["velocity_kms"]
    nch, ny, nx = data.shape
    w2 = cube["wcs2d"]
    if center_ra_deg is not None and center_dec_deg is not None:
        x0, y0 = w2.world_to_pixel_values(center_ra_deg, center_dec_deg)
    else:
        x0, y0 = w2.wcs.crpix[0] - 1, w2.wcs.crpix[1] - 1
    cd = w2.pixel_scale_matrix * 3600.0  # arcsec per pixel
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx_pix, dy_pix = xx - x0, yy - y0
    d_ra = cd[0, 0] * dx_pix + cd[0, 1] * dy_pix  # east offset (CDELT1 < 0 handles the sign)
    d_dec = cd[1, 0] * dx_pix + cd[1, 1] * dy_pix
    pa = math.radians(pa_deg)
    inc = math.radians(inclination_deg)
    if abs(math.cos(inc)) < 0.05:
        warnings.append("Inclination near 90 deg: thin-disk deprojection is ill-conditioned.")
    xd = d_ra * math.sin(pa) + d_dec * math.cos(pa)
    yd = (-d_ra * math.cos(pa) + d_dec * math.sin(pa)) / max(abs(math.cos(inc)), 1e-3)
    r_as = np.hypot(xd, yd)
    r_au = r_as * distance_pc
    with np.errstate(invalid="ignore", divide="ignore"):
        vkep = np.sqrt(G_CGS * m_star_msun * M_SUN_G / (np.maximum(r_au, 1e-3) * AU_CM)) / 1e5
        cosphi = np.where(r_as > 0, xd / r_as, 0.0)
    vlos = v_sys_kms + vkep * math.sin(inc) * cosphi
    dv = float(np.median(np.abs(np.diff(vel)))) if vel.size > 1 else 0.0
    inr = (r_au >= r_min_au) & (r_au <= r_max_au)
    mask = (np.abs(vel[:, None, None] - vlos[None]) <= line_width_kms + dv / 2.0) & inr[None]
    if dilate_beams and cube["beam"] and cube["pix_arcsec"]:
        b = cube["beam"]
        a_pix = dilate_beams * b["bmaj_arcsec"] / 2 / cube["pix_arcsec"]
        b_pix = dilate_beams * b["bmin_arcsec"] / 2 / cube["pix_arcsec"]
        n = int(math.ceil(a_pix))
        ky, kx = np.mgrid[-n:n + 1, -n:n + 1]
        th = math.radians(b["bpa_deg"])
        # beam PA east of north: major axis along (sin, cos) in (east, north); east is -x for standard images
        e_off, n_off = -kx, ky
        u = e_off * math.sin(th) + n_off * math.cos(th)
        vv = -e_off * math.cos(th) + n_off * math.sin(th)
        struct = (u / max(a_pix, 0.5)) ** 2 + (vv / max(b_pix, 0.5)) ** 2 <= 1.0
        mask = np.array([binary_dilation(mask[c], structure=struct) for c in range(nch)])
    elif dilate_beams:
        warnings.append("No beam in the header: mask not dilated.")
    h = cube["header2d"].copy()
    h["NAXIS"] = 3
    h["CTYPE3"] = "VRAD"
    h["CUNIT3"] = "km/s"
    h["CRVAL3"] = float(vel[0])
    h["CDELT3"] = float(vel[1] - vel[0]) if vel.size > 1 else 1.0
    h["CRPIX3"] = 1.0
    h["BUNIT"] = ""
    fits.PrimaryHDU(mask.astype(np.uint8), header=h).writeto(out_path, overwrite=overwrite)
    out = {"mask_path": out_path, "shape_chan_y_x": list(mask.shape), "fraction_of_voxels": float(mask.mean()),
           "channels_with_mask": int(mask.any(axis=(1, 2)).sum()),
           "max_projected_velocity_kms": float(np.nanmax(np.abs(vlos[inr] - v_sys_kms))) if inr.any() else None,
           "centre_pixel_xy": [float(x0), float(y0)]}
    warnings.append("Thin disk: elevated emission surfaces (12CO z/r ~ 0.2-0.3) are not modelled; widen line_width_kms or dilate more.")
    conv = ["PA east of north to the redshifted major axis; v_los = v_sys + sqrt(GM/r) sin i cos phi.",
            "Output spectral axis: radio velocity, increasing, same channels as the template."]
    return result(out, conv, warnings, KEP_PRACTICE)


# ------------------------------------------------------------------------------------------ masses


def _radius_cm(radius: float, unit: str, distance_pc: float | None) -> float:
    unit = unit.lower()
    if unit == "au":
        return radius * AU_CM
    if unit == "pc":
        return radius * PC_CM
    if unit == "kpc":
        return radius * KPC_CM
    if unit == "arcsec":
        if not distance_pc:
            raise ValueError("radius in arcsec needs distance_pc")
        return radius * distance_pc * AU_CM
    raise ValueError("radius_unit must be au, pc, kpc or arcsec")


def dynamical_mass(
    velocity_kms: float,
    radius: float,
    radius_unit: str = "au",
    kind: str = "keplerian",
    inclination_deg: float | None = None,
    velocity_is_projected: bool = True,
    distance_pc: float | None = None,
    velocity_err_kms: float | None = None,
    radius_err: float | None = None,
    inclination_err_deg: float | None = None,
    density_index: float = 0.0,
    velocity_is_fwhm: bool = False,
) -> dict:
    """Enclosed dynamical mass.

    kind='keplerian': M = v_rot^2 r / G with v_rot = v_obs / sin(i) when velocity_is_projected (needs inclination).
    kind='virial': M_vir = (5 / a) sigma^2 R / G, a = (1 - k/3) / (1 - 2k/5) for rho ~ r^-k (k = density_index;
    k = 0 gives 5 sigma^2 R / G, k = 1 gives 1040 sigma^2 R in Msun, km/s, pc). sigma is the 1D dispersion; set
    velocity_is_fwhm to pass a FWHM (sigma = FWHM / 2.3548). velocity_is_fwhm applies to kind='virial' only; a
    Keplerian rotation velocity is never a line width and is used as given.
    Errors are propagated linearly from velocity, radius and inclination errors.
    Limits: M scales with distance (r in arcsec); the mass-inclination degeneracy dominates for low i; pressure
    support and self-gravity are not separated from the central mass.
    """
    warnings: list[str] = []
    r = _radius_cm(radius, radius_unit, distance_pc)
    v = float(velocity_kms)
    kind = kind.lower()
    if velocity_is_fwhm:
        if kind == "virial":
            v = v * FWHM_TO_SIGMA
        else:
            warnings.append("velocity_is_fwhm ignored: it applies to kind='virial' only, not to a rotation velocity.")
    rel_v = (velocity_err_kms or 0.0) / velocity_kms
    rel_r = (radius_err or 0.0) / radius
    if kind == "keplerian":
        sini = 1.0
        rel_i = 0.0
        if velocity_is_projected:
            if inclination_deg is None:
                raise ValueError("a projected velocity needs inclination_deg (or set velocity_is_projected=False)")
            sini = math.sin(math.radians(inclination_deg))
            if inclination_err_deg:
                rel_i = 2 * abs(math.cos(math.radians(inclination_deg)) / sini) * math.radians(inclination_err_deg)
            if inclination_deg < 30:
                warnings.append("Inclination < 30 deg: M is proportional to 1/sin^2(i) and very sensitive to i.")
        vrot = v / sini
        m = (vrot * 1e5) ** 2 * r / G_CGS / M_SUN_G
        rel = math.sqrt((2 * rel_v) ** 2 + rel_r**2 + rel_i**2)
        conv = "Keplerian: M = (v_obs / sin i)^2 r / G"
        out = {"mass": q(m, "Msun", m * rel), "v_rot_kms": vrot, "radius_au": r / AU_CM, "radius_pc": r / PC_CM}
    elif kind == "virial":
        k = density_index
        a = (1 - k / 3) / (1 - 2 * k / 5)
        m = 5.0 / a * (v * 1e5) ** 2 * r / G_CGS / M_SUN_G
        rel = math.sqrt((2 * rel_v) ** 2 + rel_r**2)
        conv = f"virial: M = (5/a) sigma^2 R / G with a = {a:.4f} for rho ~ r^-{k}; coefficient {5 / a / G_PC_KMS2_MSUN:.1f} Msun per (pc (km/s)^2)"
        out = {"mass": q(m, "Msun", m * rel), "sigma_1d_kms": v, "radius_pc": r / PC_CM, "a_profile": a}
        if inclination_deg is not None:
            warnings.append("inclination ignored for kind='virial'.")
    else:
        raise ValueError("kind must be 'keplerian' or 'virial'")
    if radius_unit == "arcsec":
        warnings.append("Mass is proportional to the adopted distance.")
    return result(out, conv, warnings, DYN_PRACTICE)


def virial_parameter(
    sigma_kms: float,
    radius_pc: float,
    mass_msun: float,
    density_index: float = 0.0,
    sigma_is_fwhm: bool = False,
    radius_from_sigma_r_pc: float | None = None,
    sigma_err_kms: float | None = None,
    radius_err_pc: float | None = None,
    mass_err_msun: float | None = None,
) -> dict:
    """Virial parameter alpha_vir = 5 sigma^2 R / (G M) (Bertoldi and McKee 1992), called alpha_5 here.

    sigma: 1D velocity dispersion (km/s); pass a FWHM with sigma_is_fwhm. R: radius in pc, or give
    radius_from_sigma_r_pc so R = 1.91 sigma_r (Solomon et al. 1987 convention). For rho ~ r^-k:
    a = (1 - k/3)/(1 - 2k/5), alpha_E = 2 E_K/|E_G| = alpha_5 / a; equilibrium alpha_5 = a, bound if alpha_5 < 2a
    (equivalently alpha_E < 2). Never apply a profile-corrected coefficient and then also the 2a threshold.
    Limits: the mass conversion (alpha_CO, dust T and kappa) dominates; unresolved clouds give limits only;
    surface pressure and magnetic terms are ignored.
    """
    warnings: list[str] = []
    s = sigma_kms * FWHM_TO_SIGMA if sigma_is_fwhm else sigma_kms
    if radius_from_sigma_r_pc is not None:
        radius_pc = 1.91 * radius_from_sigma_r_pc
        rnote = "R = 1.91 sigma_r (Solomon et al. 1987)"
    else:
        rnote = "R as given"
    k = density_index
    a = (1 - k / 3) / (1 - 2 * k / 5)
    alpha5 = 5 * s**2 * radius_pc / (G_PC_KMS2_MSUN * mass_msun)
    rel = math.sqrt((2 * (sigma_err_kms or 0) / sigma_kms) ** 2 + ((radius_err_pc or 0) / radius_pc) ** 2
                    + ((mass_err_msun or 0) / mass_msun) ** 2)
    m_vir = 5 * s**2 * radius_pc / G_PC_KMS2_MSUN / a
    out = {"alpha_5": q(alpha5, "", alpha5 * rel if rel else None), "alpha_E": q(alpha5 / a, "", alpha5 / a * rel if rel else None),
           "a_profile": a, "equilibrium_alpha_5": a, "bound_limit_alpha_5": 2 * a,
           "gravitationally_bound": bool(alpha5 < 2 * a), "virial_mass_equilibrium": q(m_vir, "Msun"),
           "sigma_1d_kms": s, "radius_pc": radius_pc}
    if not rel:
        warnings.append("No uncertainties given; the mass conversion usually carries a factor of 2 or more.")
    warnings.append("Surface pressure, magnetic fields and tidal terms are not included.")
    conv = ["alpha_5 = 5 sigma_1D^2 R / (G M); G = 4.301e-3 pc (km/s)^2 / Msun.", rnote,
            f"density profile rho ~ r^-{k}: a = {a:.4f}"]
    return result(out, conv, warnings, VIR_PRACTICE)


def toomre_q(
    mode: str = "gas",
    sigma_kms: float | None = None,
    surface_density_msun_pc2: float | None = None,
    surface_density_g_cm2: float | None = None,
    kappa_kms_kpc: float | None = None,
    rotation_velocity_kms: float | None = None,
    radius_kpc: float | None = None,
    epicyclic_a: float = math.sqrt(2.0),
    m_star_msun: float | None = None,
    radius_au: float | None = None,
    temperature_k: float | None = None,
    mu: float = 2.3,
    sigma_star_kms: float | None = None,
    surface_density_star_msun_pc2: float | None = None,
    sigma_z_over_r_star: float = 0.6,
    sigma_z_over_r_gas: float = 1.0,
) -> dict:
    """Toomre Q stability parameter.

    mode='gas': Q = sigma kappa / (pi G Sigma) (gas, Toomre 1964 fluid form).
    mode='stars': Q = sigma_R kappa / (3.36 G Sigma) (collisionless).
    mode='keplerian_disk': protoplanetary disk, Q = c_s Omega / (pi G Sigma) with Omega = kappa = sqrt(G M*/r^3)
    and c_s = sqrt(k T / (mu m_H)), mu default 2.3; needs m_star_msun, radius_au, temperature_k and a surface density.
    mode='two_fluid': Romeo and Wiegert 2011 with thickness corrections T = 0.8 + 0.7 sigma_z/sigma_R:
    1/Q = W/(T_s Q_s) + 1/(T_g Q_g) if T_s Q_s >= T_g Q_g, else 1/(T_s Q_s) + W/(T_g Q_g),
    W = 2 sigma_s sigma_g / (sigma_s^2 + sigma_g^2), with Q_s = sigma_s kappa/(pi G Sigma_s) as in RW11.
    kappa: give kappa_kms_kpc, or rotation_velocity_kms and radius_kpc with kappa = epicyclic_a V/R
    with a = sqrt(2 (1 + dlnV/dlnR)): sqrt(2) for a flat curve, sqrt(3) for V ~ R^0.5, 2 for solid body.
    Limits: Sigma from dust or CO carries the conversion factor; thick or resolved-out emission makes Sigma a lower
    limit (Q an upper limit); observed dispersions include beam smearing; Q < 1-2 is not a sharp threshold.
    """
    warnings: list[str] = []
    mode = mode.lower()
    if surface_density_g_cm2 is not None:
        sig_cgs = surface_density_g_cm2
    elif surface_density_msun_pc2 is not None:
        sig_cgs = surface_density_msun_pc2 * M_SUN_G / PC_CM**2
    else:
        raise ValueError("give surface_density_msun_pc2 or surface_density_g_cm2")
    out: dict = {}
    if mode == "keplerian_disk":
        if None in (m_star_msun, radius_au, temperature_k):
            raise ValueError("keplerian_disk needs m_star_msun, radius_au and temperature_k")
        r = radius_au * AU_CM
        omega = math.sqrt(G_CGS * m_star_msun * M_SUN_G / r**3)
        cs = math.sqrt(K_CGS * temperature_k / (mu * M_H_G))
        qv = cs * omega / (math.pi * G_CGS * sig_cgs)
        out = {"Q": q(qv, ""), "c_s_kms": cs / 1e5, "omega_s-1": omega}
        conv = f"Q = c_s Omega / (pi G Sigma), Omega = sqrt(G M*/r^3), c_s = sqrt(kT/(mu m_H)), mu = {mu}"
        warnings.append("Temperature dominates: Q moves by about sqrt(T); quote Q for a temperature range.")
    else:
        if kappa_kms_kpc is None:
            if rotation_velocity_kms is None or radius_kpc is None:
                raise ValueError("give kappa_kms_kpc or rotation_velocity_kms + radius_kpc")
            kappa_kms_kpc = epicyclic_a * rotation_velocity_kms / radius_kpc
            knote = f"kappa = {epicyclic_a:.4f} V/R"
        else:
            knote = "kappa given"
        kappa = kappa_kms_kpc * 1e5 / KPC_CM  # s^-1
        if mode in ("gas", "stars"):
            if sigma_kms is None:
                raise ValueError("give sigma_kms")
            denom = math.pi if mode == "gas" else 3.36
            qv = sigma_kms * 1e5 * kappa / (denom * G_CGS * sig_cgs)
            out = {"Q": q(qv, ""), "kappa_kms_kpc": kappa_kms_kpc}
            conv = f"Q = sigma kappa / ({'pi' if mode == 'gas' else '3.36'} G Sigma); {knote}"
        elif mode == "two_fluid":
            if None in (sigma_kms, sigma_star_kms, surface_density_star_msun_pc2):
                raise ValueError("two_fluid needs sigma_kms (gas), sigma_star_kms and surface_density_star_msun_pc2")
            sig_s = surface_density_star_msun_pc2 * M_SUN_G / PC_CM**2
            qg = sigma_kms * 1e5 * kappa / (math.pi * G_CGS * sig_cgs)
            qs = sigma_star_kms * 1e5 * kappa / (math.pi * G_CGS * sig_s)
            tg = 0.8 + 0.7 * sigma_z_over_r_gas
            ts = 0.8 + 0.7 * sigma_z_over_r_star
            w = 2 * sigma_star_kms * sigma_kms / (sigma_star_kms**2 + sigma_kms**2)
            if ts * qs >= tg * qg:
                inv = w / (ts * qs) + 1 / (tg * qg)
            else:
                inv = 1 / (ts * qs) + w / (tg * qg)
            out = {"Q": q(1 / inv, ""), "Q_gas": qg, "Q_stars": qs, "W": w, "T_gas": tg, "T_stars": ts,
                   "kappa_kms_kpc": kappa_kms_kpc}
            conv = f"Romeo and Wiegert 2011 two-component Q with thickness factors; {knote}"
        else:
            raise ValueError("mode must be gas, stars, keplerian_disk or two_fluid")
    qv = out["Q"]["value"]
    out["interpretation"] = "unstable (Q < 1)" if qv < 1 else ("marginal (1 <= Q < 2)" if qv < 2 else "stable (Q >= 2)")
    warnings.append("Sigma uncertainties (conversion factor, opacity, missing flux) usually dominate Q.")
    return result(out, conv, warnings, TOOMRE_PRACTICE)
