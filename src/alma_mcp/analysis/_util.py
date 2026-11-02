"""Small shared helpers for the analysis tools: constants, JSON-safe output, file loaders."""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Iterable

import numpy as np
from astropy import constants as _c
from astropy import units as _u

# CGS constants (floats, so the pure functions stay fast and simple)
C_CGS = _c.c.cgs.value  # cm s^-1
H_CGS = _c.h.cgs.value  # erg s
K_CGS = _c.k_B.cgs.value  # erg K^-1
G_CGS = _c.G.cgs.value  # cm^3 g^-1 s^-2
M_SUN_G = _c.M_sun.cgs.value
M_EARTH_G = _c.M_earth.cgs.value
M_JUP_G = _c.M_jup.cgs.value
L_SUN_CGS = _c.L_sun.cgs.value
PC_CM = _u.pc.to(_u.cm)
AU_CM = _u.au.to(_u.cm)
MPC_CM = _u.Mpc.to(_u.cm)
KPC_CM = _u.kpc.to(_u.cm)
M_H_G = _c.m_p.cgs.value + _c.m_e.cgs.value  # hydrogen atom mass
JY_CGS = 1e-23  # erg s^-1 cm^-2 Hz^-1
ARCSEC_RAD = math.pi / 180.0 / 3600.0
T_CMB0 = 2.7255  # K, Fixsen 2009
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

# G in astronomer units: pc (km/s)^2 / Msun
G_PC_KMS2_MSUN = G_CGS * M_SUN_G / PC_CM / 1e10


def planck_nu(nu_hz, t_k):
    """Planck function B_nu(T) in erg s^-1 cm^-2 Hz^-1 sr^-1. Works on arrays."""
    nu = np.asarray(nu_hz, dtype=float)
    t = np.asarray(t_k, dtype=float)
    x = H_CGS * nu / (K_CGS * t)
    return 2.0 * H_CGS * nu**3 / C_CGS**2 / np.expm1(x)


def jnu(nu_hz, t_k):
    """Radiation temperature J_nu(T) = (h nu / k) / (exp(h nu / k T) - 1) in K."""
    t0 = H_CGS * np.asarray(nu_hz, dtype=float) / K_CGS
    with np.errstate(over="ignore"):
        return t0 / np.expm1(t0 / np.asarray(t_k, dtype=float))


def jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, NaN and inf to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return obj


def q(value, unit: str, error=None, **extra) -> dict:
    """A value with its unit and (optional) 1-sigma error."""
    out = {"value": value, "unit": unit}
    if error is not None:
        out["error"] = error
    out.update(extra)
    return out


def result(values: dict, convention: str | list, warnings: list[str], corpus_practice: list[str], **extra) -> dict:
    out = {"result": values, "convention": convention, "warnings": list(warnings), "corpus_practice": corpus_practice}
    out.update(extra)
    return jsonable(out)


def as_array(x, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


WORKDIR_ENV = "ALMA_MCP_WORKDIR"


def workdir() -> str:
    """The allowed root for every file the analysis tools read or write.

    Set by the environment variable ALMA_MCP_WORKDIR; defaults to the current working directory of the server.
    """
    return os.path.realpath(os.path.expanduser(os.environ.get(WORKDIR_ENV) or os.getcwd()))


def resolve_in_workdir(path: str) -> str:
    """Absolute, symlink-resolved path that must lie inside workdir(). Relative paths are taken from workdir()."""
    root = workdir()
    p = os.path.expanduser(str(path))
    if not os.path.isabs(p):
        p = os.path.join(root, p)
    p = os.path.realpath(p)
    try:
        inside = os.path.commonpath([os.path.normcase(root), os.path.normcase(p)]) == os.path.normcase(root)
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise PermissionError(f"path {path!r} is outside the allowed folder {root}; "
                              f"set {WORKDIR_ENV} to a folder that contains your data")
    return p


def check_path(path: str) -> str:
    """Input file: must exist and lie inside workdir()."""
    p = resolve_in_workdir(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"file not found: {p}")
    return p


def output_path(path: str, overwrite: bool = False) -> str:
    """Output file: must lie inside workdir(); an existing file is refused unless overwrite is True."""
    p = resolve_in_workdir(path)
    if os.path.isdir(p):
        raise IsADirectoryError(f"output path is a directory: {p}")
    if os.path.exists(p) and not overwrite:
        raise FileExistsError(f"{p} already exists; pass overwrite=true to replace it")
    return p


_VIS_ALIASES = {
    "u": ("u", "u_lambda", "uu"),
    "v": ("v", "v_lambda", "vv"),
    "re": ("re", "real", "vis_re", "re_jy"),
    "im": ("im", "imag", "vis_im", "im_jy"),
    "weight": ("weight", "weights", "wgt", "w"),
    "ant1": ("ant1", "antenna1", "antenna_1", "ant_1"),
    "ant2": ("ant2", "antenna2", "antenna_2", "ant_2"),
}


def _pick(keys: Iterable[str], want: str) -> str:
    low = {k.lower(): k for k in keys}
    for alias in _VIS_ALIASES[want]:
        if alias in low:
            return low[alias]
    raise KeyError(f"column for '{want}' not found; accepted names: {_VIS_ALIASES[want]}")


def load_visibilities(path: str, uv_unit: str = "lambda", freq_ghz: float | None = None) -> dict:
    """Load u, v, Re, Im, weight from .npz or .csv. Returns u, v in wavelengths.

    NPZ: arrays named u, v, Re (or real), Im (or imag), weight. A single complex array 'vis' is also accepted.
    CSV: a header row with the same names. uv_unit is 'lambda', 'klambda' or 'm' (metres; needs freq_ghz).
    Optional antenna id columns (ant1/antenna1, ant2/antenna2) are returned as "ant1", "ant2" when present.
    """
    p = check_path(path)
    if p.lower().endswith(".npz"):
        d = np.load(p)
        keys = list(d.keys())
        u = np.asarray(d[_pick(keys, "u")], float)
        v = np.asarray(d[_pick(keys, "v")], float)
        if "vis" in keys:
            vis = np.asarray(d["vis"])
            re, im = vis.real.astype(float), vis.imag.astype(float)
        else:
            re = np.asarray(d[_pick(keys, "re")], float)
            im = np.asarray(d[_pick(keys, "im")], float)
        w = np.asarray(d[_pick(keys, "weight")], float)
        ants = _optional_cols(keys, lambda k: np.asarray(d[k]))
    else:
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(r for r in fh if not r.startswith("#")))
        if not rows:
            raise ValueError("CSV has no data rows")
        keys = list(rows[0].keys())
        cols = {k: _pick(keys, k) for k in ("u", "v", "re", "im", "weight")}
        u, v, re, im, w = (np.array([float(r[cols[k]]) for r in rows]) for k in ("u", "v", "re", "im", "weight"))
        ants = _optional_cols(keys, lambda k: np.array([r[k] for r in rows]))
    uv_unit = uv_unit.lower()
    if uv_unit == "klambda":
        u, v = u * 1e3, v * 1e3
    elif uv_unit == "m":
        if not freq_ghz:
            raise ValueError("uv_unit='m' needs freq_ghz to convert metres to wavelengths")
        lam_m = C_CGS / 1e2 / (freq_ghz * 1e9)
        u, v = u / lam_m, v / lam_m
    elif uv_unit != "lambda":
        raise ValueError("uv_unit must be 'lambda', 'klambda' or 'm'")
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(re) & np.isfinite(im) & np.isfinite(w) & (w > 0)
    out = {"u": u[good], "v": v[good], "re": re[good], "im": im[good], "weight": w[good],
           "n_dropped": int((~good).sum())}
    if ants is not None:
        out["ant1"], out["ant2"] = ants[0][good], ants[1][good]
    return out


def _optional_cols(keys, get):
    """(ant1, ant2) arrays when both antenna id columns exist, else None."""
    try:
        return get(_pick(keys, "ant1")), get(_pick(keys, "ant2"))
    except KeyError:
        return None


def optional_import(name: str, extra: str):
    """Import an optional package or return (None, install message)."""
    try:
        mod = __import__(name)
        return mod, None
    except Exception:  # pragma: no cover - depends on environment
        return None, f"'{name}' is not installed. Install it with: pip install 'alma-mcp[{extra}]' (or pip install {name})"
