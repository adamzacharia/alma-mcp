import math
import os

import numpy as np
import pytest
from astropy.io import fits

from alma_mcp.analysis import imaging_checks as ic
from alma_mcp.analysis import kinematics as kin
from alma_mcp.analysis import lines
from alma_mcp.analysis._util import (
    ARCSEC_RAD, AU_CM, G_CGS, M_SUN_G, WORKDIR_ENV, check_path, output_path, resolve_in_workdir,
)


def _small_cube(path):
    rng = np.random.default_rng(3)
    data = rng.standard_normal((9, 8, 8)).astype(np.float32) * 0.01
    data[4, 3:5, 3:5] += 1.0
    h = fits.Header()
    h["CTYPE1"], h["CRVAL1"], h["CDELT1"], h["CRPIX1"], h["CUNIT1"] = "RA---SIN", 70.0, -0.1 / 3600, 4, "deg"
    h["CTYPE2"], h["CRVAL2"], h["CDELT2"], h["CRPIX2"], h["CUNIT2"] = "DEC--SIN", 25.0, 0.1 / 3600, 4, "deg"
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"], h["CUNIT3"] = "VRAD", -800.0, 200.0, 1, "m/s"
    h["BMAJ"], h["BMIN"], h["BPA"] = 0.3 / 3600, 0.3 / 3600, 0.0
    h["BUNIT"] = "Jy/beam"
    fits.PrimaryHDU(data, header=h).writeto(path)
    return str(path)


# ------------------------------------------------------------------------------------------ path guard


def test_paths_outside_workdir_are_refused(tmp_path, monkeypatch):
    inside = tmp_path / "work"
    outside = tmp_path / "elsewhere"
    inside.mkdir()
    outside.mkdir()
    monkeypatch.setenv(WORKDIR_ENV, str(inside))
    (inside / "a.txt").write_text("x")
    (outside / "b.txt").write_text("x")
    assert check_path(str(inside / "a.txt")) == os.path.realpath(inside / "a.txt")
    assert check_path("a.txt") == os.path.realpath(inside / "a.txt")  # relative paths resolve from the workdir
    with pytest.raises(PermissionError):
        check_path(str(outside / "b.txt"))
    with pytest.raises(PermissionError):
        check_path(os.path.join("..", "elsewhere", "b.txt"))
    with pytest.raises(PermissionError):
        output_path(str(outside / "new.fits"))
    with pytest.raises(PermissionError):
        resolve_in_workdir(os.path.abspath(os.sep))


def test_output_overwrite_needs_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKDIR_ENV, str(tmp_path))
    (tmp_path / "x.fits").write_text("old")
    with pytest.raises(FileExistsError):
        output_path(str(tmp_path / "x.fits"))
    assert output_path(str(tmp_path / "x.fits"), overwrite=True)
    assert output_path(str(tmp_path / "y.fits"))


def test_moment_maps_respect_workdir_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKDIR_ENV, str(tmp_path / "work"))
    (tmp_path / "work").mkdir()
    cube = _small_cube(tmp_path / "work" / "c.fits")
    r = kin.moment_maps(cube, rms=0.01)
    assert os.path.isfile(r["result"]["files"]["mom0"])
    with pytest.raises(FileExistsError):
        kin.moment_maps(cube, rms=0.01)
    r2 = kin.moment_maps(cube, rms=0.01, overwrite=True)
    assert r2["result"]["files"]["mom0"] == r["result"]["files"]["mom0"]
    with pytest.raises(PermissionError):
        kin.moment_maps(cube, rms=0.01, out_dir=str(tmp_path / "outside"))
    assert not (tmp_path / "outside").exists()
    outside_cube = _small_cube(tmp_path / "c2.fits")
    with pytest.raises(PermissionError):
        kin.moment_maps(outside_cube, rms=0.01)


def test_keplerian_mask_respects_workdir_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKDIR_ENV, str(tmp_path / "work"))
    (tmp_path / "work").mkdir()
    cube = _small_cube(tmp_path / "work" / "c.fits")
    args = (1.0, 30.0, 0.0, 100.0, 0.0, 50.0)
    r = kin.keplerian_mask(cube, *args)
    assert r["result"]["mask_path"].endswith("c.keplerian_mask.fits")
    with pytest.raises(FileExistsError):
        kin.keplerian_mask(cube, *args)
    kin.keplerian_mask(cube, *args, overwrite=True)
    with pytest.raises(PermissionError):
        kin.keplerian_mask(cube, *args, out_path=str(tmp_path / "mask.fits"))
    with pytest.raises(FileExistsError):
        kin.keplerian_mask(cube, *args, out_path=cube)  # never clobber the input cube by default


# ------------------------------------------------------------------------------------------ science fixes


def test_fwhm_flag_only_affects_virial():
    v = math.sqrt(G_CGS * M_SUN_G / (100 * AU_CM)) / 1e5
    r = kin.dynamical_mass(v, 100, "au", velocity_is_projected=False, velocity_is_fwhm=True)
    assert r["result"]["mass"]["value"] == pytest.approx(1.0, rel=1e-9)
    assert any("velocity_is_fwhm ignored" in w for w in r["warnings"])
    rv = kin.dynamical_mass(2.3548200450309493, 1.0, "pc", kind="virial", velocity_is_fwhm=True)
    assert rv["result"]["sigma_1d_kms"] == pytest.approx(1.0)


def test_alpha_error_uses_derived_alpha_with_x_co():
    r = lines.co_gas_mass(1e9, x_co=4e20, alpha_co_err=0.87)
    res = r["result"]
    alpha = res["alpha_mol_with_helium"]
    assert alpha == pytest.approx(8.7, rel=0.01)
    m = res["M_mol_with_helium"]
    assert m["error"] / m["value"] == pytest.approx(0.87 / alpha, rel=1e-9)


def test_mrs_percentiles_over_unique_antenna_pairs(tmp_path):
    # 3 antenna pairs with very different row counts: per-pair percentiles must not be pulled by the busy pair
    lam = 2.99792458e8 / 230e9
    lengths = {(0, 1): 20.0, (0, 2): 200.0, (1, 2): 400.0}
    counts = {(0, 1): 1000, (0, 2): 10, (1, 2): 10}
    u, a1, a2 = [], [], []
    for (i, j), b in lengths.items():
        u += [b / lam] * counts[(i, j)]
        a1 += [i] * counts[(i, j)]
        a2 += [j] * counts[(i, j)]
    n = len(u)
    path = tmp_path / "ants.npz"
    np.savez(path, u=np.array(u), v=np.zeros(n), Re=np.ones(n), Im=np.zeros(n), weight=np.ones(n),
             ant1=np.array(a1), ant2=np.array(a2))
    r = ic.largest_angular_scale(230.0, vis_path=str(path))
    res = r["result"]
    pairs = np.array(sorted(lengths.values()))
    assert res["L80_m"] == pytest.approx(np.percentile(pairs, 80), rel=1e-6)
    assert res["L05_m"] == pytest.approx(np.percentile(pairs, 5), rel=1e-6)
    assert "unique antenna pairs" in res["percentile_basis"]
    # without antenna ids the percentiles are over rows, and the tool says so
    path2 = tmp_path / "rows.npz"
    np.savez(path2, u=np.array(u), v=np.zeros(n), Re=np.ones(n), Im=np.zeros(n), weight=np.ones(n))
    r2 = ic.largest_angular_scale(230.0, vis_path=str(path2))
    assert "visibility rows" in r2["result"]["percentile_basis"]
    assert r2["result"]["L80_m"] == pytest.approx(20.0, rel=1e-6)
    assert any("visibility rows" in w for w in r2["warnings"])
    assert res["mrs_0p983_lambda_over_L05_arcsec"] == pytest.approx(0.983 * lam / res["L05_m"] / ARCSEC_RAD)
