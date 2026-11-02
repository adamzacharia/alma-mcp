import math

import numpy as np
import pytest
from astropy.io import fits

from alma_mcp.analysis import kinematics as kin
from alma_mcp.analysis._util import AU_CM, G_CGS, G_PC_KMS2_MSUN, M_SUN_G

M, INC, PA, D, VSYS = 1.0, 40.0, 60.0, 100.0, 5.0
NX = NY = 64
PIX = 0.05  # arcsec
DV = 0.2
V0 = -3.0
NCH = 81
SIG_LINE = 0.3
RMS = 0.002


def vlos_sky(d_ra, d_dec):
    pa, inc = math.radians(PA), math.radians(INC)
    xd = d_ra * math.sin(pa) + d_dec * math.cos(pa)
    yd = (-d_ra * math.cos(pa) + d_dec * math.sin(pa)) / math.cos(inc)
    r = np.hypot(xd, yd)
    vk = np.sqrt(G_CGS * M * M_SUN_G / (np.maximum(r * D, 1e-3) * AU_CM)) / 1e5
    return VSYS + vk * math.sin(inc) * np.where(r > 0, xd / np.maximum(r, 1e-12), 0), r


@pytest.fixture(scope="module")
def disk_cube(tmp_path_factory):
    rng = np.random.default_rng(11)
    x0 = y0 = 31.0  # 0-based reference pixel (CRPIX 32)
    yy, xx = np.mgrid[0:NY, 0:NX]
    d_ra = -(xx - x0) * PIX  # east is -x
    d_dec = (yy - y0) * PIX
    vl, r = vlos_sky(d_ra, d_dec)
    amp = 0.05 * np.exp(-r / 0.8) * (r < 1.4)
    v = V0 + DV * np.arange(NCH)
    cube = amp[None] * np.exp(-0.5 * ((v[:, None, None] - vl[None]) / SIG_LINE) ** 2)
    cube = cube + rng.standard_normal(cube.shape) * RMS
    h = fits.Header()
    h["CTYPE1"], h["CRVAL1"], h["CDELT1"], h["CRPIX1"], h["CUNIT1"] = "RA---SIN", 70.0, -PIX / 3600, 32, "deg"
    h["CTYPE2"], h["CRVAL2"], h["CDELT2"], h["CRPIX2"], h["CUNIT2"] = "DEC--SIN", 25.0, PIX / 3600, 32, "deg"
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"], h["CUNIT3"] = "VRAD", V0 * 1e3, DV * 1e3, 1, "m/s"
    h["CTYPE4"], h["CRVAL4"], h["CDELT4"], h["CRPIX4"] = "STOKES", 1, 1, 1
    h["BMAJ"], h["BMIN"], h["BPA"] = 0.1 / 3600, 0.1 / 3600, 0.0
    h["BUNIT"] = "Jy/beam"
    h["SPECSYS"] = "LSRK"
    path = tmp_path_factory.mktemp("cube") / "disk.fits"
    fits.PrimaryHDU(cube[None].astype(np.float32), header=h).writeto(path)
    return {"path": str(path), "vlos": vl, "amp": amp, "r": r, "v": v, "cube": cube}


def test_read_cube_drops_stokes(disk_cube):
    c = kin.read_cube(disk_cube["path"])
    assert c["data"].shape == (NCH, NY, NX)
    assert c["velocity_kms"][0] == pytest.approx(V0)
    assert c["pix_arcsec"] == pytest.approx(PIX)


def test_moment_maps_recover_velocity_field(disk_cube, tmp_path):
    r = kin.moment_maps(disk_cube["path"], line_free_channels=[[0, 5], [75, 80]], out_dir=str(tmp_path))
    res = r["result"]
    assert res["rms_per_channel"]["value"] == pytest.approx(RMS, rel=0.15)
    m1 = fits.getdata(res["files"]["mom1"])
    vq = fits.getdata(res["files"]["vpeak_quadratic"])
    m0 = fits.getdata(res["files"]["mom0"])
    bright = disk_cube["amp"] > 0.02
    assert np.nanmedian(np.abs(m1[bright] - disk_cube["vlos"][bright])) < 0.05
    assert np.nanmedian(np.abs(vq[bright] - disk_cube["vlos"][bright])) < 0.05
    # moment 0 without clip recovers sqrt(2 pi) A sigma
    expect = disk_cube["amp"] * math.sqrt(2 * math.pi) * SIG_LINE
    assert np.median(m0[bright] / expect[bright]) == pytest.approx(1.0, abs=0.03)
    # redshifted side lies along PA (east of north): a pixel 0.5" toward PA
    pa = math.radians(PA)
    xp = int(round(31 - 0.5 * math.sin(pa) / PIX))
    yp = int(round(31 + 0.5 * math.cos(pa) / PIX))
    assert m1[yp, xp] > VSYS + 0.5
    assert "moment-maps" in r["corpus_practice"]


def test_keplerian_mask_covers_emission(disk_cube, tmp_path):
    out = tmp_path / "mask.fits"
    r = kin.keplerian_mask(disk_cube["path"], M, INC, PA, D, VSYS, r_max_au=150, line_width_kms=0.6,
                           out_path=str(out))
    mask = fits.getdata(out).astype(bool)
    assert mask.shape == disk_cube["cube"].shape
    signal = (disk_cube["cube"] - 0) > 8 * RMS
    noise_free = disk_cube["amp"][None] * np.exp(-0.5 * ((disk_cube["v"][:, None, None] - disk_cube["vlos"][None]) / SIG_LINE) ** 2)
    strong = noise_free > 8 * RMS
    assert mask[strong].mean() > 0.97
    assert r["result"]["fraction_of_voxels"] < 0.2


def test_dynamical_mass_keplerian_and_virial():
    v = math.sqrt(G_CGS * M_SUN_G / (100 * AU_CM)) / 1e5
    r = kin.dynamical_mass(v * math.sin(math.radians(45)), 100, "au", inclination_deg=45)
    assert r["result"]["mass"]["value"] == pytest.approx(1.0, rel=1e-9)
    r2 = kin.dynamical_mass(2.0, 1.0, "pc", kind="virial")
    assert r2["result"]["mass"]["value"] == pytest.approx(5 * 4.0 * 1.0 / G_PC_KMS2_MSUN, rel=1e-9)
    r3 = kin.dynamical_mass(1.0, 1.0, "pc", kind="virial", density_index=1.0)
    assert r3["result"]["mass"]["value"] == pytest.approx(1040, rel=0.01)


def test_virial_parameter_conventions():
    r = kin.virial_parameter(1.0, 2.0, 1000.0)
    a5 = 5 * 1.0 * 2.0 / (G_PC_KMS2_MSUN * 1000)
    assert r["result"]["alpha_5"]["value"] == pytest.approx(a5)
    assert r["result"]["bound_limit_alpha_5"] == 2.0
    r2 = kin.virial_parameter(1.0, 2.0, 1000.0, density_index=1.5)
    assert r2["result"]["a_profile"] == pytest.approx(1.25)
    assert r2["result"]["alpha_E"]["value"] == pytest.approx(a5 / 1.25)
    r3 = kin.virial_parameter(2.3548200450309493, 1.0, 1000.0, sigma_is_fwhm=True, radius_from_sigma_r_pc=1.0)
    assert r3["result"]["sigma_1d_kms"] == pytest.approx(1.0)
    assert r3["result"]["radius_pc"] == pytest.approx(1.91)


def test_toomre_q_modes():
    # gas: Q = sigma kappa / (pi G Sigma); compare with a hand value in astro units
    sigma, kappa, Sig = 10.0, 50.0, 100.0  # km/s, km/s/kpc, Msun/pc2
    q_hand = sigma * kappa / 1000 / (math.pi * G_PC_KMS2_MSUN * Sig)
    r = kin.toomre_q("gas", sigma_kms=sigma, kappa_kms_kpc=kappa, surface_density_msun_pc2=Sig)
    assert r["result"]["Q"]["value"] == pytest.approx(q_hand, rel=1e-6)
    rs = kin.toomre_q("stars", sigma_kms=sigma, kappa_kms_kpc=kappa, surface_density_msun_pc2=Sig)
    assert rs["result"]["Q"]["value"] == pytest.approx(q_hand * math.pi / 3.36, rel=1e-6)
    # two-fluid with negligible stars and thin layers approaches the gas Q / T_g
    r2 = kin.toomre_q("two_fluid", sigma_kms=sigma, kappa_kms_kpc=kappa, surface_density_msun_pc2=Sig,
                      sigma_star_kms=100.0, surface_density_star_msun_pc2=1e-6)
    assert r2["result"]["Q"]["value"] == pytest.approx(q_hand * 1.5, rel=1e-3)
    # protoplanetary disk: 0.1 Msun disk-like Sigma = 10 g/cm2 at 50 au, 20 K, 1 Msun
    r3 = kin.toomre_q("keplerian_disk", surface_density_g_cm2=10.0, m_star_msun=1.0, radius_au=50.0, temperature_k=20.0)
    assert 5 < r3["result"]["Q"]["value"] < 50
