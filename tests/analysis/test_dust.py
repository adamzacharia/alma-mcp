import importlib.util
import math

import numpy as np
import pytest

from alma_mcp.analysis import dust
from alma_mcp.analysis._util import M_EARTH_G, M_SUN_G, PC_CM, planck_nu

HAS_EMCEE = importlib.util.find_spec("emcee") is not None


def test_dust_mass_hand_computed():
    # 10 mJy at 230 GHz, 140 pc, 20 K, kappa = 2.3 cm2/g: hand value
    nu = 230e9
    b = planck_nu(nu, 20.0)
    m_hand = 10e-3 * 1e-23 * (140 * PC_CM) ** 2 / (2.3 * b)
    r = dust.dust_mass_from_flux(10, 230, distance_pc=140, t_dust_k=20, gas_to_dust=100)
    m = r["result"]["dust_mass"]["value"] * M_SUN_G
    assert m == pytest.approx(m_hand, rel=1e-9)
    assert r["result"]["dust_mass"]["earth_masses"] == pytest.approx(m_hand / M_EARTH_G, rel=1e-9)
    assert r["result"]["gas_mass"]["value"] == pytest.approx(100 * m / M_SUN_G, rel=1e-9)
    # 10% calibration only
    assert r["result"]["dust_mass"]["error"] == pytest.approx(0.1 * m / M_SUN_G, rel=1e-6)
    assert "dust-continuum-mass" in r["corpus_practice"]


def test_beckwith_law_value_at_230():
    r = dust.dust_mass_from_flux(1, 230, distance_pc=100)
    assert r["result"]["kappa_nu"]["value"] == pytest.approx(2.3, rel=1e-12)


def test_kappa_includes_gas_does_not_multiply():
    r = dust.dust_mass_from_flux(5, 230, distance_pc=100, kappa_preset="custom", kappa0_cm2_g=0.01, nu0_ghz=230,
                                 beta=1.8, kappa_includes_gas=True, gas_to_dust=100)
    res = r["result"]
    assert res["dust_mass"]["value"] == pytest.approx(res["gas_plus_dust_mass"]["value"] / 100)
    assert any("do NOT multiply" in w for w in r["warnings"])


def test_cmb_contrast_raises_mass_at_high_z():
    a = dust.dust_mass_from_flux(1.0, 250, redshift=6.0, t_dust_k=40, kappa_preset="galaxy_850um")
    b = dust.dust_mass_from_flux(1.0, 250, redshift=6.0, t_dust_k=40, kappa_preset="galaxy_850um", cmb="contrast")
    c = dust.dust_mass_from_flux(1.0, 250, redshift=6.0, t_dust_k=40, kappa_preset="galaxy_850um", cmb="heating_and_contrast")
    ma, mb = a["result"]["dust_mass"]["value"], b["result"]["dust_mass"]["value"]
    assert mb > ma
    assert b["result"]["cmb_contrast_factor"] < 1
    assert c["result"]["t_dust_used"]["value"] > 40
    # da Cunha eq 12 by hand
    e = 4 + 1.8
    t_hand = (40**e + 2.7255**e * (7**e - 1)) ** (1 / e)
    assert c["result"]["t_dust_used"]["value"] == pytest.approx(t_hand)


def _sed(t=35.0, beta=1.8, logm=8.0, z=2.0, form="thin", freqs=None, noise=0.03, seed=3):
    from alma_mcp.analysis.dust import _mbb_model, luminosity_distance_cm, KAPPA_PRESETS
    p = KAPPA_PRESETS["galaxy_850um"]
    d_cm, _ = luminosity_distance_cm(z, None)
    freqs = np.array(freqs if freqs is not None else [100, 150, 230, 345, 450, 650, 850, 1200, 1900, 3000.0])
    s = _mbb_model(freqs * 1e9, logm, t, beta, z, d_cm, p["kappa0_cm2_g"], p["nu0_ghz"] * 1e9, form, 100.0, False)
    rng = np.random.default_rng(seed)
    err = noise * s
    return freqs, s + rng.standard_normal(s.size) * err, err


def test_fit_mbb_recovers_injected():
    f, s, e = _sed()
    r = dust.fit_modified_blackbody(f, s, e, redshift=2.0, flux_cal_frac=0.0)
    p = r["result"]["parameters"]
    assert abs(p["t_dust_k"]["value"] - 35) < 3 * p["t_dust_k"]["error"] + 0.5
    assert abs(p["beta"]["value"] - 1.8) < 3 * p["beta"]["error"] + 0.05
    assert abs(p["log10_mass_msun"]["value"] - 8.0) < 3 * p["log10_mass_msun"]["error"] + 0.03
    assert r["result"]["L_IR_8_1000um"]["value"] > 1e11
    assert r["result"]["reduced_chi2"] < 3


def test_fit_mbb_general_opacity():
    f, s, e = _sed(t=50, beta=2.0, logm=8.5, form="general", noise=0.02)
    r = dust.fit_modified_blackbody(f, s, e, redshift=2.0, form="general", flux_cal_frac=0.0)
    p = r["result"]["parameters"]
    assert abs(p["t_dust_k"]["value"] - 50) < 3 * p["t_dust_k"]["error"] + 1
    assert abs(p["beta"]["value"] - 2.0) < 3 * p["beta"]["error"] + 0.05


def test_lir_matches_analytic_for_blackbody_limit():
    # optically thin, beta = 0 at z small: compare L_IR with a direct numerical integral of the same model
    f, s, e = _sed(t=30, beta=1.5, logm=7.0, z=0.5, noise=0.0)
    r = dust.fit_modified_blackbody(f, s, np.maximum(e, 1e-3 * s) + 1e-3 * s, redshift=0.5, flux_cal_frac=0.0)
    from alma_mcp.analysis.dust import luminosity_distance_cm, KAPPA_PRESETS
    from alma_mcp.analysis._util import C_CGS, L_SUN_CGS
    k = KAPPA_PRESETS["galaxy_850um"]
    nu = np.logspace(np.log10(C_CGS / 0.1), np.log10(C_CGS / 8e-4), 20000)
    kap = k["kappa0_cm2_g"] * (nu / (k["nu0_ghz"] * 1e9)) ** 1.5
    l_hand = 4 * math.pi * 1e7 * M_SUN_G * np.trapezoid(kap * planck_nu(nu, 30.0), nu) / L_SUN_CGS
    assert r["result"]["L_IR_8_1000um"]["value"] == pytest.approx(l_hand, rel=0.02)


@pytest.mark.skipif(not HAS_EMCEE, reason="emcee not installed")
def test_fit_mbb_mcmc():
    f, s, e = _sed()
    r = dust.fit_modified_blackbody(f, s, e, redshift=2.0, flux_cal_frac=0.0, mcmc=True, mcmc_steps=800)
    p = r["result"]["parameters"]["t_dust_k"]
    assert p["lower_16"] < p["value"] < p["upper_84"]
    assert abs(p["value"] - 35) < 4


def test_spectral_index_two_point_and_fit():
    r = dust.spectral_index([100, 300], [1.0, 1.0 * 3**2.5], flux_cal_frac=0.1)
    a = r["result"]["alpha"]
    assert a["value"] == pytest.approx(2.5)
    assert a["error"] == pytest.approx(math.sqrt(0.02) / math.log(3))
    nu = np.array([90, 140, 230, 340.0])
    r2 = dust.spectral_index(nu, 2.0 * (nu / 100) ** 3.2, flux_cal_frac=0.05)
    assert r2["result"]["alpha"]["value"] == pytest.approx(3.2, abs=1e-9)
    assert r2["result"]["beta_rayleigh_jeans_thin"]["value"] == pytest.approx(1.2, abs=1e-9)


def test_spectral_index_planck_correction():
    # pure modified blackbody with beta = 1.7 at 20 K: Planck-corrected beta recovers 1.7 exactly for two points
    nu = np.array([100e9, 345e9])
    s = nu**1.7 * planck_nu(nu, 20.0)
    r = dust.spectral_index(nu / 1e9, s / s[0], t_dust_k=20.0)
    assert r["result"]["beta_planck_corrected"]["value"] == pytest.approx(1.7, abs=1e-9)
    assert r["result"]["beta_rayleigh_jeans_thin"]["value"] < 1.7
