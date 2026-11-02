import importlib.util
import math

import numpy as np
import pytest

from alma_mcp.analysis import imaging_checks as ic
from alma_mcp.analysis import uv
from alma_mcp.analysis._util import ARCSEC_RAD

HAS_EMCEE = importlib.util.find_spec("emcee") is not None
HAS_FRANK = importlib.util.find_spec("frank") is not None


def make_vis(path, model, params, n=4000, sigma=0.002, seed=5, weight_scale=1.0, qmax=1.5e6):
    rng = np.random.default_rng(seed)
    r = qmax * np.sqrt(rng.uniform(0.002, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    u, v = r * np.cos(th), r * np.sin(th)
    m = uv.model_visibilities(model, params, u, v)
    re = m.real + rng.standard_normal(n) * sigma
    im = m.imag + rng.standard_normal(n) * sigma
    w = np.full(n, weight_scale / sigma**2)
    np.savez(path, u=u, v=v, Re=re, Im=im, weight=w)
    return path


def within(p, truth, k=4, floor=0.0):
    return abs(p["value"] - truth) < k * p["error"] + floor


def test_fit_offset_gaussian(tmp_path):
    truth = {"flux_jy": 0.05, "dra_arcsec": 0.12, "ddec_arcsec": -0.08, "fwhm_arcsec": 0.3}
    path = make_vis(tmp_path / "g.npz", "gaussian", truth)
    r = uv.fit_uv_model(str(path), "gaussian")
    p = r["result"]["parameters"]
    for k, val in truth.items():
        assert within(p[k], val), (k, p[k], val)
    assert 0.9 < r["result"]["reduced_chi2"] < 1.1
    assert "uv-plane-fitting" in r["corpus_practice"]


def test_fit_elliptical_gaussian_and_weight_rescale(tmp_path):
    truth = {"flux_jy": 0.08, "dra_arcsec": 0.0, "ddec_arcsec": 0.0, "fwhm_major_arcsec": 0.4, "axis_ratio": 0.5, "pa_deg": 30.0}
    # pipeline weights 4x too large
    path = make_vis(tmp_path / "e.npz", "elliptical_gaussian", truth, weight_scale=4.0)
    r0 = uv.fit_uv_model(str(path), "elliptical_gaussian")
    assert r0["result"]["reduced_chi2"] > 3
    assert any("mis-scaled" in w for w in r0["warnings"])
    r = uv.fit_uv_model(str(path), "elliptical_gaussian", rescale_weights="residual_scatter")
    assert r["result"]["weight_scale_applied"] == pytest.approx(0.25, rel=0.1)
    assert r["result"]["reduced_chi2"] == pytest.approx(1.0, abs=0.1)
    p = r["result"]["parameters"]
    for k, val in truth.items():
        assert within(p[k], val, floor=0.5 if k == "pa_deg" else 0.0), (k, p[k], val)


def test_fit_ring_and_point(tmp_path):
    truth = {"flux_jy": 0.1, "dra_arcsec": 0.0, "ddec_arcsec": 0.0, "radius_arcsec": 0.5, "width_fwhm_arcsec": 0.08,
             "inclination_deg": 50.0, "pa_deg": 110.0}
    path = make_vis(tmp_path / "ring.npz", "ring", truth, qmax=1.0e6)
    r = uv.fit_uv_model(str(path), "ring", initial={"radius_arcsec": 0.4, "inclination_deg": 40, "pa_deg": 100})
    p = r["result"]["parameters"]
    for k in ("flux_jy", "radius_arcsec", "inclination_deg", "pa_deg"):
        assert within(p[k], truth[k], floor=0.01), (k, p[k], truth[k])
    pt = {"flux_jy": 0.01, "dra_arcsec": -0.2, "ddec_arcsec": 0.3}
    path2 = make_vis(tmp_path / "pt.npz", "point", pt)
    r2 = uv.fit_uv_model(str(path2), "point")
    for k, val in pt.items():
        assert within(r2["result"]["parameters"][k], val)


def test_fit_from_csv_in_metres(tmp_path):
    truth = {"flux_jy": 0.02, "dra_arcsec": 0.0, "ddec_arcsec": 0.0, "diameter_arcsec": 0.6, "inclination_deg": 0.0, "pa_deg": 0.0}
    path = make_vis(tmp_path / "d.npz", "uniform_disk", truth, qmax=6e5)
    d = np.load(path)
    lam = 2.99792458e8 / 230e9
    csv = tmp_path / "d.csv"
    with open(csv, "w") as fh:
        fh.write("u,v,Re,Im,weight\n")
        for row in zip(d["u"] * lam, d["v"] * lam, d["Re"], d["Im"], d["weight"]):
            fh.write(",".join(f"{x:.10g}" for x in row) + "\n")
    r = uv.fit_uv_model(str(csv), "uniform_disk", uv_unit="m", freq_ghz=230, fixed={"inclination_deg": 0, "pa_deg": 0})
    p = r["result"]["parameters"]
    assert within(p["flux_jy"], 0.02)
    assert within(p["diameter_arcsec"], 0.6)


@pytest.mark.skipif(not HAS_EMCEE, reason="emcee not installed")
def test_fit_uv_mcmc(tmp_path):
    truth = {"flux_jy": 0.05, "dra_arcsec": 0.0, "ddec_arcsec": 0.0, "fwhm_arcsec": 0.3}
    path = make_vis(tmp_path / "m.npz", "gaussian", truth, n=1500)
    r = uv.fit_uv_model(str(path), "gaussian", mcmc=True, mcmc_steps=600)
    p = r["result"]["parameters"]["fwhm_arcsec"]
    assert p["mcmc_lower_16"] < p["mcmc_median"] < p["mcmc_upper_84"]
    assert abs(p["mcmc_median"] - 0.3) < 4 * (p["mcmc_upper_84"] - p["mcmc_lower_16"]) / 2


def test_deproject_and_bin(tmp_path):
    truth = {"flux_jy": 0.05, "dra_arcsec": 0.2, "ddec_arcsec": 0.1, "fwhm_major_arcsec": 0.4, "axis_ratio": math.cos(math.radians(60)), "pa_deg": 40.0}
    path = make_vis(tmp_path / "dp.npz", "elliptical_gaussian", truth, n=20000, sigma=0.001)
    r = uv.deproject_and_bin(str(path), inclination_deg=60, pa_deg=40, dra_arcsec=0.2, ddec_arcsec=0.1, n_bins=20)
    bins = r["result"]["bins"]
    th = 0.4 * ARCSEC_RAD
    for b in bins:
        expect = 0.05 * math.exp(-math.pi**2 * th**2 * (b["uv_klambda"] * 1e3) ** 2 / (4 * math.log(2)))
        assert abs(b["re_jy"] - expect) < 5 * b["err_jy"] + 0.02 * expect
        assert abs(b["im_jy"]) < 5 * b["err_jy"]


def test_frank_message_or_run(tmp_path):
    truth = {"flux_jy": 0.05, "dra_arcsec": 0.0, "ddec_arcsec": 0.0, "fwhm_arcsec": 0.3}
    path = make_vis(tmp_path / "f.npz", "gaussian", truth, n=500)
    r = uv.frank_profile(str(path), 0.0, 0.0)
    if HAS_FRANK:
        assert r["result"]["available"] is True
    else:
        assert r["result"]["available"] is False
        assert "pip install" in r["warnings"][0]


# ---------------------------------------------------------------------------------------- imaging checks


def test_jy_to_k_rayleigh_jeans_hand_value():
    r = ic.beam_and_brightness(0.01, 230.0, 0.5, 0.4)
    t_hand = 1.222e6 * 0.01 / (230.0**2 * 0.5 * 0.4)
    assert r["result"]["brightness_temperature"]["value"] == pytest.approx(t_hand, rel=1e-3)
    back = ic.beam_and_brightness(t_hand, 230.0, 0.5, 0.4, direction="k_to_jy")
    assert back["result"]["flux_density_per_beam"]["value"] == pytest.approx(0.01, rel=1e-3)
    om = math.pi * 0.5 * 0.4 / (4 * math.log(2)) * ARCSEC_RAD**2
    assert r["result"]["beam_solid_angle"]["value"] == pytest.approx(om)
    # Planck temperature is above the RJ one and round-trips
    p = ic.beam_and_brightness(0.01, 230.0, 0.5, 0.4, planck=True)
    tp = p["result"]["brightness_temperature"]["value"]
    assert tp > t_hand
    pb = ic.beam_and_brightness(tp, 230.0, 0.5, 0.4, direction="k_to_jy", planck=True)
    assert pb["result"]["flux_density_per_beam"]["value"] == pytest.approx(0.01, rel=1e-9)


def test_rms_from_spectrum():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(2000) * 0.5
    s[900:1100] += 10.0
    r = ic.rms_from_line_free(spectrum=s.tolist(), line_free_channels=[[0, 800], [1200, 1999]])
    assert r["result"]["rms_robust_mad"]["value"] == pytest.approx(0.5, rel=0.08)


def test_mrs_conventions():
    b = np.linspace(15, 500, 1000)
    r = ic.largest_angular_scale(230.0, baselines_m=b.tolist())
    lam = 2.99792458e8 / 230e9
    res = r["result"]
    assert res["mrs_0p6_lambda_over_Bmin_arcsec"] == pytest.approx(0.6 * lam / 15 / ARCSEC_RAD)
    assert res["mrs_0p983_lambda_over_L05_arcsec"] == pytest.approx(0.983 * lam / np.percentile(b, 5) / ARCSEC_RAD)
    f = ic.flux_recovery(res["mrs_0p6_lambda_over_Bmin_arcsec"], 230.0, 15.0)
    assert 0 < f["result"]["visible_fraction_at_shortest_baseline"] < 1
    g = ic.flux_recovery(0.01, 230.0, 15.0)
    assert g["result"]["visible_fraction_at_shortest_baseline"] == pytest.approx(1.0, abs=1e-4)
    half = f["result"]["size_for_0.5_arcsec"]
    assert ic.flux_recovery(half, 230.0, 15.0)["result"]["visible_fraction_at_shortest_baseline"] == pytest.approx(0.5, abs=1e-6)
