import math

import numpy as np
import pytest

from alma_mcp.analysis import lines
from alma_mcp.analysis._util import H_CGS, K_CGS


def test_three_gaussians_recovered():
    rng = np.random.default_rng(7)
    v = np.arange(-60, 60, 0.5)
    truth = [(10.0, -25.0, 3.0), (6.0, 0.0, 5.0), (8.0, 22.0, 2.5)]
    y = sum(a * np.exp(-0.5 * ((v - c) / s) ** 2) for a, c, s in truth)
    rms = 0.3
    y = y + rng.standard_normal(v.size) * rms
    r = lines.fit_gaussian_lines(v.tolist(), y.tolist(), n_components=3, rms=rms)
    comps = r["result"]["components"]
    assert len(comps) == 3
    for (a, c, s), comp in zip(truth, comps):
        assert abs(comp["centre"]["value"] - c) < 4 * comp["centre"]["error"] + 0.05
        assert abs(comp["sigma"]["value"] - s) < 4 * comp["sigma"]["error"] + 0.05
        integ = math.sqrt(2 * math.pi) * a * s
        gi = comp["integrated_flux_gaussian"]
        assert abs(gi["value"] - integ) < 4 * gi["error"]
        cs = comp["integrated_flux_channel_sum"]
        assert cs["error"] == pytest.approx(rms * 0.5 * math.sqrt(cs["n_channels"]))
    assert 0.6 < r["result"]["reduced_chi2"] < 1.5


def test_line_luminosity_solomon_hand_values():
    # S dv = 1 Jy km/s, CO(1-0) at z = 1 with an explicit D_L = 6700 Mpc
    r = lines.line_luminosity(1.0, 115.271, redshift=1.0, distance_mpc=6700.0)
    nu_obs = 115.271 / 2
    lp = 3.25e7 * 1.0 * nu_obs**-2 * 6700.0**2 / 8.0
    ll = 1.04e-3 * nu_obs * 6700.0**2
    assert r["result"]["L_prime"]["value"] == pytest.approx(lp)
    assert r["result"]["L_line"]["value"] == pytest.approx(ll)
    # consistency L = 3.18e-11 nu_rest^3 L'
    assert ll == pytest.approx(3.18e-11 * 115.271**3 * lp, rel=0.01)
    assert r["warnings"] == []  # 6700 Mpc is within 2 % of Planck18 D_L(z=1)


def test_line_luminosity_warns_when_distance_disagrees_with_redshift():
    # seen with gpt-oss as the MCP client: it passed a rounded 20000 Mpc next to z = 2.5
    r = lines.line_luminosity(1.2, 345.796, redshift=2.5, distance_mpc=20000.0)
    assert len(r["warnings"]) == 1 and "Planck18" in r["warnings"][0] and "-8." in r["warnings"][0]
    assert lines.line_luminosity(1.2, 345.796, redshift=2.5)["warnings"] == []


def test_co_gas_mass_conventions():
    r = lines.co_gas_mass(1e9, j_upper=2, r_j1=0.8, alpha_co=4.35)
    res = r["result"]
    assert res["L_prime_CO10"]["value"] == pytest.approx(1.25e9)
    assert res["M_mol_with_helium"]["value"] == pytest.approx(4.35 * 1.25e9)
    assert res["M_H2_without_helium"]["value"] == pytest.approx(4.35 * 1.25e9 / 1.36)
    # flux-density ratio 3.2 for 2-1 means r21 = 0.8
    r2 = lines.co_gas_mass(1e9, j_upper=2, r_j1=3.2, ratio_is_flux_density=True)
    assert r2["result"]["r_J1_used"] == pytest.approx(0.8)
    # X_CO = 2e20 (H2 only) -> alpha_H2 about 3.2, with He about 4.35
    r3 = lines.co_gas_mass(1.0, x_co=2e20)
    assert r3["result"]["alpha_H2_without_helium"] == pytest.approx(3.2, rel=0.01)
    assert r3["result"]["alpha_mol_with_helium"] == pytest.approx(4.35, rel=0.01)


def _ladder(t_rot=80.0, n_tot=1e15, b_ghz=9.0, fwhm=None, noise=0.0, seed=2):
    """Synthetic linear-rotor-like ladder with made-up but self-consistent constants."""
    rng = np.random.default_rng(seed)
    J = np.arange(3, 13)
    nu = 2 * b_ghz * J  # GHz
    eu = H_CGS * b_ghz * 1e9 * J * (J + 1) / K_CGS
    gu = 2 * J + 1
    a_ul = 1e-5 * (J / 5.0) ** 3
    qv = K_CGS * t_rot / (H_CGS * b_ghz * 1e9) + 1 / 3
    n_u = n_tot / qv * gu * np.exp(-eu / t_rot)
    from alma_mcp.analysis._util import C_CGS
    w = n_u * H_CGS * C_CGS**3 * a_ul / (8 * math.pi * K_CGS * (nu * 1e9) ** 2) / 1e5
    werr = 0.05 * w
    w_obs = w * (1 + noise * rng.standard_normal(w.size))
    out = []
    for i in range(J.size):
        d = {"freq_ghz": nu[i], "a_ul": a_ul[i], "e_u_k": eu[i], "g_u": gu[i], "w_k_kms": w_obs[i], "w_err_k_kms": werr[i]}
        if fwhm:
            d["fwhm_kms"] = fwhm
        out.append(d)
    return out, w


def test_rotational_diagram_recovers_t_and_n():
    ladder, _ = _ladder(noise=0.05)
    r = lines.rotational_diagram(ladder, linear_rotor_b_ghz=9.0, flux_cal_frac=0.0)
    t = r["result"]["T_rot"]
    n = r["result"]["N_tot"]
    assert abs(t["value"] - 80) < 3 * t["error"]
    assert abs(n["value"] - 1e15) < 3 * n["error"]


def test_rotational_diagram_partition_table_matches_rotor():
    ladder, _ = _ladder()
    tab = [[T, 1.380649e-16 * T / (6.62607015e-27 * 9e9) + 1 / 3] for T in (9.375, 18.75, 37.5, 75, 150, 225, 300)]
    r = lines.rotational_diagram(ladder, partition_table=tab, flux_cal_frac=0.0)
    assert r["result"]["T_rot"]["value"] == pytest.approx(80, rel=1e-6)
    assert r["result"]["N_tot"]["value"] == pytest.approx(1e15, rel=1e-3)


def test_rotational_diagram_tau_correction():
    # build an optically thick ladder: observed W = W_thin (1 - e^-tau)/tau
    t_rot, n_tot, fwhm = 50.0, 1e15, 1.0
    ladder, w_thin = _ladder(t_rot=t_rot, n_tot=n_tot, fwhm=fwhm)
    nu = np.array([l["freq_ghz"] for l in ladder])
    a = np.array([l["a_ul"] for l in ladder])
    n_u = lines.upper_level_column(w_thin, nu, a)
    tau = lines._tau_from_nu(n_u, nu, a, t_rot, fwhm)
    assert tau.max() > 0.3
    for l, wt, tt in zip(ladder, w_thin, tau):
        l["w_k_kms"] = wt * (-np.expm1(-tt)) / tt
    thin = lines.rotational_diagram(ladder, linear_rotor_b_ghz=9.0, flux_cal_frac=0.0)
    corr = lines.rotational_diagram(ladder, linear_rotor_b_ghz=9.0, flux_cal_frac=0.0, tau_correction=True)
    assert corr["result"]["N_tot"]["value"] == pytest.approx(n_tot, rel=0.02)
    assert corr["result"]["T_rot"]["value"] == pytest.approx(t_rot, rel=0.02)
    assert thin["result"]["N_tot"]["value"] < 0.95 * n_tot


def test_lte_column_density_consistent_with_ladder():
    ladder, w = _ladder(t_rot=80.0, n_tot=1e15)
    l = ladder[4]
    q80 = K_CGS * 80 / (H_CGS * 9e9) + 1 / 3
    r = lines.lte_column_density(l["w_k_kms"], l["freq_ghz"], l["a_ul"], l["e_u_k"], l["g_u"], 80.0,
                                 partition_q=q80, t_bg_k=0.0001)
    # with a negligible background and RJ-equivalent J ratio, the column matches the injected one up to
    # J(Tex)/(J(Tex)-J(Tbg)) ~ 1
    assert r["result"]["N_tot"]["value"] == pytest.approx(1e15, rel=1e-3)
    # CMB background raises N
    r2 = lines.lte_column_density(l["w_k_kms"], l["freq_ghz"], l["a_ul"], l["e_u_k"], l["g_u"], 80.0, partition_q=q80)
    assert r2["result"]["N_tot"]["value"] > r["result"]["N_tot"]["value"]
    # tau correction
    r3 = lines.lte_column_density(l["w_k_kms"], l["freq_ghz"], l["a_ul"], l["e_u_k"], l["g_u"], 80.0, partition_q=q80, tau=1.0)
    assert r3["result"]["C_tau"] == pytest.approx(1 / (1 - math.exp(-1)))
