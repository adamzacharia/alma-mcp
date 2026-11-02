from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import tarfile
from pathlib import Path

import pytest

from alma_mcp.corpus import cards as C
from alma_mcp.corpus import data

HELD = "2022ApJ...004..004H"


# ------------------------------------------------------------------------------------------------ build: filters
def test_exclude_heldout_filter(build_mod):
    rows = [{"bibcode": "a", "heldout": 0}, {"bibcode": "b", "heldout": 1}, {"bibcode": "c", "heldout": None},
            {"bibcode": "d", "heldout": "True"}, {"bibcode": "e"}]
    held = build_mod.heldout_set(rows)
    assert held == {"b", "d"}
    kept = [r["bibcode"] for r in build_mod.exclude_heldout(rows, held)]
    assert kept == ["a", "c", "e"]
    # a row without its own flag is still dropped when its bibcode is in the held-out set
    assert build_mod.exclude_heldout([{"bibcode": "b"}], held) == []


def test_redact_heldout_text(build_mod):
    t, n = build_mod.redact_heldout_text("see 2022ApJ...004..004H and 2022ApJ...004..004H", {HELD})
    assert n == 2 and HELD not in t


def test_heldout_absent_everywhere(bundles):
    for variant in ("full", "public"):
        db = bundles[variant] / "alma_methods.sqlite"
        con = sqlite3.connect(db)
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%fts%'").fetchall():
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            for c in cols:
                n = con.execute(f"SELECT count(*) FROM {t} WHERE CAST({c} AS TEXT) LIKE ?", (f"%{HELD}%",)).fetchone()[0]
                assert n == 0, (variant, t, c)
        assert con.execute("SELECT count(*) FROM papers").fetchone()[0] == 3  # held-out and non-evidence_ok dropped
        con.close()
        for p in (bundles[variant] / "skills").rglob("*"):
            if p.is_file():
                assert HELD not in p.read_text(encoding="utf-8")
    assert bundles["full_manifest"]["skills_redacted_heldout_mentions"] == 1
    assert bundles["full_manifest"]["cards_redacted_heldout_mentions"] == 1


def test_manifest_and_tarball(bundles):
    m = bundles["full_manifest"]
    assert m["counts"]["papers"] == 3 and m["sha256"] == data.sha256_file(bundles["full"] / "alma_methods.sqlite")
    with tarfile.open(m["tarball"]) as tar:
        names = tar.getnames()
    assert "alma_methods.sqlite" in names and "manifest.json" in names and "skills/self-calibration/SKILL.md" in names
    # full and public tarballs go to separate folders (only the public one is meant for public upload)
    rel = bundles["root"] / "rel"
    assert Path(m["tarball"]).parent == rel / "full"
    assert Path(bundles["public_manifest"]["tarball"]).parent == rel / "public"
    assert "-public-" in Path(bundles["public_manifest"]["tarball"]).name


# ------------------------------------------------------------------------------------------------ internal metadata
def _all_text(db):
    """Every text cell of every non-FTS table."""
    con = sqlite3.connect(db)
    out = []
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%fts%'").fetchall():
        for row in con.execute(f"SELECT * FROM {t}"):
            out += [v for v in row if isinstance(v, str)]
    con.close()
    return out


def test_bundle_cards_carry_minimal_provenance(bundles):
    for variant in ("full", "public"):
        con = sqlite3.connect(bundles[variant] / "alma_methods.sqlite")
        for (j,) in con.execute("SELECT card_json FROM cards"):
            c = json.loads(j)
            assert set(c["provenance"]) <= set(C.PROVENANCE_KEYS) and c["provenance"]["model"] == "claude-sonnet-x"
            assert "repair_log" not in c
            if variant == "public":
                assert all(isinstance(v, (bool, int, float)) for v in c["checks"].values())
                assert "quotes_too_long" not in c["checks"]
        con.close()
        text = "\n".join(_all_text(bundles[variant] / "alma_methods.sqlite"))
        for secret in ("msgbatch_01SECRET", "card-sonnet-abc123", "repair-sonnet-xyz", "sonnet_run2", "input_tokens"):
            assert secret not in text, (variant, secret)


def test_public_bundle_has_no_closed_license_quote(bundles):
    """No verbatim quote longer than 40 chars from a closed-license paper appears anywhere in the public bundle."""
    from conftest import CLOSED_LONG_QUOTE, CLOSED_VERBATIM_VALUE, OTHER_CHANNEL_QUOTE

    corpus = bundles["root"] / "corpus"
    con = sqlite3.connect(corpus / "manifest.sqlite")
    closed = {b: t for b, lic, t in con.execute("SELECT bibcode, license, title FROM papers WHERE heldout = 0")
              if not C.license_is_open(lic)}
    con.close()
    quotes = set()
    for p in (corpus / "cards").glob("*/*/card.json"):  # every extraction channel of each closed paper
        card = json.loads(p.read_text(encoding="utf-8"))
        bib = card["identity"]["bibcode"]
        if bib in closed:
            # a quote of the paper title is bibliographic metadata and a bare list of ALMA project codes is a list
            # of identifiers; neither is body text
            quotes |= {q for q in _quote_strings(card) if q not in closed[bib] and not PROJECT_CODES.fullmatch(q)}
    quotes = {q for q in quotes if len(q) > 40}
    assert {CLOSED_LONG_QUOTE, CLOSED_VERBATIM_VALUE, OTHER_CHANNEL_QUOTE} <= quotes and len(quotes) >= 5
    texts = _all_text(bundles["public"] / "alma_methods.sqlite")
    texts += [p.read_text(encoding="utf-8") for p in (bundles["public"] / "skills").rglob("*") if p.is_file()]
    leaks = [q for q in quotes if any(q in t for t in texts)]
    assert leaks == []
    # values copied verbatim are replaced by a marker in the public card only
    assert any(C.WITHHELD_TEXT in t for t in texts)
    # the full bundle does keep them (private or team use only)
    full = _all_text(bundles["full"] / "alma_methods.sqlite")
    for q in (CLOSED_LONG_QUOTE, CLOSED_VERBATIM_VALUE, OTHER_CHANNEL_QUOTE):
        assert any(q in t for t in full)


def test_redact_verbatim_unit():
    card = {"identity": {"title": "A survey of dense gas in nearby spiral galaxies with ALMA"},
            "body": {"q": {"value": "We ask how the dense gas fraction varies with radius in the disks."},
                     "short": "dense gas", "t": "A survey of dense gas in nearby spiral galaxies with ALMA"}}
    quotes = {"how the dense gas fraction varies with radius in the disks", "dense gas",
              "A survey of dense gas in nearby spiral galaxies with ALMA"}
    out, n = C.redact_verbatim(card, quotes)
    assert n == 1 and out["body"]["q"]["value"] == f"We ask {C.WITHHELD_TEXT}."
    assert out["body"]["short"] == "dense gas"  # 40 chars or fewer are left alone
    assert out["body"]["t"] == card["identity"]["title"]  # title quotes are metadata
    assert card["body"]["q"]["value"].startswith("We ask how")  # input untouched


PROJECT_CODES = re.compile(r"(?:\s*(?:and\s+)?\d{4}\.[\dA-Z]\.\d{5}\.[A-Z]+\s*[,;]?)+")


def _quote_strings(card):
    """Every verbatim string in a source card: evidence quotes, repair-log old/new, checks.quotes_too_long."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("quote", "old", "new") and isinstance(v, str):
                    out.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(card)
    too_long = (card.get("checks") or {}).get("quotes_too_long")
    out |= {q for q in too_long if isinstance(q, str)} if isinstance(too_long, list) else set()
    return out


def test_skill_paths_neutralized_and_scan_clean(bundles, build_mod):
    for variant in ("full", "public"):
        md = (bundles[variant] / "skills" / "self-calibration" / "SKILL.md").read_text(encoding="utf-8")
        assert "`skills/packs/self-calibration.md`" in md and "Users" not in md
        par = (bundles[variant] / "skills" / "self-calibration" / "references" / "parameters.md").read_text(encoding="utf-8")
        assert "`skills/packs/self-calibration.md`" in par and "/home/" not in par
        assert build_mod.scan_bundle(bundles[variant]) == []
    t, n = build_mod.neutralize_paths(r"see C:\Users\bob\data\x.fits and /Users/al/notes/y.md")
    assert n == 2 and t == "see <local>/x.fits and <local>/y.md"


def test_scan_text_patterns(build_mod):
    assert build_mod.scan_text("disk-averaged brightness, risk-free, sk-short") == []
    assert build_mod.scan_text("key sk-" + "a" * 24) == ["sk- API key"]
    assert build_mod.scan_text("AIza" + "B" * 35) == ["Google API key"]
    assert build_mod.scan_text(r"C:\Users\x") == ["windows user path"]


# ------------------------------------------------------------------------------------------------ public variant
def test_license_rules():
    assert C.license_is_open("http://creativecommons.org/licenses/by/4.0/")
    assert C.license_is_open("http://creativecommons.org/licenses/by-sa/4.0/")
    assert C.license_is_open("http://creativecommons.org/publicdomain/zero/1.0/")
    assert not C.license_is_open("http://creativecommons.org/licenses/by-nc-nd/4.0/")
    assert not C.license_is_open("http://creativecommons.org/licenses/by-nc-sa/4.0/")
    assert not C.license_is_open("http://arxiv.org/licenses/nonexclusive-distrib/1.0/")
    assert not C.license_is_open(None)


def test_strip_quotes_keeps_values_and_locations():
    card = {"body": {"x": {"value": "robust 0.5", "evidence": [{"loc": "Sec 2", "quote": "we used robust 0.5"}]}}}
    out, n = C.strip_quotes(card)
    assert n == 1
    e = out["body"]["x"]["evidence"][0]
    assert e == {"loc": "Sec 2", "quote_withheld": True}
    assert out["body"]["x"]["value"] == "robust 0.5"
    assert card["body"]["x"]["evidence"][0]["quote"] == "we used robust 0.5"  # input untouched


def test_public_bundle_strips_quotes_for_non_cc_only(bundles):
    con = sqlite3.connect(bundles["public"] / "alma_methods.sqlite")
    rows = {b: (j, md, n) for b, j, md, n in con.execute("SELECT bibcode, card_json, card_md, quotes_withheld FROM cards")}
    abstracts = dict(con.execute("SELECT bibcode, abstract FROM papers"))
    con.close()
    cc, arxiv, nolic = rows["2019ApJ...001..001A"], rows["2020A&A...002..002B"], rows["2021MNRAS.003..003C"]
    assert cc[2] == 0 and '"quote"' in cc[0] and "we used" in cc[1]
    for j, md, n in (arxiv, nolic):
        assert n > 0 and '"quote"' not in j and "we used" not in md and "quote withheld" in md
        assert "Briggs" in md or "robust" in md  # structured values survive
        assert "2 Observations" in md  # the location survives
    assert abstracts["2020A&A...002..002B"] is None and abstracts["2019ApJ...001..001A"]
    # full bundle keeps all quotes
    con = sqlite3.connect(bundles["full"] / "alma_methods.sqlite")
    assert con.execute("SELECT sum(quotes_withheld) FROM cards").fetchone()[0] == 0
    assert "we used" in con.execute("SELECT card_md FROM cards WHERE bibcode='2020A&A...002..002B'").fetchone()[0]
    con.close()


# ------------------------------------------------------------------------------------------------ parsing
def test_extract_settings():
    body = {"processing": {"calibration": {"self_calibration": {"value": "phase-only, solint inf, 60s then 2 min"}},
                           "imaging": {"weighting_robust": {"value": "Briggs robust=0.5 and natural"},
                                       "cell_size": {"value": "30 mas"},
                                       "continuum_subtraction": {"value": "uvcontsub with fitorder=1"}}},
            "data": {"bands": {"value": "Band 3 and Band 6"}, "arrays": {"value": "12m + ACA"}}}
    s = C.extract_settings(body)
    assert ("robust", "0.5") in s and ("weighting", "natural") in s and ("weighting", "briggs") in s
    assert ("cell_size", "0.03") in s
    assert {("selfcal_solint", "inf"), ("selfcal_solint", "60"), ("selfcal_solint", "120")} <= set(s)
    assert ("selfcal_mode", "phase-only") in s
    assert ("contsub_order", "1") in s and ("contsub_domain", "uv plane") in s
    assert ("band", "3") in s and ("band", "6") in s and ("array", "12m") in s and ("array", "7m/ACA") in s


def test_normalize_setting():
    assert C.normalize_setting("self-cal solint") == "selfcal_solint"
    assert C.normalize_setting("Cell size") == "cell_size"
    assert C.normalize_setting("continuum subtraction order") == "contsub_order"
    assert C.normalize_setting("robust") == "robust"


def test_fts_query():
    assert C.fts_query("self-calibration solint") == '"self calibration" AND "solint"'
    assert C.fts_query("a OR b", "OR") == '"a" OR "OR" OR "b"'
    assert C.fts_query("  ") == ""


# ------------------------------------------------------------------------------------------------ tools (store)
def test_search_papers(store):
    r = store.search_papers("self-calibration solint")
    bibs = [x["bibcode"] for x in r["results"]]
    assert bibs and set(bibs) <= {"2019ApJ...001..001A", "2021MNRAS.003..003C"} and HELD not in bibs
    assert r["results"][0]["ads_url"].startswith("https://ui.adsabs.harvard.edu/abs/")
    r = store.search_papers("galaxy", field="nearby_galaxies")
    assert [x["bibcode"] for x in r["results"]] == ["2020A&A...002..002B"]
    assert "%26" in r["results"][0]["ads_url"]
    assert store.search_papers("disk", year_min=2020)["n"] == 0 or all(
        x["year"] >= 2020 for x in store.search_papers("disk", year_min=2020)["results"])
    r = store.search_papers("", technique="self-calibration")
    assert {x["bibcode"] for x in r["results"]} == {"2019ApJ...001..001A", "2021MNRAS.003..003C"}
    assert store.search_papers("zzzunfindable")["n"] == 0


def test_get_methods_card(store):
    r = store.get_methods_card("2019ApJ...001..001A")
    assert "# A ringed protoplanetary disk" in r["markdown"] and r["ads_url"].endswith("2019ApJ...001..001A")
    r = store.get_methods_card("2019ApJ...001..001A", section="processing")
    assert "## Processing" in r["markdown"] and "## Analysis recipe" not in r["markdown"]
    for s in C.SECTIONS:
        assert store.get_methods_card("2019ApJ...001..001A", section=s)["markdown"].startswith("## ")
    assert "error" in store.get_methods_card("2019ApJ...001..001A", section="bogus")
    assert "error" in store.get_methods_card(HELD)


def test_get_recipe(store):
    r = store.get_recipe("2020A&A...002..002B")
    assert r["n_steps"] == 1 and r["steps"][0]["software"] == ["CASA", "3DBarolo"]
    assert r["steps"][0]["key_parameters"] == ["robust 2 (natural-like)"]
    assert r["processing_stated"]["imaging.cell_size"] == "0.1 arcsec"


def test_find_papers_using(store):
    r = store.find_papers_using(software="3dbarolo")
    assert [x["bibcode"] for x in r["results"]] == ["2020A&A...002..002B"]
    r = store.find_papers_using(parameter_regex=r"robust\s*=\s*-0\.5")
    assert [x["bibcode"] for x in r["results"]] == ["2021MNRAS.003..003C"] and r["results"][0]["matches"]
    r = store.find_papers_using(technique="self-calibration", software="casa")
    assert len(r["results"]) == 2
    assert "error" in store.find_papers_using()
    assert "error" in store.find_papers_using(parameter_regex="(")


def test_parameter_census(store):
    r = store.parameter_census("robust")
    vals = {v["value"]: v["n_papers"] for v in r["values"]}
    assert vals == {"0.5": 1, "2": 1, "-0.5": 1}
    assert r["n_papers_in_scope"] == 3 and r["n_papers_stating_field"] == 3 and r["fraction_not_stated"] == 0
    r = store.parameter_census("self-cal solint")
    vals = {v["value"]: v["n_papers"] for v in r["values"]}
    assert vals == {"inf": 1, "60": 1, "30": 1, "120": 1}
    assert r["n_papers_not_stating_field"] == 1 and r["corpus_fraction_unstated_published_stats"] == 0.722
    assert r["values"][0]["examples"][0]["ads_url"].startswith("https://")
    r = store.parameter_census("cell size", field="nearby_galaxies")
    assert [v["value"] for v in r["values"]] == ["0.1"] and r["n_papers_in_scope"] == 1
    r = store.parameter_census("contsub_order")
    assert {v["value"] for v in r["values"]} == {"1"} and r["n_papers_with_parsed_value"] == 2
    r = store.parameter_census("imaging.final_rms")
    assert r["values"][0]["value"] == "20 ujy/beam" and r["values"][0]["n_papers"] == 3
    assert "error" in store.parameter_census("nonsense_setting")


def test_software_usage(store):
    r = store.software_usage("casa")
    assert r["n_papers"] == 3 and r["fraction_of_papers"] == 1.0 and ("5.6", 3) in [tuple(v) for v in r["versions_stated"]]
    r = store.software_usage("casa", by_year=True)
    assert [y["year"] for y in r["by_year"]] == [2019, 2020, 2021]
    r = store.software_usage(field="nearby_galaxies")
    assert {x["software"] for x in r["top_software"]} == {"casa", "3dbarolo"}
    assert "by_year_top8" in store.software_usage(by_year=True)
    # "which software do papers use for technique X": the scope narrows to papers tagged with it
    r = store.software_usage(technique="self-calibration")
    assert r["technique"] == "self-calibration" and r["n_papers_in_scope"] == 2
    # distinctive_software needs >= 3 papers per package in scope; this 2-paper scope is below the floor
    assert r["distinctive_software"] == [] and r["top_software"]
    assert "distinctive_software" not in store.software_usage()  # no scope, nothing to compare against

def test_skills(store):
    r = store.list_techniques()
    assert r["techniques"][0]["slug"] == "self-calibration" and r["techniques"][0]["papers_tagged_in_bundle"] == 2
    s = store.get_skill("self-calibration")
    assert "Phase first" in s["skill_md"] and "parameters_md" not in s
    s = store.get_skill("self-calibration", include_parameters=True)
    assert "solint" in s["parameters_md"]
    assert "error" in store.get_skill("nope")


def test_compare_papers(store):
    r = store.compare_papers(["2019ApJ...001..001A", "2020A&A...002..002B", HELD])
    assert r["missing"] == [HELD] and len(r["papers"]) == 2
    assert r["table"]["imaging.cell_size"]["2020A&A...002..002B"] == "0.1 arcsec"
    assert r["markdown"].startswith("| | 2019ApJ...001..001A | 2020A&A...002..002B |")
    assert "error" in store.compare_papers(["2019ApJ...001..001A"])


def test_corpus_info(store):
    r = store.corpus_info()
    assert r["version"] == "test" and r["counts"]["papers"] == 3 and "how_to_cite" in r
    assert r["variant"] == "full" and "CC BY 4.0" in r["licenses"]


# ------------------------------------------------------------------------------------------------ data + server
def test_bundle_missing_message(monkeypatch, tmp_path):
    monkeypatch.setenv(data.ENV_VAR, str(tmp_path / "nowhere"))
    with pytest.raises(data.BundleMissing) as e:
        data.find_bundle()
    assert "alma-corpus-mcp download" in str(e.value)


def test_download_checks_sha256(monkeypatch, tmp_path, bundles):
    tgz = bundles["full_manifest"]["tarball"]
    good = data.sha256_file(__import__("pathlib").Path(tgz))
    monkeypatch.setattr(data, "_fetch", lambda url, dest: __import__("shutil").copyfile(tgz, dest))
    out = data.download("https://example.invalid/b.tar.gz", tmp_path / "d1", sha256=good)
    assert (out / "alma_methods.sqlite").is_file() and (out / "skills" / "self-calibration" / "SKILL.md").is_file()
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        data.download("https://example.invalid/b.tar.gz", tmp_path / "d2", sha256="0" * 64)
    assert not (tmp_path / "d2" / "alma_methods.sqlite").exists()


def test_server_tools_prompts_resources(monkeypatch, bundles):
    from alma_mcp.corpus import server
    monkeypatch.setenv(data.ENV_VAR, str(bundles["full"]))
    monkeypatch.setattr(server, "_store", None)
    mcp = server.build_server()

    async def run():
        tools = {t.name for t in await mcp.list_tools()}
        assert tools == {"search_papers", "get_methods_card", "get_recipe", "find_papers_using", "parameter_census",
                         "software_usage", "list_techniques", "get_skill", "compare_papers", "corpus_info"}
        res = await mcp.call_tool("parameter_census", {"setting": "robust"})
        payload = res[1] if isinstance(res, tuple) else res
        text = json.dumps(payload, default=str)
        assert "0.5" in text
        prompts = {p.name for p in await mcp.list_prompts()}
        assert "use-self-calibration" in prompts
        got = await mcp.get_prompt("use-self-calibration", {"task": "plan selfcal"})
        assert "Phase first" in got.messages[0].content.text and "plan selfcal" in got.messages[0].content.text
        uris = {str(r.uri) for r in await mcp.list_resources()}
        assert "alma://skills/self-calibration" in uris
        templates = {t.uriTemplate for t in await mcp.list_resource_templates()}
        assert "alma://cards/{bibcode}" in templates
        card = await mcp.read_resource("alma://cards/2020A%26A...002..002B")
        assert "Dense gas" in list(card)[0].content

    asyncio.run(run())


def test_server_without_bundle(monkeypatch, tmp_path):
    from alma_mcp.corpus import server
    monkeypatch.setenv(data.ENV_VAR, str(tmp_path / "none"))
    monkeypatch.setattr(server, "_store", None)
    mcp = server.build_server()
    assert "error" in server._call("corpus_info") and "download" in server._call("corpus_info")["error"]
    assert asyncio.run(mcp.list_prompts()) == []
