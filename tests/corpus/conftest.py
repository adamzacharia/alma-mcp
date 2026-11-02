"""Fixture: a tiny fake corpus on disk, built into a bundle with the real scripts/build_bundle.py (no network,
no pandas needed: the fake corpus has no catalog parquet, so technique tags stay empty unless a card sets them)."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_build_module():
    spec = importlib.util.spec_from_file_location("build_bundle", ROOT / "scripts" / "build_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_bundle"] = mod
    spec.loader.exec_module(mod)
    return mod


CC_BY = "http://creativecommons.org/licenses/by/4.0/"
ARXIV = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def ev(value, loc="2 Observations", quote=None):
    return {"value": value, "evidence": [{"loc": loc, "quote": quote or f"we used {value}"}] if value != "not stated" else []}


def make_card(bib, title, year, fields, *, robust="Briggs robust = 0.5", cell="0.02''", selfcal="not stated",
              contsub="not stated", software=("CASA",), steps=None, heldout=False):
    steps = steps or [{"order": 1, "goal": "Image continuum", "method": ev("tclean with Briggs weighting"),
                       "software": list(software), "key_parameters": [ev(robust)], "assumptions": [],
                       "fitting": ev("not stated"), "validation": ev("not stated")}]
    return {
        "schema_version": "1.0",
        "identity": {"bibcode": bib, "title": title, "year": year, "journal": "ApJ", "first_author": "Doe, J.",
                     "arxiv_id": "2101.00001", "doi": "10.0/x", "heldout": heldout},
        "body": {
            "depth": "full", "paper_type": "observational_alma_primary",
            "science": {"fields": fields, "question": ev("What is the disk mass?", "abstract"), "targets": []},
            "data": {"alma_project_codes": [{"code": "2016.1.00001.S", "evidence": [{"loc": "Ack", "quote": "2016.1.00001.S"}]}],
                     "bands": ev("Band 6 and Band 7"), "arrays": ev("12m and 7m (ACA)"),
                     "angular_resolution": ev("0.05 arcsec"), "sensitivity": ev("20 uJy/beam")},
            "processing": {
                "calibration": {"pipeline_or_manual": ev("ALMA pipeline"), "casa_version": ev("5.6"),
                                "self_calibration": ev(selfcal), "flux_scale_uncertainty": ev("10%")},
                "imaging": {"task": ev("tclean"), "deconvolver": ev("multiscale"), "weighting_robust": ev(robust),
                            "cell_size": ev(cell), "continuum_subtraction": ev(contsub), "final_rms": ev("20 uJy/beam")},
            },
            "analysis_steps": steps,
            "software": [{"name": s, "version": "5.6" if s == "CASA" else "not stated", "used_for": ev("imaging")} for s in software],
            "results_numeric": [{"quantity": "dust mass", "value": "12", "uncertainty": "3", "units": "Earth masses",
                                 "from_step": 1, "evidence": [{"loc": "Sec 4", "quote": "M_dust = 12 +/- 3 M_earth"}]}],
            "findings_qualitative": [ev("Ring at 40 au")],
            "claims": [{"hypothesis": "h", "experiment": "e", "result": "r", "finding": "a ring", "support": "strong"}],
            "reproducibility": {"score_inferred": 4, "rediscovery_candidate": True, "missing_for_reproduction": ["mask"]},
            "links": {},
        },
        "provenance": {"model": "claude-sonnet-x", "custom_id": "card-sonnet-abc123", "batch_id": "msgbatch_01SECRET",
                       "run": "sonnet_run2", "usage": {"input_tokens": 1000}, "usd": 0.04, "text_source": "latex",
                       "extractor": "1", "collected_at": "2026-10-28T18:36:46"},
        "checks": {"evidence_ok": True, "quotes": 5, "quote_pass_rate": 1.0,
                   "quotes_too_long": ["this quote was much too long to keep in the card as evidence text"],
                   "codes_not_in_text": ["2016.1.00001.S"]},
    }


# Verbatim text that only appears in quotes and in the repair log of the closed-license paper (all > 40 chars).
CLOSED_LONG_QUOTE = "the dense gas fraction rises steeply toward the galaxy centre in all three fields"
CLOSED_REPAIR_OLD = "we imaged the cube with natural weighting and a 0.1 arcsec pixel"
CLOSED_REPAIR_NEW = "we imaged the cube with natural weighting and a 0.1 arcsec pixel."
# A structured value the extractor copied word for word from its own quote, and one that matches a quote found only
# in another extraction channel's card for the same paper.
CLOSED_VERBATIM_VALUE = "How does the dense gas fraction vary with radius across the galaxy disk?"
OTHER_CHANNEL_QUOTE = "the HCN to CO ratio was measured in elliptical annuli of constant width"


PAPERS = [
    # bibcode, title, year, fields, license, heldout, card kwargs
    ("2019ApJ...001..001A", "A ringed protoplanetary disk", 2019, ["protoplanetary_disks"], CC_BY, 0,
     dict(selfcal="phase-only self-calibration with solint inf, 60s and 30s", contsub="uvcontsub fitorder=1")),
    ("2020A&A...002..002B", "Dense gas in a nearby galaxy", 2020, ["nearby_galaxies"], ARXIV, 0,
     dict(robust="robust 2 (natural-like)", cell="0.1 arcsec", software=("CASA", "3DBarolo"),
          contsub="first-order polynomial in the image plane")),
    ("2021MNRAS.003..003C", "Self-calibration of a bright quasar host", 2021, ["high_z_galaxies"], None, 0,
     dict(selfcal="phase and amplitude self-calibration, solint 2 min", robust="Briggs weighting, robust=-0.5")),
    ("2022ApJ...004..004H", "Held out benchmark disk", 2022, ["protoplanetary_disks"], CC_BY, 1,
     dict(selfcal="phase self-calibration solint 10s")),
]


def write_fake_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    (corpus / "skills" / "stats").mkdir(parents=True)
    con = sqlite3.connect(corpus / "manifest.sqlite")
    con.execute("CREATE TABLE papers (bibcode TEXT PRIMARY KEY, heldout INTEGER, status TEXT, license TEXT, "
                "abstract TEXT, arxiv_id TEXT, doi TEXT, title TEXT, year INTEGER, journal TEXT, first_author TEXT)")
    for bib, title, year, fields, lic, held, kw in PAPERS:
        con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (bib, held, "evidence_ok", lic, f"We present ALMA observations of {title.lower()}.", "2101.00001",
                     "10.0/x", title, year, "ApJ", "Doe, J."))
        mod = load_build_module()
        d = corpus / "cards" / mod.bib_key(bib) / "sonnet"
        d.mkdir(parents=True)
        card = make_card(bib, title, year, fields, **kw)
        if bib.startswith("2019"):  # cites the held-out paper inside a quote; the build must redact it
            card["body"]["findings_qualitative"].append(ev("Same ring as before", "Sec 5", "as in (cite: 2022ApJ...004..004H)"))
        if bib.startswith("2020"):  # closed license: long quote plus a repair log with verbatim old/new text
            card["body"]["findings_qualitative"].append(ev("Dense gas fraction rises inward", "Sec 4", CLOSED_LONG_QUOTE))
            card["repair_log"] = {"at": "2026-10-29T00:40:00", "custom_id": "repair-sonnet-xyz",
                                  "fixed": [{"path": "findings_qualitative[0].evidence[0]", "old": CLOSED_REPAIR_OLD,
                                             "new": CLOSED_REPAIR_NEW}]}
            card["body"]["science"]["question"] = ev(CLOSED_VERBATIM_VALUE, "abstract", CLOSED_VERBATIM_VALUE)
            card["body"]["findings_qualitative"].append(ev(OTHER_CHANNEL_QUOTE, "Sec 3", "ratio in annuli"))
            other = make_card(bib, title, year, fields, **kw)
            other["checks"] = {"evidence_ok": False}
            other["body"]["findings_qualitative"].append(ev("HCN/CO in annuli", "Sec 3", OTHER_CHANNEL_QUOTE))
            (corpus / "cards" / mod.bib_key(bib) / "gpt61sol").mkdir(parents=True)
            (corpus / "cards" / mod.bib_key(bib) / "gpt61sol" / "card.json").write_text(json.dumps(other), encoding="utf-8")
        (d / "card.json").write_text(json.dumps(card), encoding="utf-8")
    # a paper that is not evidence_ok must never appear
    con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("2023ApJ...005..005X", 0, "extracted", CC_BY, "", "", "", "Not shredded", 2023, "ApJ", "X"))
    con.commit()
    con.close()
    sk = corpus / "skills" / "self-calibration"
    (sk / "references").mkdir(parents=True)
    (sk / "SKILL.md").write_text('---\nname: alma-self-calibration\ndescription: "Plan phase and amplitude self-cal."\n---\n\n'
                                 "# Self-calibration\n\nPhase first. Example 2019ApJ...001..001A; held out 2022ApJ...004..004H.\n"
                                 "Counts are from the pack (`C:\\Users\\someone\\alma-corpus\\skills\\packs\\self-calibration.md`).\n",
                                 encoding="utf-8")
    (sk / "references" / "parameters.md").write_text("From `/home/someone/alma-corpus/skills/packs/self-calibration.md`.\n\n"
                                                     "| Paper | solint |\n|---|---|\n| 2019ApJ...001..001A | inf |\n",
                                                     encoding="utf-8")
    (corpus / "skills" / "index.json").write_text(json.dumps({"techniques": [
        {"slug": "self-calibration", "title": "Self-calibration strategy", "papers": 2, "state": "reviewed"}]}),
        encoding="utf-8")
    st = corpus / "skills" / "stats"
    (st / "settings_long.csv").write_text("setting,value,field,year\nrobust,0.5,protoplanetary_disks,2019\n", encoding="utf-8")
    (st / "unstated_rates.csv").write_text(",fraction_unstated\nimaging.cell_size,0.836\ncalibration.self_calibration,0.722\n",
                                           encoding="utf-8")
    (st / "software_overall.csv").write_text("software_norm,mentions\ncasa,3\n3dbarolo,1\n", encoding="utf-8")
    return corpus


@pytest.fixture(scope="session")
def build_mod():
    return load_build_module()


@pytest.fixture(scope="session")
def bundles(tmp_path_factory, build_mod):
    root = tmp_path_factory.mktemp("alma")
    corpus = write_fake_corpus(root)
    full = build_mod.build(corpus, root / "full", public=False, quasar_root=None, version="test", tar_dir=root / "rel", tagger=fake_tagger)
    pub = build_mod.build(corpus, root / "public", public=True, quasar_root=None, version="test", tar_dir=root / "rel", tagger=fake_tagger)
    return {"root": root, "full": root / "full", "public": root / "public", "full_manifest": full, "public_manifest": pub}


@pytest.fixture()
def store(bundles):
    from alma_mcp.corpus.store import CorpusStore
    return CorpusStore(bundles["full"])


def fake_tagger(card):
    """Tags 'self-calibration' when the card states self-cal, on step index 0."""
    sc = card["body"]["processing"]["calibration"]["self_calibration"]["value"]
    return {"self-calibration": {"steps": [0], "processing": {}}} if sc != "not stated" else {}
