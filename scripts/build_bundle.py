"""Build the alma-corpus data bundle from the ALMA paper corpus (read-only).

Usage (pandas + pyarrow are needed to read catalog/*.parquet technique tags, e.g. Quasar's venv or
`pip install alma-mcp[build]`; without them, or without a catalog, techniques come from the optional tagger):

    python scripts/build_bundle.py --corpus /path/to/alma-corpus                  # full + public variants
    python scripts/build_bundle.py --corpus /path/to/alma-corpus --only full --out dist-data

The corpus folder comes from --corpus or the ALMA_CORPUS environment variable (one of them is required); the
optional Quasar checkout for the technique tagger from --quasar-root or QUASAR_ROOT.

Writes, per variant:
    <out>/alma_methods.sqlite   tables + FTS5 indexes (see SCHEMA below)
    <out>/skills/<slug>/SKILL.md, references/*.md, index.json
    <out>/manifest.json         version, build date, counts, sha256 of the sqlite file
and a release tarball <out>/release/full/alma-mcp-data-<version>.tar.gz or
<out>/release/public/alma-mcp-data-public-<version>.tar.gz with a .sha256 sidecar. Only the public tarball is meant
for public upload; the full one carries verbatim quotes from arXiv non-exclusive papers (private or team use only).
After writing, every variant is scanned for local paths, user names and API-key-like strings; a hit fails the build.

Held-out benchmark papers (manifest heldout = 1) are excluded from every table, every card and every skill file.
The public variant (--public, or the second variant of a default build) withholds verbatim evidence quotes and
abstracts for papers whose arXiv license is not CC BY, CC BY-SA, CC0 or public domain.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tarfile
import urllib.parse
from pathlib import Path
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alma_mcp.corpus import cards as C  # noqa: E402

SCHEMA_VERSION = "1"
# Order of card channels tried for each paper: the production model first, then the other shred channels. The first
# card whose checks pass wins; if none pass, the first card found is used (same rule as Quasar's load_cards).
CARD_MODELS = ("sonnet", "gpt61sol", "sonnetcc", "gemini4argon", "gptoss120b", "deepseekv32", "minimaxm3")

SCHEMA = """
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE papers (bibcode TEXT PRIMARY KEY, title TEXT, year INTEGER, journal TEXT, first_author TEXT,
  fields TEXT, techniques TEXT, alma_project_codes TEXT, arxiv_id TEXT, doi TEXT, license TEXT,
  license_open INTEGER, abstract TEXT, depth TEXT, paper_type TEXT, bands TEXT, arrays TEXT,
  repro_score INTEGER, rediscovery_candidate INTEGER, card_model TEXT);
CREATE TABLE steps (bibcode TEXT, ord INTEGER, goal TEXT, method TEXT, software TEXT, key_parameters TEXT,
  assumptions TEXT, fitting TEXT, validation TEXT, techniques TEXT);
CREATE TABLE software (bibcode TEXT, name TEXT, name_norm TEXT, version TEXT, used_for TEXT);
CREATE TABLE results (bibcode TEXT, quantity TEXT, value TEXT, uncertainty TEXT, units TEXT, from_step TEXT);
CREATE TABLE claims (bibcode TEXT, hypothesis TEXT, experiment TEXT, result TEXT, finding TEXT, support TEXT);
CREATE TABLE datasets (bibcode TEXT, project_code TEXT);
CREATE TABLE cards (id INTEGER PRIMARY KEY, bibcode TEXT UNIQUE, model TEXT, card_md TEXT, card_json TEXT, quotes_withheld INTEGER);
CREATE TABLE processing (bibcode TEXT, grp TEXT, key TEXT, value TEXT, stated INTEGER);
CREATE TABLE paper_settings (bibcode TEXT, setting TEXT, value TEXT);
CREATE TABLE settings (setting TEXT, value TEXT, field TEXT, year INTEGER);
CREATE TABLE unstated_rates (setting TEXT PRIMARY KEY, fraction_unstated REAL);
CREATE TABLE software_overall (software TEXT PRIMARY KEY, mentions INTEGER);
CREATE TABLE skills (slug TEXT PRIMARY KEY, title TEXT, description TEXT, state TEXT, paper_count INTEGER,
  skill_md TEXT, parameters_md TEXT);
CREATE VIRTUAL TABLE papers_fts USING fts5(bibcode UNINDEXED, title, abstract, fields, techniques,
  tokenize='porter unicode61');
CREATE VIRTUAL TABLE cards_fts USING fts5(bibcode UNINDEXED, card_md, content='cards', content_rowid='id',
  tokenize='porter unicode61');
CREATE VIRTUAL TABLE steps_fts USING fts5(bibcode UNINDEXED, ord UNINDEXED, text, tokenize='porter unicode61');
"""

INDEXES = """
CREATE INDEX steps_bib ON steps(bibcode);
CREATE INDEX software_bib ON software(bibcode);
CREATE INDEX software_norm ON software(name_norm);
CREATE INDEX results_bib ON results(bibcode);
CREATE INDEX claims_bib ON claims(bibcode);
CREATE INDEX datasets_bib ON datasets(bibcode);
CREATE INDEX processing_bib ON processing(bibcode);
CREATE INDEX processing_key ON processing(grp, key);
CREATE INDEX psettings_setting ON paper_settings(setting);
"""


# ------------------------------------------------------------------------------------------------ helpers
def bib_key(bibcode: str) -> str:
    """Card directory name for a bibcode (same encoding as the corpus: URL-quoted, trailing dot escaped)."""
    key = urllib.parse.quote(bibcode, safe="")
    return key[:-1] + "%2E" if key.endswith(".") else key


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def heldout_set(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    """Bibcodes flagged held-out (manifest heldout = 1, or a card identity that says heldout)."""
    return {r["bibcode"] for r in rows if r.get("heldout")}


def exclude_heldout(rows: Iterable[Dict[str, Any]], heldout: Set[str], key: str = "bibcode") -> List[Dict[str, Any]]:
    """Drop every row whose bibcode is held out or whose own heldout flag is set."""
    out = []
    for r in rows:
        if r.get(key) in heldout:
            continue
        h = r.get("heldout")
        if h not in (None, False, 0, "0", "False", "false", "<NA>") and str(h).lower() not in ("nan", "none"):
            continue
        out.append(r)
    return out


def redact_heldout_text(text: str, heldout: Set[str]) -> Tuple[str, int]:
    """Replace any held-out bibcode inside free text (skills) with a neutral marker."""
    n = 0
    for b in heldout:
        if b in text:
            n += text.count(b)
            text = text.replace(b, "[held-out paper]")
    return text, n


def redact_heldout_card(card: Dict[str, Any], heldout: Set[str]) -> Tuple[Dict[str, Any], int]:
    """A card with held-out bibcodes (e.g. citations of a held-out paper inside a quote) replaced by a marker."""
    text = json.dumps(card, ensure_ascii=False)
    red, n = redact_heldout_text(text, heldout)
    return (json.loads(red), n) if n else (card, 0)


def pick_card(card_dir: Path, models: Sequence[str] = CARD_MODELS) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """The card for one paper: first channel whose checks.evidence_ok is true, else the first card found."""
    found = []
    for m in models:
        c = read_json(card_dir / m / "card.json")
        if c is not None:
            found.append((m, c))
    for m, c in found:
        if (c.get("checks") or {}).get("evidence_ok"):
            return m, c
    return found[0] if found else (None, None)


# Absolute local paths (Windows user folders, macOS /Users, Linux /home) inside skill text.
_LOCAL_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]+Users[\\/]+|/Users/|/home/)[^\s`'\"<>|()\[\]]+")


def neutral_path(m: "re.Match[str]") -> str:
    """A local corpus path becomes its corpus-relative form (e.g. skills/packs/<file>); other paths '<local>/<name>'."""
    p = re.sub(r"[\\/]+", "/", m.group(0))
    i = p.lower().find("alma-corpus/")
    if i >= 0:
        return p[i + len("alma-corpus/"):]
    return "<local>/" + p.rstrip("/").rsplit("/", 1)[-1]


def neutralize_paths(text: str) -> Tuple[str, int]:
    """Replace absolute local paths with a neutral relative form. Returns (text, number_replaced)."""
    return _LOCAL_PATH.subn(neutral_path, text)


# Patterns the final scan refuses: local user paths, the builder's user name, API-key-like tokens. The key
# patterns need a token after the prefix, so ordinary words such as "disk-averaged" do not match.
LEAK_PATTERNS = {
    "windows user path": re.compile(r"[A-Za-z]:\\+Users\\", re.I),
    "user name": re.compile(re.escape(Path.home().name)),
    "sk- API key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
}


def scan_text(text: str) -> List[str]:
    """Names of LEAK_PATTERNS found in text."""
    return [name for name, rx in LEAK_PATTERNS.items() if rx.search(text)]


def scan_bundle(out: Path) -> List[str]:
    """Every leak hit in a built variant: each text cell of each sqlite table (FTS tables skipped, they copy the
    same text), each skill file and manifest.json. Returns 'where: pattern' strings (empty means clean)."""
    hits: List[str] = []
    con = sqlite3.connect(f"file:{(out / 'alma_methods.sqlite').as_posix()}?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if "_fts" not in r[0]]
    for t in tables:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        for row in con.execute(f"SELECT * FROM {t}"):
            for c, v in zip(cols, row):
                if isinstance(v, str):
                    hits += [f"{t}.{c} ({row[0]}): {name}" for name in scan_text(v)]
    con.close()
    for p in sorted((out / "skills").rglob("*")) + [out / "manifest.json"]:
        if p.is_file():
            hits += [f"{p.relative_to(out).as_posix()}: {name}" for name in scan_text(p.read_text(encoding="utf-8"))]
    return hits


def norm_software(name: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", (name or "").strip().lower())


def _int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _bool(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    s = str(x).strip().lower()
    return 1 if s in ("true", "1", "yes") else 0 if s in ("false", "0", "no") else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rmtree_retry(path: Path, tries: int = 5) -> None:
    """shutil.rmtree that also removes read-only files and folders (Windows refuses those) and retries briefly
    (an indexer or virus scanner can hold a just-emptied folder)."""
    import stat
    import time

    def clear_readonly(func, p, _exc):
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        func(p)

    for k in range(tries):
        try:
            shutil.rmtree(path, onerror=clear_readonly)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ loading
def load_manifest(corpus: Path) -> List[Dict[str, Any]]:
    con = sqlite3.connect(f"file:{(corpus / 'manifest.sqlite').as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT bibcode, heldout, status, license, abstract, arxiv_id, doi, title, year, journal, first_author "
        "FROM papers")]
    con.close()
    return rows


def load_catalog_techniques(corpus: Path, heldout: Set[str]):
    """Technique tags from catalog/*.parquet: {bibcode: 'a;b'} and {bibcode: {step order: 'a;b'}}."""
    if not (corpus / "catalog" / "papers.parquet").exists():
        return {}, {}
    try:
        import pandas as pd
    except ImportError:
        print("warning: pandas not installed; catalog technique tags skipped")
        return {}, {}

    papers = pd.read_parquet(corpus / "catalog" / "papers.parquet").astype(object).where(lambda d: d.notna(), None)
    steps = pd.read_parquet(corpus / "catalog" / "steps.parquet").astype(object).where(lambda d: d.notna(), None)
    p_rows = exclude_heldout(papers.to_dict("records"), heldout)
    s_rows = exclude_heldout(steps.to_dict("records"), heldout)
    ptech = {r["bibcode"]: r.get("techniques") or "" for r in p_rows}
    stech: Dict[str, Dict[int, str]] = {}
    for r in s_rows:
        stech.setdefault(r["bibcode"], {})[_int(r.get("order"))] = r.get("techniques") or ""
    return ptech, stech


def load_tagger(quasar_root: Optional[Path]):
    """Quasar's tag_card (technique regexes), used only for papers the catalog has not tagged yet. Optional."""
    if not quasar_root or not (quasar_root / "scripts" / "alma_corpus" / "skills.py").exists():
        return None
    sys.path.insert(0, str(quasar_root))
    try:
        from scripts.alma_corpus.skills import tag_card  # type: ignore
        return tag_card
    except Exception as e:  # pragma: no cover - depends on the local checkout
        print(f"warning: could not import Quasar tag_card ({e}); untagged papers keep empty techniques")
        return None
    finally:
        sys.path.remove(str(quasar_root))


# ------------------------------------------------------------------------------------------------ build
def build_rows(card: Dict[str, Any], man: Dict[str, Any], model: str, ptech: str, stech: Dict[int, str],
               public: bool, verbatim: Iterable[str] = ()) -> Dict[str, List[Dict[str, Any]]]:
    """All table rows for one paper. Pure function (tested). ``verbatim``: extra quote strings of this paper (other
    extraction channels) that must not survive in a public, closed-license card."""
    i = card.get("identity") or {}
    bib = man["bibcode"]
    lic = man.get("license")
    lic_open = C.license_is_open(lic)
    withheld = public and not lic_open
    shown, n_quotes = C.bundle_card(card, public=public, withhold_quotes=withheld, verbatim=verbatim)
    b = shown.get("body") or {}  # every table row comes from the shipped (sanitised) body
    d = b.get("data") or {}
    codes = [pc.get("code") for pc in d.get("alma_project_codes") or [] if isinstance(pc, dict) and pc.get("code")]
    rep = b.get("reproducibility") or {}
    rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in
                                             ("papers", "steps", "software", "results", "claims", "datasets",
                                              "cards", "processing", "paper_settings")}
    rows["papers"].append({
        "bibcode": bib, "title": man.get("title") or i.get("title"), "year": _int(man.get("year") or i.get("year")),
        "journal": man.get("journal") or i.get("journal"), "first_author": man.get("first_author") or i.get("first_author"),
        "fields": ";".join((b.get("science") or {}).get("fields") or []), "techniques": ptech,
        "alma_project_codes": ";".join(C.uniq(codes)), "arxiv_id": man.get("arxiv_id") or i.get("arxiv_id"),
        "doi": man.get("doi") or i.get("doi"), "license": lic, "license_open": int(lic_open),
        "abstract": None if withheld else man.get("abstract"),
        "depth": b.get("depth"), "paper_type": b.get("paper_type"),
        "bands": C.fact_value(d.get("bands")), "arrays": C.fact_value(d.get("arrays")),
        "repro_score": _int(rep.get("score_inferred")), "rediscovery_candidate": _bool(rep.get("rediscovery_candidate")),
        "card_model": model})
    for pc in C.uniq(codes):
        rows["datasets"].append({"bibcode": bib, "project_code": pc})
    for k, s in enumerate(sorted(b.get("analysis_steps") or [], key=lambda s: _int(s.get("order")) or 0)):
        order = _int(s.get("order")) or k + 1
        rows["steps"].append({
            "bibcode": bib, "ord": order, "goal": s.get("goal"), "method": C.fact_value(s.get("method")),
            "software": ";".join(s.get("software") or []),
            "key_parameters": json.dumps([C.fact_value(p) for p in s.get("key_parameters") or []], ensure_ascii=False),
            "assumptions": json.dumps([C.fact_value(a) for a in s.get("assumptions") or []], ensure_ascii=False),
            "fitting": C.fact_value(s.get("fitting")), "validation": C.fact_value(s.get("validation")),
            "techniques": stech.get(order, "")})
    for s in b.get("software") or []:
        if not isinstance(s, dict) or not s.get("name"):
            continue
        rows["software"].append({"bibcode": bib, "name": s.get("name"), "name_norm": norm_software(s.get("name")),
                                 "version": C.fact_value(s.get("version")) or None,
                                 "used_for": C.fact_value(s.get("used_for"))})
    for r in b.get("results_numeric") or []:
        if isinstance(r, dict):
            rows["results"].append({"bibcode": bib, **{k: None if r.get(k) is None else str(r.get(k))
                                                       for k in ("quantity", "value", "uncertainty", "units", "from_step")}})
    for cl in b.get("claims") or []:
        if isinstance(cl, dict):
            rows["claims"].append({"bibcode": bib, **{k: None if cl.get(k) is None else C.fact_value(cl.get(k))
                                                      for k in ("hypothesis", "experiment", "result", "finding", "support")}})
    for grp, facts in (b.get("processing") or {}).items():
        if isinstance(facts, dict):
            for k, v in facts.items():
                if isinstance(v, (dict, str)) or v is None:
                    rows["processing"].append({"bibcode": bib, "grp": grp, "key": k, "value": C.fact_value(v),
                                               "stated": int(C.stated(v))})
    for k in ("bands", "arrays", "angular_resolution", "sensitivity", "time_on_source", "configurations"):
        v = d.get(k)
        if isinstance(v, (dict, str)):
            rows["processing"].append({"bibcode": bib, "grp": "data", "key": k, "value": C.fact_value(v),
                                       "stated": int(C.stated(v))})
    for setting, value in dict.fromkeys(C.extract_settings(b)):
        rows["paper_settings"].append({"bibcode": bib, "setting": setting, "value": value})
    rows["cards"].append({"bibcode": bib, "model": model,
                          "card_md": C.render_card(shown, license_url=lic, model=model),
                          "card_json": json.dumps(shown, ensure_ascii=False), "quotes_withheld": n_quotes})
    return rows


def load_skills(corpus: Path, heldout: Set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]], int]:
    """Skill rows plus {slug: {relpath: text}} files to copy. Held-out bibcodes are redacted (count returned)."""
    sk = corpus / "skills"
    index = read_json(sk / "index.json") or {}
    meta = {t["slug"]: t for t in index.get("techniques") or []}
    rows, files, redacted = [], {}, 0
    for d in sorted(p for p in sk.iterdir() if p.is_dir() and (p / "SKILL.md").exists()):
        slug = d.name
        texts: Dict[str, str] = {}
        for f in [d / "SKILL.md"] + sorted((d / "references").glob("*.md")) if (d / "references").exists() else [d / "SKILL.md"]:
            t, n = redact_heldout_text(f.read_text(encoding="utf-8"), heldout)
            redacted += n
            t, _ = neutralize_paths(t)
            texts[f.relative_to(d).as_posix()] = t
        md = texts["SKILL.md"]
        fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", md, re.S)
        desc = ""
        if fm:
            m = re.search(r'^description:\s*"?(.*?)"?\s*$', fm.group(1), re.M)
            desc = m.group(1) if m else ""
        title_m = re.search(r"^#\s+(.+)$", md, re.M)
        t = meta.get(slug, {})
        rows.append({"slug": slug, "title": t.get("title") or (title_m.group(1) if title_m else slug),
                     "description": desc, "state": t.get("state"), "paper_count": _int(t.get("papers")),
                     "skill_md": md, "parameters_md": texts.get("references/parameters.md")})
        files[slug] = texts
    return rows, files, redacted


def build(corpus: Path, out: Path, *, public: bool, quasar_root: Optional[Path], version: str,
          tar_dir: Optional[Path] = None, tagger: Any = None) -> Dict[str, Any]:
    """Build one bundle variant into ``out``. ``tagger(card) -> {slug: {"steps": [idx...]}}`` tags papers the
    catalog lacks (default: Quasar's tag_card from ``quasar_root`` when importable)."""
    man_rows = load_manifest(corpus)
    held = heldout_set(man_rows)
    candidates = [r for r in exclude_heldout(man_rows, held) if r["status"] == "evidence_ok"]
    ptech, stech = load_catalog_techniques(corpus, held)
    tables: Dict[str, List[Dict[str, Any]]] = {}
    n_tagged_live = n_skipped_card = n_card_heldout = n_card_mentions = 0
    for man in sorted(candidates, key=lambda r: r["bibcode"]):
        model, card = pick_card(corpus / "cards" / bib_key(man["bibcode"]))
        if card is None:
            n_skipped_card += 1
            continue
        if (card.get("identity") or {}).get("heldout"):
            n_card_heldout += 1
            continue
        bib = man["bibcode"]
        pt = ptech.get(bib)
        st = stech.get(bib, {}) if pt is not None else {}
        if pt is None:
            if tagger is None:
                tagger = load_tagger(quasar_root) or False
            if tagger:
                tags = tagger(card)
                pt = ";".join(sorted(tags))
                orders = [_int(s.get("order")) or k + 1 for k, s in enumerate(card["body"].get("analysis_steps") or [])]
                st = {}
                for slug, v in tags.items():
                    for idx in v["steps"]:
                        o = orders[idx]
                        st[o] = ";".join(sorted(set(C.split_list(st.get(o, ""))) | {slug}))
                n_tagged_live += 1
            else:
                pt = ""
        card, n_red = redact_heldout_card(card, held)
        n_card_mentions += n_red
        verbatim: Set[str] = set()
        if public and not C.license_is_open(man.get("license")):
            for m in CARD_MODELS:
                other = read_json(corpus / "cards" / bib_key(bib) / m / "card.json")
                if other is not None:
                    verbatim |= C.quote_strings(other)
        for k, v in build_rows(card, man, model, pt, st, public, verbatim).items():
            tables.setdefault(k, []).extend(v)

    stats = corpus / "skills" / "stats"
    settings = read_csv(stats / "settings_long.csv")
    unstated = {r[""]: r["fraction_unstated"] for r in read_csv(stats / "unstated_rates.csv")}
    sw_overall = read_csv(stats / "software_overall.csv")
    skills, skill_files, redacted = load_skills(corpus, held)

    out.mkdir(parents=True, exist_ok=True)
    db = out / "alma_methods.sqlite"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)

    def ins(table: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cols = list(rows[0].keys())
        con.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [tuple(r.get(c) for c in cols) for r in rows])

    for t in ("papers", "steps", "software", "results", "claims", "datasets", "cards", "processing", "paper_settings"):
        ins(t, tables.get(t, []))
    ins("settings", [{"setting": r["setting"], "value": str(r["value"]), "field": r["field"], "year": _int(r["year"])}
                     for r in settings])
    ins("unstated_rates", [{"setting": k, "fraction_unstated": float(v)} for k, v in unstated.items()])
    ins("software_overall", [{"software": r["software_norm"], "mentions": int(r["mentions"])} for r in sw_overall
                             if isinstance(r.get("software_norm"), str)])
    ins("skills", skills)
    con.execute("INSERT INTO papers_fts (bibcode, title, abstract, fields, techniques) "
                "SELECT bibcode, title, coalesce(abstract,''), replace(fields,';',' '), "
                "replace(replace(techniques,';',' '),'-',' ') FROM papers")
    con.execute("INSERT INTO cards_fts (rowid, bibcode, card_md) SELECT id, bibcode, card_md FROM cards")
    con.executemany("INSERT INTO steps_fts (bibcode, ord, text) VALUES (?,?,?)",
                    [(s["bibcode"], s["ord"], " ".join(x for x in (
                        s["goal"], s["method"], s["software"].replace(";", " "),
                        " ".join(json.loads(s["key_parameters"])), " ".join(json.loads(s["assumptions"])),
                        s["fitting"], s["validation"]) if x)) for s in tables.get("steps", [])])
    con.executescript(INDEXES)
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in
              ("papers", "steps", "software", "results", "claims", "datasets", "cards", "processing",
               "paper_settings", "settings", "skills")}
    counts["papers_open_license"] = con.execute("SELECT count(*) FROM papers WHERE license_open=1").fetchone()[0]
    counts["quotes_withheld"] = con.execute("SELECT coalesce(sum(quotes_withheld),0) FROM cards").fetchone()[0]
    years = con.execute("SELECT min(year), max(year) FROM papers").fetchone()
    built = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"version": version, "schema_version": SCHEMA_VERSION, "variant": "public" if public else "full",
            "build_date": built, "year_min": str(years[0]), "year_max": str(years[1]),
            "heldout_excluded": str(len(held))}
    con.executemany("INSERT INTO meta (k, v) VALUES (?, ?)", list(meta.items()) + [("counts", json.dumps(counts))])
    con.commit()
    con.execute("VACUUM")
    con.close()

    sk_out = out / "skills"
    if sk_out.exists():
        rmtree_retry(sk_out)
    for slug, texts in skill_files.items():
        for rel, txt in texts.items():
            p = sk_out / slug / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt, encoding="utf-8", newline="\n")
    (sk_out / "index.json").write_text(json.dumps(
        [{k: s[k] for k in ("slug", "title", "description", "state", "paper_count")} for s in skills], indent=1),
        encoding="utf-8")

    manifest = {**meta, "counts": counts, "sqlite": db.name, "sha256": sha256_file(db),
                "size_bytes": db.stat().st_size, "skills_redacted_heldout_mentions": redacted,
                "papers_without_card": n_skipped_card, "papers_tagged_at_build": n_tagged_live,
                "card_heldout_flagged": n_card_heldout, "cards_redacted_heldout_mentions": n_card_mentions,
                "source": "ALMA paper corpus (methods cards from arXiv/ADS full texts)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if tar_dir is not None:
        tar_dir = tar_dir / ("public" if public else "full")
        tar_dir.mkdir(parents=True, exist_ok=True)
        tar_path = tar_dir / f"alma-mcp-data{'-public' if public else ''}-{version}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for p in [db, out / "manifest.json"] + sorted(sk_out.rglob("*")):
                if p.is_file():
                    tar.add(p, arcname=p.relative_to(out).as_posix())
        digest = sha256_file(tar_path)
        tar_path.with_name(tar_path.name + ".sha256").write_text(f"{digest}  {tar_path.name}\n", encoding="utf-8")
        manifest["tarball"] = str(tar_path)
        manifest["tarball_size_bytes"] = tar_path.stat().st_size
        manifest["tarball_sha256"] = digest
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env_corpus, env_quasar = os.environ.get("ALMA_CORPUS"), os.environ.get("QUASAR_ROOT")
    ap.add_argument("--corpus", type=Path, default=Path(env_corpus) if env_corpus else None,
                    required=not env_corpus, help="ALMA paper corpus folder (default: $ALMA_CORPUS)")
    ap.add_argument("--out", type=Path, default=ROOT / "dist-data", help="full variant goes here, public in <out>/public")
    ap.add_argument("--public", action="store_true", help="build only the public (quote-stripped) variant")
    ap.add_argument("--only", choices=("full", "public", "both"), default=None)
    ap.add_argument("--quasar-root", type=Path, default=Path(env_quasar) if env_quasar else None,
                    help="optional: Quasar checkout whose technique tagger fills tags for papers the catalog lacks "
                         "(default: $QUASAR_ROOT)")
    ap.add_argument("--version", default=_dt.date.today().strftime("%Y.%m.%d"))
    ap.add_argument("--no-tarball", action="store_true")
    a = ap.parse_args(argv)
    which = a.only or ("public" if a.public else "both")
    variants = [("full", False), ("public", True)] if which == "both" else [(which, which == "public")]
    for name, public in variants:
        out = a.out / "public" if (public and which == "both") else a.out
        m = build(a.corpus, out, public=public, quasar_root=a.quasar_root, version=a.version,
                  tar_dir=None if a.no_tarball else a.out / "release")
        print(f"[{name}] {out / 'alma_methods.sqlite'}: {m['size_bytes'] / 1e6:.1f} MB"
              + (f", tarball {m['tarball_size_bytes'] / 1e6:.1f} MB" if "tarball" in m else ""))
        print(f"[{name}] counts: {json.dumps(m['counts'])}")
        print(f"[{name}] sha256 {m['sha256']}; held-out excluded {m['heldout_excluded']}; "
              f"held-out mentions redacted: skills {m['skills_redacted_heldout_mentions']}, cards {m['cards_redacted_heldout_mentions']}; "
              f"papers without a card {m['papers_without_card']}; tagged at build {m['papers_tagged_at_build']}")
        hits = scan_bundle(out)
        if hits:
            print(f"[{name}] LEAK SCAN FAILED: {len(hits)} hits")
            for h in hits[:50]:
                print("  " + h)
            return 1
        print(f"[{name}] leak scan clean (local paths, user name, sk-/AIza keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
