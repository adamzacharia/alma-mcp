"""Pure helpers for methods cards: license rules, quote stripping, markdown rendering, settings extraction.

Used both by ``scripts/build_bundle.py`` (at build time) and by the server (at query time). No I/O here.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

ADS_ABS = "https://ui.adsabs.harvard.edu/abs/"

SECTIONS = ("data", "processing", "recipe", "results", "reproducibility")


def ads_url(bibcode: str) -> str:
    """ADS abstract page for a bibcode (the '&' in A&A bibcodes is URL-encoded)."""
    return ADS_ABS + bibcode.replace("&", "%26")


# ------------------------------------------------------------------------------------------------ licenses
_OPEN_LICENSE = re.compile(
    r"creativecommons\.org/(licenses/(by|by-sa)/[\d.]+|publicdomain/zero/[\d.]+|licenses/publicdomain)/?$", re.I)


def license_is_open(license_url: Optional[str]) -> bool:
    """True when quotes may be redistributed: CC BY, CC BY-SA, CC0 or public domain.

    NC and ND variants, the arXiv non-exclusive distribution license, and a missing license all return False.
    """
    return bool(license_url) and bool(_OPEN_LICENSE.search(license_url.strip()))


def license_label(license_url: Optional[str]) -> str:
    """Short human label for an arXiv license URL."""
    if not license_url:
        return "unknown"
    u = license_url.lower()
    if "nonexclusive-distrib" in u:
        return "arXiv non-exclusive"
    if "publicdomain/zero" in u:
        return "CC0"
    if "publicdomain" in u:
        return "public domain"
    m = re.search(r"licenses/([a-z-]+)/([\d.]+)", u)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    return license_url


# ------------------------------------------------------------------------------------------------ quotes
def strip_quotes(card: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Copy of a card with every verbatim evidence quote removed.

    An evidence item is any dict holding a ``quote`` key (normally ``{"loc": ..., "quote": ...}``). The quote text
    is dropped and ``quote_withheld: true`` is set, so the structured value and the location of the evidence
    (section, table or figure) stay. Returns (new_card, number_of_quotes_removed).
    """
    out = copy.deepcopy(card)
    n = 0

    def walk(x: Any) -> None:
        nonlocal n
        if isinstance(x, dict):
            if "quote" in x and isinstance(x.get("quote"), (str, type(None))):
                if x.get("quote"):
                    n += 1
                x.pop("quote", None)
                x["quote_withheld"] = True
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(out)
    return out, n


# Provenance fields shipped in a bundle; batch ids, custom ids, run names, token usage and cost stay out.
PROVENANCE_KEYS = ("model", "text_source", "extractor")
# In the public bundle no verbatim span longer than this (in characters) from a closed-license paper survives.
VERBATIM_MIN_CHARS = 40
WITHHELD_TEXT = "[verbatim text withheld, see paper]"


def quote_strings(card: Dict[str, Any]) -> set:
    """Every verbatim paper string in a source card: evidence quotes, repair-log old/new text and any
    checks.quotes_too_long strings."""
    out = set()

    def walk(x: Any) -> None:
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
    if isinstance(too_long, list):
        out |= {q for q in too_long if isinstance(q, str)}
    return out


def redact_verbatim(card: Dict[str, Any], verbatim: Iterable[str],
                    min_chars: int = VERBATIM_MIN_CHARS) -> Tuple[Dict[str, Any], int]:
    """Replace every span of ``verbatim`` longer than min_chars inside the card's body strings with WITHHELD_TEXT.

    Catches structured values that an extractor copied word for word from the paper (a science question, a list of
    codes). Quotes that are part of the paper title are skipped: the title is bibliographic metadata. Returns
    (card, number_of_replacements); the input is not modified.
    """
    title = ((card.get("identity") or {}).get("title") or "")
    qs = sorted({q for q in verbatim if isinstance(q, str) and len(q) > min_chars and q not in title},
                key=len, reverse=True)
    if not qs:
        return card, 0
    out = copy.deepcopy(card)
    n = 0

    def fix(s: str) -> str:
        nonlocal n
        for q in qs:
            if q in s:
                n += s.count(q)
                s = s.replace(q, WITHHELD_TEXT)
        return s

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str) and len(x) > min_chars:
            return fix(x)
        return x

    out["body"] = walk(out.get("body") or {})
    return out, n


def bundle_card(card: Dict[str, Any], *, public: bool, withhold_quotes: bool,
                verbatim: Iterable[str] = ()) -> Tuple[Dict[str, Any], int]:
    """The card as shipped in a bundle. Returns (card, number_of_quotes_removed).

    Both variants keep schema_version, identity, body and a minimal provenance (model, text_source, extractor);
    the internal repair log (which carries verbatim old/new quote text) is always dropped. The full variant keeps
    the checks block; the public variant keeps only its numeric and boolean fields. With withhold_quotes the body
    goes through strip_quotes, then redact_verbatim with the card's own quotes plus ``verbatim`` (quotes of the
    same paper from other extraction channels).
    """
    prov = card.get("provenance") or {}
    out: Dict[str, Any] = {"schema_version": card.get("schema_version"),
                           "identity": copy.deepcopy(card.get("identity") or {}),
                           "body": copy.deepcopy(card.get("body") or {}),
                           "provenance": {k: prov[k] for k in PROVENANCE_KEYS if k in prov}}
    checks = card.get("checks")
    if isinstance(checks, dict):
        if public:
            out["checks"] = {k: v for k, v in checks.items()
                             if isinstance(v, (bool, int, float)) and not isinstance(v, str)}
        else:
            out["checks"] = copy.deepcopy(checks)
    n = 0
    if withhold_quotes:
        out, n = strip_quotes(out)
        out, _ = redact_verbatim(out, quote_strings(card) | set(verbatim))
    return out, n


# ------------------------------------------------------------------------------------------------ facts
def fact_value(fact: Any) -> str:
    """The plain value of a card fact (``{"value": ..., "evidence": [...]}``) or a bare string."""
    if isinstance(fact, dict):
        v = fact.get("value")
        return "" if v is None else str(v)
    return "" if fact is None else str(fact)


def stated(fact: Any) -> bool:
    """False for empty values and the card schema's 'not stated' style placeholders."""
    v = fact_value(fact).strip().lower()
    return bool(v) and not v.startswith(("not stated", "none", "n/a", "not applicable", "unknown", "no ")) \
        and v not in ("no", "-")


def _evidence_md(fact: Any) -> str:
    if not isinstance(fact, dict):
        return ""
    parts = []
    for ev in fact.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        loc = ev.get("loc") or "?"
        q = ev.get("quote")
        parts.append(f'"{q}" ({loc})' if q else f"({loc}; quote withheld, see paper)")
    return ("  \n  > " + "; ".join(parts)) if parts else ""


def _fact_line(label: str, fact: Any) -> str:
    v = fact_value(fact)
    if not stated(fact):
        return f"- {label}: _not stated_"
    return f"- {label}: {v}{_evidence_md(fact)}"


def _human(key: str) -> str:
    return key.replace("_", " ")


# ------------------------------------------------------------------------------------------------ markdown
def render_section(card: Dict[str, Any], section: str) -> str:
    """Markdown for one section of a card: data, processing, recipe, results or reproducibility."""
    b = card.get("body") or {}
    lines: List[str] = []
    if section == "data":
        sci = b.get("science") or {}
        lines += ["## Science", "", f"- Fields: {', '.join(sci.get('fields') or []) or 'not tagged'}"]
        if sci.get("question"):
            lines.append(_fact_line("Question", sci.get("question")))
        for t in sci.get("targets") or []:
            lines.append(f"- Target **{t.get('name', '?')}**: "
                         f"{fact_value(t.get('redshift_or_distance')) or 'distance not stated'}")
        d = b.get("data") or {}
        lines += ["", "## Data provenance", ""]
        codes = [pc.get("code") for pc in d.get("alma_project_codes") or [] if isinstance(pc, dict)]
        lines.append(f"- ALMA projects: {', '.join(c for c in codes if c) or 'none given'}")
        for k, v in d.items():
            if k in ("alma_project_codes", "other_facilities", "external_data_from_literature", "lines_targeted"):
                continue
            if isinstance(v, (dict, str)):
                lines.append(_fact_line(_human(k), v))
        lt = d.get("lines_targeted") or []
        if lt:
            names = [fact_value(x.get("line") if isinstance(x, dict) and "line" in x else x) or str(x)
                     for x in lt]
            lines.append("- lines targeted: " + "; ".join(n for n in names if n))
        for f in d.get("other_facilities") or []:
            if isinstance(f, dict):
                lines.append(f"- other facility {f.get('name', '?')}: {fact_value(f.get('provided'))}")
    elif section == "processing":
        p = b.get("processing") or {}
        lines += ["## Processing", ""]
        for grp, facts in p.items():
            if isinstance(facts, dict):
                lines += [f"### {_human(grp).capitalize()}", ""]
                for k, v in facts.items():
                    if isinstance(v, (dict, str)) or v is None:
                        lines.append(_fact_line(_human(k), v))
                    elif isinstance(v, list):
                        lines.append(f"- {_human(k)}: " + ("; ".join(fact_value(x) or str(x) for x in v) or "none"))
                lines.append("")
            elif isinstance(facts, list) and facts:
                lines += [f"### {_human(grp).capitalize()}", ""]
                lines += [f"- {fact_value(x) or x}" for x in facts]
                lines.append("")
    elif section == "recipe":
        lines += ["## Analysis recipe", ""]
        for s in sorted(b.get("analysis_steps") or [], key=lambda s: (s.get("order") is None, s.get("order") or 0)):
            sw = ", ".join(s.get("software") or []) or "software not stated"
            lines.append(f"{s.get('order', '?')}. **{s.get('goal', '')}** ({sw})")
            lines.append(f"   - method: {fact_value(s.get('method')) or 'not stated'}{_evidence_md(s.get('method'))}")
            for kp in s.get("key_parameters") or []:
                lines.append(f"   - parameter: {fact_value(kp)}{_evidence_md(kp)}")
            for a in s.get("assumptions") or []:
                lines.append(f"   - assumption: {fact_value(a)}")
            if stated(s.get("fitting")):
                lines.append(f"   - fitting: {fact_value(s.get('fitting'))}")
            if stated(s.get("validation")):
                lines.append(f"   - validation: {fact_value(s.get('validation'))}")
        sw = b.get("software") or []
        if sw:
            lines += ["", "## Software", ""]
            for s in sw:
                ver = f" {s.get('version')}" if s.get("version") and stated(s.get("version")) else ""
                lines.append(f"- {s.get('name', '?')}{ver}: {fact_value(s.get('used_for'))}")
    elif section == "results":
        lines += ["## Numeric results", ""]
        for r in b.get("results_numeric") or []:
            unc = f" +/- {r.get('uncertainty')}" if r.get("uncertainty") not in (None, "", "not stated") else ""
            lines.append(f"- {r.get('quantity', '?')}: {r.get('value', '?')}{unc} {r.get('units') or ''}".rstrip()
                         + _evidence_md(r))
        fq = b.get("findings_qualitative") or []
        if fq:
            lines += ["", "## Qualitative findings", ""]
            lines += [f"- {fact_value(f) or f}" for f in fq]
        cl = b.get("claims") or []
        if cl:
            lines += ["", "## Claims", ""]
            for c in cl:
                if isinstance(c, dict):
                    lines.append(f"- {c.get('finding') or c.get('result') or ''} (support: {c.get('support', '?')})")
    elif section == "reproducibility":
        r = b.get("reproducibility") or {}
        lines += ["## Reproducibility", ""]
        for k, v in r.items():
            if isinstance(v, list):
                lines.append(f"- {_human(k)}: " + ("; ".join(fact_value(x) or str(x) for x in v) or "none"))
            else:
                lines.append(_fact_line(_human(k), v) if isinstance(v, dict) else f"- {_human(k)}: {v}")
        links = b.get("links") or {}
        if links:
            lines += ["", "## Links", ""]
            if isinstance(links, dict):
                for k, v in links.items():
                    lines.append(f"- {_human(k)}: {v if not isinstance(v, (dict, list)) else fact_value(v) or v}")
            else:
                lines += [f"- {x}" for x in links]
    else:
        raise ValueError(f"unknown section {section!r}; use one of {', '.join(SECTIONS)}")
    return "\n".join(lines).rstrip() + "\n"


def render_card(card: Dict[str, Any], *, license_url: Optional[str] = None, model: Optional[str] = None) -> str:
    """Full markdown for a card: header plus the five sections."""
    i = card.get("identity") or {}
    b = card.get("body") or {}
    bib = i.get("bibcode", "?")
    head = [f"# {i.get('title') or bib}", "",
            f"`{bib}` | {i.get('journal') or '?'} {i.get('year') or ''} | first author: {i.get('first_author') or '?'}"
            f" | arXiv: {i.get('arxiv_id') or '-'} | DOI: {i.get('doi') or '-'}",
            f"ADS: {ads_url(bib)} | license: {license_label(license_url)} | card model: {model or '?'}"
            f" | depth {b.get('depth', '?')} | type {b.get('paper_type', '?')}", ""]
    return "\n".join(head) + "\n" + "\n".join(render_section(card, s) for s in SECTIONS)


def step_text(step: Dict[str, Any]) -> str:
    """One searchable string for an analysis step (goal, method, fitting, software, parameters, assumptions)."""
    parts = [step.get("goal") or "", fact_value(step.get("method")), fact_value(step.get("fitting")),
             fact_value(step.get("validation")), " ".join(step.get("software") or [])]
    parts += [fact_value(p) for p in step.get("key_parameters") or []]
    parts += [fact_value(a) for a in step.get("assumptions") or []]
    return " ".join(p for p in parts if p)


# ------------------------------------------------------------------------------------------------ settings
ROBUST = re.compile(r"robust\w*\s*(?:=|of|parameter of|value of|parameter|value)?\s*\(?\s*([+-]?\d+(?:\.\d+)?)", re.I)
_NUM = r"(\d+(?:\.\d+)?)"
CELL = re.compile(r"(?<![/\d.])" + _NUM + r"(?!\s*/)(?!\s*(?:of|x\s*(?:the\s*)?beam|times))\s*(mas|milliarcsec\w*|arcsec\w*|asec|''|\"|″|′′|\$''\$)?", re.I)
SOLINT = re.compile(r"\b(inf|int)\b|" + _NUM + r"\s*(?:-|\s)?\s*(s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hours?)\b",
                    re.I)
FITORDER = re.compile(r"fitorder\s*=?\s*(\d)|(?:polynomial\s+)?order\s*(?:=|of)?\s*(\d)\b|"
                      r"\b(zeroth|zero|first|second|third|0th|1st|2nd|3rd)[- ]order\b|\b(linear|constant)\b", re.I)
_ORDER_WORDS = {"zeroth": 0, "zero": 0, "0th": 0, "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
                "linear": 1, "constant": 0}

# setting name -> (card processing field it comes from, description with units)
SETTINGS: Dict[str, Tuple[str, str]] = {
    "robust": ("imaging.weighting_robust", "Briggs robust parameter (dimensionless, -2 uniform-like to +2 natural-like)"),
    "weighting": ("imaging.weighting_robust", "Visibility weighting scheme: natural, uniform or briggs"),
    "cell_size": ("imaging.cell_size", "Image pixel (cell) size in arcsec"),
    "uv_taper": ("imaging.uv_taper", "Outer uv taper, as stated (arcsec or klambda)"),
    "deconvolver": ("imaging.deconvolver", "tclean deconvolver: hogbom, clark, multiscale, mtmfs"),
    "selfcal_solint": ("calibration.self_calibration", "Self-calibration solution interval in seconds ('inf' = scan, 'int' = integration)"),
    "selfcal_mode": ("calibration.self_calibration", "Self-calibration mode: phase-only or phase+amplitude"),
    "contsub_order": ("imaging.continuum_subtraction", "Continuum subtraction polynomial order (fitorder)"),
    "contsub_domain": ("imaging.continuum_subtraction", "Continuum subtraction domain: uv plane or image plane"),
    "band": ("data.bands", "ALMA receiver band number"),
    "array": ("data.arrays", "Arrays used: 12m, 7m/ACA, TP"),
}

ALIASES = {
    "briggs": "robust", "robustness": "robust", "weighting_robust": "robust",
    "cell": "cell_size", "cellsize": "cell_size", "pixel": "cell_size", "pixel_size": "cell_size",
    "solint": "selfcal_solint", "self_cal_solint": "selfcal_solint", "self_calibration_solint": "selfcal_solint",
    "selfcal": "selfcal_solint", "self_calibration": "selfcal_solint",
    "fitorder": "contsub_order", "continuum_subtraction_order": "contsub_order", "contsub_fitorder": "contsub_order",
    "continuum_order": "contsub_order", "continuum_subtraction": "contsub_order",
    "taper": "uv_taper", "uvtaper": "uv_taper", "bands": "band", "arrays": "array",
}


def normalize_setting(name: str) -> str:
    """Map free text like 'self-cal solint' or 'Cell size' to a canonical setting name."""
    k = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    k = k.replace("self_cal", "selfcal").replace("selfcalibration", "selfcal")
    if k in SETTINGS:
        return k
    if k in ALIASES:
        return ALIASES[k]
    k2 = k.replace("selfcal", "self_cal")
    if k2 in ALIASES:
        return ALIASES[k2]
    if "solint" in k or ("selfcal" in k and "interval" in k):
        return "selfcal_solint"
    if "order" in k and ("cont" in k or "fit" in k):
        return "contsub_order"
    if "cell" in k or "pixel" in k:
        return "cell_size"
    if "robust" in k or "briggs" in k:
        return "robust"
    return k


def _fmt(x: float) -> str:
    return f"{x:g}"


def extract_settings(body: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(setting, value) pairs parsed from one card body. One paper can give several values for one setting.

    Values are normalized: robust as a number, cell size in arcsec, solint in seconds or 'inf'/'int',
    continuum order as an integer. Unparseable statements give nothing (they still count as stated).
    """
    proc = body.get("processing") or {}
    cal, img = proc.get("calibration") or {}, proc.get("imaging") or {}
    data = body.get("data") or {}
    out: List[Tuple[str, str]] = []

    wr = fact_value(img.get("weighting_robust"))
    if stated(img.get("weighting_robust")):
        for m in ROBUST.finditer(wr):
            v = float(m.group(1))
            if -2.0 <= v <= 2.0:
                out.append(("robust", _fmt(v)))
        low = wr.lower()
        for w in ("natural", "uniform", "briggs"):
            if w in low or (w == "briggs" and "robust" in low):
                out.append(("weighting", w))

    cs = img.get("cell_size")
    if stated(cs):
        for m in CELL.finditer(fact_value(cs)):
            v, unit = float(m.group(1)), (m.group(2) or "").lower()
            if unit.startswith(("mas", "milli")):
                v /= 1000.0
            elif not unit and v >= 1:  # bare numbers >= 1 are usually pixel counts or beam fractions, not arcsec
                continue
            if 0 < v <= 5:
                out.append(("cell_size", _fmt(round(v, 5))))
                break

    tp = img.get("uv_taper")
    if stated(tp):
        out.append(("uv_taper", fact_value(tp).strip()[:60]))

    dc = fact_value(img.get("deconvolver")).lower()
    if stated(img.get("deconvolver")):
        for d in ("mtmfs", "multiscale", "hogbom", "clark", "asp"):
            if d in dc.replace("multi-scale", "multiscale").replace("multi scale", "multiscale"):
                out.append(("deconvolver", d))

    sc = cal.get("self_calibration")
    if stated(sc):
        txt = fact_value(sc)
        seen = set()
        for m in SOLINT.finditer(txt):
            if m.group(1):
                v = m.group(1).lower()
            else:
                num, unit = float(m.group(2)), m.group(3).lower()
                sec = num * (60 if unit.startswith("m") else 3600 if unit.startswith("h") else 1)
                if sec <= 0 or sec > 36000:
                    continue
                v = _fmt(sec)
            if v not in seen:
                seen.add(v)
                out.append(("selfcal_solint", v))
        low = txt.lower()
        if re.search(r"amplitude|\bamp\b|\ba&p\b|\bap\b|phase\s*(and|\+|&)\s*amp", low):
            out.append(("selfcal_mode", "phase+amplitude"))
        elif "phase" in low:
            out.append(("selfcal_mode", "phase-only"))

    cso = img.get("continuum_subtraction")
    if stated(cso):
        txt = fact_value(cso)
        for m in FITORDER.finditer(txt):
            g = m.group(1) or m.group(2)
            v = int(g) if g is not None else _ORDER_WORDS.get((m.group(3) or m.group(4) or "").lower())
            if v is not None and 0 <= v <= 5:
                out.append(("contsub_order", str(v)))
                break
        low = txt.lower()
        if re.search(r"uvcontsub|uv[- ]?(plane|domain|space)|visibilit", low):
            out.append(("contsub_domain", "uv plane"))
        elif re.search(r"imcontsub|image[- ]?(plane|domain)|statcont|cube", low):
            out.append(("contsub_domain", "image plane"))

    for bd in sorted(set(re.findall(r"band\s*(\d{1,2})\b", fact_value(data.get("bands")), re.I)), key=int):
        if 1 <= int(bd) <= 10:
            out.append(("band", bd))
    arr = fact_value(data.get("arrays")).lower()
    for name, rx in (("12m", r"12[- ]?m"), ("7m/ACA", r"7[- ]?m|\baca\b|morita"), ("TP", r"total power|\btp\b")):
        if re.search(rx, arr):
            out.append(("array", name))
    return out


# ------------------------------------------------------------------------------------------------ FTS
def fts_query(text: str, mode: str = "AND") -> str:
    """Turn free text into a safe FTS5 query: each whitespace token becomes a quoted phrase of its word parts
    ('self-calibration' -> "self calibration"), joined with AND or OR. Empty text gives ''."""
    phrases = []
    for tok in text.split():
        words = re.findall(r"[A-Za-z0-9]+", tok)
        if words:
            phrases.append('"' + " ".join(words) + '"')
    return f" {mode} ".join(phrases)


def split_list(s: Optional[str]) -> List[str]:
    return [x for x in (s or "").split(";") if x]


def uniq(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
