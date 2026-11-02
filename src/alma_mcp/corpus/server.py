"""alma-corpus MCP server: what ALMA papers did, with which settings, and where they wrote it.

Run:  alma-corpus-mcp                 (stdio, for Claude Desktop / Claude Code / Codex / Cursor)
      alma-corpus-mcp --http 8765     (streamable HTTP, so a team can share one copy)
      alma-corpus-mcp download [--url URL] [--sha256 HEX]
      alma-corpus-mcp info
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import Prompt

from . import data
from .store import CorpusStore

INSTRUCTIONS = (
    "Knowledge server over methods cards from ~3,500 refereed ALMA papers (2012 onward): data used, calibration "
    "and imaging settings, ordered analysis steps, software, numeric results, plus 38 reviewed technique skills and "
    "corpus-wide parameter statistics. Use it to answer 'how do ALMA papers usually do X', 'what value of Y do they "
    "use', and 'which papers did Z'. Every result carries bibcodes and ADS links; cite them and tell the user to "
    "check numbers on ADS, since cards were extracted by an LLM. Held-out benchmark papers are not included."
)

_store: Optional[CorpusStore] = None


def get_store() -> CorpusStore:
    global _store
    if _store is None:
        _store = CorpusStore(data.find_bundle())
    return _store


def _call(fn_name: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    try:
        store = get_store()
    except data.BundleMissing as e:
        return {"error": str(e)}
    return getattr(store, fn_name)(*args, **kwargs)


def build_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("alma-corpus", instructions=INSTRUCTIONS, host=host, port=port)

    @mcp.tool()
    def search_papers(query: str, technique: Optional[str] = None, field: Optional[str] = None,
                      year_min: Optional[int] = None, year_max: Optional[int] = None, limit: int = 20) -> Dict[str, Any]:
        """Full-text search over ALMA paper titles, abstracts and full methods cards (BM25 ranking).

        Inputs: query (free text, e.g. "self-calibration solint", "CO(2-1) Keplerian disk"; all terms are tried
        with AND first, then OR), technique (a skill slug from list_techniques, e.g. "self-calibration"),
        field (science field tag, e.g. "protoplanetary_disks", "high_z_galaxies", "nearby_galaxies",
        "star_formation", "astrochemistry", "evolved_stars"), year_min / year_max (publication year, inclusive),
        limit (1 to 100, default 20).
        Returns bibcode, title, year, first author, ADS link, fields, techniques and the matching snippet.
        Limits: ranking is lexical, not semantic; an empty query with filters lists the newest papers.
        """
        return _call("search_papers", query, technique=technique, field=field, year_min=year_min,
                     year_max=year_max, limit=limit)

    @mcp.tool()
    def get_methods_card(bibcode: str, section: Optional[str] = None) -> Dict[str, Any]:
        """The methods card of one paper as markdown, with the location (and normally a short verbatim quote) of
        the evidence for each value.

        Inputs: bibcode (ADS bibcode, e.g. "2018ApJ...869L..41A"), section (optional): "data" (science goal,
        targets, bands, arrays, resolution, sensitivity, project codes), "processing" (calibration and imaging
        settings), "recipe" (ordered analysis steps and software), "results" (numeric results with units,
        findings, claims) or "reproducibility". No section returns the whole card.
        Limits: values were extracted by an LLM from the paper text and checked against the quotes; "not stated"
        means the paper does not give it. Quotes are withheld in the public bundle for non-CC papers.
        """
        return _call("get_methods_card", bibcode, section=section)

    @mcp.tool()
    def get_recipe(bibcode: str) -> Dict[str, Any]:
        """Ordered analysis steps of one paper: goal, method, software, key parameters (as stated, with units),
        assumptions, fitting and validation, plus the stated calibration/imaging settings and software versions.

        Input: bibcode. Use it to reproduce or adapt a paper's workflow. Steps are in the order the paper gives.
        """
        return _call("get_recipe", bibcode)

    @mcp.tool()
    def find_papers_using(software: Optional[str] = None, technique: Optional[str] = None,
                          parameter_regex: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Papers that used a software package, a technique, and/or a parameter pattern.

        Inputs (at least one): software (substring of the normalized package name, e.g. "uvmultifit", "radmc",
        "3dbarolo", "casa"), technique (skill slug, e.g. "uv-plane-fitting"), parameter_regex (Python regex,
        case-insensitive, run over step key parameters, step methods and stated processing values, e.g.
        r"robust\\s*=?\\s*-?0\\.5", r"solint", r"nterms\\s*=\\s*2"), limit (1 to 500, default 50).
        Returns papers newest first with ADS links and the matching lines.
        """
        return _call("find_papers_using", software=software, technique=technique, parameter_regex=parameter_regex,
                     limit=limit)

    @mcp.tool()
    def parameter_census(setting: str, field: Optional[str] = None) -> Dict[str, Any]:
        """What values ALMA papers used for a processing setting, how many papers used each, example bibcodes,
        and how often the setting is not stated at all.

        Parsed settings (normalized values): "robust" (Briggs robust, dimensionless), "weighting"
        (natural/uniform/briggs), "cell_size" (arcsec), "uv_taper" (as stated), "deconvolver"
        (hogbom/clark/multiscale/mtmfs), "selfcal_solint" (seconds, or "inf" = per scan, "int" = per integration),
        "selfcal_mode" (phase-only / phase+amplitude), "contsub_order" (continuum fit polynomial order),
        "contsub_domain" (uv plane / image plane), "band" (ALMA band number), "array" (12m, 7m/ACA, TP).
        Free text such as "self-cal solint" or "cell size" is mapped to these names. Any other card processing field
        (e.g. "imaging.masking", "calibration.flux_calibrator", "primary_beam_correction") returns its raw values
        grouped by wording.
        field: optional science field filter (e.g. "protoplanetary_disks").
        Limits: counts are papers, not images; a paper that states a setting in words our parser does not
        recognise counts as stated but has no parsed value.
        """
        return _call("parameter_census", setting, field=field)

    @mcp.tool()
    def software_usage(name: Optional[str] = None, field: Optional[str] = None, by_year: bool = False,
                       technique: Optional[str] = None) -> Dict[str, Any]:
        """How many ALMA papers name a software package (or the top 40 packages when name is empty), optionally
        within one science field or technique and split by publication year.

        Inputs: name (substring of the normalized name, e.g. "casa", "emcee", "galario"), field (science field
        tag), technique (technique slug from list_techniques, e.g. "keplerian-disk-kinematics": answers "which
        software do papers use for X" when name is empty), by_year (true adds counts and fractions per year).
        Returns paper counts, fractions, stated versions and example bibcodes with ADS links. With a technique or
        field and no name it also returns distinctive_software, ranked by lift over the whole corpus: general
        packages (casa, astropy, numpy) top every subset, so use distinctive_software to name the packages that are
        specific to that technique.
        """
        return _call("software_usage", name=name, field=field, by_year=by_year, technique=technique)

    @mcp.tool()
    def list_techniques() -> Dict[str, Any]:
        """The reviewed technique skills (38 in the current bundle: self-calibration, array combination, continuum subtraction, imaging
        choices, dust mass, SED fitting, Keplerian disk kinematics, uv-plane fitting, ...) with slugs, paper counts,
        prompt names (use-<slug>) and resource URIs (alma://skills/<slug>)."""
        return _call("list_techniques")

    @mcp.tool()
    def get_skill(slug: str, include_parameters: bool = False) -> Dict[str, Any]:
        """A technique skill as markdown: when to use it, the recipe the corpus converges on, parameter choices,
        pitfalls and example bibcodes. include_parameters=true adds the per-paper parameter table
        (references/parameters.md) where the skill has one. Get slugs from list_techniques."""
        return _call("get_skill", slug, include_parameters=include_parameters)

    @mcp.tool()
    def compare_papers(bibcodes: List[str]) -> Dict[str, Any]:
        """Side-by-side table (markdown plus structured) of 2 to 8 papers: year, fields, techniques, project codes,
        bands, arrays, resolution, sensitivity, calibration and imaging settings, analysis steps, software and the
        first numeric results. Unknown bibcodes are listed under "missing"."""
        return _call("compare_papers", bibcodes)

    @mcp.tool()
    def corpus_info() -> Dict[str, Any]:
        """Bundle version, build date, table counts, license breakdown, data-licensing notes, held-out policy and
        how to cite."""
        return _call("corpus_info")

    # ------------------------------------------------------------------------------------------ resources
    @mcp.resource("alma://cards/{bibcode}", name="methods-card", mime_type="text/markdown",
                  description="Methods card of one ALMA paper (markdown). Use the ADS bibcode, URL-encoded if needed.")
    def card_resource(bibcode: str) -> str:
        bib = urllib.parse.unquote(bibcode)
        try:
            md = get_store().get_card_markdown(bib)
        except data.BundleMissing as e:
            return str(e)
        return md if md is not None else f"No card for {bib!r} (not in the corpus, or a held-out paper)."

    try:
        skills = get_store().skill_slugs()
    except data.BundleMissing:
        print(data.missing_message(), file=sys.stderr)
        skills = []
    for sk in skills:
        _register_skill(mcp, sk["slug"], sk["title"], sk["description"] or sk["title"])
    return mcp


def _register_skill(mcp: FastMCP, slug: str, title: str, description: str) -> None:
    def skill_text() -> str:
        return get_store().get_skill(slug)["skill_md"]

    mcp.resource(f"alma://skills/{slug}", name=f"skill-{slug}", title=title, description=description[:500],
                 mime_type="text/markdown")(skill_text)

    def use_skill(task: str = "") -> str:
        s = get_store().get_skill(slug, include_parameters=False)
        ask = f"\n\nTask: {task}" if task else ""
        return (f"Use the following ALMA technique skill ({title}), built from the ALMA paper corpus, to plan or "
                f"check the work. Cite the bibcodes it gives (ADS: https://ui.adsabs.harvard.edu/abs/<bibcode>) and "
                f"use the alma-corpus tools (get_methods_card, parameter_census) to check specific values.{ask}\n\n"
                f"{s['skill_md']}")

    mcp.add_prompt(Prompt.from_function(use_skill, name=f"use-{slug}", title=f"Use skill: {title}",
                                        description=description[:500]))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="alma-corpus-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP on PORT instead of stdio")
    ap.add_argument("--host", default="127.0.0.1", help="bind address for --http (default 127.0.0.1)")
    sub = ap.add_subparsers(dest="cmd")
    dl = sub.add_parser("download", help="download and verify the data bundle")
    dl.add_argument("--url", help=f"release tarball URL (default {data.DEFAULT_URL})")
    dl.add_argument("--sha256", help="expected sha256 (default: read <url>.sha256)")
    dl.add_argument("--dest", help="target directory (default $ALMA_MCP_DATA or ~/.alma-mcp/data)")
    sub.add_parser("info", help="print bundle info and exit")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        from pathlib import Path
        try:
            data.download(a.url, Path(a.dest) if a.dest else None, a.sha256)
        except Exception as e:
            print(f"download failed: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "info":
        try:
            print(json.dumps(get_store().corpus_info(), indent=1))
        except data.BundleMissing as e:
            print(e, file=sys.stderr)
            return 1
        return 0
    server = build_server(host=a.host, port=a.http or 8000)
    if a.http:
        server.run(transport="streamable-http")
    else:
        server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
