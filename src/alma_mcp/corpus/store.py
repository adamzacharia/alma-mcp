"""Read-only access to the alma-corpus bundle (SQLite + FTS5). Every public method returns a JSON-safe dict."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import cards as C
from .data import DB_NAME, read_manifest

LABEL = {"papers_fts": "title/abstract", "steps_fts": "analysis step", "cards_fts": "card"}

CITE = ("Cite the original papers (bibcodes and ADS links are in every result). For the corpus itself, cite "
        "alma-mcp (https://github.com/adamzacharia/alma-mcp) and give the bundle version from corpus_info().")


def _paper_ref(row: sqlite3.Row) -> Dict[str, Any]:
    return {"bibcode": row["bibcode"], "title": row["title"], "year": row["year"],
            "first_author": row["first_author"], "ads_url": C.ads_url(row["bibcode"])}


class CorpusStore:
    """Thin query layer over alma_methods.sqlite. Opened read-only; safe to share across threads per call."""

    def __init__(self, bundle_dir: Path):
        self.dir = Path(bundle_dir)
        self.path = self.dir / DB_NAME
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.manifest = read_manifest(self.dir)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def q(self, sql: str, args: Sequence[Any] = ()) -> List[sqlite3.Row]:
        con = self._con()
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    # ------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _filters(technique: Optional[str], field: Optional[str], year_min: Optional[int], year_max: Optional[int],
                 alias: str = "p"):
        where, args = [], []
        if technique:
            where.append(f"(';' || {alias}.techniques || ';') LIKE ?")
            args.append(f"%;{technique.strip().lower()};%")
        if field:
            where.append(f"(';' || {alias}.fields || ';') LIKE ?")
            args.append(f"%;{field.strip().lower()};%")
        if year_min is not None:
            where.append(f"{alias}.year >= ?")
            args.append(int(year_min))
        if year_max is not None:
            where.append(f"{alias}.year <= ?")
            args.append(int(year_max))
        return where, args

    def paper(self, bibcode: str) -> Optional[sqlite3.Row]:
        rows = self.q("SELECT * FROM papers WHERE bibcode = ?", (bibcode.strip(),))
        return rows[0] if rows else None

    def _not_found(self, bibcode: str) -> Dict[str, Any]:
        like = self.q("SELECT bibcode FROM papers WHERE bibcode LIKE ? LIMIT 5", (f"%{bibcode.strip()[:12]}%",))
        return {"error": f"bibcode {bibcode!r} is not in the corpus (held-out benchmark papers are excluded).",
                "did_you_mean": [r["bibcode"] for r in like]}

    # ------------------------------------------------------------------------------------------ tools
    def search_papers(self, query: str, technique: Optional[str] = None, field: Optional[str] = None,
                      year_min: Optional[int] = None, year_max: Optional[int] = None, limit: int = 20) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        where, fargs = self._filters(technique, field, year_min, year_max)
        fsql = (" AND " + " AND ".join(where)) if where else ""
        hits: Dict[str, Dict[str, Any]] = {}
        mode_used = None
        for mode in ("AND", "OR"):
            fq = C.fts_query(query or "", mode)
            if not fq:
                break
            for table, weight in (("papers_fts", 2.0), ("steps_fts", 1.0), ("cards_fts", 1.0)):
                sql = (f"SELECT p.*, bm25({table}) AS score, snippet({table}, -1, "
                       f"'[', ']', ' ... ', 18) AS snip FROM {table} JOIN papers p ON p.bibcode = {table}.bibcode "
                       f"WHERE {table} MATCH ?{fsql} ORDER BY score LIMIT ?")
                try:
                    rows = self.q(sql, [fq] + fargs + [limit * 3])
                except sqlite3.OperationalError as e:
                    return {"error": f"search syntax problem: {e}", "query": query}
                for r in rows:
                    h = hits.setdefault(r["bibcode"], {**_paper_ref(r), "score": 0.0, "fields": C.split_list(r["fields"]),
                                                       "techniques": C.split_list(r["techniques"]), "match": []})
                    h["score"] += -float(r["score"]) * weight
                    h["match"].append(f"{LABEL[table]}: {r['snip']}")
            if hits:
                mode_used = mode
                break
        if not query or not query.strip():
            rows = self.q(f"SELECT p.* FROM papers p WHERE 1=1{fsql} ORDER BY p.year DESC LIMIT ?", fargs + [limit])
            hits = {r["bibcode"]: {**_paper_ref(r), "score": None, "fields": C.split_list(r["fields"]),
                                   "techniques": C.split_list(r["techniques"]), "match": []} for r in rows}
            mode_used = "filters only"
        ranked = sorted(hits.values(), key=lambda h: -(h["score"] or 0))[:limit]
        for h in ranked:
            if h["score"] is not None:
                h["score"] = round(h["score"], 3)
        return {"query": query, "match_mode": mode_used, "n": len(ranked), "results": ranked,
                "note": "Ranked by BM25 over titles, abstracts and full methods cards. AND of all terms is tried "
                        "first, then OR. Check details with get_methods_card(bibcode)."}

    def get_methods_card(self, bibcode: str, section: Optional[str] = None) -> Dict[str, Any]:
        rows = self.q("SELECT c.*, p.license, p.title FROM cards c JOIN papers p USING (bibcode) WHERE bibcode = ?",
                      (bibcode.strip(),))
        if not rows:
            return self._not_found(bibcode)
        r = rows[0]
        if section:
            sec = section.strip().lower()
            if sec not in C.SECTIONS:
                return {"error": f"unknown section {section!r}", "sections": list(C.SECTIONS)}
            md = C.render_section(json.loads(r["card_json"]), sec)
        else:
            md = r["card_md"]
        return {"bibcode": r["bibcode"], "title": r["title"], "ads_url": C.ads_url(r["bibcode"]),
                "section": section or "all", "card_model": r["model"], "license": C.license_label(r["license"]),
                "quotes_withheld": bool(r["quotes_withheld"]), "markdown": md,
                "note": "Values were extracted from the paper by an LLM and each carries the location of its evidence. "
                        "Quotes are verbatim from the paper unless withheld. Verify on ADS before relying on a number."}

    def get_card_markdown(self, bibcode: str) -> Optional[str]:
        rows = self.q("SELECT card_md FROM cards WHERE bibcode = ?", (bibcode.strip(),))
        return rows[0]["card_md"] if rows else None

    def get_recipe(self, bibcode: str) -> Dict[str, Any]:
        p = self.paper(bibcode)
        if p is None:
            return self._not_found(bibcode)
        steps = []
        for s in self.q("SELECT * FROM steps WHERE bibcode = ? ORDER BY ord", (p["bibcode"],)):
            steps.append({"order": s["ord"], "goal": s["goal"], "method": s["method"],
                          "software": C.split_list(s["software"]), "key_parameters": json.loads(s["key_parameters"] or "[]"),
                          "assumptions": json.loads(s["assumptions"] or "[]"),
                          "fitting": s["fitting"] if C.stated(s["fitting"]) else None,
                          "validation": s["validation"] if C.stated(s["validation"]) else None,
                          "techniques": C.split_list(s["techniques"])})
        proc = {f"{r['grp']}.{r['key']}": r["value"] for r in
                self.q("SELECT grp, key, value FROM processing WHERE bibcode = ? AND stated = 1 AND grp != 'data'",
                       (p["bibcode"],))}
        sw = [{"name": r["name"], "version": r["version"], "used_for": r["used_for"]}
              for r in self.q("SELECT name, version, used_for FROM software WHERE bibcode = ?", (p["bibcode"],))]
        return {**_paper_ref(p), "n_steps": len(steps), "steps": steps, "processing_stated": proc, "software": sw,
                "note": "Steps are in the order the paper reports them. Settings missing here were not stated in the paper."}

    def find_papers_using(self, software: Optional[str] = None, technique: Optional[str] = None,
                          parameter_regex: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        if not (software or technique or parameter_regex):
            return {"error": "give at least one of software, technique, parameter_regex"}
        limit = max(1, min(int(limit), 500))
        rx = None
        if parameter_regex:
            try:
                rx = re.compile(parameter_regex, re.I)
            except re.error as e:
                return {"error": f"bad regex: {e}"}
        where, args = self._filters(technique, None, None, None)
        if software:
            s = software.strip().lower()
            where.append("(p.bibcode IN (SELECT bibcode FROM software WHERE name_norm LIKE ?) OR "
                         "p.bibcode IN (SELECT bibcode FROM steps WHERE lower(software) LIKE ?))")
            args += [f"%{s}%", f"%{s}%"]
        sql = "SELECT p.* FROM papers p" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY p.year DESC, p.bibcode"
        papers = self.q(sql, args)
        total_before_regex = len(papers)
        out = []
        for p in papers:
            matches: List[str] = []
            if rx is not None:
                for s in self.q("SELECT ord, goal, key_parameters, method FROM steps WHERE bibcode = ?", (p["bibcode"],)):
                    for kp in json.loads(s["key_parameters"] or "[]") + [s["method"] or ""]:
                        if rx.search(kp):
                            matches.append(f"step {s['ord']} ({s['goal']}): {kp}")
                for r in self.q("SELECT grp, key, value FROM processing WHERE bibcode = ? AND stated = 1", (p["bibcode"],)):
                    if rx.search(r["value"] or ""):
                        matches.append(f"{r['grp']}.{r['key']}: {r['value']}")
                if not matches:
                    continue
            ref = {**_paper_ref(p), "fields": C.split_list(p["fields"]), "techniques": C.split_list(p["techniques"])}
            if software:
                ref["software_used"] = [f"{r['name']}{(' ' + r['version']) if r['version'] else ''}: {r['used_for']}"
                                        for r in self.q("SELECT name, version, used_for FROM software WHERE bibcode = ? "
                                                        "AND name_norm LIKE ?", (p["bibcode"], f"%{software.strip().lower()}%"))][:5]
            if matches:
                ref["matches"] = C.uniq(matches)[:8]
            out.append(ref)
        return {"software": software, "technique": technique, "parameter_regex": parameter_regex,
                "n_total": len(out), "n_returned": min(len(out), limit), "n_before_regex": total_before_regex,
                "results": out[:limit],
                "note": "Software names match by substring on the normalized name (e.g. 'casa' also hits 'casa 5.6'). "
                        "The regex runs over step key parameters, step methods and stated processing values."}

    def parameter_census(self, setting: str, field: Optional[str] = None) -> Dict[str, Any]:
        canon = C.normalize_setting(setting)
        fwhere, fargs = self._filters(None, field, None, None)
        fsql = (" AND " + " AND ".join(fwhere)) if fwhere else ""
        n_papers = self.q(f"SELECT count(*) AS n FROM papers p WHERE 1=1{fsql}", fargs)[0]["n"]
        res: Dict[str, Any] = {"setting": canon, "asked": setting, "field": field, "n_papers_in_scope": n_papers}
        if canon in C.SETTINGS:
            src, desc = C.SETTINGS[canon]
            grp, key = src.split(".", 1)
            res["description"] = desc
            res["source_field"] = src
            rows = self.q(f"SELECT s.value, s.bibcode, p.year FROM paper_settings s JOIN papers p USING (bibcode) "
                          f"WHERE s.setting = ?{fsql} ORDER BY p.year DESC", [canon] + fargs)
            by_val: Dict[str, List[str]] = defaultdict(list)
            for r in rows:
                by_val[r["value"]].append(r["bibcode"])
            n_with_value = len({r["bibcode"] for r in rows})
            stated = self.q(f"SELECT count(DISTINCT pr.bibcode) AS n FROM processing pr JOIN papers p USING (bibcode) "
                            f"WHERE pr.grp = ? AND pr.key = ? AND pr.stated = 1{fsql}", [grp, key] + fargs)[0]["n"]

            def sort_key(kv):
                try:
                    return (0, -len(kv[1]), float(kv[0]))
                except ValueError:
                    return (0, -len(kv[1]), 0.0)
            values = [{"value": v, "n_papers": len(set(b)), "fraction_of_papers_with_value": round(len(set(b)) / n_with_value, 3)
                       if n_with_value else None, "examples": [{"bibcode": x, "ads_url": C.ads_url(x)} for x in C.uniq(b)[:5]]}
                      for v, b in sorted(by_val.items(), key=sort_key)]
            res.update({"values": values[:40], "n_distinct_values": len(values),
                        "n_papers_with_parsed_value": n_with_value, "n_papers_stating_field": stated,
                        "n_papers_not_stating_field": n_papers - stated,
                        "fraction_not_stated": round(1 - stated / n_papers, 3) if n_papers else None})
            nums = []
            for v, b in by_val.items():
                try:
                    nums += [float(v)] * len(set(b))
                except ValueError:
                    pass
            if nums and len(nums) == sum(len(set(b)) for b in by_val.values()):
                nums.sort()
                res["numeric_summary"] = {"median": nums[len(nums) // 2], "min": nums[0], "max": nums[-1], "n": len(nums)}
            ur = self.q("SELECT fraction_unstated FROM unstated_rates WHERE setting = ?", (src,))
            if ur:
                res["corpus_fraction_unstated_published_stats"] = ur[0]["fraction_unstated"]
            res["note"] = ("Counts are papers (one paper can report several values, e.g. a solint ladder). "
                           "'Stating field' means the paper says anything about it; a parsed value needs a number or "
                           "keyword our parser recognises, so n_papers_with_parsed_value <= n_papers_stating_field.")
            return res
        # any other processing field, e.g. 'imaging.masking' or 'flux_calibrator': raw value counts
        raw = re.sub(r"[^a-z0-9.]+", "_", setting.strip().lower()).strip("_.")
        key = raw.split(".")[-1]
        grp_rows = self.q("SELECT DISTINCT grp, key FROM processing WHERE grp || '.' || key = ? OR key = ? "
                          "ORDER BY grp || '.' || key = ? DESC", (raw, key, raw))
        if not grp_rows:
            keys = [f"{r['grp']}.{r['key']}" for r in self.q("SELECT DISTINCT grp, key FROM processing ORDER BY grp, key")]
            return {"error": f"unknown setting {setting!r}", "parsed_settings": sorted(C.SETTINGS),
                    "processing_fields": keys}
        grp, key = grp_rows[0]["grp"], grp_rows[0]["key"]
        rows = self.q(f"SELECT pr.value, pr.bibcode FROM processing pr JOIN papers p USING (bibcode) "
                      f"WHERE pr.grp = ? AND pr.key = ? AND pr.stated = 1{fsql}", [grp, key] + fargs)
        cnt: Counter = Counter()
        ex: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            v = re.sub(r"\s+", " ", (r["value"] or "").strip().lower())[:120]
            cnt[v] += 1
            ex[v].append(r["bibcode"])
        ur = self.q("SELECT fraction_unstated FROM unstated_rates WHERE setting = ?", (f"{grp}.{key}",))
        res.update({"source_field": f"{grp}.{key}", "n_papers_stating_field": len(rows),
                    "fraction_not_stated": round(1 - len(rows) / n_papers, 3) if n_papers else None,
                    "values": [{"value": v, "n_papers": n, "examples": [{"bibcode": x, "ads_url": C.ads_url(x)}
                                                                        for x in ex[v][:3]]} for v, n in cnt.most_common(30)],
                    "note": "Free-text values grouped by exact (lowercased) wording; similar wordings are not merged."})
        if ur:
            res["corpus_fraction_unstated_published_stats"] = ur[0]["fraction_unstated"]
        return res

    def software_usage(self, name: Optional[str] = None, field: Optional[str] = None, by_year: bool = False,
                       technique: Optional[str] = None) -> Dict[str, Any]:
        fwhere, fargs = self._filters(technique, field, None, None)
        fsql = (" AND " + " AND ".join(fwhere)) if fwhere else ""
        n_papers = self.q(f"SELECT count(*) AS n FROM papers p WHERE 1=1{fsql}", fargs)[0]["n"]
        res: Dict[str, Any] = {"name": name, "field": field, "technique": technique, "n_papers_in_scope": n_papers}
        if name:
            s = name.strip().lower()
            rows = self.q(f"SELECT DISTINCT sw.bibcode, sw.name, sw.version, p.year, p.title, p.first_author FROM software sw "
                          f"JOIN papers p USING (bibcode) WHERE sw.name_norm LIKE ?{fsql} ORDER BY p.year DESC",
                          [f"%{s}%"] + fargs)
            bibs = C.uniq(r["bibcode"] for r in rows)
            versions = Counter(r["version"] for r in rows if r["version"] and C.stated(r["version"]))
            res.update({"n_papers": len(bibs), "fraction_of_papers": round(len(bibs) / n_papers, 3) if n_papers else None,
                        "matched_names": Counter(r["name"] for r in rows).most_common(10),
                        "versions_stated": versions.most_common(15),
                        "examples": [{"bibcode": b, "ads_url": C.ads_url(b)} for b in bibs[:10]]})
            if by_year:
                yr: Counter = Counter()
                seen = set()
                for r in rows:
                    if r["bibcode"] not in seen:
                        seen.add(r["bibcode"])
                        yr[r["year"]] += 1
                tot = {r["year"]: r["n"] for r in self.q(f"SELECT year, count(*) AS n FROM papers p WHERE 1=1{fsql} GROUP BY year", fargs)}
                res["by_year"] = [{"year": y, "n_papers": yr[y], "n_papers_that_year": tot.get(y),
                                   "fraction": round(yr[y] / tot[y], 3) if tot.get(y) else None} for y in sorted(tot) if y]
            return res
        rows = self.q(f"SELECT sw.name_norm AS n, count(DISTINCT sw.bibcode) AS c FROM software sw JOIN papers p USING (bibcode) "
                      f"WHERE 1=1{fsql} GROUP BY sw.name_norm ORDER BY c DESC LIMIT 40", fargs)
        res["top_software"] = [{"software": r["n"], "n_papers": r["c"],
                                "fraction_of_papers": round(r["c"] / n_papers, 3) if n_papers else None} for r in rows]
        if fwhere and n_papers:
            # General packages (casa, astropy, numpy) top every subset; lift = share of papers in scope that name the
            # package / share of all papers that do, so >1 means it is characteristic of this technique or field.
            n_all = self.q("SELECT count(*) AS n FROM papers")[0]["n"]
            overall = {r["n"]: r["c"] for r in self.q(
                "SELECT name_norm AS n, count(DISTINCT bibcode) AS c FROM software GROUP BY name_norm")}
            scoped = self.q(f"SELECT sw.name_norm AS n, count(DISTINCT sw.bibcode) AS c FROM software sw "
                            f"JOIN papers p USING (bibcode) WHERE 1=1{fsql} GROUP BY sw.name_norm", fargs)
            min_n = max(3, round(0.01 * n_papers))
            lifted = [{"software": r["n"], "n_papers": r["c"],
                       "fraction_of_papers": round(r["c"] / n_papers, 3),
                       "lift_vs_corpus": round((r["c"] / n_papers) / (overall[r["n"]] / n_all), 2)}
                      for r in scoped if r["c"] >= min_n and overall.get(r["n"])]
            lifted.sort(key=lambda x: (-x["lift_vs_corpus"], -x["n_papers"]))
            res["distinctive_software"] = lifted[:20]
            res["distinctive_note"] = (f"Packages named by at least {min_n} papers in scope, ranked by lift_vs_corpus "
                                       "(how much more often they appear here than in the whole corpus). Use this, "
                                       "not top_software, to say which tools are specific to the technique or field.")
        if by_year:
            top = [r["n"] for r in rows[:8]]
            yrows = self.q(f"SELECT p.year AS y, sw.name_norm AS n, count(DISTINCT sw.bibcode) AS c FROM software sw "
                           f"JOIN papers p USING (bibcode) WHERE sw.name_norm IN ({','.join('?' * len(top))}){fsql} "
                           f"GROUP BY p.year, sw.name_norm ORDER BY p.year", top + fargs) if top else []
            by: Dict[Any, Dict[str, int]] = defaultdict(dict)
            for r in yrows:
                by[r["y"]][r["n"]] = r["c"]
            res["by_year_top8"] = [{"year": y, **v} for y, v in sorted(by.items(), key=lambda kv: kv[0] or 0)]
        res["note"] = "Counts are distinct papers naming the package in their software list (normalized, lowercased)."
        return res

    def list_techniques(self) -> Dict[str, Any]:
        rows = self.q("SELECT slug, title, description, state, paper_count FROM skills ORDER BY paper_count DESC")
        tagged = Counter()
        for r in self.q("SELECT techniques FROM papers"):
            for t in C.split_list(r["techniques"]):
                tagged[t] += 1
        return {"n": len(rows), "techniques": [
            {"slug": r["slug"], "title": r["title"], "state": r["state"], "papers_in_skill": r["paper_count"],
             "papers_tagged_in_bundle": tagged.get(r["slug"], 0), "prompt": f"use-{r['slug']}",
             "resource": f"alma://skills/{r['slug']}", "description": r["description"]} for r in rows],
            "note": "Use a slug as the technique filter in search_papers / find_papers_using, or read it with get_skill."}

    def get_skill(self, slug: str, include_parameters: bool = False) -> Dict[str, Any]:
        rows = self.q("SELECT * FROM skills WHERE slug = ?", (slug.strip().lower(),))
        if not rows:
            return {"error": f"unknown skill {slug!r}", "slugs": [r["slug"] for r in self.q("SELECT slug FROM skills ORDER BY slug")]}
        r = rows[0]
        out = {"slug": r["slug"], "title": r["title"], "state": r["state"], "paper_count": r["paper_count"],
               "skill_md": r["skill_md"]}
        if include_parameters:
            out["parameters_md"] = r["parameters_md"] or "(no parameters reference for this skill)"
        out["note"] = ("Skills were drafted from the corpus and reviewed. Bibcodes in the text can be checked on ADS "
                       "(https://ui.adsabs.harvard.edu/abs/<bibcode>) or with get_methods_card.")
        return out

    def skill_slugs(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.q("SELECT slug, title, description FROM skills ORDER BY slug")]

    def compare_papers(self, bibcodes: Sequence[str]) -> Dict[str, Any]:
        bibcodes = [b.strip() for b in bibcodes if b and b.strip()][:8]
        if len(bibcodes) < 2:
            return {"error": "give 2 to 8 bibcodes"}
        found, missing = [], []
        for b in bibcodes:
            p = self.paper(b)
            (found.append(p) if p is not None else missing.append(b))
        rows_spec = [
            ("year", lambda p, pr: str(p["year"])), ("first author", lambda p, pr: p["first_author"] or ""),
            ("fields", lambda p, pr: ", ".join(C.split_list(p["fields"]))),
            ("techniques", lambda p, pr: ", ".join(C.split_list(p["techniques"]))),
            ("ALMA projects", lambda p, pr: ", ".join(C.split_list(p["alma_project_codes"]))),
        ]
        keys = [("data", "bands"), ("data", "arrays"), ("data", "angular_resolution"), ("data", "sensitivity"),
                ("data", "time_on_source"), ("calibration", "pipeline_or_manual"), ("calibration", "casa_version"),
                ("calibration", "self_calibration"), ("calibration", "flux_scale_uncertainty"), ("imaging", "deconvolver"),
                ("imaging", "weighting_robust"), ("imaging", "cell_size"), ("imaging", "continuum_subtraction"),
                ("imaging", "final_beam"), ("imaging", "final_rms")]
        table: Dict[str, Dict[str, str]] = {}
        per: Dict[str, Dict[str, Any]] = {}
        for p in found:
            b = p["bibcode"]
            pr = {(r["grp"], r["key"]): (r["value"] if r["stated"] else "not stated")
                  for r in self.q("SELECT grp, key, value, stated FROM processing WHERE bibcode = ?", (b,))}
            for label, fn in rows_spec:
                table.setdefault(label, {})[b] = fn(p, pr)
            for g, k in keys:
                table.setdefault(f"{g}.{k}", {})[b] = pr.get((g, k), "not stated")
            steps = self.q("SELECT ord, goal, software FROM steps WHERE bibcode = ? ORDER BY ord", (b,))
            table.setdefault("analysis steps", {})[b] = " | ".join(f"{s['ord']}. {s['goal']}" for s in steps)
            sw = self.q("SELECT DISTINCT name FROM software WHERE bibcode = ?", (b,))
            table.setdefault("software", {})[b] = ", ".join(r["name"] for r in sw)
            res = self.q("SELECT quantity, value, uncertainty, units FROM results WHERE bibcode = ? LIMIT 6", (b,))
            table.setdefault("key results", {})[b] = "; ".join(
                f"{r['quantity']} = {r['value']}{(' +/- ' + r['uncertainty']) if r['uncertainty'] and C.stated(r['uncertainty']) else ''}"
                f" {r['units'] or ''}".strip() for r in res)
            per[b] = _paper_ref(p)

        def cell(x: str) -> str:
            return (x or "").replace("|", "/").replace("\n", " ")[:160]
        bibs = [p["bibcode"] for p in found]
        md = ["| | " + " | ".join(bibs) + " |", "|---|" + "---|" * len(bibs)]
        md += [f"| {label} | " + " | ".join(cell(vals.get(b, "")) for b in bibs) + " |" for label, vals in table.items()]
        return {"papers": [per[b] for b in bibs], "missing": missing, "table": table, "markdown": "\n".join(md),
                "note": "Cells are truncated to 160 characters in the markdown; the 'table' field has full values."}

    def corpus_info(self) -> Dict[str, Any]:
        meta = {r["k"]: r["v"] for r in self.q("SELECT k, v FROM meta")}
        counts = json.loads(meta.pop("counts", "{}"))
        lic = Counter(C.license_label(r["license"]) for r in self.q("SELECT license FROM papers"))
        return {"name": "alma-corpus", **meta, "counts": counts, "sha256": self.manifest.get("sha256"),
                "bundle_dir": str(self.dir), "licenses": dict(lic.most_common()),
                "license_notes": ("Code is MIT. Cards are LLM-extracted summaries of each paper's methods with short "
                                  "verbatim evidence quotes and their locations. Quotes stay the copyright of the "
                                  "authors/publishers; the public bundle withholds them for papers that are not CC BY, "
                                  "CC BY-SA, CC0 or public domain. Check every number on ADS before citing it."),
                "heldout": "Held-out benchmark papers are excluded from every table, card and skill.",
                "how_to_cite": CITE}
