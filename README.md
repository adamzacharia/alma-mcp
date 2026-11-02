# alma-mcp

[![CI](https://github.com/adamzacharia/alma-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/adamzacharia/alma-mcp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/adamzacharia/alma-mcp/blob/main/pyproject.toml)
[![Data](https://img.shields.io/badge/papers-3%2C537-informational.svg)](https://github.com/adamzacharia/alma-mcp/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/adamzacharia/alma-mcp/blob/main/LICENSE)

Two [Model Context Protocol](https://modelcontextprotocol.io) servers for ALMA science that you can add to Claude
(Desktop or Code), Codex, Cursor or any other MCP client.

- **`alma-corpus-mcp`** (knowledge): what ALMA papers actually did, with which settings, and where they wrote it.
  It serves methods cards extracted from refereed ALMA papers, reviewed technique skills and corpus-wide parameter
  statistics. Ask things like "what robust value do disk papers use?", "which papers fit visibilities with
  galario?", or "show me the self-calibration recipe of 2018ApJ...869L..41A".
- **`alma-analysis-mcp`** (compute): the analysis steps those papers use most, as tested tools (see
  [Analysis tools](#analysis-tools)).

Every corpus result carries bibcodes and ADS links (`https://ui.adsabs.harvard.edu/abs/<bibcode>`) so you can check
the source.

## What is in the data bundle

Numbers from the bundle built on 2026-11-02 (version `2026.11.02`):

| | count |
|---|---|
| Papers (methods cards) | 3,537 (published 2012 to 2026) |
| Analysis steps | 19,482 |
| Software mentions | 14,812 |
| Numeric results (with units) | 43,069 |
| Claim tuples | 11,291 |
| ALMA project codes linked | 7,505 |
| Processing facts (calibration, imaging, data) | 99,036 |
| Parsed settings (robust, cell size, solint, ...) | 12,337 (in 3,109 papers) |
| Reviewed technique skills | 38 |
| Papers with a CC BY / CC BY-SA / CC0 license | 707 |

Each card records, for one paper: science fields, targets, bands, arrays, resolution and sensitivity, project codes,
calibration and imaging settings (pipeline or manual, CASA version, self-calibration, weighting, cell size,
deconvolver, continuum subtraction, beam, rms), the ordered analysis steps with software and key parameters,
numeric results with units, claims and a reproducibility assessment. Each value carries the location of its
evidence in the paper (section, table or figure) and, in the full bundle, a short verbatim quote.

Cards were extracted from the paper full texts by LLMs and every quote was checked against the text. Only papers
whose evidence checks passed are included. Treat a card as a pointer into the paper, not as the paper: check a
number on ADS before you rely on it.

Bundle sizes: full 279 MB (tarball 79 MB), public 223 MB (tarball 60 MB).

## Install

With [uv](https://docs.astral.sh/uv/), nothing to install first:

```bash
uvx --from git+https://github.com/adamzacharia/alma-mcp alma-corpus-mcp download   # fetch the data bundle once
uvx --from git+https://github.com/adamzacharia/alma-mcp alma-corpus-mcp info       # check: version and counts
```

Or with pip:

```bash
pip install git+https://github.com/adamzacharia/alma-mcp                      # both servers, light core
pip install "alma-mcp[fit] @ git+https://github.com/adamzacharia/alma-mcp"    # + emcee for MCMC posteriors
pip install "alma-mcp[all] @ git+https://github.com/adamzacharia/alma-mcp"    # + emcee, frank, bettermoments
alma-corpus-mcp download
```

From a checkout:

```bash
git clone https://github.com/adamzacharia/alma-mcp && cd alma-mcp
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[fit,test]"
pytest
```

The servers need Python 3.10 or newer and use the `mcp` Python SDK 1.x (`FastMCP`; the 2.x SDK renamed it, so the
dependency is pinned below 2).

## Get the data

The corpus server needs the data bundle (a directory with `alma_methods.sqlite`, `manifest.json` and `skills/`).

```bash
alma-corpus-mcp download                    # into ~/.alma-mcp/data, sha256-checked (60 MB)
alma-corpus-mcp info                        # version, counts, licenses
```

The download comes from the latest [GitHub release](https://github.com/adamzacharia/alma-mcp/releases/latest)
(`alma-mcp-data-public.tar.gz`, the public bundle) and is checked against the release's `.sha256` file; a mismatch
aborts without touching an existing bundle. `--url` or `ALMA_MCP_DATA_URL` selects another tarball.

To use a bundle somewhere else, set `ALMA_MCP_DATA` to its directory. If no bundle is found the server still
starts, and every tool returns an error that says how to get one.

Maintainers build the bundle from the corpus with:

```bash
python scripts/build_bundle.py --corpus /path/to/alma-corpus            # full -> dist-data/, public -> dist-data/public/
python scripts/build_bundle.py --corpus /path/to/alma-corpus --public   # public variant only
```

`--corpus` can also come from the `ALMA_CORPUS` environment variable. The build writes each release tarball and its
`.sha256` file to its own folder:

- `dist-data/release/public/alma-mcp-data-public-<version>.tar.gz`: the only one meant for public upload (for
  example a GitHub release).
- `dist-data/release/full/alma-mcp-data-<version>.tar.gz`: private or team use only. It carries verbatim quotes and
  abstracts from papers under the arXiv non-exclusive license, which may not be redistributed.

At the end the build scans every table and skill file for local user paths, the builder's user name and
API-key-like strings, and fails if it finds one.

## Tools (alma-corpus)

| Tool | What it does |
|---|---|
| `search_papers(query, technique, field, year_min, year_max, limit)` | BM25 full-text search over titles, abstracts and full cards, with filters |
| `get_methods_card(bibcode, section)` | The card as markdown; sections `data`, `processing`, `recipe`, `results`, `reproducibility` |
| `get_recipe(bibcode)` | Ordered analysis steps with software, key parameters, assumptions, plus stated settings |
| `find_papers_using(software, technique, parameter_regex, limit)` | Papers by package, technique or a regex over parameters |
| `parameter_census(setting, field)` | Values papers used (robust, cell size, self-cal solint and mode, continuum order and domain, deconvolver, band, array, or any processing field), counts, example bibcodes, and how often it is not stated |
| `software_usage(name, field, technique, by_year)` | How many papers name a package, versions, trends by year; with a technique or field, the packages most specific to it (ranked by lift over the corpus) |
| `list_techniques()` / `get_skill(slug, include_parameters)` | The reviewed technique skills |
| `compare_papers(bibcodes)` | Side-by-side table of data, processing and results |
| `corpus_info()` | Bundle version, counts, license notes, how to cite |

Each skill is also an MCP prompt (`use-<slug>`, with an optional `task` argument) and a resource
(`alma://skills/<slug>`); each card is a resource (`alma://cards/<bibcode>`, URL-encode the `&` of A&A bibcodes as
`%26`).

## Analysis tools

`alma-analysis-mcp` (FastMCP server "alma-analysis") wraps the analysis steps ALMA papers use most. Inputs are plain
numbers and lists, or local file paths: FITS cubes, and visibility tables (`.npz` or `.csv` with columns `u, v, Re, Im,
weight`; `u, v` in wavelengths, kilo-wavelengths or metres). Every tool returns `result` (values with units and
errors), `convention` (the formula and constants used), `warnings`, and `corpus_practice` (skill slugs you can pass to
the corpus server to see the settings published papers chose). The functions are also importable without MCP from
`alma_mcp.analysis.<module>`.

File access: every input and output path must lie inside one allowed folder, set by `ALMA_MCP_WORKDIR` (default:
the folder the server was started in). Relative paths are taken from that folder, and paths that resolve outside it
(including through `..` or symlinks) are refused. `moment_maps` and `keplerian_mask` refuse to replace an existing
file unless you pass `overwrite=true`. Point `ALMA_MCP_WORKDIR` at your data folder in the client config, for example
`"env": {"ALMA_MCP_WORKDIR": "/path/to/data"}`.

| Tool | What it does | Convention stated |
|---|---|---|
| `dust_mass_from_flux` | Optically thin dust (and gas) mass from one continuum flux, local or at redshift z | Hildebrand 1983 with the full Planck function; opacity presets (Beckwith 1990 disks, OH94 cores, 0.077 m2/kg galaxies) or custom; per-dust vs per-gas opacity; CMB contrast and heating (da Cunha 2013) |
| `fit_modified_blackbody` | SED fit for T_d, beta, M_dust, L_IR (8-1000 um) and L_FIR | Optically thin or general-opacity form (tau = 1 at lambda_0); calibration floor in quadrature; scipy least squares, optional emcee posterior |
| `spectral_index` | Two-point or multi-point alpha, beta = alpha - 2 and Planck-corrected beta | Per-band calibration error in quadrature (calibration dominates the absolute alpha) |
| `fit_gaussian_lines` | 1..N Gaussian components on a spectrum | Integrated flux from the fit and from a channel sum with error rms dv sqrt(N) |
| `line_luminosity` | L'_line (K km/s pc^2) and L_line (Lsun) | Solomon et al. 1997; Solomon and Vanden Bout 2005 |
| `co_gas_mass` | Molecular gas mass from L'_CO with r_J1 | alpha_CO with or without helium (1.36) or X_CO; flux-density ratios converted to brightness ratios (divide by J^2) |
| `rotational_diagram` | LTE population diagram: T_rot and N_tot | N_u from W (thin, RJ); Q(T) from a user table or linear rotor; optional joint opacity fit (C_tau, Goldsmith and Langer 1999) |
| `lte_column_density` | Column density from one line at an assumed T_ex | Mangum and Shirley 2015 with the J(T_bg) term; optional tau correction from tau or peak T_B |
| `moment_maps` | Moment 0/1/2, peak, quadratic peak velocity and a moment-0 noise map from a FITS cube; writes FITS | Robust rms from line-free channels; separate clips for moment 0 and moments 1/2 or a user mask; Teague and Foreman-Mackey 2018 quadratic peak |
| `keplerian_mask` | Keplerian mask cube for a thin disk on a template cube's grid | PA east of north to the redshifted major axis; beam dilation |
| `dynamical_mass` | Keplerian or virial enclosed mass | v / sin i for Keplerian; (5/a) sigma^2 R / G for rho ~ r^-k |
| `virial_parameter` | alpha_vir and boundedness | alpha_5 = 5 sigma^2 R / (G M) (Bertoldi and McKee 1992), alpha_E = alpha_5 / a, R = 1.91 sigma_r option |
| `toomre_q` | Toomre Q for gas, stars, protoplanetary disks, or two fluids | sigma kappa / (pi G Sigma); 3.36 for stars; Romeo and Wiegert 2011 two-component form |
| `fit_uv_model` | Point, Gaussian, elliptical Gaussian, uniform disk or ring fitted directly to visibilities | chi^2 with CASA WEIGHT = 1/sigma^2 per real and per imaginary part; optional weight rescaling from the residual scatter; multi-start least squares, optional emcee |
| `deproject_and_bin` | Phase shift, deprojection and binned Re/Im vs uv distance (klambda) | q = sqrt(q_maj^2 + (q_min cos i)^2); bin error 1/sqrt(sum w) |
| `frank_profile` | Non-parametric radial brightness profile | Wraps frank (Jennings et al. 2020) if installed, else returns the install command |
| `beam_and_brightness` | Jy/beam to K and back, beam solid angle, pixels per beam | Rayleigh-Jeans 1.222e6 form or full Planck inversion |
| `rms_from_line_free` | Noise from line-free channels of a cube or spectrum | 1.4826 x MAD, with plain std for comparison |
| `largest_angular_scale` | Maximum recoverable scale and resolution from baselines | 0.6 lambda / B_min (older ALMA Technical Handbook) and 0.983 lambda / L05, 0.574 lambda / L80 (current Handbook) |
| `flux_recovery` | Rough fraction of a smooth source's flux seen on the shortest baseline | Gaussian or uniform-disk visibility at B_min / lambda (a heuristic, not a measurement) |

Optional extras: `[fit]` enables the emcee posteriors, `[uv]` enables `frank_profile`, `[cube]` adds bettermoments
next to the built-in quadratic peak method (install as `"alma-mcp[fit] @ git+https://github.com/adamzacharia/alma-mcp"`).

## Client configuration

Replace the command with the full path to the executable if it is not on your `PATH` (for a venv on Windows:
`C:\path\to\alma-mcp\.venv\Scripts\alma-corpus-mcp.exe`). Add `"env": {"ALMA_MCP_DATA": "/path/to/bundle"}` if the
bundle is not in `~/.alma-mcp/data`.

**Claude Desktop** (`claude_desktop_config.json`; Settings > Developer > Edit Config):

```json
{
  "mcpServers": {
    "alma-corpus": { "command": "uvx", "args": ["--from", "git+https://github.com/adamzacharia/alma-mcp", "alma-corpus-mcp"] },
    "alma-analysis": { "command": "uvx", "args": ["--from", "alma-mcp[fit] @ git+https://github.com/adamzacharia/alma-mcp", "alma-analysis-mcp"] }
  }
}
```

**Claude Code**:

```bash
claude mcp add alma-corpus -- uvx --from git+https://github.com/adamzacharia/alma-mcp alma-corpus-mcp
claude mcp add alma-analysis -- uvx --from "alma-mcp[fit] @ git+https://github.com/adamzacharia/alma-mcp" alma-analysis-mcp
# with a local bundle:
claude mcp add alma-corpus -e ALMA_MCP_DATA=/path/to/dist-data -- alma-corpus-mcp
```

**Codex** (`~/.codex/config.toml`):

```toml
[mcp_servers.alma-corpus]
command = "uvx"
args = ["--from", "git+https://github.com/adamzacharia/alma-mcp", "alma-corpus-mcp"]
# env = { ALMA_MCP_DATA = "/path/to/dist-data" }

[mcp_servers.alma-analysis]
command = "uvx"
args = ["--from", "alma-mcp[fit] @ git+https://github.com/adamzacharia/alma-mcp", "alma-analysis-mcp"]
```

**Cursor** (`~/.cursor/mcp.json`, or `.cursor/mcp.json` in a project):

```json
{
  "mcpServers": {
    "alma-corpus": { "command": "uvx", "args": ["--from", "git+https://github.com/adamzacharia/alma-mcp", "alma-corpus-mcp"] }
  }
}
```

## Hosting one copy for a team

```bash
alma-corpus-mcp --http 8765                 # binds 127.0.0.1 by default
```

This serves streamable HTTP at `http://127.0.0.1:8765/mcp`. Clients that support remote servers connect with that
URL (Claude Code: `claude mcp add --transport http alma-corpus http://127.0.0.1:8765/mcp`). There is no
authentication. To share it, keep the default `127.0.0.1` binding and put an authenticating reverse proxy or a VPN
in front; do not bind `--host 0.0.0.0` on a reachable network without one.

**Be careful with `alma-analysis-mcp --http`.** The analysis server reads and writes files (FITS cubes, visibility
tables, moment maps, masks) inside `ALMA_MCP_WORKDIR`, so anyone who can reach its HTTP port can read data files in
that folder and write new files there. Bind it to `127.0.0.1` (the default), set `ALMA_MCP_WORKDIR` to a folder that
holds only the data you mean to share, and never expose it on `0.0.0.0` without an authenticating proxy.

## Companion server: quasar-mcp

[quasar-mcp](https://github.com/adamzacharia/quasar-mcp) serves about 200 live-data astronomy tools: ALMA/NRAO,
MAST, ESO, CADC and HEASARC archives; VO TAP/SIA/cone search; SIMBAD, NED, VizieR and Gaia; Splatalogue; NASA ADS;
NOIRLab Data Lab; ZTF/ALeRCE; HiPS cutouts and plotting.

```bash
claude mcp add quasar -- uvx --from git+https://github.com/adamzacharia/quasar-mcp quasar-mcp
```

The alma-corpus server says how published papers did a step; quasar-mcp fetches the archive data and papers to do
it. Running all three servers together is the intended setup.

## Data licensing

- The code in this repository is MIT licensed (see `LICENSE`).
- The cards are structured summaries of each paper's methods. In the **full** bundle they also contain short
  verbatim quotes from the papers as evidence, and paper abstracts for search. Those quotes and abstracts remain the
  copyright of the authors and publishers.
- The **public** bundle (`build_bundle.py --public`) keeps every structured value and the location of its evidence
  but withholds the quote text and the abstract for every paper whose arXiv license is not CC BY, CC BY-SA, CC0 or
  public domain (most papers use the arXiv non-exclusive license). For those papers it also replaces any structured
  value that repeats more than 40 characters of a quote word for word with `[verbatim text withheld, see paper]`.
  Each paper's license is stored in
  `papers.license` so you can filter. Distribute the public bundle (`dist-data/release/public/`); keep the full
  one (`dist-data/release/full/`) for local or team use. Public cards also drop the internal repair log and the
  text fields of the extraction checks; both bundles keep only the model, text source and extractor version as
  provenance.

## Held-out papers

Papers reserved as held-out benchmark items in the corpus manifest are excluded from every table, card, statistic
and skill file in both bundles, so the servers can be used next to benchmark runs without contaminating them.

## How to cite

Cite the original papers: every result gives their bibcodes and ADS links. For the corpus and servers, cite this
repository (`https://github.com/adamzacharia/alma-mcp`) and the bundle version reported by `corpus_info()`.
