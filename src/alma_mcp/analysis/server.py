"""FastMCP server "alma-analysis": the analysis steps ALMA papers use most, as tested tools.

Run: alma-analysis-mcp (stdio) or alma-analysis-mcp --http 8001 (streamable HTTP).
Every tool returns {"result", "convention", "warnings", "corpus_practice"}; corpus_practice names the skill slugs in
the alma-corpus server that describe typical published settings for that step.
"""

from __future__ import annotations

import argparse
import os
import sys

from mcp.server.fastmcp import FastMCP

from . import dust, imaging_checks, kinematics, lines, uv
from ._util import WORKDIR_ENV, workdir

INSTRUCTIONS = (
    "Compute tools for ALMA analysis. Inputs are plain numbers/lists or local file paths (FITS cubes, or .npz/.csv "
    "visibility tables with columns u, v, Re, Im, weight). Every result states units, the convention used and "
    "warnings; read the warnings before quoting numbers. corpus_practice lists alma-corpus skill slugs to consult "
    "for the settings papers typically use. File paths must lie inside the server's allowed folder "
    "(ALMA_MCP_WORKDIR, default its working directory); relative paths start there. Tools that write files refuse "
    "to replace an existing file unless overwrite=true."
)

TOOLS = [
    dust.dust_mass_from_flux,
    dust.fit_modified_blackbody,
    dust.spectral_index,
    lines.fit_gaussian_lines,
    lines.line_luminosity,
    lines.co_gas_mass,
    lines.rotational_diagram,
    lines.lte_column_density,
    kinematics.moment_maps,
    kinematics.keplerian_mask,
    kinematics.dynamical_mass,
    kinematics.virial_parameter,
    kinematics.toomre_q,
    uv.fit_uv_model,
    uv.deproject_and_bin,
    uv.frank_profile,
    imaging_checks.beam_and_brightness,
    imaging_checks.rms_from_line_free,
    imaging_checks.largest_angular_scale,
    imaging_checks.flux_recovery,
]


def build_server(host: str = "127.0.0.1", port: int = 8001) -> FastMCP:
    mcp = FastMCP("alma-analysis", instructions=INSTRUCTIONS, host=host, port=port)
    for fn in TOOLS:
        mcp.tool(name=fn.__name__, description=(fn.__doc__ or "").strip())(fn)
    return mcp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="alma-analysis-mcp", description=__doc__)
    ap.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP on PORT instead of stdio")
    ap.add_argument("--host", default="127.0.0.1",
                    help="host for --http (default 127.0.0.1; the server reads and writes files, so do not bind "
                         "0.0.0.0 without an authenticating proxy)")
    ap.add_argument("--workdir", help=f"allowed folder for all file inputs and outputs (default ${WORKDIR_ENV}, "
                                      "else the current directory)")
    args = ap.parse_args(argv)
    if args.workdir:
        os.environ[WORKDIR_ENV] = os.path.abspath(os.path.expanduser(args.workdir))
    print(f"alma-analysis: file access limited to {workdir()}", file=sys.stderr)
    if args.http:
        build_server(args.host, args.http).run(transport="streamable-http")
    else:
        build_server().run()


if __name__ == "__main__":
    main()
