"""Locate or download the alma-corpus data bundle.

The bundle is a directory holding ``alma_methods.sqlite``, ``manifest.json`` and ``skills/``. It is looked up in
``$ALMA_MCP_DATA`` (a directory) first, then in ``~/.alma-mcp/data``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

DB_NAME = "alma_methods.sqlite"
ENV_VAR = "ALMA_MCP_DATA"
# The public bundle attached to the latest GitHub release (with a .sha256 next to it). Override with --url or
# ALMA_MCP_DATA_URL.
DEFAULT_URL = "https://github.com/adamzacharia/alma-mcp/releases/latest/download/alma-mcp-data-public.tar.gz"


class BundleMissing(RuntimeError):
    """Raised when no data bundle is found. The message tells the user how to get one."""


def default_dir() -> Path:
    return Path.home() / ".alma-mcp" / "data"


def data_dir() -> Path:
    """Directory the bundle is expected in (it may not exist yet)."""
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else default_dir()


def missing_message(where: Optional[Path] = None) -> str:
    where = where or data_dir()
    return (f"alma-corpus data bundle not found: expected {where / DB_NAME}.\n"
            f"Get it with:  alma-corpus-mcp download            (into {default_dir()})\n"
            f"or point {ENV_VAR} at a directory that holds {DB_NAME}, manifest.json and skills/ "
            f"(for example the dist-data/ folder made by scripts/build_bundle.py).")


def find_bundle() -> Path:
    """Path to the bundle directory, or BundleMissing with instructions."""
    d = data_dir()
    if (d / DB_NAME).is_file():
        return d
    raise BundleMissing(missing_message(d))


def read_manifest(d: Path) -> Dict[str, Any]:
    try:
        return json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "alma-mcp"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:  # noqa: S310 (user-chosen URL)
        shutil.copyfileobj(r, f)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not (target == root or root in target.parents) or m.issym() or m.islnk() or m.isdev():
            raise RuntimeError(f"refusing unsafe archive member {m.name!r}")
    if sys.version_info >= (3, 12):
        tar.extractall(dest, filter="data")
    else:  # members were checked above
        tar.extractall(dest)  # noqa: S202


def download(url: Optional[str] = None, dest: Optional[Path] = None, sha256: Optional[str] = None) -> Path:
    """Download a release tarball, verify its sha256, and unpack it into ``dest`` (default: data_dir()).

    The expected digest comes from ``sha256`` or from ``<url>.sha256`` (``<hex>  <name>`` format). A mismatch or a
    missing digest aborts without touching an existing bundle.
    """
    url = url or os.environ.get("ALMA_MCP_DATA_URL") or DEFAULT_URL
    dest = (dest or data_dir()).expanduser()
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "bundle.tar.gz"
        print(f"downloading {url}", file=sys.stderr)
        _fetch(url, tgz)
        expected = sha256
        if not expected:
            side = Path(tmp) / "bundle.sha256"
            try:
                _fetch(url + ".sha256", side)
                expected = side.read_text(encoding="utf-8").split()[0]
            except Exception as e:
                raise RuntimeError(f"no checksum: could not fetch {url}.sha256 ({e}); pass --sha256") from e
        got = sha256_file(tgz)
        if got.lower() != expected.strip().lower():
            raise RuntimeError(f"sha256 mismatch for {url}: expected {expected}, got {got}")
        stage = Path(tmp) / "stage"
        stage.mkdir()
        with tarfile.open(tgz, "r:gz") as tar:
            _safe_extract(tar, stage)
        if not (stage / DB_NAME).is_file():
            raise RuntimeError(f"archive has no {DB_NAME} at its top level")
        dest.mkdir(parents=True, exist_ok=True)
        for p in stage.iterdir():
            target = dest / p.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(p), str(target))
    print(f"bundle ready in {dest} (sha256 {got[:12]}...)", file=sys.stderr)
    return dest
