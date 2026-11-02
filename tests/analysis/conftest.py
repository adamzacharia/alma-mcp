import os

import pytest

from alma_mcp.analysis._util import WORKDIR_ENV


@pytest.fixture(scope="session", autouse=True)
def _workdir_is_basetemp(tmp_path_factory):
    """The analysis tools only touch files inside ALMA_MCP_WORKDIR; point it at pytest's base temp folder."""
    old = os.environ.get(WORKDIR_ENV)
    os.environ[WORKDIR_ENV] = str(tmp_path_factory.getbasetemp())
    yield
    if old is None:
        os.environ.pop(WORKDIR_ENV, None)
    else:
        os.environ[WORKDIR_ENV] = old
