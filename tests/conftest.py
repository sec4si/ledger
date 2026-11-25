"""Shared pytest fixtures for Ledger tests."""

import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE = Path(__file__).resolve().parent / "fixtures" / "captures.tar.xz"


@pytest.fixture(scope="session")
def captures(tmp_path_factory):
    """Unpack tests/fixtures/captures.tar.xz once per session and return the captures directory."""
    dest = tmp_path_factory.mktemp("ledger-captures")
    with tarfile.open(ARCHIVE, "r:xz") as tf:
        tf.extractall(dest, filter="data")
    return dest / "captures"
