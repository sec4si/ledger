"""Tests for ledger.transfers on captures containing memory copies."""

import io

from ledger import transfers


def test_transfers_runs_on_capture_with_copies(captures):
    """transfers.build and transfers.report complete cleanly on a capture with memory copies."""
    capture = str(captures / "decode-q38")
    rows, nregions, have_kernels = transfers.build(capture)
    assert rows
    assert nregions > 0
    assert have_kernels

    buf = io.StringIO()
    transfers.report(capture, ceilings={"HtoD": 24.0, "DtoH": 24.0}, steps=3, out=buf)
    text = buf.getvalue()

    assert "# Transfers:" in text
    assert "No copies in the window." not in text
    assert "## By direction" in text
    assert "## By region" in text
    for direction in ("HtoD", "DtoH", "DtoD"):
        assert f"| {direction} |" in text
