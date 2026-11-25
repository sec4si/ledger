"""Tests for ledger.diff comparing two captures and self-comparison."""

import re

from ledger import diff


def _section(text, heading):
    """Return the body of a level-2 markdown section, excluding its heading."""
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert m, f"missing section: {heading}"
    return m.group(1)


def _table_rows(section_text):
    """Parse markdown table data rows into lists of stripped cell strings."""
    rows = []
    for line in section_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows[1:] if rows else []


def test_diff_decode_q38_vs_bf16state(captures, tmp_path):
    """BF16 recurrent state changes GDN decode (paired as another variant) and out_proj beyond drift."""
    out = tmp_path / "diff-bf16.md"
    diff.report(
        str(captures / "decode-q38"),
        str(captures / "decode-q38-bf16state"),
        ceiling=252,
        min_ms=0.05,
        tolerance=0.03,
        out=str(out),
    )
    text = out.read_text()

    changed = _table_rows(_section(text, "Changed beyond drift"))
    gdn = [r for r in changed if "gdn_decode_post_conv_mtp" in r[0]]
    out_proj = [r for r in changed if "linear_attn.out_proj.weight" in r[2]]

    assert gdn, "expected GDN decode kernel in changed groups"
    assert "(other variant)" in gdn[0][0]
    assert float(gdn[0][6]) < 0

    assert out_proj, "expected linear_attn.out_proj.weight in changed groups"
    assert float(out_proj[0][6]) < 0


def test_diff_self_reports_nothing(captures, tmp_path):
    """Comparing a capture against itself reports no changes beyond drift."""
    out = tmp_path / "diff-self.md"
    path = str(captures / "decode-q38")
    diff.report(path, path, ceiling=252, min_ms=0.05, tolerance=0.03, out=str(out))
    text = out.read_text()

    assert _section(text, "Changed beyond drift").strip() == "Nothing."
    assert "## In A only" not in text
    assert "## In B only" not in text
