"""Tests for ledger.diff comparing two captures and self-comparison."""

import json
import shutil

from ledger import diff
from report import section, table_rows


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

    changed = table_rows(section(text, "Changed beyond drift"))
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

    assert section(text, "Changed beyond drift").strip() == "Nothing."
    assert "## In A only" not in text
    assert "## In B only" not in text


def test_diff_works_without_steps_in_run_json(captures, tmp_path):
    """diff succeeds when run.json omits 'steps' by counting step markers via window_steps."""
    dst = tmp_path / "decode-no-steps"
    shutil.copytree(captures / "decode-q38", dst)
    run_path = dst / "run.json"
    run = json.loads(run_path.read_text())
    run.pop("steps", None)
    run_path.write_text(json.dumps(run))

    out = tmp_path / "diff-no-steps.md"
    diff.report(
        str(dst),
        str(captures / "decode-q38-bf16state"),
        ceiling=252,
        min_ms=0.05,
        tolerance=0.03,
        out=str(out),
    )
    changed = table_rows(section(out.read_text(), "Changed beyond drift"))
    assert any("gdn_decode_post_conv_mtp" in r[0] for r in changed)
    assert any("linear_attn.out_proj.weight" in r[2] for r in changed)
