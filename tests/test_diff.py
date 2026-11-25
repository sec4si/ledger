"""Tests for ledger.diff comparing two captures and self-comparison."""

import json
import shutil
import subprocess
import sys

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


def _scale_capture_durations(src, dst, factor):
    """Copy a capture and scale every kernel duration and inter-kernel timestamp by `factor`."""
    shutil.copytree(src, dst)
    kernels_path = next(dst.glob("kernels-*.jsonl"))
    records = [json.loads(line) for line in kernels_path.read_text().splitlines() if line.strip()]
    t0 = min(r["start"] for r in records if r["start"] > 0)
    with kernels_path.open("w") as f:
        for r in records:
            if r["start"] > 0:
                r["start"] = t0 + int((r["start"] - t0) * factor)
                r["end"] = t0 + int((r["end"] - t0) * factor)
            f.write(json.dumps(r) + "\n")


def test_diff_gate_kernel_regression_and_thresholds(captures, tmp_path):
    """Gate fails when shared kernel groups regress past --fail-above and --fail-min-ms, and passes
    when thresholds are raised or when B is faster."""
    fast = str(captures / "decode-q38-bf16state")
    slow = str(captures / "decode-q38")

    # Fails on gdn_decode_post_conv_mtp (~+86%, +0.92 ms) and out_proj (~+9%, +0.78 ms).
    fail_out = tmp_path / "gate-fail.md"
    failures = diff.report(fast, slow, ceiling=252, min_ms=0.05, tolerance=0.03,
                           out=str(fail_out), fail_above=5, fail_min_ms=0.5)
    assert len(failures) == 2
    gate = section(fail_out.read_text(), "Gate")
    assert "Failed" in gate
    assert "gdn_decode_post_conv_mtp" in gate
    assert "linear_attn.out_proj.weight" in gate

    # Passes when --fail-min-ms filters out both groups.
    min_ms_out = tmp_path / "gate-min-ms.md"
    assert not diff.report(fast, slow, ceiling=252, min_ms=0.05, tolerance=0.03,
                           out=str(min_ms_out), fail_above=5, fail_min_ms=2.0)
    assert "Passed" in section(min_ms_out.read_text(), "Gate")

    # Passes when --fail-above is above the largest regression.
    high_out = tmp_path / "gate-high.md"
    assert not diff.report(fast, slow, ceiling=252, min_ms=0.05, tolerance=0.03,
                           out=str(high_out), fail_above=100, fail_min_ms=0.5)
    assert "Passed" in section(high_out.read_text(), "Gate")

    # Speedups never trigger the gate.
    faster_out = tmp_path / "gate-faster.md"
    assert not diff.report(slow, fast, ceiling=252, min_ms=0.05, tolerance=0.03,
                           out=str(faster_out), fail_above=5, fail_min_ms=0.5)
    assert "Passed" in section(faster_out.read_text(), "Gate")


def test_diff_gate_overall_step_slowdown(captures, tmp_path):
    """A uniform 10% slowdown has 0% beyond-drift per kernel, but triggers the overall step gate."""
    base = captures / "decode-q38"
    slow = tmp_path / "decode-10pct-slower"
    _scale_capture_durations(base, slow, 1.10)

    out = tmp_path / "gate-overall.md"
    failures = diff.report(str(base), str(slow), ceiling=252, min_ms=0.05, tolerance=0.03,
                           out=str(out), fail_above=5, fail_min_ms=0.5)
    assert len(failures) == 1
    assert "GPU busy per step" in failures[0]
    gate = section(out.read_text(), "Gate")
    assert "Failed" in gate
    assert "GPU busy per step" in gate


def test_diff_gate_cli_exit_codes(captures, tmp_path):
    """python -m ledger.diff exits 1 when the gate fails and 0 when it passes."""
    fast = str(captures / "decode-q38-bf16state")
    slow = str(captures / "decode-q38")

    fail_res = subprocess.run(
        [sys.executable, "-m", "ledger.diff", fast, slow, "--ceiling-gbps", "252", "--fail-above", "5"],
        capture_output=True, text=True,
    )
    assert fail_res.returncode == 1
    assert "## Gate" in fail_res.stdout
    assert "Failed" in fail_res.stdout

    pass_res = subprocess.run(
        [sys.executable, "-m", "ledger.diff", slow, fast, "--ceiling-gbps", "252", "--fail-above", "5"],
        capture_output=True, text=True,
    )
    assert pass_res.returncode == 0
    assert "## Gate" in pass_res.stdout
    assert "Passed" in pass_res.stdout
