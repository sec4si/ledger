"""Tests for ledger.cli CPU paths and cached GPU ceiling resolution."""

import json
import sys

import pytest

from ledger import ceilings, cli
from report import section


def _write_cache(ceiling_gbps=252, peaks=None):
    """Write a synthetic ceiling cache JSON for NVIDIA GB10 under the current XDG_CACHE_HOME."""
    path = ceilings.cache_path("NVIDIA GB10")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "device": "NVIDIA GB10",
        "ceiling_gbps": ceiling_gbps,
        "peak_tflops": peaks or {"bf16": 91, "fp8": 181},
    }
    path.write_text(json.dumps(data))
    return path


def _run_cli(monkeypatch, *args):
    """Invoke ledger.cli.main() with the given CLI arguments."""
    monkeypatch.setattr(sys, "argv", ["ledger", *args])
    cli.main()


def test_cli_analyze_uses_cached_ceilings_and_peaks(captures, tmp_path, monkeypatch):
    """Without --ceiling-gbps or --peak-tflops, ledger analyze fills both from the cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_cache(ceiling_gbps=252, peaks={"bf16": 91, "fp8": 181})

    out = tmp_path / "analyze.md"
    _run_cli(monkeypatch, "analyze", str(captures / "decode-q38"), "--out", str(out))
    text = out.read_text()

    assert "Bandwidth ceiling 252 GB/s." in text
    assert section(text, "Compute")


def test_cli_explicit_ceiling_overrides_cache(captures, tmp_path, monkeypatch):
    """An explicit --ceiling-gbps takes precedence over the cached value."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_cache(ceiling_gbps=252, peaks={"bf16": 91, "fp8": 181})

    out = tmp_path / "analyze-override.md"
    _run_cli(monkeypatch, "analyze", str(captures / "decode-q38"), "--ceiling-gbps", "300", "--out", str(out))
    text = out.read_text()

    assert "Bandwidth ceiling 300 GB/s." in text


def test_cli_missing_cache_and_ceiling_exits_with_hint(captures, tmp_path, monkeypatch):
    """Without cache or --ceiling-gbps, ledger analyze exits and points to 'ledger ceilings'."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, "analyze", str(captures / "decode-q38"))
    assert "ledger ceilings" in str(exc.value)


def test_cli_steps_and_diff_use_cached_ceiling(captures, tmp_path, monkeypatch):
    """ledger steps and ledger diff also resolve --ceiling-gbps from the cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_cache(ceiling_gbps=252)

    steps_out = tmp_path / "steps.md"
    _run_cli(monkeypatch, "steps", str(captures / "decode-q38"), "--out", str(steps_out))
    assert section(steps_out.read_text(), "Time per step")

    diff_out = tmp_path / "diff.md"
    _run_cli(
        monkeypatch,
        "diff",
        str(captures / "decode-q38"),
        str(captures / "decode-q38-bf16state"),
        "--out",
        str(diff_out),
    )
    assert section(diff_out.read_text(), "Changed beyond drift")


def test_cli_diff_gate_uses_cached_ceiling_and_exits_nonzero(captures, tmp_path, monkeypatch):
    """ledger diff --fail-above resolves cached ceilings and exits non-zero on regression."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _write_cache(ceiling_gbps=252)

    out = tmp_path / "diff-gate.md"
    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            "diff",
            str(captures / "decode-q38-bf16state"),
            str(captures / "decode-q38"),
            "--fail-above",
            "5",
            "--out",
            str(out),
        )
    assert exc.value.code == 1
    assert "Failed" in section(out.read_text(), "Gate")
