"""Tests for ledger.steps on decode, long-context and single-step prefill captures."""

import re

import pytest

from ledger import steps
from report import section, table_rows


def test_steps_decode_q38_speculative(captures, tmp_path):
    """Verify speculative decoding metrics on decode-q38."""
    out = tmp_path / "steps-decode.md"
    steps.report(str(captures / "decode-q38"), ceiling=252, out=str(out))
    text = out.read_text()

    spec = dict(table_rows(section(text, "Speculative decoding")))
    tokens = float(spec["tokens gained (all requests)"])
    draft_ms = float(re.search(r"([\d.]+)\s*ms", spec["draft steps, GPU busy"]).group(1))
    assert tokens == pytest.approx(4.0, abs=0.1)
    assert draft_ms > 0


def test_steps_longctx_q38_attention(captures, tmp_path):
    """Verify attention section on longctx-q38: main model launches 16 times/step and reads context."""
    out = tmp_path / "steps-longctx.md"
    steps.report(str(captures / "longctx-q38"), ceiling=252, out=str(out))
    text = out.read_text()

    attn_section = section(text, "Attention over the KV cache")
    rows = table_rows(attn_section)
    main_reads = [
        r for r in rows
        if r[1] == "main" and "(not a context read)" not in r[6]
    ]
    assert main_reads, "expected at least one main-model context-reading attention kernel"
    assert any(float(r[2]) == pytest.approx(16.0) for r in main_reads)
    assert "Context reads:" in attn_section


def test_steps_prefill_q38_single_step(captures, tmp_path):
    """Single-step prefill capture completes without crashing."""
    out = tmp_path / "steps-prefill.md"
    steps.report(str(captures / "prefill-q38"), ceiling=252, out=str(out))
    text = out.read_text()

    assert "1 steps," in text
    timing = dict(table_rows(section(text, "Time per step")))
    assert float(timing["GPU busy"]) > 0
