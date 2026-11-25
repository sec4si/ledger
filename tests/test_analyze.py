"""Tests for ledger.analyze: Memory lookup and report sections on real captures."""

import re

import pytest

from ledger import analyze
from report import section, table_rows


def test_memory_prefers_weight_over_segment_at_same_start():
    """When a segment and a weight share a start address, addresses inside the weight resolve to it
    and addresses elsewhere in the segment fall back to the segment."""
    base = 0x7000_0000_0000
    weight = {"kind": "weight", "name": "layer.0.weight", "addr": base, "size": 4096}
    segment = {"kind": "segment", "name": "segment", "addr": base, "size": 65536}

    for regions in ([segment, weight], [weight, segment]):
        mem = analyze.Memory(regions)
        for addr in (base, base + 128, base + 4095):
            idx = mem.find(addr)
            assert idx is not None
            assert mem.regions[idx]["kind"] == "weight"
            assert mem.regions[idx]["name"] == "layer.0.weight"

        for addr in (base + 4096, base + 10000, base + 65535):
            idx = mem.find(addr)
            assert idx is not None
            assert mem.regions[idx]["kind"] == "segment"

        assert mem.find(base - 1) is None
        assert mem.find(base + 65536) is None


def test_analyze_decode_q38(captures, tmp_path):
    """Verify step summary, speed-split diagnosis, repeated reads and underfilled grids on decode-q38."""
    out = tmp_path / "decode-q38.md"
    analyze.report(str(captures / "decode-q38"), ceiling=252, steps=3, out=str(out))
    text = out.read_text()

    # Step summary: weight bytes and effective bandwidth.
    step_rows = dict(table_rows(section(text, "Step")))
    gb = float(re.search(r"([\d.]+)\s*GB", step_rows["weight bytes streamed"]).group(1))
    gbps = float(re.search(r"([\d.]+)\s*GB/s", step_rows["effective bandwidth"]).group(1))
    assert gb == pytest.approx(25.47, abs=0.2)
    assert gbps == pytest.approx(212, abs=5)

    # Same kernel, same shape, different speed: GDN out_proj slower than self_attn.o_proj,
    # preceded by gdn_decode_post_conv_mtp.
    split = section(text, "Same kernel, same shape, different speed")
    m_out = re.search(r"`[^`]*linear_attn\.out_proj\.weight`: median \*\*([\d.]+) us\*\*", split)
    m_attn = re.search(r"`[^`]*self_attn\.o_proj\.weight`: median ([\d.]+) us", split)
    assert m_out and m_attn
    assert float(m_out.group(1)) > float(m_attn.group(1))
    assert re.search(
        r"last kernel touching caches before `[^`]*linear_attn\.out_proj\.weight`: `[^`]*gdn_decode_post_conv_mtp",
        split,
    )

    # Repeated reads: lm_head read 4 times per step.
    repeated = {r[0].strip("`"): r for r in table_rows(section(text, "Weights read more than once per step"))}
    assert "language_model.lm_head.weight" in repeated
    assert float(repeated["language_model.lm_head.weight"][3]) == pytest.approx(4.0, abs=0.1)

    # Underfilled grids: NVFP4 down_proj runs with 40 blocks (< 48 SMs).
    under = table_rows(section(text, "Launches with fewer blocks than SMs (48)"))
    down_proj = [r for r in under if "mlp.down_proj.weight_packed" in r[7]]
    assert down_proj, "expected NVFP4 down_proj in underfilled launches"
    assert int(down_proj[0][2]) == 40
    assert int(down_proj[0][2]) < 48


def test_analyze_prefill_q38_compute_bounds(captures, tmp_path):
    """With bf16 and fp8 peaks on prefill-q38, FP8 GEMMs are compute-bound, lm_head is memory-bound
    and FP4 has no peak."""
    out = tmp_path / "prefill-q38.md"
    peaks = analyze.parse_peaks("bf16=91,fp8=181")
    analyze.report(str(captures / "prefill-q38"), ceiling=252, steps=1, out=str(out), peaks=peaks)
    text = out.read_text()

    rows = table_rows(section(text, "Compute"))
    assert rows

    fp8_gemms = [r for r in rows if r[1] == "fp8" and "lm_head" not in r[0]]
    lm_head = [r for r in rows if "lm_head" in r[0]]
    fp4_rows = [r for r in rows if r[1] == "fp4"]

    assert fp8_gemms and all(r[8] == "compute" for r in fp8_gemms)
    assert len(lm_head) == 1 and lm_head[0][8] == "memory"
    assert fp4_rows and all(r[8] == "no fp4 peak" for r in fp4_rows)
