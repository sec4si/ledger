"""Tests for ledger.analyze: Memory lookup, window_steps and report sections on real captures."""

import json
import re
import shutil

import pytest

from ledger import analyze
from report import section, table_rows


def _findings(text):
    """Split the Findings section into (ranked, notes) where ranked is [(ms, body), ...]."""
    ranked, notes = [], []
    for line in section(text, "Findings").splitlines():
        if not line.startswith("- "):
            continue
        m = re.match(r"^- \*\*([\d.]+) ms per step\*\*: (.*)$", line)
        if m:
            ranked.append((float(m.group(1)), m.group(2)))
        else:
            notes.append(line[2:])
    return ranked, notes


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


def test_analyze_findings_decode_q38(captures, tmp_path):
    """Findings on decode-q38 rank lm_head first, include GDN out_proj after gdn_decode_post_conv_mtp,
    exclude near-ceiling NVFP4 down_proj, and stay sorted with at most 5 ranked items."""
    out = tmp_path / "findings-decode.md"
    analyze.report(str(captures / "decode-q38"), ceiling=252, steps=3, out=str(out))
    text = out.read_text()

    ranked, _notes = _findings(text)
    assert 1 <= len(ranked) <= 5
    ms_values = [ms for ms, _ in ranked]
    assert ms_values == sorted(ms_values, reverse=True)

    # First finding is lm_head repeated reads.
    assert "language_model.lm_head.weight" in ranked[0][1]
    assert "Weights read more than once" in ranked[0][1]

    # GDN out_proj same-kernel finding names gdn_decode_post_conv_mtp.
    same_kernel = [
        body for _, body in ranked
        if "linear_attn.out_proj.weight" in body and "same kernel" in body.lower()
    ]
    assert same_kernel
    assert "gdn_decode_post_conv_mtp" in same_kernel[0]

    # NVFP4 down_proj runs near the bandwidth ceiling despite 40 blocks, so it is omitted.
    assert all("down_proj.weight_packed" not in body for _, body in ranked)


def test_analyze_findings_prefill_q38_without_peaks(captures, tmp_path):
    """Without --peak-tflops on prefill-q38, GEMMs are not flagged as bandwidth-short and a note
    asks for --peak-tflops."""
    out = tmp_path / "findings-prefill-no-peaks.md"
    analyze.report(str(captures / "prefill-q38"), ceiling=252, steps=1, out=str(out))
    text = out.read_text()

    ranked, notes = _findings(text)
    assert len(ranked) <= 5
    ms_values = [ms for ms, _ in ranked]
    assert ms_values == sorted(ms_values, reverse=True)
    assert all("bandwidth floor" not in body for _, body in ranked)
    assert any(
        re.search(r"\d+ weight groups multiply more than 64 rows; pass --peak-tflops", n)
        for n in notes
    )


def test_analyze_findings_prefill_q38_with_peaks(captures, tmp_path):
    """With bf16 and fp8 peaks on prefill-q38, FP8 mlp.down_proj.weight is flagged against its
    compute floor at ~65%."""
    out = tmp_path / "findings-prefill-peaks.md"
    peaks = {"bf16": 91, "fp8": 181}
    analyze.report(str(captures / "prefill-q38"), ceiling=252, steps=1, out=str(out), peaks=peaks)
    text = out.read_text()

    ranked, _notes = _findings(text)
    assert 1 <= len(ranked) <= 5
    ms_values = [ms for ms, _ in ranked]
    assert ms_values == sorted(ms_values, reverse=True)

    down_proj = [body for _, body in ranked if "`language_model.model.layers.N.mlp.down_proj.weight`" in body]
    assert down_proj
    m = re.search(r"runs at (\d+)% of its compute floor", down_proj[0])
    assert m
    assert int(m.group(1)) == pytest.approx(65, abs=2)


def test_window_steps_falls_back_to_markers_and_errors_without_both(captures, tmp_path):
    """window_steps counts step markers when run.json lacks 'steps', and exits with --steps hint
    when neither is present."""
    dst = tmp_path / "decode-no-steps"
    shutil.copytree(captures / "decode-q38", dst)
    run_path = dst / "run.json"
    run = json.loads(run_path.read_text())
    run.pop("steps", None)
    run_path.write_text(json.dumps(run))

    assert analyze.window_steps(str(dst)) == 3

    no_markers = tmp_path / "no-markers"
    no_markers.mkdir()
    (no_markers / "run.json").write_text(json.dumps({"batch": 1}))
    (no_markers / "kernels-1.jsonl").write_text(
        json.dumps({"start": 0, "end": 1000, "name": "some_gemm", "grid": [1, 1, 1], "node": 0, "corr": 1}) + "\n"
    )
    with pytest.raises(SystemExit) as exc:
        analyze.window_steps(str(no_markers))
    assert "--steps" in str(exc.value)


def test_analyze_drops_untimed_kernels_and_markers(captures, tmp_path, capsys):
    """Kernels with zero timestamps (still running when the window closed) are dropped on load with
    a stderr notice, do not skew wall time, and do not count as step markers."""
    base_out = tmp_path / "base.md"
    analyze.report(str(captures / "decode-q38"), ceiling=252, steps=3, out=str(base_out))
    base_wall = dict(table_rows(section(base_out.read_text(), "Step")))["wall time (first launch to last)"]

    dst = tmp_path / "decode-untimed"
    shutil.copytree(captures / "decode-q38", dst)
    run_path = dst / "run.json"
    run = json.loads(run_path.read_text())
    run.pop("steps", None)
    run_path.write_text(json.dumps(run))

    kernels_path = next(dst.glob("kernels-*.jsonl"))
    untimed = [
        {"start": 0, "end": 0, "name": "spin_kernel", "grid": [1, 1, 1], "node": 0, "corr": 999901},
        {"start": 0, "end": 0, "name": "in_flight_gemm_1", "grid": [1, 1, 1], "node": 0, "corr": 999902},
        {"start": 0, "end": 0, "name": "in_flight_gemm_2", "grid": [1, 1, 1], "node": 0, "corr": 999903},
    ]
    with kernels_path.open("a") as f:
        for rec in untimed:
            f.write(json.dumps(rec) + "\n")

    steps = analyze.window_steps(str(dst))
    assert steps == 3

    capsys.readouterr()
    out = tmp_path / "untimed.md"
    analyze.report(str(dst), ceiling=252, steps=steps, out=str(out))
    err = capsys.readouterr().err
    text = out.read_text()

    assert "over 3 steps," in text
    assert dict(table_rows(section(text, "Step")))["wall time (first launch to last)"] == base_wall
    assert re.search(r"dropped 3 of \d+ kernel records without timestamps", err)
