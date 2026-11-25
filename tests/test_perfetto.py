"""Tests for ledger.perfetto Chrome Trace Event export."""

from collections import defaultdict
import json

import pytest

from ledger import analyze, perfetto


@pytest.mark.parametrize(
    ("capture_name", "expected_steps"),
    [("decode-q38", 3), ("prefill-q38", 1)],
)
def test_perfetto_export_structure_and_lanes(captures, tmp_path, capture_name, expected_steps):
    """Verify valid JSON, kernel count, non-overlapping lanes, Steps/Memcpy tracks and bandwidth args."""
    capture = str(captures / capture_name)
    out = tmp_path / f"{capture_name}.json"

    n_kernels, n_copies, n_steps = perfetto.export(capture, ceiling=252, out=str(out))
    data = json.loads(out.read_text())
    events = data["traceEvents"]

    # Kernel count matches launches after removing step markers.
    rows, _, _ = analyze.build(capture, 252)
    expected_kernels = sum(analyze.MARKER not in r["name"] for r in rows)
    kernel_events = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "weight", "draft")]
    assert n_kernels == expected_kernels
    assert len(kernel_events) == expected_kernels

    # Events on the same tid never overlap.
    by_tid = defaultdict(list)
    for e in events:
        if e.get("ph") == "X":
            by_tid[e["tid"]].append((e["ts"], e["ts"] + e["dur"]))
    for tid, spans in by_tid.items():
        spans.sort()
        for (_s1, e1), (s2, _e2) in zip(spans, spans[1:]):
            assert e1 <= s2, f"overlapping events on tid {tid}: {e1} > {s2}"

    # Track names by tid.
    track_names = {
        e["tid"]: e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }

    # Steps track matches expected step count.
    step_tids = [tid for tid, name in track_names.items() if name == "Steps"]
    assert len(step_tids) == 1
    step_events = [e for e in events if e.get("ph") == "X" and e["tid"] == step_tids[0]]
    assert n_steps == expected_steps
    assert len(step_events) == expected_steps

    # Weight-streaming events carry gbps and pct_ceiling when ceiling is supplied.
    weight_events = [e for e in kernel_events if "weight" in e["args"] and e["dur"] > 0]
    assert weight_events
    assert all("gbps" in e["args"] and "pct_ceiling" in e["args"] for e in weight_events)

    # Memcpy track exists and holds copy events.
    memcpy_tids = {tid for tid, name in track_names.items() if name.startswith("Memcpy")}
    assert memcpy_tids
    memcpy_events = [e for e in events if e.get("ph") == "X" and e["tid"] in memcpy_tids]
    assert n_copies > 0
    assert len(memcpy_events) == n_copies


def test_perfetto_without_ceiling_omits_bandwidth_args(captures, tmp_path):
    """Without a bandwidth ceiling, weight events omit gbps and pct_ceiling."""
    out = tmp_path / "decode-no-ceiling.json"
    perfetto.export(str(captures / "decode-q38"), ceiling=None, out=str(out))
    events = json.loads(out.read_text())["traceEvents"]
    weight_events = [e for e in events if e.get("ph") == "X" and "weight" in e.get("args", {})]
    assert weight_events
    assert all("gbps" not in e["args"] and "pct_ceiling" not in e["args"] for e in weight_events)
