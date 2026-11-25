"""Export a Ledger capture to Chrome Trace Event JSON for Perfetto or chrome://tracing.

    python -m ledger.perfetto <capture dir> [--ceiling-gbps 252] --out trace.json

Each CUDA stream gets its own track (splitting into extra lanes only when concurrent graph branches
on the same stream overlap in time, so timestamps stay exact and Perfetto reports no slice errors).
Kernel events are named by the short kernel name plus the folded weight group they stream. When
step markers are present, a "Steps" track shows each step span. When memcpys-*.jsonl is present,
copies appear on a "Memcpy" track.

Without --ceiling-gbps, per-launch GB/s and % ceiling are omitted and partial-read filtering
(such as embedding gathers) is disabled.
"""

import argparse
import glob
import json
import os

from ledger import analyze as A
from ledger import transfers as T


def _us(ns, t0):
    return round((ns - t0) / 1e3, 3)


def _meta(tid, name, sort_index):
    return [
        {"ph": "M", "pid": 1, "tid": tid, "name": "thread_name", "args": {"name": name}},
        {"ph": "M", "pid": 1, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": sort_index}},
    ]


def _assign_lanes(items, start_key, end_key):
    """Greedy interval coloring so no two events on the same track partially overlap."""
    lane_ends = []
    out = []
    for item in items:
        s, e = item[start_key], item[end_key]
        lane = next((i for i, end in enumerate(lane_ends) if end <= s), None)
        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(e)
        else:
            lane_ends[lane] = e
        out.append((lane, item))
    return len(lane_ends), out


def _step_events(steps, t0, next_tid, sort_idx):
    """Build metadata and complete events for the Steps track."""
    tid = next_tid
    events = _meta(tid, "Steps", sort_idx)
    for i, step in enumerate(steps):
        if not step:
            continue
        s_start = step[0]["start"]
        s_end = max(r["end"] for r in step)
        span, busy = A.busy_and_gaps(step)
        events.append({
            "ph": "X",
            "pid": 1,
            "tid": tid,
            "cat": "step",
            "name": f"Step {i}",
            "ts": _us(s_start, t0),
            "dur": round((s_end - s_start) / 1e3, 3),
            "args": {
                "launches": len(step),
                "busy_us": round(busy / 1e3, 1),
                "idle_us": round((span - busy) / 1e3, 1),
                "weight_bytes": sum(r["weight_bytes"] for r in step),
            },
        })
    return events, next_tid + 1, sort_idx + 1


def _kernel_events(kernel_rows, pretty, t0, ceiling, has_steps, next_tid, sort_idx):
    """Build per-stream tracks and kernel events, splitting concurrent overlaps into extra lanes."""
    events = []
    by_stream = {}
    for r in kernel_rows:
        by_stream.setdefault(r["stream"], []).append(r)

    for s in sorted(by_stream):
        n_lanes, laned = _assign_lanes(by_stream[s], "start", "end")
        lane_tids = []
        for lane in range(n_lanes):
            tid = next_tid
            next_tid += 1
            lane_tids.append(tid)
            tname = f"Stream {s}" if lane == 0 else f"Stream {s} (lane {lane + 1})"
            events += _meta(tid, tname, sort_idx)
            sort_idx += 1
        for lane, r in laned:
            kname = A.short(pretty[r["name"]], 60)
            w = r["weights"][0] if r["weights"] else None
            name = f"{kname} [{A.group_name(w['name'])}]" if w else kname
            args = {"grid": list(r["grid"])}
            if w:
                args["weight"] = w["name"]
                args["bytes"] = r["weight_bytes"]
                if ceiling is not None and r["us"] > 0:
                    gbps = r["weight_bytes"] / r["us"] / 1e3
                    args["gbps"] = round(gbps, 1)
                    args["pct_ceiling"] = round(gbps / ceiling * 100.0, 1)
            if r["caches"]:
                args["caches"] = list(dict.fromkeys(r["caches"]))
            if has_steps:
                args["draft"] = bool(r.get("draft"))
            cat = "draft" if r.get("draft") else ("weight" if w else "kernel")
            events.append({
                "ph": "X",
                "pid": 1,
                "tid": lane_tids[lane],
                "cat": cat,
                "name": name,
                "ts": _us(r["start"], t0),
                "dur": round(r["us"], 3),
                "args": args,
            })
    return events, next_tid, sort_idx


def _copy_events(copies, t0, next_tid, sort_idx):
    """Build Memcpy track(s) and memory copy events."""
    events = []
    n_lanes, laned = _assign_lanes(copies, "start", "end")
    copy_tids = []
    for lane in range(n_lanes):
        tid = next_tid
        next_tid += 1
        copy_tids.append(tid)
        tname = "Memcpy" if lane == 0 else f"Memcpy (lane {lane + 1})"
        events += _meta(tid, tname, sort_idx)
        sort_idx += 1
    for lane, c in laned:
        cname = f"{c['kind']} [{c['region']}]" if c["region"] else c["kind"]
        args = {
            "kind": c["kind"],
            "bytes": c["bytes"],
            "gbps": round(T.gbps(c["bytes"], c["ns"]), 2),
            "exposed_us": round(c["exposed_ns"] / 1e3, 3),
            "graph": c["graph"],
        }
        if c["region"]:
            args["region"] = c["region"]
        if c["host"]:
            args["host"] = c["host"]
        events.append({
            "ph": "X",
            "pid": 1,
            "tid": copy_tids[lane],
            "cat": "memcpy",
            "name": cname,
            "ts": _us(c["start"], t0),
            "dur": round(c["ns"] / 1e3, 3),
            "args": args,
        })
    return events, next_tid, sort_idx


def export(capture, ceiling, out):
    """Build Chrome Trace Event JSON from a capture directory and write it to `out`."""
    rows, _unresolved, _n_regions = A.build(capture, ceiling if ceiling is not None else float("inf"))
    if not rows:
        raise SystemExit("no kernel records in range")

    _markers, steps = A.split_steps(rows)
    has_steps = steps is not None
    kernel_rows = [r for r in rows if A.MARKER not in r["name"]]
    pretty = A.demangle([r["name"] for r in kernel_rows])

    has_memcpys = bool(glob.glob(os.path.join(capture, "memcpys-*.jsonl")))
    copies = T.build(capture)[0] if has_memcpys else []
    for c in copies:
        c["end"] = c["start"] + c["ns"]

    starts = [r["start"] for r in kernel_rows] + [c["start"] for c in copies]
    if not starts:
        raise SystemExit("no kernel or memcpy records in range")
    t0 = min(starts)

    label = os.path.basename(os.path.abspath(capture))
    # Avoid tid 0 and tid == pid (1), which Perfetto treats as the process main thread.
    events = [{"ph": "M", "pid": 1, "tid": 1, "name": "process_name", "args": {"name": label}}]
    next_tid, sort_idx = 2, 0

    if has_steps:
        ev, next_tid, sort_idx = _step_events(steps, t0, next_tid, sort_idx)
        events += ev
    ev, next_tid, sort_idx = _kernel_events(kernel_rows, pretty, t0, ceiling, has_steps, next_tid, sort_idx)
    events += ev
    if copies:
        ev, next_tid, sort_idx = _copy_events(copies, t0, next_tid, sort_idx)
        events += ev

    with open(out, "w") as f:
        json.dump({"displayTimeUnit": "ns", "traceEvents": events}, f, separators=(",", ":"))
    return len(kernel_rows), len(copies), len(steps) if has_steps else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="capture directory written under LEDGER_OUT")
    ap.add_argument(
        "--ceiling-gbps",
        type=float,
        default=None,
        help="measured DRAM read bandwidth (GB/s); without it, GB/s, %% ceiling and partial-read filtering are omitted",
    )
    ap.add_argument("--out", required=True, help="output Chrome Trace Event JSON path")
    a = ap.parse_args()
    n_k, n_c, n_s = export(a.capture, a.ceiling_gbps, a.out)
    print(f"wrote {a.out}: {n_k} kernels, {n_c} copies, {n_s} steps")


if __name__ == "__main__":
    main()
