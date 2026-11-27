"""Import Huawei Ascend CANN msprof + aclGraph traces into a Ledger capture directory.

    python -m ledger.ascend <msprof dir or op_summary.csv> <out capture dir> \\
        [--graph capture.json ...] [--weights target_weights.json ...] \\
        [--draft-weights draft_weights.json ...] [--owners owners.json ...] \\
        [--regions regions.json ...] [--stream-map 23:18 ...] \\
        [--device "Ascend 910B1"] [--sms 24] [--steps 1] [--batch 1] [--draft-tokens 0] [--prefill 0]

CANN `msprof --export=on` writes `op_summary_*.csv` (per-task hardware timestamps, stream/task IDs,
block counts) and `task_time_*.csv` (including `MEMCPY_ASYNC` transfers). When an aclGraph debug
dump (`aclmdlRIDebugPrint`) is also provided, launches are joined by `(Stream ID, Task ID)` to
recover the exact 64-bit device pointer arguments (`Kernel Args`) for every AI Core, AI Vector, and
HCCL AIV task. Combining that with named weight and KV-cache maps (such as Sibyl's `kv-census`
descriptors or existing Ledger `regions-*.json`) lets `ledger analyze`, `ledger transfers`,
`ledger diff`, and `ledger perfetto` account for Ascend runs identically to CUDA captures.
"""

import argparse
import csv
import glob
import json
import os
import struct

from ledger import analyze as A
from ledger import sibyl_regions

# CUPTI_ACTIVITY_MEMCPY_KIND_DTOD; used for device async copies when msprof does not record direction.
_MEMCPY_DTOD = 8


def _find_csv(source, prefix):
    """Locate an msprof CSV inside a PROF_* tree, mindstudio_profiler_output, or direct file."""
    if os.path.isfile(source):
        return source if os.path.basename(source).startswith(prefix) else None
    patterns = (
        os.path.join(source, f"{prefix}_*.csv"),
        os.path.join(source, "mindstudio_profiler_output", f"{prefix}_*.csv"),
        os.path.join(source, "PROF_*", "mindstudio_profiler_output", f"{prefix}_*.csv"),
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _parse_us_ns(text):
    """Parse an msprof microsecond timestamp/duration string (which may carry a trailing tab) to ns."""
    text = (text or "").strip()
    if not text or text == "N/A":
        return None
    try:
        return round(float(text) * 1000.0)
    except ValueError:
        return None


def _parse_int(text, default=0):
    text = (text or "").strip()
    if not text or text == "N/A":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_kernel_args(text):
    """Encode space-separated hex words from aclmdlRIDebugPrint `Kernel Args` as Ledger params."""
    words = []
    for tok in (text or "").split():
        try:
            words.append(int(tok, 16) & 0xFFFFFFFFFFFFFFFF)
        except ValueError:
            continue
    if not words:
        return []
    raw = struct.pack(f"<{len(words)}Q", *words)
    return [[0, len(raw), raw.hex()]]


def load_acl_graphs(paths, stream_map=None):
    """Load aclmdlRIDebugPrint JSON dumps into a `(stream_id, task_id) -> node` lookup."""
    by_st = {}
    for path in paths:
        events = json.load(open(path))
        for ev in events:
            if not isinstance(ev, dict):
                continue
            args = ev.get("args") or {}
            if "Stream Id" not in args or "Task Id" not in args:
                continue
            ka = args.get("Kernel Args")
            if not ka:
                continue
            params = parse_kernel_args(ka)
            if not params:
                continue
            stream_id = int(args["Stream Id"])
            task_id = int(args["Task Id"])
            by_st[(stream_id, task_id)] = {
                "name": ev.get("name", ""),
                "blocks": int(args.get("Numblocks") or 1),
                "params": params,
            }
    if stream_map:
        for runtime_stream, graph_stream in stream_map.items():
            for (gs, tid), node in list(by_st.items()):
                if gs == graph_stream and (runtime_stream, tid) not in by_st:
                    by_st[(runtime_stream, tid)] = node
    return by_st


def _region_entry(kind, name, info):
    addr = info.get("address", info.get("addr"))
    size = info.get("bytes", info.get("size", 0))
    if addr is None or not size or size <= 0:
        return None
    dtype = str(info.get("dtype", "")).removeprefix("torch.")
    shape = list(info.get("shape") or [])
    entry = {
        "kind": kind,
        "name": name,
        "addr": int(addr),
        "size": int(size),
        "shape": shape,
        "dtype": dtype,
    }
    if kind == "weight" and len(shape) == 2 and all(isinstance(d, int) and d > 0 for d in shape):
        entry["out_features"] = shape[0]
        entry["in_features"] = shape[1]
    return entry


def load_regions(weight_paths=(), draft_weight_paths=(), owner_paths=(), region_paths=()):
    """Build Ledger regions from Ascend kv-census JSONs and/or existing region files."""
    regions = []
    seen = set()

    def add(entry):
        if entry is None:
            return
        key = (entry["kind"], entry["addr"], entry["size"], entry["name"])
        if key not in seen:
            seen.add(key)
            regions.append(entry)

    for path in weight_paths:
        data = json.load(open(path))
        for name, info in data.items():
            if isinstance(info, dict):
                add(_region_entry("weight", name, info))

    for path in draft_weight_paths:
        data = json.load(open(path))
        for name, info in data.items():
            if isinstance(info, dict):
                prefixed = name if name.startswith(A.DRAFT) else f"draft.{name}"
                add(_region_entry("weight", prefixed, info))

    for path in owner_paths:
        data = json.load(open(path))
        for name, items in data.items():
            if isinstance(items, list):
                for idx, info in enumerate(items):
                    if isinstance(info, dict):
                        label = name if len(items) == 1 else f"{name}[{idx}]"
                        add(_region_entry("cache", label, info))
            elif isinstance(items, dict):
                add(_region_entry("cache", name, items))

    for path in region_paths:
        data = json.load(open(path))
        if isinstance(data, list) and data and isinstance(data[0], list):
            for entry in sibyl_regions.convert(data):
                add(entry)
        elif isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    add(entry)

    return regions


def _load_op_summary(csv_path):
    """Read msprof op_summary CSV, deduplicating paired `Stream ID == N/A` HCCL summary rows.

    CANN msprof emits a high-level `Stream ID = N/A` COMMUNICATION row for every collective.
    For AIV-expanded collectives that row duplicates a device `AivKernel` row with the same
    timestamp and duration; for SDMA/RDMA collectives only the `N/A` row exists. We enrich
    paired `AivKernel` rows with the collective name and retain unmatched `N/A` collectives on
    a dedicated virtual stream (-1) so they are neither double-counted nor silently dropped.
    """
    raw_rows = list(csv.DictReader(open(csv_path, newline="")))
    na_rows = {}
    device_rows = []
    for idx, row in enumerate(raw_rows):
        stream_text = (row.get("Stream ID") or "").strip()
        start_ns = _parse_us_ns(row.get("Task Start Time(us)"))
        dur_ns = _parse_us_ns(row.get("Task Duration(us)"))
        if start_ns is None or dur_ns is None or dur_ns <= 0:
            continue
        model_id = (row.get("Model ID") or "").strip()
        if stream_text in ("", "N/A"):
            key = (model_id, start_ns // 1000, dur_ns // 1000)
            na_rows.setdefault(key, []).append((idx, row, start_ns, dur_ns))
            continue
        device_rows.append((row, start_ns, dur_ns))

    out = []
    matched_na = set()
    for row, start_ns, dur_ns in device_rows:
        model_id = (row.get("Model ID") or "").strip()
        stream_id = _parse_int(row.get("Stream ID"))
        task_id = _parse_int(row.get("Task ID"))
        op_name = (row.get("Op Name") or "").strip()
        task_type = (row.get("Task Type") or "").strip()
        if op_name in ("", "AivKernel", "N/A"):
            key = (model_id, start_ns // 1000, dur_ns // 1000)
            bucket = na_rows.get(key)
            if bucket:
                na_idx, na_row, _, _ = bucket.pop(0)
                matched_na.add(na_idx)
                enriched = (na_row.get("Op Name") or "").strip()
                if enriched:
                    op_name = enriched
        block_num = _parse_int(row.get("Block Num"), 0)
        mix_blocks = _parse_int(row.get("Mix Block Num"), 0)
        blocks = max(block_num, mix_blocks, 1)
        out.append({
            "model_id": model_id,
            "stream": stream_id,
            "task": task_id,
            "name": op_name or task_type or "unknown",
            "task_type": task_type,
            "start": start_ns,
            "end": start_ns + dur_ns,
            "blocks": blocks,
        })

    for bucket in na_rows.values():
        for na_idx, row, start_ns, dur_ns in bucket:
            if na_idx in matched_na:
                continue
            op_name = (row.get("Op Name") or "").strip()
            task_type = (row.get("Task Type") or "").strip()
            out.append({
                "model_id": (row.get("Model ID") or "").strip(),
                "stream": -1,
                "task": 0,
                "name": op_name or task_type or "communication",
                "task_type": task_type,
                "start": start_ns,
                "end": start_ns + dur_ns,
                "blocks": max(_parse_int(row.get("Block Num"), 0), 1),
            })

    out.sort(key=lambda r: r["start"])
    return out


def _load_memcpys(task_time_csv):
    """Extract MEMCPY_ASYNC rows from msprof task_time_*.csv as Ledger memcpy records."""
    if not task_time_csv or not os.path.isfile(task_time_csv):
        return []
    memcpys = []
    for idx, row in enumerate(csv.DictReader(open(task_time_csv, newline="")), start=1):
        if (row.get("kernel_type") or "").strip() != "MEMCPY_ASYNC":
            continue
        start_ns = _parse_us_ns(row.get("task_start(us)"))
        stop_ns = _parse_us_ns(row.get("task_stop(us)"))
        dur_ns = _parse_us_ns(row.get("task_time(us)"))
        if start_ns is None:
            continue
        if stop_ns is None or stop_ns <= start_ns:
            if dur_ns is None or dur_ns <= 0:
                continue
            stop_ns = start_ns + dur_ns
        memcpys.append({
            "start": start_ns,
            "end": stop_ns,
            "bytes": 0,
            "kind": _MEMCPY_DTOD,
            "src_kind": 3,
            "dst_kind": 3,
            "stream": _parse_int(row.get("stream_id")),
            "node": 0,
            "corr": idx,
            "rcorr": 0,
        })
    return memcpys


def import_capture(
    source,
    out_dir,
    graph_paths=(),
    weight_paths=(),
    draft_weight_paths=(),
    owner_paths=(),
    region_paths=(),
    stream_map=None,
    device="Ascend 910B",
    sms=24,
    steps=1,
    batch=1,
    draft_tokens=0,
    prefill=0,
):
    """Convert an Ascend msprof export into a standard Ledger capture directory."""
    op_csv = _find_csv(source, "op_summary")
    if not op_csv:
        raise SystemExit(f"ledger ascend-import: no op_summary_*.csv found under {source}")
    task_csv = _find_csv(source, "task_time") if not os.path.isfile(source) else None

    ops = _load_op_summary(op_csv)
    graph_nodes = load_acl_graphs(graph_paths, stream_map=stream_map)
    regions = load_regions(weight_paths, draft_weight_paths, owner_paths, region_paths)
    memcpys = _load_memcpys(task_csv)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "regions-ascend.json"), "w") as f:
        json.dump(regions, f)

    node_ids = {}
    next_node_id = 1
    nodes_out = []
    kernels_out = []
    matched = 0

    for corr, op in enumerate(ops, start=1):
        key = (op["stream"], op["task"])
        gnode = graph_nodes.get(key)
        node_id = 0
        blocks = op["blocks"]
        name = op["name"]
        if gnode is not None:
            matched += 1
            if key not in node_ids:
                node_ids[key] = next_node_id
                nodes_out.append({"id": next_node_id, "params": gnode["params"]})
                next_node_id += 1
            node_id = node_ids[key]
            if gnode["blocks"] > 0:
                blocks = gnode["blocks"]
            if name == "AivKernel" or name.startswith("hcom_"):
                # Keep collective semantic prefix alongside AIV kernel name when both exist.
                name = f"{name}:{gnode['name']}" if name.startswith("hcom_") else gnode["name"]
        kernels_out.append({
            "start": op["start"],
            "end": op["end"],
            "name": name,
            "grid": [blocks, 1, 1],
            "stream": op["stream"],
            "node": node_id,
            "corr": corr,
        })

    with open(os.path.join(out_dir, "nodes-0.jsonl"), "w") as f:
        for rec in nodes_out:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(out_dir, "kernels-0.jsonl"), "w") as f:
        for rec in kernels_out:
            f.write(json.dumps(rec) + "\n")
    if memcpys:
        with open(os.path.join(out_dir, "memcpys-0.jsonl"), "w") as f:
            for rec in memcpys:
                f.write(json.dumps(rec) + "\n")

    run_meta = {
        "device": device,
        "backend": "ascend",
        "steps": steps,
        "batch": batch,
        "draft_tokens": draft_tokens,
        "prefill": prefill,
        "sms": sms,
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
        f.write("\n")

    return {
        "kernels": len(kernels_out),
        "matched_nodes": matched,
        "regions": len(regions),
        "memcpys": len(memcpys),
    }


def _parse_stream_maps(specs):
    mapping = {}
    for spec in specs or ():
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair:
                continue
            src, dst = pair.split(":", 1)
            mapping[int(src)] = int(dst)
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="msprof PROF_* directory, mindstudio_profiler_output dir, or op_summary_*.csv")
    ap.add_argument("out", help="output Ledger capture directory")
    ap.add_argument("--graph", action="append", default=[], help="aclmdlRIDebugPrint JSON dump (repeatable)")
    ap.add_argument("--stream-map", action="append", default=[],
                    help="map runtime stream IDs to graph dump stream IDs, e.g. 23:18,24:19")
    ap.add_argument("--weights", action="append", default=[], help="Ascend target_weights.json descriptor map")
    ap.add_argument("--draft-weights", action="append", default=[], help="Ascend draft_weights.json descriptor map")
    ap.add_argument("--owners", action="append", default=[], help="Ascend KV/state owners.json descriptor map")
    ap.add_argument("--regions", action="append", default=[], help="existing Ledger regions.json or Sibyl ranges.json")
    ap.add_argument("--device", default="Ascend 910B", help="Ascend device name written to run.json")
    ap.add_argument("--sms", type=int, default=24, help="AI Core count for underfilled-grid accounting (default: 24)")
    ap.add_argument("--steps", type=int, default=1, help="steps in the profiled window (default: 1)")
    ap.add_argument("--batch", type=int, default=1, help="batch size in run.json")
    ap.add_argument("--draft-tokens", type=int, default=0, help="speculative draft tokens per request")
    ap.add_argument("--prefill", type=int, default=0, help="prefill token count (0 for decode)")
    a = ap.parse_args()
    summary = import_capture(
        a.source,
        a.out,
        graph_paths=a.graph,
        weight_paths=a.weights,
        draft_weight_paths=a.draft_weights,
        owner_paths=a.owners,
        region_paths=a.regions,
        stream_map=_parse_stream_maps(a.stream_map),
        device=a.device,
        sms=a.sms,
        steps=a.steps,
        batch=a.batch,
        draft_tokens=a.draft_tokens,
        prefill=a.prefill,
    )
    print(
        f"{summary['kernels']} kernels ({summary['matched_nodes']} with aclGraph params), "
        f"{summary['regions']} regions, {summary['memcpys']} memcpys -> {a.out}"
    )


if __name__ == "__main__":
    main()
