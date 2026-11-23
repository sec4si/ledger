"""Ledger transfer accounting: every memory copy in the window, joined with the memory map.

    python -m ledger.transfers <capture dir> [--h2d-gbps 24] [--d2h-gbps 24] [--steps 40] [--out report.md]

Engines that keep part of the model off the GPU (expert offload, KV offload, weight streaming)
pay for it in copies, and a copy costs the step only the part of it no kernel runs alongside.
For each copy this joins the CUPTI timing with the addresses taken from the copy call (or the
memcpy node of a graph), names the region on the device side, and measures how much of the copy
overlaps kernel execution. The report sums bytes, time and exposed time per region group (layer
index folded) and per direction, flags pageable host memory and compares achieved bandwidth with
the ceilings passed in (measure them with bench/bwtest).
"""

import argparse
import bisect
import os
import sys
from collections import defaultdict

from ledger.analyze import group_name, load, read_jsonl

# CUPTI_ACTIVITY_MEMCPY_KIND_* and CUPTI_ACTIVITY_MEMORY_KIND_*
COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH", 10: "PtoP"}
MEMORY_KINDS = {1: "pageable", 2: "pinned", 3: "device", 4: "array", 5: "managed", 6: "device_static", 7: "managed_static"}
DEVICE_SIDES = {"HtoD": ("dst",), "DtoH": ("src",), "DtoD": ("dst", "src"), "PtoP": ("dst", "src")}


class Busy:
    """Union of kernel execution intervals, for asking how much of [start, end) a kernel covers."""

    def __init__(self, kernels):
        merged = []
        for k in sorted(kernels, key=lambda k: k["start"]):
            if merged and k["start"] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], k["end"])
            else:
                merged.append([k["start"], k["end"]])
        self.spans = merged
        self.starts = [s for s, _ in merged]

    def covered(self, start, end):
        total = 0
        i = max(bisect.bisect_right(self.starts, start) - 1, 0)
        for s, e in self.spans[i:]:
            if s >= end:
                break
            total += max(0, min(e, end) - max(s, start))
        return total


def copy_address(m, copies, copy_nodes, clones):
    """Addresses of one timed copy: from the API call that issued it, or from its graph node."""
    if m["node"]:
        node = m["node"]
        for _ in range(4):  # an exec node may be a clone of a clone
            if node in copy_nodes:
                return copy_nodes[node]
            node = clones.get(node)
            if node is None:
                break
        return None
    return copies.get(m["corr"]) or copies.get(m["rcorr"])


def device_region(mem, kind, call):
    """The region the copy fills or drains on the device; for device-to-device the destination."""
    for side in DEVICE_SIDES.get(kind, ()):
        i = mem.find(call[side])
        if i is not None:
            return mem.regions[i]
    return None


def build(capture):
    mem, _nodes, clones, _direct, kernels = load(capture)
    copies = {r["corr"]: r for r in read_jsonl(os.path.join(capture, "copies-*.jsonl"))}
    copy_nodes = {r["id"]: r for r in read_jsonl(os.path.join(capture, "copynodes-*.jsonl"))}
    busy = Busy(kernels)
    rows = []
    for m in sorted(read_jsonl(os.path.join(capture, "memcpys-*.jsonl")), key=lambda m: m["start"]):
        kind = COPY_KINDS.get(m["kind"], str(m["kind"]))
        call = copy_address(m, copies, copy_nodes, clones)
        region = device_region(mem, kind, call) if call else None
        ns = max(m["end"] - m["start"], 1)
        host_side = "src_kind" if kind.startswith("H") else "dst_kind" if kind.endswith("H") else None
        rows.append(
            {
                "start": m["start"],
                "ns": ns,
                "exposed_ns": ns - busy.covered(m["start"], m["end"]),
                "bytes": m["bytes"],
                "kind": kind,
                "host": MEMORY_KINDS.get(m[host_side]) if host_side else None,
                "graph": bool(m["node"]),
                "region": group_name(region["name"]) if region else None,
                "addressed": call is not None,
            }
        )
    return rows, len(mem.regions), bool(kernels)


def gbps(nbytes, ns):
    return nbytes / ns if ns else 0.0  # bytes per ns == GB/s


def report(capture, ceilings, steps, out):
    rows, nregions, have_kernels = build(capture)
    lines = [f"# Transfers: {capture}", ""]
    if not rows:
        lines.append("No copies in the window.")
        print("\n".join(lines), file=out)
        return
    per = steps or 1
    unit = "per step" if steps else "in the window"
    lines.append(
        f"{len(rows) / per:g} copies, {sum(r['bytes'] for r in rows) / per / 1e6:.2f} MB and "
        f"{sum(r['ns'] for r in rows) / per / 1e3:.1f} us {unit}; "
        f"{sum(r['exposed_ns'] for r in rows) / per / 1e3:.1f} us of it with no kernel running."
    )
    if not have_kernels:
        lines.append("(No kernel timings in the capture, so every copy counts as exposed.)")
    unaddressed = sum(not r["addressed"] for r in rows)
    unmapped = sum(r["addressed"] and r["region"] is None for r in rows)
    if unaddressed:
        lines.append(f"{unaddressed} copies came from entry points Ledger does not decode (2D/3D, peer, symbol copies).")
    if unmapped:
        lines.append(f"{unmapped} copies touch no region of the memory map ({nregions} regions).")
    lines.append("")

    lines += ["## By direction", "", "| direction | host memory | copies | MB | us | exposed us | GB/s | of ceiling |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    by_dir = defaultdict(list)
    for r in rows:
        by_dir[(r["kind"], r["host"] or "-")].append(r)
    for (kind, host), rs in sorted(by_dir.items(), key=lambda kv: -sum(r["ns"] for r in kv[1])):
        nbytes, ns = sum(r["bytes"] for r in rs), sum(r["ns"] for r in rs)
        ceiling = ceilings.get(kind)
        share = f"{gbps(nbytes, ns) / ceiling:.0%}" if ceiling else "-"
        lines.append(
            f"| {kind} | {host} | {len(rs) / per:g} | {nbytes / per / 1e6:.2f} | {ns / per / 1e3:.1f} | "
            f"{sum(r['exposed_ns'] for r in rs) / per / 1e3:.1f} | {gbps(nbytes, ns):.1f} | {share} |"
        )
    if any(r["host"] == "pageable" for r in rows):
        lines += ["", "Pageable host memory: such copies return to the host only when done and do not overlap "
                  "kernels the way async copies from pinned memory can. Whether they also run slower depends on "
                  "the machine (bwtest prints both)."]
    lines.append("")

    lines += ["## By region (layer index folded), most exposed time first", "",
              "| region | direction | copies | MB | us | exposed us | GB/s |", "|---|---|---:|---:|---:|---:|---:|"]
    by_region = defaultdict(list)
    for r in rows:
        by_region[(r["region"] or "(unmapped)", r["kind"])].append(r)
    for (region, kind), rs in sorted(by_region.items(), key=lambda kv: -sum(r["exposed_ns"] for r in kv[1]))[:40]:
        nbytes, ns = sum(r["bytes"] for r in rs), sum(r["ns"] for r in rs)
        lines.append(
            f"| `{region}` | {kind} | {len(rs) / per:g} | {nbytes / per / 1e6:.2f} | {ns / per / 1e3:.1f} | "
            f"{sum(r['exposed_ns'] for r in rs) / per / 1e3:.1f} | {gbps(nbytes, ns):.1f} |"
        )
    print("\n".join(lines), file=out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("capture")
    p.add_argument("--h2d-gbps", type=float, help="measured host-to-device ceiling (pinned)")
    p.add_argument("--d2h-gbps", type=float, help="measured device-to-host ceiling (pinned)")
    p.add_argument("--steps", type=int, help="divide totals by this many steps")
    p.add_argument("--out", help="write the report here instead of stdout")
    a = p.parse_args()
    ceilings = {k: v for k, v in (("HtoD", a.h2d_gbps), ("DtoH", a.d2h_gbps)) if v}
    out = open(a.out, "w") if a.out else sys.stdout
    report(a.capture, ceilings, a.steps, out)


if __name__ == "__main__":
    main()
