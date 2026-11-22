"""Follow data through the launches of a step: who wrote what, who read it, and what nobody read.

Input is an access trace (`access/`: for every launch, the 1 KiB granules it read and wrote, with
32-byte sub-block masks, plus the byte ranges of memcpy/memset calls) and a capture of the same
program (`capture/`), whose pointer parameters stand in for the launches the tracer cannot see into
(tensor-map copies). Such a launch is taken to read and write everything its pointers land in: the
named region, or else the whole allocation segment. Memory the host maps for kernels (pinned
segments) is written and read by the CPU unseen and counts as live. Weights are not traced.

Reported per step, over the full steps of the window:
- dead writes: bytes written and then overwritten without anyone reading them in between;
- dead launches: launches whose every written byte is dead;
- hand-offs: bytes one launch writes and only the next launch reads (fusion candidates);
- constant inputs: launches that read nothing written during the window (hoisting candidates);
- independent neighbours: adjacent launches without a read/write conflict (overlap candidates).
Writes of the last step are left unresolved (the window ends before they could be read).

    python -m ledger.dataflow <access dir> --capture <capture dir> --anchor <kernel name>
        [--layout <mem_layout.json>] [--out report.md]
"""

import argparse
import bisect
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from ledger import analyze as A

SUB = 32  # sub-blocks per granule


def popcount(x):
    return bin(x).count("1")


def load_access(path):
    meta = [json.loads(line) for line in open(glob.glob(os.path.join(path, "access-launches-*.jsonl"))[0])]
    info = json.load(open(glob.glob(os.path.join(path, "access-meta-*.json"))[0]))
    ev = np.fromfile(glob.glob(os.path.join(path, "access-[0-9]*.bin"))[0], dtype=np.uint64)
    masks = np.fromfile(glob.glob(os.path.join(path, "access-masks-*.bin"))[0], dtype=np.uint32)
    ok = ev != np.uint64(2**64 - 1)
    ev, masks = ev[ok], masks[ok]
    lid = (ev >> np.uint64(44)).astype(np.int64)
    kind = ((ev >> np.uint64(43)) & np.uint64(1)).astype(np.int64)
    gran = (ev & np.uint64((1 << 43) - 1)).astype(np.int64)
    return meta, info["granule_shift"], lid, kind, gran, masks.astype(np.int64)


def range_events(meta, shift):
    """memcpy/memset ranges as (launch, kind, granule, mask) rows."""
    rows = []
    size = 1 << shift
    sub = size // SUB
    for m in meta:
        for kind, key in ((0, "reads"), (1, "writes")):
            for b, e in m.get(key, []):
                for g in range(b >> shift, ((e - 1) >> shift) + 1):
                    lo, hi = max(b, g << shift) - (g << shift), min(e, (g + 1) << shift) - (g << shift)
                    first, last = lo // sub, (hi - 1) // sub
                    rows.append((m["id"], kind, g, ((1 << (last + 1)) - 1) & ~((1 << first) - 1)))
    return rows


class Containers:
    """Where a pointer lands: a named non-weight region, else an allocation segment."""

    def __init__(self, regions, segments, pinned):
        spans = [(r["addr"], r["addr"] + r["size"], "region:" + r["name"]) for r in regions if r["kind"] != "weight"]
        spans += [(s["addr"], s["addr"] + s["size"], "segment:%x" % s["addr"]) for s in segments]
        self.named = sorted(spans[: len(spans) - len(segments)])
        self.segments = sorted(spans[len(spans) - len(segments) :])
        self.pinned = sorted((s["addr"], s["addr"] + s["size"]) for s in pinned)

    @staticmethod
    def _find(spans, addr):
        i = bisect.bisect_right(spans, (addr, float("inf"), "")) - 1
        return spans[i] if i >= 0 and addr < spans[i][1] else None

    def of(self, addr):
        return self._find(self.named, addr) or self._find(self.segments, addr)

    def all_of(self, addr):
        """Every container holding addr (a region and the segment around it)."""
        return [c for c in (self._find(self.named, addr), self._find(self.segments, addr)) if c]

    def pinned_at(self, addr):
        i = bisect.bisect_right(self.pinned, (addr, float("inf"))) - 1
        return i >= 0 and addr < self.pinned[i][1]


def pointer_words(params):
    for _off, size, hexs in params:
        raw = bytes.fromhex(hexs)
        for o in range(0, size - 7, 8):
            yield int.from_bytes(raw[o : o + 8], "little")


def split_steps(names, anchor):
    steps, cur = [], None
    for i, n in enumerate(names):
        if n == anchor:
            cur = []
            steps.append(cur)
        if cur is not None:
            cur.append(i)
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("access")
    ap.add_argument("--capture", help="capture of the same program; without it opaque launches are ignored")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--layout", help="mem_layout.json of the recording (allocation and pinned segments)")
    ap.add_argument("--ceiling-gbps", type=float, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    meta, shift, lid, kind, gran, mask = load_access(args.access)
    extra = range_events(meta, shift)
    if extra:
        e = np.array(extra, dtype=np.int64)
        lid, kind, gran, mask = (np.concatenate([a, e[:, i]]) for i, a in enumerate((lid, kind, gran, mask)))
    kernel_ids = [m["id"] for m in meta if m["name"] not in ("memcpy", "memset")]
    names = {m["id"]: m["name"] for m in meta}
    pretty = A.demangle(list(set(names.values())))
    label = lambda i: A.short(pretty[names[i]], 70)

    # Steps of the trace and of the capture; positions must match one to one.
    steps = split_steps([names[i] for i in kernel_ids], args.anchor)
    steps = [[kernel_ids[j] for j in s] for s in steps]
    params, durations = [None] * len(steps[0]), [0.0] * len(steps[0])
    if args.capture:
        _mem, nodes, clones, direct, kernels = A.load(args.capture)
        cap_steps = split_steps([k["name"] for k in kernels], args.anchor)
        cap_full = [s for s in cap_steps if len(s) == len(steps[0])]
        if not cap_full or any([names[i] for i in s] != [kernels[j]["name"] for j in cap_full[0]] for s in steps):
            raise SystemExit("trace and capture steps do not line up")
        params = [A.resolve(kernels[j], nodes, clones, direct) for j in cap_full[0]]
        durations = [float(np.mean([(kernels[s[p]]["end"] - kernels[s[p]]["start"]) / 1e3 for s in cap_full]))
                     for p in range(len(steps[0]))]
    position = {i: p for s in steps for p, i in enumerate(s)}
    step_of = {}
    for n, s in enumerate(steps):
        for i in range(s[0], (steps[n + 1][0] if n + 1 < len(steps) else max(names) + 1)):
            step_of[i] = n
    n_steps = len(steps)
    full = n_steps - 1  # the last step's writes stay unresolved

    regions = []
    for path in glob.glob(os.path.join(args.capture or args.access, "regions-*.json")):
        regions += json.load(open(path))
    layout = json.load(open(args.layout)) if args.layout else {"std_dev_segments": [], "host_pinned_segments": []}
    cont = Containers(regions, layout["std_dev_segments"], layout["host_pinned_segments"])

    # Launches the tracer saw nothing of, and what their pointers land in.
    seen = Counter(lid.tolist())
    blind = defaultdict(list)  # container -> launch ids
    blind_launches = [i for i in kernel_ids if not seen[i]]
    for i in blind_launches:
        p = params[position[i]]
        hits = set()
        for v in pointer_words(p["params"] if p else []):
            for c in cont.all_of(v):
                hits.add(c[2])
        for c in hits:
            blind[c].append(i)

    # Per granule, events in launch order; a launch's reads come before its writes.
    order = np.lexsort((kind, lid, gran))
    gran, lid, kind, mask = gran[order], lid[order], kind[order], mask[order]
    bounds = np.flatnonzero(np.diff(gran)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(gran)]])
    sub_bytes = (1 << shift) // SUB

    written = Counter()   # launch -> bytes written
    dead = Counter()      # launch -> bytes overwritten unread
    live = Counter()      # launch -> bytes read later
    open_ = Counter()     # launch -> bytes unresolved at the end of the window
    dead_by_region = Counter()
    handoff = Counter()   # (writer position, reader position) -> bytes
    next_kernel = dict(zip(kernel_ids, kernel_ids[1:]))

    def settle(p):
        """A write all of whose bytes are read or overwritten: was it a hand-off to the next launch?"""
        nxt = next_kernel.get(p["w"])
        if not p["dead"] and p["readers"] == {nxt}:
            handoff[(position.get(p["w"]), position.get(nxt))] += popcount(p["m0"]) * sub_bytes

    for s, e in zip(starts.tolist(), ends.tolist()):
        g = int(gran[s])
        addr = g << shift
        if cont.pinned_at(addr):
            continue
        c = cont.of(addr)
        region = c[2] if c else "unnamed"
        blinds = sorted(i for cc in cont.all_of(addr) for i in blind.get(cc[2], []))
        pending = []
        for j in range(s, e):
            L, k, m = int(lid[j]), int(kind[j]), int(mask[j])
            for p in pending:  # a launch we cannot see into may have read it
                b = bisect.bisect_right(blinds, p["w"])
                if p["un"] and b < len(blinds) and blinds[b] < L:
                    live[p["w"]] += popcount(p["un"]) * sub_bytes
                    p["readers"].add(-1)
                    p["un"] = 0
            if k == 0:
                for p in pending:
                    hit = p["un"] & m
                    if hit and p["w"] != L:
                        live[p["w"]] += popcount(hit) * sub_bytes
                        p["readers"].add(L)
                        p["un"] &= ~m
            else:
                for p in pending:
                    hit = p["un"] & m
                    if hit and p["w"] != L:
                        nbytes = popcount(hit) * sub_bytes
                        dead[p["w"]] += nbytes
                        p["dead"] += nbytes
                        if step_of.get(p["w"], full) < full:
                            dead_by_region[A.group_name(region)] += nbytes
                        p["un"] &= ~m
                written[L] += popcount(m) * sub_bytes
                pending.append({"w": L, "m0": m, "un": m, "readers": set(), "dead": 0})
            for p in pending:
                if not p["un"]:
                    settle(p)
            pending = [p for p in pending if p["un"]]
        for p in pending:
            open_[p["w"]] += popcount(p["un"]) * sub_bytes

    report(args, steps, full, names, label, position, durations, written, dead, live, open_, dead_by_region,
           handoff, blind_launches, kernel_ids, lid, kind, gran, mask, shift, cont, params)


def report(args, steps, full, names, label, position, durations, written, dead, live, open_, dead_by_region,
           handoff, blind_launches, kernel_ids, lid, kind, gran, mask, shift, cont, params):
    out = []
    emit = out.append
    in_full = lambda i: any(i in s for s in steps[:full])
    per = lambda c: sum(v for i, v in c.items() if in_full(i)) / full
    emit(f"# Ledger dataflow: `{os.path.basename(os.path.abspath(args.access))}`\n")
    emit(f"{len(steps)} steps in the window, {full} resolved; {len(steps[0])} kernel launches per step, "
         f"{len(blind_launches) // len(steps)} of them opaque to the tracer (tensor-map copies). "
         f"Granule {1 << shift} B, sub-block {(1 << shift) // SUB} B. Weights not traced.\n")
    emit("## Writes per step\n")
    emit("| | MB per step |\n|---|---|")
    emit(f"| written | {per(written) / 1e6:.1f} |")
    emit(f"| read later | {per(live) / 1e6:.1f} |")
    emit(f"| **dead** (overwritten unread) | **{per(dead) / 1e6:.1f}** |")
    emit("")

    by_kernel = defaultdict(lambda: [0, 0, 0])
    for s in steps[:full]:
        for i in s:
            k = by_kernel[label(i)]
            k[0] += written[i]
            k[1] += dead[i]
            k[2] += 1
    emit("## Dead writes by kernel\n")
    emit("| kernel | launches/step | written MB/step | dead MB/step | dead % |")
    emit("|---|---|---|---|---|")
    for name, (w, d, n) in sorted(by_kernel.items(), key=lambda x: -x[1][1])[:15]:
        if d:
            emit(f"| `{name}` | {n / full:.0f} | {w / full / 1e6:.2f} | {d / full / 1e6:.2f} | {d / w * 100:.0f}% |")
    emit("")
    emit("## Dead writes by region\n")
    emit("| region | dead MB/step |\n|---|---|")
    for name, d in dead_by_region.most_common(10):
        emit(f"| `{name}` | {d / full / 1e6:.2f} |")
    emit("")

    dead_launch = Counter()
    for s in steps[:full]:
        for i in s:
            if written[i] and dead[i] == written[i]:
                dead_launch[label(i)] += 1
    emit("## Dead launches (every written byte dead)\n")
    if dead_launch:
        emit("| kernel | per step |\n|---|---|")
        for name, n in dead_launch.most_common(10):
            emit(f"| `{name}` | {n / full:.1f} |")
    else:
        emit("None.")
    emit("")

    emit("## Hand-offs to the next launch only (fusion candidates)\n")
    pairs = sorted(((v, a, b) for (a, b), v in handoff.items() if v and a is not None and b is not None), reverse=True)
    if pairs:
        emit("| writer | reader | KB/step |\n|---|---|---|")
        agg = Counter()
        for v, a, b in pairs:
            agg[(label(steps[0][a]), label(steps[0][b]))] += v
        for (a, b), v in agg.most_common(12):
            emit(f"| `{a}` | `{b}` | {v / len(steps) / 1e3:.0f} |")
    emit("")

    emit("## Constant inputs (hoisting candidates)\n")
    emit("Launches whose traced reads all land on memory nobody wrote during the window.\n")
    written_g = set(gran[kind == 1].tolist())
    reads = defaultdict(set)
    for L, g in zip(lid[kind == 0].tolist(), gran[kind == 0].tolist()):
        reads[L].add(g)
    const = Counter()
    for s in steps:
        for i in s:
            gs = reads.get(i)
            if gs and not (gs & written_g) and not any(cont.pinned_at(g << shift) for g in gs):
                const[label(i)] += 1
    if const:
        emit("| kernel | per step | us each |\n|---|---|---|")
        us = defaultdict(list)
        for p, i in enumerate(steps[0]):
            us[label(i)].append(durations[p])
        for name, n in const.most_common(12):
            emit(f"| `{name}` | {n / len(steps):.1f} | {np.mean(us[name]):.1f} |")
    else:
        emit("None.")
    emit("")

    emit("## Independent neighbours (overlap candidates)\n")
    sets = defaultdict(lambda: (set(), set()))
    for L, k, g, m in zip(lid.tolist(), kind.tolist(), gran.tolist(), mask.tolist()):
        sets[L][k].add(g)
    blind_set = set(blind_launches)
    free = []
    s = steps[0]
    for p in range(len(s) - 1):
        a, b = s[p], s[p + 1]
        if a in blind_set or b in blind_set:
            continue
        ra, wa = sets[a]
        rb, wb = sets[b]
        if not (wa & (rb | wb)) and not (ra & wb):
            free.append((min(durations[p], durations[p + 1]), label(a), label(b)))
    emit(f"{len(free)} of {len(s) - 1} adjacent pairs in a step have no conflicting access "
         f"(pairs involving an opaque launch are skipped). Upper bound of the overlap: "
         f"{sum(f[0] for f in free) / 1e3:.2f} ms per step.\n")
    if free:
        emit("| shorter launch us | first | second |\n|---|---|---|")
        for t, a, b in sorted(free, reverse=True)[:10]:
            emit(f"| {t:.1f} | `{a}` | `{b}` |")
    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
