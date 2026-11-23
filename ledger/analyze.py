"""Ledger analysis: join kernel timings with launch parameters and the memory map, then keep the
books per launch.

    python -m ledger.analyze <capture dir> --ceiling-gbps 252 [--steps 40] [--out report.md]

What it reports
  * step summary: time, GPU busy time, idle gaps, launches, weight bytes and effective bandwidth;
  * per weight group (layer index folded): how close each weight-streaming launch runs to the
    measured bandwidth ceiling;
  * same kernel, same shape, different speed: launches of one kernel on same-shaped weights whose
    durations split by which weight they read. Such splits come from the surroundings rather than
    the kernel, most often from the preceding kernel's writes being flushed from L2 on our time;
    the report lists the preceding kernels and the caches they touched;
  * repeated reads: weights streamed more than once per step.

Byte counts assume a decode step reads each referenced weight tensor once per launch and ignore
activations, which is accurate when the batch is a handful of tokens.
"""

import argparse
import bisect
import glob
import json
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict


def read_jsonl(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass  # a line cut short by a killed process


class Memory:
    """Address -> region lookup over the exported memory map."""

    def __init__(self, regions):
        self.regions = sorted(regions, key=lambda r: r["addr"])
        self.starts = [r["addr"] for r in self.regions]

    def find(self, addr):
        i = bisect.bisect_right(self.starts, addr) - 1
        if i >= 0 and addr < self.regions[i]["addr"] + self.regions[i]["size"]:
            return i
        return None

    def touched(self, params):
        """Regions referenced by any 8-byte word of the launch parameters, mapped to whether some
        word points at the region's first byte. Words landing mid-region are often not pointers
        (strides, packed fields) and are trusted less."""
        hits = {}
        for _off, size, hexs in params:
            raw = bytes.fromhex(hexs)
            for o in range(0, size - 7, 8):
                v = int.from_bytes(raw[o : o + 8], "little")
                i = self.find(v)
                if i is not None:
                    hits[i] = hits.get(i, False) or v == self.regions[i]["addr"]
        return hits


def group_name(name):
    return re.sub(r"\.\d+\.", ".N.", name)


def unmangle_head(name):
    """First nested identifiers of an Itanium-mangled name, for names c++filt gives up on."""
    if not name.startswith("_ZN"):
        return name
    parts, i = [], 3
    while i < len(name) and len(parts) < 3:
        m = re.match(r"\d+", name[i:])
        if not m:
            break
        n = int(m.group())
        i += len(m.group())
        parts.append(name[i : i + n])
        i += n
        if i < len(name) and name[i] == "I":  # template arguments follow
            return "::".join(parts[:-1] + [parts[-1] + "<…>"])
    return "::".join(parts)


def short(name, n=70):
    name = unmangle_head(name)
    name = name.replace("(anonymous namespace)::", "")
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"<.*>", "<…>", name)
    return name if len(name) <= n else name[: n - 3] + "..."


_demangled = {}


def demangle(names):
    """Batch c++filt; names that fail stay mangled."""
    todo = [n for n in set(names) if n not in _demangled]
    if todo:
        try:
            out = subprocess.run(["c++filt"], input="\n".join(todo), capture_output=True, text=True).stdout
            for n, d in zip(todo, out.splitlines()):
                _demangled[n] = d
        except OSError:
            pass
    return {n: _demangled.get(n, n) for n in names}


def module(name):
    return name.rsplit(".", 1)[0]


def load(capture):
    regions = []
    for path in glob.glob(os.path.join(capture, "regions-*.json")):
        regions += json.load(open(path))
    mem = Memory(regions)
    nodes = {r["id"]: r for r in read_jsonl(os.path.join(capture, "nodes-*.jsonl"))}
    clones = {r["id"]: r["orig"] for r in read_jsonl(os.path.join(capture, "clones-*.jsonl"))}
    # A runtime launch reaches both the runtime and the driver callback with one correlation id;
    # keep the record that carries the parameters.
    direct = {}
    for r in read_jsonl(os.path.join(capture, "launches-*.jsonl")):
        if r["params"] or r["corr"] not in direct:
            direct[r["corr"]] = r
    kernels = sorted(read_jsonl(os.path.join(capture, "kernels-*.jsonl")), key=lambda k: k["start"])
    return mem, nodes, clones, direct, kernels


def resolve(k, nodes, clones, direct):
    if k["node"]:
        node = k["node"]
        for _ in range(4):  # an exec node may be a clone of a clone
            if node in nodes:
                return nodes[node]
            node = clones.get(node)
            if node is None:
                break
        return None
    return direct.get(k["corr"])


def build(capture, ceiling):
    """One row per kernel launch. A launch streams its largest referenced weight and the tensors of
    the same module (scales); other weight hits are incidental pointers. A launch too short to have
    read its weight at the ceiling only gathered from it (embedding lookups) and streams nothing."""
    mem, nodes, clones, direct, kernels = load(capture)
    rows, unresolved = [], 0
    for k in kernels:
        launch = resolve(k, nodes, clones, direct)
        hits = mem.touched(launch["params"]) if launch and launch["params"] else {}
        unresolved += not (launch and launch["params"])
        weight_hits = [i for i in hits if mem.regions[i]["kind"] == "weight"]
        exact = [i for i in weight_hits if hits[i]]
        weights = sorted((mem.regions[i] for i in (exact or weight_hits)), key=lambda r: -r["size"])
        us = (k["end"] - k["start"]) / 1e3
        streamed = [w for w in weights if module(w["name"]) == module(weights[0]["name"])] if weights else []
        nbytes = sum(w["size"] for w in streamed)
        partial = nbytes > us * 1e3 * ceiling * 1.3
        rows.append(
            {
                "start": k["start"],
                "end": k["end"],
                "us": us,
                "name": k["name"],
                "grid": tuple(k["grid"]),
                "weights": [] if partial else streamed,
                "weight_bytes": 0 if partial else nbytes,
                "partial": [w["name"] for w in streamed] if partial else [],
                "caches": [mem.regions[i]["name"] for i in hits if mem.regions[i]["kind"] == "cache"],
            }
        )
    return rows, unresolved, len(mem.regions)


def busy_and_gaps(rows):
    busy, cur_s, cur_e = 0, None, None
    for r in rows:
        if cur_e is None or r["start"] > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = r["start"], r["end"]
        else:
            cur_e = max(cur_e, r["end"])
    if cur_e is not None:
        busy += cur_e - cur_s
    span = rows[-1]["end"] - rows[0]["start"] if rows else 0
    return span, busy


def weight_key(r):
    """Group a weight-streaming launch by its main weight: folded name + shape + dtype."""
    w = r["weights"][0]
    return group_name(w["name"]), tuple(w["shape"]), w["dtype"]


def report(capture, ceiling, steps, out):
    rows, unresolved, n_regions = build(capture, ceiling)
    pretty = demangle([r["name"] for r in rows])
    if not rows:
        raise SystemExit("no kernel records in range")
    span, busy = busy_and_gaps(rows)
    total_bytes = sum(r["weight_bytes"] for r in rows)
    lines = []
    emit = lines.append

    emit(f"# Ledger report: `{os.path.basename(os.path.abspath(capture))}`\n")
    emit(f"{len(rows)} kernel launches over {steps} steps, {n_regions} memory regions, "
         f"{unresolved} launches without parameters. Bandwidth ceiling {ceiling:.0f} GB/s.\n")
    emit("## Step\n")
    emit("| | per step |\n|---|---|")
    emit(f"| wall time (first launch to last) | {span / steps / 1e6:.2f} ms |")
    emit(f"| GPU busy | {busy / steps / 1e6:.2f} ms |")
    emit(f"| idle between kernels | {(span - busy) / steps / 1e6:.2f} ms ({(span - busy) / span * 100:.1f}%) |")
    emit(f"| launches | {len(rows) / steps:.0f} |")
    emit(f"| weight bytes streamed | {total_bytes / steps / 1e9:.2f} GB |")
    emit(f"| effective bandwidth | {total_bytes / busy:.0f} GB/s ({total_bytes / busy / ceiling * 100:.0f}% of ceiling) |")
    emit(f"| floor at ceiling | {total_bytes / steps / ceiling / 1e6:.2f} ms |\n")

    # Per weight group.
    groups = defaultdict(list)
    for r in rows:
        if r["weight_bytes"] >= 1 << 20:
            groups[weight_key(r)].append(r)
    emit("## Weight-streaming launches by weight\n")
    emit("| weight | shape | dtype | launches/step | avg us | MB | GB/s | % ceiling | lost ms/step |")
    emit("|---|---|---|---|---|---|---|---|---|")
    lost_rows = []
    for key, rs in groups.items():
        t = statistics.mean(r["us"] for r in rs)
        b = statistics.mean(r["weight_bytes"] for r in rs)
        lost = sum(r["us"] - r["weight_bytes"] / ceiling / 1e3 for r in rs) / steps / 1e3
        lost_rows.append((lost, key, rs, t, b))
    for lost, key, rs, t, b in sorted(lost_rows, key=lambda x: -x[0]):
        gbps = b / t / 1e3
        emit(f"| `{key[0]}` | {list(key[1])} | {key[2]} | {len(rs) / steps:.1f} | {t:.1f} | {b / 1e6:.1f} | "
             f"{gbps:.0f} | {gbps / ceiling * 100:.0f}% | {lost:.2f} |")
    emit("")

    # Same kernel, same shape, different speed.
    emit("## Same kernel, same shape, different speed\n")
    by_shape = defaultdict(lambda: defaultdict(list))
    index = {id(r): i for i, r in enumerate(rows)}
    for r in rows:
        if r["weight_bytes"] >= 1 << 20:
            w = r["weights"][0]
            by_shape[(r["name"], r["grid"], tuple(w["shape"]), w["dtype"])][group_name(w["name"])].append(r)
    found = False
    for (name, grid, shape, dtype), subs in by_shape.items():
        if len(subs) < 2:
            continue
        stats = {g: statistics.median(r["us"] for r in rs) for g, rs in subs.items()}
        fast = min(stats, key=stats.get)
        for g, t in sorted(stats.items(), key=lambda x: -x[1]):
            if g == fast or t < stats[fast] * 1.15:
                continue
            found = True
            extra = (t - stats[fast]) * len(subs[g]) / steps / 1e3
            emit(f"### `{short(pretty[name])}` grid {list(grid)} on {list(shape)} {dtype}\n")
            emit(f"- `{g}`: median **{t:.1f} us** over {len(subs[g])} launches")
            emit(f"- `{fast}`: median {stats[fast]:.1f} us over {len(subs[fast])} launches")
            emit(f"- excess: {t - stats[fast]:.1f} us per launch, **{extra:.2f} ms per step**")
            for label, grp in ((g, subs[g]), (fast, subs[fast])):
                prev = Counter()
                for r in grp:
                    i = index[id(r)]
                    for j in range(i - 1, max(-1, i - 4), -1):  # walk back to the last kernel touching caches
                        if rows[j]["caches"]:
                            prev[short(pretty[rows[j]["name"]], 80)] += 1
                            break
                top = prev.most_common(1)
                if top:
                    emit(f"- last kernel touching caches before `{label}`: `{top[0][0]}` "
                         f"({top[0][1]}/{len(grp)} launches)")
            emit("")
    if not found:
        emit("None above 15%.\n")

    # Repeated reads of the same weight within a step.
    emit("## Weights read more than once per step\n")
    reads = Counter()
    size = {}
    for r in rows:
        if r["weights"]:
            w = r["weights"][0]
            reads[w["name"]] += 1
            size[w["name"]] = w["size"]
    folded = defaultdict(lambda: [0, 0, 0.0])  # group -> tensors, bytes each, reads per tensor per step
    for n, c in reads.items():
        per = c / steps
        if per > 1.5 and size[n] >= 1 << 20:
            f = folded[(group_name(n), size[n])]
            f[0] += 1
            f[2] = max(f[2], per)
    if folded:
        emit("| weight | tensors | MB each | reads/step | extra GB/step |\n|---|---|---|---|---|")
        for (n, sz), (cnt, _, per) in sorted(folded.items(), key=lambda kv: -kv[1][0] * kv[0][1] * (kv[1][2] - 1)):
            emit(f"| `{n}` | {cnt} | {sz / 1e6:.1f} | {per:.1f} | {cnt * (per - 1) * sz / 1e9:.2f} |")
    else:
        emit("None.")
    emit("")
    gathered = Counter(n for r in rows for n in r["partial"])
    if gathered:
        emit("Partial reads (too short to stream the whole tensor, not counted above): "
             + ", ".join(f"`{n}` ×{c / steps:.1f}/step" for n, c in gathered.most_common(5)) + "\n")

    text = "\n".join(lines)
    if out:
        open(out, "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--ceiling-gbps", type=float, required=True, help="measured DRAM read bandwidth")
    ap.add_argument("--steps", type=int, default=None, help="steps in range (default: run.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    steps = a.steps or json.load(open(os.path.join(a.capture, "run.json")))["steps"]
    report(a.capture, a.ceiling_gbps, steps, a.out)


if __name__ == "__main__":
    main()
