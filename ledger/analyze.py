"""Ledger analysis: join kernel timings with launch parameters and the memory map, then keep the
books per launch.

    python -m ledger.analyze <capture dir> --ceiling-gbps 252 [--steps 40] [--out report.md]

What it reports
  * findings: the largest costs found below, ranked by time per step, each against its reference
    (the bandwidth or compute floor, an idle SM's share, a faster twin of the same kernel);
  * step summary: time, GPU busy time, idle gaps, launches, weight bytes and effective bandwidth;
  * per weight group (layer index folded): how close each weight-streaming launch runs to the
    measured bandwidth ceiling;
  * same kernel, same shape, different speed: launches of one kernel on same-shaped weights whose
    durations split by which weight they read. Such splits come from the surroundings rather than
    the kernel, most often from the preceding kernel's writes being flushed from L2 on our time;
    the report lists the preceding kernels and the caches they touched;
  * repeated reads: weights streamed more than once per step;
  * launches with fewer blocks than SMs, with the time each ran alone (needs the SM count: run.json
    or --sms);
  * with --peak-tflops, weight-streaming launches against both ceilings, bandwidth and compute.

Step markers (`spin_kernel`, launched by examples/vllm_decode.py) are left out of the books; they
cut the window into steps and mark launches from the first draft-model weight of a step on as draft.


Byte counts assume a decode step reads each referenced weight tensor once per launch and ignore
activations, which is accurate when the batch is a handful of tokens.
"""

import argparse
import bisect
import glob
import json
import math
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
    """Address -> region lookup over the exported memory map. Named regions (weights, caches, buffers)
    win over the allocation segments that contain them: a segment and the first tensor in it start at
    the same address."""

    def __init__(self, regions):
        self.regions = sorted(regions, key=lambda r: (r["kind"] == "segment", r["addr"]))
        self.split = sum(r["kind"] != "segment" for r in self.regions)
        self.starts = [r["addr"] for r in self.regions]

    def _find(self, addr, lo, hi):
        i = bisect.bisect_right(self.starts, addr, lo, hi) - 1
        if i >= lo and addr < self.regions[i]["addr"] + self.regions[i]["size"]:
            return i
        return None

    def find(self, addr):
        i = self._find(addr, 0, self.split)
        return i if i is not None else self._find(addr, self.split, len(self.regions))

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
                "stream": k.get("stream", 0),
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


MARKER = "spin_kernel"  # torch.cuda._sleep, launched by examples/vllm_decode.py at the start of each step
DRAFT = ("speculator.", "drafter.", "spec_decode_model.")


def split_steps(rows):
    """Cut the window at the step markers: (markers, steps), steps a list of row lists without the
    markers. Rows of a step from its first launch streaming a draft-model weight on are marked
    "draft". Without markers, (None, None)."""
    marks = [i for i, r in enumerate(rows) if MARKER in r["name"]]
    if not marks:
        return None, None
    steps = []
    for a, b in zip(marks, marks[1:] + [len(rows)]):
        step = rows[a + 1 : b]
        first = next((i for i, r in enumerate(step) if r["weights"] and r["weights"][0]["name"].startswith(DRAFT)), len(step))
        for i, r in enumerate(step):
            r["draft"] = i >= first
        steps.append(step)
    return [rows[i] for i in marks], steps


def precision(dtype):
    """Compute class of a weight dtype. uint8 weights are taken to be packed FP4 (two per byte)."""
    if dtype.startswith("float8"):
        return "fp8", 1
    if dtype in ("bfloat16", "float16"):
        return "bf16", 1
    if dtype == "uint8":
        return "fp4", 2
    return dtype, 1


def gemm_flops(r, run):
    """2 * M * elements of the streamed weight, M the rows the launch multiplies: in a prefill step the
    prompt (the draft layers run over it too), in decode batch * (1 + draft tokens) for the main model
    and batch for a draft step. The vocabulary projection only runs over the positions that are
    sampled: batch in a prefill or draft step. vLLM's layout; other engines may differ."""
    w = r["weights"][0]
    cls, per_byte = precision(w["dtype"])
    elements = math.prod(w["shape"]) * per_byte
    batch, prefill = run.get("batch", 1), run.get("prefill", 0)
    sampled = "lm_head" in w["name"]
    if prefill:
        m = batch if sampled else prefill
    elif r.get("draft"):
        m = batch
    else:
        m = batch * (1 + run.get("draft_tokens", 0))
    return cls, m, 2 * m * elements


def weight_finding(name, rs, t, b, steps, ceiling, peaks, run):
    """(ms per step, text) for a weight group well short of its floor: the bandwidth floor, or with
    peaks the larger of the bandwidth and compute floors. None when it is close, or when it multiplies
    more than 64 rows and no peaks were given (it may be compute-bound)."""
    cls, m, flops = gemm_flops(rs[0], run)
    mem_us = b / ceiling / 1e3
    if peaks and cls in peaks:
        comp_us = flops / peaks[cls] / 1e6
        floor, what = max(mem_us, comp_us), "compute" if comp_us > mem_us else "bandwidth"
    elif m > 64:
        return None
    else:
        floor, what = mem_us, "bandwidth"
    lost = (t - floor) * len(rs) / steps / 1e3
    if floor >= 0.8 * t or lost < 0.5:
        return None
    return lost, (f"`{name}` runs at {floor / t * 100:.0f}% of its {what} floor, {lost:.2f} ms per step "
                  "short of it (Weight-streaming launches by weight)")


def weight_key(r):
    """Group a weight-streaming launch by its main weight: folded name + shape + dtype."""
    w = r["weights"][0]
    return group_name(w["name"]), tuple(w["shape"]), w["dtype"]


def load_run(capture):
    path = os.path.join(capture, "run.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def window_steps(capture):
    """Steps in the window: run.json's count (examples/vllm_decode.py), else the step markers (the vLLM
    plugin under `vllm serve`, where the window holds however many steps the load produced)."""
    steps = load_run(capture).get("steps")
    if steps:
        return steps
    steps = sum(MARKER in k["name"] for k in read_jsonl(os.path.join(capture, "kernels-*.jsonl")))
    if not steps:
        raise SystemExit(f"{capture}: no step count in run.json and no step markers; pass --steps")
    return steps


def report(capture, ceiling, steps, out, peaks=None, sms=None):
    rows, unresolved, n_regions = build(capture, ceiling)
    run = load_run(capture)
    split_steps(rows)  # marks draft launches
    rows = [r for r in rows if MARKER not in r["name"]]
    sms = sms or run.get("sms")
    pretty = demangle([r["name"] for r in rows])
    if not rows:
        raise SystemExit("no kernel records in range")
    span, busy = busy_and_gaps(rows)
    total_bytes = sum(r["weight_bytes"] for r in rows)
    lines = []
    emit = lines.append
    findings = []  # (ms per step, text): the largest costs found below, listed first

    emit(f"# Ledger report: `{os.path.basename(os.path.abspath(capture))}`\n")
    emit(f"{len(rows)} kernel launches over {steps} steps, {n_regions} memory regions, "
         f"{unresolved} launches without parameters. Bandwidth ceiling {ceiling:.0f} GB/s.\n")
    findings_at = len(lines)
    idle = (span - busy) / steps / 1e6
    if idle >= 1 or idle * 1e6 * steps >= 0.02 * span:
        findings.append((idle, f"the GPU idles {idle:.2f} ms per step between kernels (Step)"))
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
    unjudged = 0
    for lost, key, rs, t, b in sorted(lost_rows, key=lambda x: -x[0]):
        gbps = b / t / 1e3
        finding = weight_finding(key[0], rs, t, b, steps, ceiling, peaks, run)
        if finding:
            findings.append(finding)
        unjudged += finding is None and gemm_flops(rs[0], run)[1] > 64 and not peaks
        emit(f"| `{key[0]}` | {list(key[1])} | {key[2]} | {len(rs) / steps:.1f} | {t:.1f} | {b / 1e6:.1f} | "
             f"{gbps:.0f} | {gbps / ceiling * 100:.0f}% | {lost:.2f} |")
    emit("")
    if unjudged:
        findings.append((0, f"{unjudged} weight groups multiply more than 64 rows; pass --peak-tflops to judge "
                            "them against the compute ceiling as well (Compute)"))

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
                if label == g:
                    after = f", after `{top[0][0]}`" if top else ""
                    findings.append((extra, f"`{g}` takes {t - stats[fast]:.1f} us longer per launch than `{fast}` "
                                            f"on the same kernel and shape{after} (Same kernel, same shape)"))
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
            extra_bytes = cnt * (per - 1) * sz
            emit(f"| `{n}` | {cnt} | {sz / 1e6:.1f} | {per:.1f} | {extra_bytes / 1e9:.2f} |")
            if extra_bytes / ceiling / 1e6 >= 0.5:
                findings.append((extra_bytes / ceiling / 1e6, f"`{n}` is read {per:.1f} times per step; the extra "
                                 f"{extra_bytes / 1e9:.2f} GB cost at least that long at the ceiling (Weights read more than once)"))
    else:
        emit("None.")
    emit("")
    gathered = Counter(n for r in rows for n in r["partial"])
    if gathered:
        emit("Partial reads (too short to stream the whole tensor, not counted above): "
             + ", ".join(f"`{n}` ×{c / steps:.1f}/step" for n, c in gathered.most_common(5)) + "\n")

    if sms:
        for alone_ms, name, grid, gbps in underfilled(rows, steps, sms, pretty, emit):
            # A launch streaming near the ceiling loses little to idle SMs; otherwise count their share.
            idle_ms = alone_ms * (1 - math.prod(grid) / sms)
            if idle_ms >= 0.5 and (gbps is None or gbps < 0.8 * ceiling):
                findings.append((idle_ms, f"`{short(pretty[name], 50)}` runs on {math.prod(grid)} blocks for {sms} SMs "
                                          f"for {alone_ms:.2f} ms per step; the idle SMs' share is {idle_ms:.2f} ms "
                                          "(Launches with fewer blocks than SMs)"))
    if peaks:
        compute(rows, steps, ceiling, peaks, run, emit)

    notes = [text for ms, text in findings if ms == 0]
    top = sorted((f for f in findings if f[0] > 0), key=lambda x: -x[0])[:5]
    summary = ["## Findings\n"]
    if top:
        summary += [f"- **{ms:.2f} ms per step**: {text}" for ms, text in top]
        summary += [f"- {text}" for text in notes]
        summary.append("\nEach is the time the item costs against its reference, an upper bound on what fixing it "
                       "saves; the section named in parentheses has the details.\n")
    else:
        summary.append("Nothing above 0.5 ms per step.\n")
    lines[findings_at:findings_at] = summary


    text = "\n".join(lines)
    if out:
        open(out, "w").write(text)
    print(text)


def alone(rows):
    """Set each row's "alone" to the microseconds it ran with no other kernel running: on a parallel
    graph branch a launch's duration overstates what it costs the step."""
    events = sorted([(r["start"], 1, i) for i, r in enumerate(rows)] + [(r["end"], -1, i) for i, r in enumerate(rows)])
    active, last = set(), None
    for r in rows:
        r["alone"] = 0.0
    for t, kind, i in events:
        if len(active) == 1 and last is not None:
            rows[next(iter(active))]["alone"] += (t - last) / 1e3
        if kind == 1:
            active.add(i)
        else:
            active.discard(i)
        last = t


def underfilled(rows, steps, sms, pretty, emit):
    """Launches with fewer blocks than SMs: whatever their occupancy, some SMs sit idle. That costs a
    compute-bound kernel time; a bandwidth-bound one may still saturate memory with fewer SMs, so read
    it with the launch's bandwidth. "alone" is the time no other kernel ran next to it."""
    alone(rows)
    groups = defaultdict(list)
    for r in rows:
        if math.prod(r["grid"]) < sms:
            groups[(r["name"], r["grid"])].append(r)
    found = sorted(((sum(r["alone"] for r in rs) / steps / 1e3, k, rs) for k, rs in groups.items()), key=lambda x: -x[0])
    found = [f for f in found if f[0] >= 0.05]
    emit(f"## Launches with fewer blocks than SMs ({sms})\n")
    if not found:
        emit("None above 0.05 ms per step of time alone.\n")
        return []
    emit(f"{sum(f[0] for f in found):.2f} ms per step in launches that cannot use every SM and run alone.\n")
    emit("| kernel | grid | blocks | launches/step | median us | ms/step | alone ms/step | streams | GB/s |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for _, (name, grid), rs in found[:15]:
        ms = sum(r["us"] for r in rs) / steps / 1e3
        w = Counter(group_name(r["weights"][0]["name"]) for r in rs if r["weights"])
        streams = f"`{w.most_common(1)[0][0]}`" if w else ""
        nbytes, us = sum(r["weight_bytes"] for r in rs), sum(r["us"] for r in rs)
        gbps = f"{nbytes / us / 1e3:.0f}" if nbytes >= len(rs) << 20 else ""
        emit(f"| `{short(pretty[name], 60)}` | {list(grid)} | {math.prod(grid)} | {len(rs) / steps:.1f} | "
             f"{statistics.median(r['us'] for r in rs):.1f} | {ms:.2f} | {sum(r['alone'] for r in rs) / steps / 1e3:.2f} | "
             f"{streams} | {gbps} |")
    emit("")
    out = []
    for ms, (name, grid), rs in found:
        nbytes, us = sum(r["weight_bytes"] for r in rs), sum(r["us"] for r in rs)
        out.append((ms, name, grid, nbytes / us / 1e3 if nbytes >= len(rs) << 20 else None))
    return out



def compute(rows, steps, ceiling, peaks, run, emit):
    """Weight-streaming launches against both ceilings: bytes at the bandwidth ceiling and FLOPs at the
    measured peak of their precision. The larger floor says what bounds the launch."""
    groups = defaultdict(list)
    for r in rows:
        if r["weight_bytes"] >= 1 << 20:
            cls, m, flops = gemm_flops(r, run)
            groups[(weight_key(r), cls, m)].append((r, flops))
    emit("## Compute\n")
    emit("M is the tokens of the forward pass (" + ", ".join(f"{k} {v:.0f} TFLOP/s" for k, v in peaks.items()) + ").\n")
    emit("| weight | precision | M | GFLOP | avg us | TFLOP/s | memory floor us | compute floor us | bound |")
    emit("|---|---|---|---|---|---|---|---|---|")
    floor_mem = floor_both = 0.0
    lines = []
    for ((name, _shape, _dtype), cls, m), items in groups.items():
        t = statistics.mean(r["us"] for r, _ in items)
        b = statistics.mean(r["weight_bytes"] for r, _ in items)
        f = statistics.mean(fl for _, fl in items)
        mem_us = b / ceiling / 1e3
        comp_us = f / peaks[cls] / 1e6 if cls in peaks else None
        floor_mem += mem_us * len(items) / steps
        floor_both += max(mem_us, comp_us or 0) * len(items) / steps
        bound = f"no {cls} peak" if comp_us is None else "compute" if comp_us > mem_us else "memory"
        lines.append((t * len(items), f"| `{name}` | {cls} | {m} | {f / 1e9:.2f} | {t:.1f} | {f / t / 1e6:.1f} | "
                                     f"{mem_us:.1f} | {'' if comp_us is None else f'{comp_us:.1f}'} | {bound} |"))
    for _, line in sorted(lines, key=lambda x: -x[0]):
        emit(line)
    emit(f"\nFloor of the weight-streaming launches: {floor_mem / 1e3:.2f} ms per step from bandwidth alone, "
         f"{floor_both / 1e3:.2f} ms with compute.\n")


def parse_peaks(text):
    return {k: float(v) for k, v in (kv.split("=") for kv in text.split(","))} if text else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--ceiling-gbps", type=float, required=True, help="measured DRAM read bandwidth")
    ap.add_argument("--steps", type=int, default=None, help="steps in range (default: run.json)")
    ap.add_argument("--peak-tflops", default=None, help="measured peaks per precision, e.g. fp8=200,bf16=100,fp4=400")
    ap.add_argument("--sms", type=int, default=None, help="SM count (default: run.json)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    steps = a.steps or window_steps(a.capture)
    report(a.capture, a.ceiling_gbps, steps, a.out, parse_peaks(a.peak_tflops), a.sms)


if __name__ == "__main__":
    main()
