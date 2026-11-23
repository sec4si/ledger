"""Books per step: where the time between two steps goes, what a generated token costs, how attention
reads the KV cache, and what the clock did.

    python -m ledger.steps <capture dir> --ceiling-gbps 252 [--out report.md]

Needs a capture with step markers and steps.jsonl (examples/vllm_decode.py writes both):

- time per step: the cycle from one step's first kernel to the next, split into GPU busy, idle
  between kernels inside the step, and idle between steps (host work the GPU waits for: scheduling,
  sampling, output processing);
- speculative decoding: tokens gained per step, main-model and draft time, cost per generated token;
- attention: launches over a layer's KV cache against the bytes of K and V the context needs (the
  attention-*.json file gives each attention layer's cache view and K and V bytes per token);
- clock and power: SM clock, power and temperature sampled during each step, and steps that ran
  slow because the clock dropped.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

from ledger import analyze as A


def load_steps(capture):
    path = os.path.join(capture, "steps.jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def per_step(markers, steps):
    """Per step: cycle start, first and last kernel, busy (ns)."""
    out = []
    for i, (m, rows) in enumerate(zip(markers, steps)):
        if not rows:
            continue
        span, busy = A.busy_and_gaps(rows)
        draft = [r for r in rows if r.get("draft")]
        out.append({
            "marker_end": m["end"], "first": rows[0]["start"], "last": max(r["end"] for r in rows),
            "busy": busy, "draft_busy": A.busy_and_gaps(draft)[1] if draft else 0, "span": span, "index": i,
        })
    for a, b in zip(out, out[1:]):
        a["cycle"] = b["first"] - a["first"]
        a["between"] = b["first"] - a["last"]
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def attention_layers(capture, rows):
    """Mark each row with the attention layers whose cache view starts at one of its pointers:
    {layer: K and V bytes per cached token}. Uses attention-*.json (examples/vllm_decode.py writes it
    next to the regions); rows must come from analyze.build on the same capture."""
    views = []
    for path in glob.glob(os.path.join(capture, "attention-*.json")):
        views += json.load(open(path))
    if not views:
        return False
    mem = A.Memory(views)
    _, nodes, clones, direct, kernels = A.load(capture)
    for r, k in zip(rows, kernels):
        launch = A.resolve(k, nodes, clones, direct)
        hits = mem.touched(launch["params"]) if launch and launch["params"] else {}
        r["attention"] = {mem.regions[i]["name"]: mem.regions[i]["kv_bytes_per_token"] for i, exact in hits.items() if exact}
    return True


def report(capture, ceiling, out):
    rows, _, _ = A.build(capture, ceiling)
    has_attention = attention_layers(capture, rows)
    markers, steps = A.split_steps(rows)
    if not markers:
        raise SystemExit("no step markers in the capture (run it with examples/vllm_decode.py)")
    run, host = A.load_run(capture), load_steps(capture)
    books = per_step(markers, steps)
    # Steps followed by another have a cycle; a single-step window (a prefill) only has its own books.
    full = [b for b in books if "cycle" in b] or books
    ms = lambda ns: ns / 1e6  # noqa: E731
    lines = []
    emit = lines.append
    emit(f"# Ledger steps: `{os.path.basename(os.path.abspath(capture))}`\n")
    emit(f"{len(books)} steps, batch {run.get('batch', '?')}, {run.get('draft_tokens', 0)} draft tokens. "
         f"Medians over the {len(full)} steps followed by another.\n")

    emit("## Time per step\n")
    emit("| | ms |\n|---|---|")
    emit(f"| cycle (first kernel to first kernel) | {ms(med(b.get('cycle') for b in full)):.2f} |")
    emit(f"| GPU busy | {ms(med(b['busy'] for b in full)):.2f} |")
    emit(f"| idle between kernels in the step | {ms(med(b['span'] - b['busy'] for b in full)):.2f} |")
    emit(f"| idle between steps (host) | {ms(med(b.get('between') for b in full)):.2f} |")
    if host:
        emit(f"| host time of `engine.step()` | {med(h['host_ms'] for h in host):.2f} |")
    emit("")

    if host and len(host) == len(steps):
        # A prefill step "gains" its whole prompt; tokens per step only mean something in decode.
        gained = {b["index"]: 0 if run.get("prefill") else sum(host[b["index"]]["gained"].values()) for b in books}
        batch = run.get("batch", 1)
        tokens = statistics.mean(gained[b["index"]] for b in full)
        if tokens and any(b["draft_busy"] for b in books):
            cycle, draft = med(b.get("cycle") for b in full), med(b["draft_busy"] for b in full)
            emit("## Speculative decoding\n")
            emit("| | per step |\n|---|---|")
            emit(f"| tokens gained (all requests) | {statistics.mean(gained[b['index']] for b in full):.2f} |")
            emit(f"| tokens per request | {statistics.mean(gained[b['index']] for b in full) / batch:.2f} |")
            emit(f"| main model and sampling, GPU busy | {ms(med(b['busy'] - b['draft_busy'] for b in full)):.2f} ms |")
            emit(f"| draft steps, GPU busy | {ms(draft):.2f} ms |")
            emit(f"| cycle per generated token | {ms(cycle) / tokens:.2f} ms |")
            emit(f"| of which drafting | {ms(draft) / tokens:.2f} ms |")
            emit(f"| cycle per token without drafts, if the main pass cost the same | {ms(cycle - draft) / batch:.2f} ms |")
            emit("")
            emit("The last row is an estimate: a bandwidth-bound main pass costs about the same for one token per "
                 "request as for one plus the drafts.\n")
        elif tokens:
            emit(f"Cycle per generated token: {ms(med(b.get('cycle') for b in full)) / tokens:.2f} ms.\n")

        if has_attention:
            attention(books, steps, host, ceiling, pretty_names(rows), emit)
        clocks(books, host, emit)

    text = "\n".join(lines)
    if out:
        open(out, "w").write(text)
    print(text)


def pretty_names(rows):
    return A.demangle([r["name"] for r in rows])


def attention(books, steps, host, ceiling, pretty, emit):
    """Launches over attention caches, by kernel, against one read of K and V for the context. A kernel
    that would need more than 1.3x the ceiling for that does not read the whole context (it writes new
    tokens, or reads a part) and is left out of the total."""
    per = defaultdict(lambda: [0, 0.0, 0.0])  # (kernel, draft) -> launches, us, needed bytes
    contexts = []
    for b in books:
        context = sum(host[b["index"]]["context"].values())
        contexts.append(context)
        for r in steps[b["index"]]:
            if r.get("attention"):
                p = per[(r["name"], bool(r.get("draft")))]
                p[0] += 1
                p[1] += r["us"]
                p[2] += sum(r["attention"].values()) * context
    if not per:
        return
    n = len(books)
    emit("## Attention over the KV cache\n")
    emit(f"Context: {statistics.mean(contexts):.0f} cached tokens per step (all requests). Needed bytes are one "
         "read of K and V for the context, per layer.\n")
    emit("| kernel | part | launches/step | ms/step | needed MB/step | GB/s | % ceiling |\n|---|---|---|---|---|---|---|")
    total = [0.0, 0.0]
    for (name, draft), (k, us, need) in sorted(per.items(), key=lambda x: -x[1][1]):
        gbps = need / us / 1e3
        reads = gbps <= 1.3 * ceiling
        if reads:
            total[0] += us
            total[1] += need
        emit(f"| `{A.short(pretty[name], 50)}` | {'draft' if draft else 'main'} | {k / n:.1f} | {us / n / 1e3:.3f} | "
             f"{need / n / 1e6:.1f} | {gbps:.0f} | {gbps / ceiling * 100:.0f}%{'' if reads else ' (not a context read)'} |")
    if total[0]:
        gbps = total[1] / total[0] / 1e3
        emit(f"\nContext reads: {total[0] / n / 1e3:.2f} ms per step for {total[1] / n / 1e6:.1f} MB, {gbps:.0f} GB/s, "
             f"{gbps / ceiling * 100:.0f}% of the ceiling.\n")


def clocks(books, host, emit):
    xs = [(b, host[b["index"]]) for b in books if "sm_mhz" in host[b["index"]]]
    if not xs:
        return
    mhz = [h["sm_mhz"] for _, h in xs]
    m = statistics.median(mhz)
    emit("## Clock and power\n")
    emit("| | median | min | max |\n|---|---|---|---|")
    for key, label in (("sm_mhz", "SM clock (MHz)"), ("power_w", "power (W)"), ("temp_c", "temperature (C)")):
        v = [h[key] for _, h in xs]
        emit(f"| {label} | {statistics.median(v):.0f} | {min(v):.0f} | {max(v):.0f} |")
    slow = [(b, h) for b, h in xs if h["sm_mhz"] < 0.97 * m]
    emit("")
    if slow:
        busy = statistics.median(b["busy"] for b, _ in xs)
        emit(f"{len(slow)} of {len(xs)} steps ran at least 3% below the median clock; their GPU busy time is "
             f"{statistics.median(b['busy'] for b, _ in slow) / busy * 100 - 100:+.1f}% against the median step.\n")
    else:
        emit("No step ran 3% or more below the median clock.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--ceiling-gbps", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report(a.capture, a.ceiling_gbps, a.out)


if __name__ == "__main__":
    main()
