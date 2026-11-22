"""Summarize value-level experiments: which writes the output does not depend on.

An experiment poisons part of what one launch wrote (first with 0xFF, then, if the tokens did not move,
with 0x00) and replays a few steps. Bytes whose poisoning leaves the generated tokens unchanged under
both patterns were computed for nothing. Joined with the dataflow view (was the memory read at all?),
they split into dead writes, which nobody reads, and wasted reads: read, computed on, and then dropped.

    python -m ledger.perturb_report <plan.json> <result.jsonl> [--access <trace dir>] [--out report.md]
"""

import argparse
import bisect
import json
from collections import defaultdict

from ledger import analyze as A
from ledger.dataflow import load_access


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("results")
    ap.add_argument("--access", help="the access trace the plan came from: tell read from never-read bytes")
    ap.add_argument("--out")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    key = lambda x: (x["kernel"], x["occurrence"], x["bytes"], x.get("addr", x["ranges"][0][0] if "ranges" in x else 0))
    verdict = defaultdict(list)  # experiment -> results of its patterns
    crashed = set()  # experiments whose poison crashed the process: the output depends on them
    for line in open(args.results):
        r = json.loads(line)
        if r.get("crashed"):
            crashed.add(r["i"])
            continue
        verdict[r["i"] if "i" in r else (r["kernel"], r["occurrence"], r["bytes"], r["addr"])].append(r)

    readers = defaultdict(list)  # granule -> (launch id, kind, sub-block mask) of its accesses, by launch
    shift, step_len = 10, 0
    if args.access:
        _meta, shift, lid, kind, gran, mask, _pcs = load_access(args.access)
        for L, k, g, m in zip(lid.tolist(), kind.tolist(), gran.tolist(), mask.tolist()):
            readers[g].append((L, k, m))
        for v in readers.values():
            v.sort()
        step_len = len({x["position"] for x in plan}) and max(x["position"] for x in plan) + 1

    sub = (1 << shift) // 32

    def read_later(x):
        """Bytes of the experiment a later launch reads before anyone overwrites them (32-byte sub-blocks)."""
        n = 0
        for b, size in x["ranges"]:
            for g in range(b >> shift, ((b + size - 1) >> shift) + 1):
                lo, hi = max(b, g << shift) - (g << shift), min(b + size, (g + 1) << shift) - (g << shift)
                want = ((1 << ((hi - 1) // sub + 1)) - 1) & ~((1 << (lo // sub)) - 1)
                got = killed = 0
                rs = readers.get(g, [])
                for L, k, m in rs[bisect.bisect_right(rs, (x["id"], 2, 1 << 32)):]:
                    if L >= x["id"] + step_len:
                        break
                    if k == 0:
                        got |= m & want & ~killed
                    else:
                        killed |= m & want
                n += bin(got).count("1") * sub
        return n

    pretty = A.demangle(list({x["kernel"] for x in plan}))
    per = defaultdict(lambda: {"launches": set(), "tested": 0, "free": 0, "free_read": 0, "logits_only": 0, "done": 0, "crashed": 0})
    rows = []
    for i, x in enumerate(plan):
        rs = verdict.get(i) or verdict.get(key(x))
        k = per[A.short(pretty[x["kernel"]], 60)]
        k["launches"].add(x["position"])
        if i in crashed:
            k["done"] += 1
            k["tested"] += x["bytes"]
            k["crashed"] += 1
            continue
        if not rs:
            continue
        k["done"] += 1
        k["tested"] += x["bytes"]
        tokens_same = all(r["first_token_diff_step"] < 0 for r in rs) and len(rs) == 2
        logits_same = tokens_same and all(r["first_logits_diff_step"] < 0 for r in rs)
        if tokens_same:
            k["free"] += x["bytes"]
            if readers:
                k["free_read"] += read_later(x)
            if not logits_same:
                k["logits_only"] += x["bytes"]
        rows.append((x, tokens_same, logits_same))

    out = []
    emit = out.append
    done = sum(k["done"] for k in per.values())
    emit("# Ledger value experiments\n")
    emit(f"{done} of {len(plan)} experiments finished ({sum(k['crashed'] for k in per.values())} crashed the process: "
         f"the poisoned data was used as an index). A write is **unused** when poisoning it with 0xFF and with "
         f"0x00 leaves the generated tokens of the following steps unchanged.\n")
    emit("| kernel | launches/step | tested MB | unused MB | unused % | of it, read later | of it, logits moved |")
    emit("|---|---|---|---|---|---|---|")
    for name, k in sorted(per.items(), key=lambda kv: -kv[1]["free"]):
        if not k["tested"]:
            continue
        emit(f"| `{name}` | {len(k['launches'])} | {k['tested'] / 1e6:.2f} | {k['free'] / 1e6:.2f} | "
             f"{k['free'] / k['tested'] * 100:.0f}% | {k['free_read'] / 1e6:.2f} | {k['logits_only'] / 1e6:.2f} |")
    total = sum(k["tested"] for k in per.values())
    free = sum(k["free"] for k in per.values())
    free_read = sum(k["free_read"] for k in per.values())
    emit(f"\nTested {total / 1e6:.1f} MB of writes in one step, {free / 1e6:.1f} MB unused"
         + (f", of which {free_read / 1e6:.1f} MB is read by a later launch: computed on and then dropped." if readers else ".")
         + "\n")
    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
