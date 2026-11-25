"""Compare two captures launch group by launch group: what got faster, what got slower, and how much
of it is drift.

    python -m ledger.diff <capture A> <capture B> --ceiling-gbps 252 [--out report.md]

Launches are grouped by kernel, grid and the weight they stream (folded name; weights of at least
1 MB, since a stray parameter word can hit a small scale tensor). Every group present in both runs gives a ratio of median launch times; their
time-weighted median is the drift, the change a run-to-run effect (clock, power cap) applies to
everything. A group is reported as changed when its time per step moved by at least --min-ms and
its ratio departs from the drift by more than --tolerance; groups found in one run only are listed
apart. With steps.jsonl in both captures the SM clock and power are compared, and with tokens.json
in both, the generated tokens.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

from ledger import analyze as A
from ledger.steps import load_steps


def groups(capture, ceiling):
    rows, _, _ = A.build(capture, ceiling)
    run = A.load_run(capture)
    A.split_steps(rows)
    rows = [r for r in rows if A.MARKER not in r["name"]]
    steps = A.window_steps(capture)
    out = defaultdict(list)
    for r in rows:
        # Only a streamed weight names a group: a small tensor (a scale) can be hit by a stray word, and
        # hybrid models carve all caches out of one storage whose name changes from run to run.
        what = A.group_name(r["weights"][0]["name"]) if r["weight_bytes"] >= 1 << 20 else ""
        out[(r["name"], r["grid"], what)].append(r["us"])
    _, busy = A.busy_and_gaps(rows)
    return out, steps, busy / steps / 1e6, run


def weighted_median(pairs):
    """Median of values weighted by their weights: pairs of (value, weight)."""
    pairs = sorted(pairs)
    half, acc = sum(w for _, w in pairs) / 2, 0.0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return v
    return float("nan")


def tokens_equal(a, b):
    pa, pb = os.path.join(a, "tokens.json"), os.path.join(b, "tokens.json")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    ta, tb = json.load(open(pa)), json.load(open(pb))
    return ta == tb


def check_gate(busy_a, busy_b, group_deltas, pretty, variants, fail_above, fail_min_ms):
    """Return regression descriptions that exceed --fail-above (percent).

    A run fails when (a) overall GPU busy per step regresses by more than fail_above percent, or
    (b) any shared kernel group regresses by more than fail_above percent beyond drift with a
    beyond-drift increase of at least fail_min_ms ms/step.
    """
    failures = []
    overall_pct = (busy_b / busy_a - 1) * 100
    if overall_pct > fail_above:
        failures.append(
            f"GPU busy per step: {overall_pct:+.1f}% ({busy_a:.2f} → {busy_b:.2f} ms, {busy_b - busy_a:+.2f} ms)"
        )
    slow_groups = [
        (extra, pct, k, ma, mb)
        for extra, pct, k, ma, mb in group_deltas
        if pct > fail_above and extra >= fail_min_ms
    ]
    for extra, pct, (kname, _grid, what), ma, mb in sorted(slow_groups, key=lambda x: -x[0]):
        mark = " (other variant)" if (kname, _grid, what) in variants else ""
        label = f"`{A.short(pretty[kname], 50)}`{mark}"
        if what:
            label += f" (`{what}`)"
        failures.append(f"{label}: {pct:+.1f}% beyond drift ({ma:.2f} → {mb:.2f} ms/step, {extra:+.2f} ms)")
    return failures


def report(a, b, ceiling, min_ms, tolerance, out, fail_above=None, fail_min_ms=0.5):
    ga, sa, busy_a, run_a = groups(a, ceiling)
    gb, sb, busy_b, run_b = groups(b, ceiling)
    pretty = A.demangle([k[0] for k in set(ga) | set(gb)])
    per = lambda us, s: sum(us) / s / 1e3  # noqa: E731  ms per step
    # A kernel whose template arguments changed (a state dtype, say) keeps its short name: pair groups
    # found in one run only by short name, grid and weight, and rename B's to A's.
    loose = lambda k: (A.short(pretty[k[0]], 120), k[1], k[2])  # noqa: E731
    only_a = {loose(k): k for k in ga if k not in gb}
    variants = set()
    for k in [k for k in gb if k not in ga]:
        if loose(k) in only_a:
            ka = only_a.pop(loose(k))
            gb[ka] = gb.pop(k)
            variants.add(ka)
    both = [k for k in ga if k in gb]
    drift = weighted_median([(statistics.median(gb[k]) / statistics.median(ga[k]), per(ga[k], sa)) for k in both])
    lines = []
    emit = lines.append
    name = lambda p: os.path.basename(os.path.abspath(p))  # noqa: E731
    emit(f"# Ledger diff: `{name(a)}` → `{name(b)}`\n")
    emit("| | A | B | change |\n|---|---|---|---|")
    emit(f"| GPU busy per step | {busy_a:.2f} ms | {busy_b:.2f} ms | {busy_b - busy_a:+.2f} ms ({(busy_b / busy_a - 1) * 100:+.1f}%) |")
    host_a, host_b = load_steps(a), load_steps(b)
    for key, label, unit in (("sm_mhz", "SM clock", "MHz"), ("power_w", "power", "W")):
        va = [h[key] for h in host_a if key in h]
        vb = [h[key] for h in host_b if key in h]
        if va and vb:
            ma, mb = statistics.median(va), statistics.median(vb)
            emit(f"| {label} (median) | {ma:.0f} {unit} | {mb:.0f} {unit} | {(mb / ma - 1) * 100:+.1f}% |")
    emit("")
    emit(f"Drift: launches common to both runs take {(drift - 1) * 100:+.1f}% (time-weighted median of "
         f"{len(both)} groups). Changes below are measured against it.\n")
    same = tokens_equal(a, b)
    if same is not None:
        emit(f"Generated tokens: {'identical' if same else 'different'}.\n")

    changed = []
    group_deltas = []
    for k in both:
        ma, mb = per(ga[k], sa), per(gb[k], sb)
        ratio = statistics.median(gb[k]) / statistics.median(ga[k])
        # Time per step B would have if only the drift applied; the rest is the change.
        expected = ma * drift * (len(gb[k]) / sb) / (len(ga[k]) / sa)
        extra = mb - expected
        pct = (ratio / drift - 1) * 100
        group_deltas.append((extra, pct, k, ma, mb))
        if abs(mb - ma) >= min_ms and abs(ratio / drift - 1) > tolerance:
            changed.append((extra, k, ma, mb, ratio))
    emit("## Changed beyond drift\n")
    if changed:
        emit("| kernel | grid | streams | launches/step A → B | median us A → B | ms/step A → B | beyond drift ms/step |")
        emit("|---|---|---|---|---|---|---|")
        for extra, (kname, grid, what), ma, mb, ratio in sorted(changed, key=lambda x: -abs(x[0])):
            mark = " (other variant)" if (kname, grid, what) in variants else ""
            emit(f"| `{A.short(pretty[kname], 50)}`{mark} | {list(grid)} | `{what}` | {len(ga[(kname, grid, what)]) / sa:.1f} → "
                 f"{len(gb[(kname, grid, what)]) / sb:.1f} | {statistics.median(ga[(kname, grid, what)]):.1f} → "
                 f"{statistics.median(gb[(kname, grid, what)]):.1f} | {ma:.2f} → {mb:.2f} | {extra:+.2f} |")
        emit(f"\nTogether {sum(c[0] for c in changed):+.2f} ms per step beyond drift.\n")
    else:
        emit("Nothing.\n")

    for label, ours, theirs, s in (("A only", ga, gb, sa), ("B only", gb, ga, sb)):
        only = sorted(((per(ours[k], s), k) for k in ours if k not in theirs), key=lambda x: -x[0])
        only = [o for o in only if o[0] >= min_ms]
        if only:
            emit(f"## In {label}\n")
            emit("| kernel | grid | streams | launches/step | ms/step |\n|---|---|---|---|---|")
            for ms, (kname, grid, what) in only[:15]:
                emit(f"| `{A.short(pretty[kname], 50)}` | {list(grid)} | `{what}` | {len(ours[(kname, grid, what)]) / s:.1f} | {ms:.2f} |")
            emit("")

    failures = []
    if fail_above is not None:
        failures = check_gate(busy_a, busy_b, group_deltas, pretty, variants, fail_above, fail_min_ms)
        emit("## Gate\n")
        if failures:
            emit(f"Failed (`--fail-above {fail_above:g}%`, `--fail-min-ms {fail_min_ms:g} ms`):\n")
            for item in failures:
                emit(f"- {item}")
            emit("")
        else:
            emit(f"Passed (`--fail-above {fail_above:g}%`, `--fail-min-ms {fail_min_ms:g} ms`).\n")

    text = "\n".join(lines)
    if out:
        open(out, "w").write(text)
    print(text)
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ceiling-gbps", type=float, required=True)
    ap.add_argument("--min-ms", type=float, default=0.05, help="smallest change per step to report")
    ap.add_argument("--tolerance", type=float, default=0.03, help="departure from the drift to report")
    ap.add_argument("--fail-above", type=float, default=None,
                    help="exit non-zero if step time or a shared kernel group regresses by more than this percent")
    ap.add_argument("--fail-min-ms", type=float, default=0.5,
                    help="minimum beyond-drift ms/step increase for a kernel group to trigger --fail-above")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.fail_above is not None and a.fail_above < 0:
        ap.error("--fail-above must be non-negative")
    if a.fail_min_ms < 0:
        ap.error("--fail-min-ms must be non-negative")
    failures = report(a.a, a.b, a.ceiling_gbps, a.min_ms, a.tolerance, a.out, a.fail_above, a.fail_min_ms)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
