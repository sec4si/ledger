"""Tell dead writes that happen every step from those that depend on the input.

A write found dead in one step of one request may be dead only because of what that step did: a draft
token rejected, a branch not taken. Given the masks `ledger.dataflow --masks-out` saves for several
traces of the same program (other requests, other steps), this splits each launch position's mean dead
bytes three ways:

- fixed: the same bytes, dead in every resolved step of every trace. These can be dropped.
- moving: the rest of what every step leaves dead (the least any step does). As much is dead each step,
  but which bytes depends on the input, as when a speculative step keeps the state of whichever position
  was accepted last: nothing can be dropped without knowing the outcome in advance. A buffer that rotates
  between steps also lands here.
- varies: the rest of the mean, dead in some steps only.

    python -m ledger.dead_across <masks.npz> ... [--out report.md]
"""

import argparse
from collections import defaultdict

import numpy as np

from ledger import analyze as A


def popcount(x):
    return bin(x).count("1")


def mb(b):
    return f"{b / 1e6:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()

    names, shift = None, None
    samples = []  # per (trace, step): {(position, granule): [written, dead]}
    for path in args.masks:
        z = np.load(path)
        if names is None:
            names, shift = list(z["names"]), int(z["shift"])
        elif list(z["names"]) != names:
            raise SystemExit(f"{path}: its steps launch other kernels than the first trace's")
        per = [defaultdict(lambda: [0, 0]) for _ in range(int(z["steps"]))]
        for st, pos, g, m, d in zip(*(z[k].tolist() for k in ("step", "position", "granule", "mask", "dead"))):
            per[st][(pos, g)][d] |= m
        samples += per
    sub = (1 << shift) // 32

    # Per sample, the bytes each launch position wrote and left dead.
    w_at = [defaultdict(int) for _ in samples]
    d_at = [defaultdict(int) for _ in samples]
    for i, s in enumerate(samples):
        for (pos, _g), (w, d) in s.items():
            w_at[i][pos] += popcount(w) * sub
            d_at[i][pos] += popcount(d) * sub
    written = {pos: sum(x[pos] for x in w_at) for pos in range(len(names))}
    dead = {pos: sum(x[pos] for x in d_at) for pos in range(len(names))}
    least = {pos: min(x[pos] for x in d_at) for pos in range(len(names))}
    fixed = defaultdict(int)  # position -> bytes dead in every sample, the same ones
    for key in set().union(*samples):
        both = 0xFFFFFFFF
        for s in samples:
            both &= s[key][1] if key in s else 0
        fixed[key[0]] += popcount(both) * sub

    n = len(samples)
    pretty = A.demangle(list(set(names)))
    per = defaultdict(lambda: [0, 0, 0, 0, 0])  # kernel -> launches, written, dead, least, fixed (per step)
    for pos, name in enumerate(names):
        k = per[A.short(pretty[name], 60)]
        k[0] += 1
        k[1] += written[pos] / n
        k[2] += dead[pos] / n
        k[3] += least[pos]
        k[4] += fixed[pos]
    out = []
    emit = out.append
    emit("# Ledger: which dead writes can be dropped\n")
    emit(f"{len(args.masks)} traces, {n} resolved steps. **Fixed**: the same bytes dead in every step, can be "
         f"dropped. **Moving**: as much dead every step, but which bytes depends on the input. **Varies**: dead "
         f"in some steps only. The three add up to the mean.\n")
    emit("| kernel | launches/step | written MB/step | dead MB/step | fixed | moving | varies |")
    emit("|---|---|---|---|---|---|---|")
    for name, (launches, w, d, lo, fx) in sorted(per.items(), key=lambda kv: -kv[1][2]):
        if d >= 1e3:
            emit(f"| `{name}` | {launches} | {mb(w)} | {mb(d)} | {mb(fx)} | {mb(lo - fx)} | {mb(d - lo)} |")
    tw, td, tl, tf = (sum(v[i] for v in per.values()) for i in (1, 2, 3, 4))
    emit(f"\nPer step: {mb(tw)} MB written, {mb(td)} MB dead on average: {mb(tf)} MB fixed, {mb(tl - tf)} MB "
         f"moving, {mb(td - tl)} MB varying.\n")
    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
