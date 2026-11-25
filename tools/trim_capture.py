"""Cut a Ledger capture down to its first N steps (by step markers), for test fixtures.
Keeps only the graph nodes, clones and launch records those kernels reference; regions,
attention views and copies stay whole or are cut to the same time span.

    python tools/trim_capture.py <capture dir> <out dir> <steps>
"""
import glob, json, os, shutil, sys

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
rows = lambda pat: [json.loads(l) for f in glob.glob(os.path.join(src, pat)) for l in open(f) if l.strip()]
kernels = sorted(rows("kernels-*.jsonl"), key=lambda k: k["start"])
marks = [k["start"] for k in kernels if "spin_kernel" in k["name"]]
end = marks[n] if len(marks) > n else (kernels[-1]["end"] + 1)
keep = [k for k in kernels if k["start"] < end]
clones = {r["id"]: r["orig"] for r in rows("clones-*.jsonl")}
node_ids, clone_ids = set(), set()
for k in keep:
    node = k["node"]
    while node:
        node_ids.add(node)
        if node in clones:
            clone_ids.add(node)
            node = clones[node]
        else:
            break
corrs = {k["corr"] for k in keep if not k["node"]}
pid = os.path.basename(glob.glob(os.path.join(src, "kernels-*.jsonl"))[0]).split("-")[1].split(".")[0]
def write(name, items):
    with open(os.path.join(dst, f"{name}-{pid}.jsonl"), "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
write("kernels", keep)
write("nodes", [r for r in rows("nodes-*.jsonl") if r["id"] in node_ids])
write("clones", [r for r in rows("clones-*.jsonl") if r["id"] in clone_ids])
write("launches", [r for r in rows("launches-*.jsonl") if r["corr"] in corrs])
t0 = kernels[0]["start"]
for name in ("memcpys", "copies"):
    if glob.glob(os.path.join(src, f"{name}-*.jsonl")):
        write(name, [r for r in rows(f"{name}-*.jsonl") if t0 <= r.get("start", t0) < end])
for pat in ("regions-*.json", "attention-*.json", "copynodes-*.jsonl"):
    for f in glob.glob(os.path.join(src, pat)):
        shutil.copy(f, dst)
run = json.load(open(os.path.join(src, "run.json"))); run["steps"] = min(n, run["steps"])
json.dump(run, open(os.path.join(dst, "run.json"), "w"))
if os.path.exists(os.path.join(src, "steps.jsonl")):
    open(os.path.join(dst, "steps.jsonl"), "w").writelines(open(os.path.join(src, "steps.jsonl")).readlines()[:n])
print(dst, len(keep), "kernels", len(node_ids), "nodes", len(corrs), "direct")
