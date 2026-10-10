#!/usr/bin/env python3
"""Verdict card for one cell of the multi-vendor OLMo experiment.

Reads what the cell already wrote (training.log, controller.log,
timeline.json) and prints one card ending in GO or STOP, so the decision
to spend the next rented cell is made from the artifacts in seconds rather
than improvised while the meter runs. Exit 0 = GO, 3 = STOP, 2 = the cell
could not be read.

    mv_verdict.py <cell dir> [--seed N] [--band <band root>]

The band root holds s<seed>/olmo-graph/cpu-async-diloco/ cells; with it
the card places this cell's final eval in the band (mean, sample SD, z)
and pairs it with the band member of the same seed. Without it, or with
fewer than --min-band members, the eval is reported and not judged.

Two things the card knows that a log skim does not: a cadence eval's
epoch tag is where it was ARMED, so the curve is printed by the reduce's
step (the controller's `-v` eval line carries it), and two identical
consecutive evals are one measurement, not a plateau.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

EVAL_LINE = re.compile(
    r"ddp: eval \(epoch (\d+)\) on rank (\d+): ([0-9.]+|error) in (\d+)ms at step (\d+)"
)
BUDGET_LINE = re.compile(r"overshoot budget \[([^\]]*)\] covers a (\d+)ms reduce")
ROSTER_LINE = re.compile(
    r'host "([^"]+)" joined with ranks \[([^\]]*)\] \((\d+) GPU\(s\), libtorch "([^"]+)"\)'
)
DONE_LINE = re.compile(r"done: loss=([0-9.]+), total=([0-9.]+)s, syncs=(\d+)")
WARN = re.compile(r"\[W\d* ")


def read_training_log(path):
    out = {"epochs": [], "evals": {}, "final_eval": None, "total_s": None,
           "roster": [], "eval_set": None, "shares": None}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# gpu r"):
                out["roster"].append(line[2:])
            elif line.startswith("# eval:"):
                out["eval_set"] = line[len("# eval:"):].strip()
            elif line.startswith("# total:"):
                m = re.search(r"# total: ([0-9.]+)s", line)
                out["total_s"] = float(m.group(1)) if m else None
            elif line.startswith("final eval="):
                out["final_eval"] = float(line.split("=", 1)[1])
            elif line.startswith("epoch "):
                m = re.match(r"epoch (\d+): (.*)", line)
                ep = int(m.group(1))
                kv = dict(p.split("=", 1) for p in m.group(2).split(", ") if "=" in p)
                if "loss" in kv:
                    out["epochs"].append(ep)
                if "eval" in kv:
                    out["evals"][ep] = float(kv["eval"])
            elif line.startswith("per-rank:"):
                out["shares"] = line[len("per-rank:"):].strip()
    return out


def read_controller_log(path):
    out = {"done": None, "warns": 0, "degraded": 0, "budgets": [], "eval_points": [],
           "roster": []}
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        for line in f:
            if WARN.search(line):
                out["warns"] += 1
            if "finished DEGRADED" in line or "device-side assert" in line:
                out["degraded"] += 1
            m = DONE_LINE.search(line)
            if m:
                out["done"] = {"loss": float(m.group(1)), "total_s": float(m.group(2)),
                               "syncs": int(m.group(3))}
            m = BUDGET_LINE.search(line)
            if m:
                budget = [int(x) for x in m.group(1).split(",") if x.strip()]
                out["budgets"].append((budget, int(m.group(2))))
            m = EVAL_LINE.search(line)
            if m:
                metric = None if m.group(3) == "error" else float(m.group(3))
                out["eval_points"].append({"tag": int(m.group(1)), "rank": int(m.group(2)),
                                           "metric": metric, "ms": int(m.group(4)),
                                           "step": int(m.group(5))})
            m = ROSTER_LINE.search(line)
            if m:
                out["roster"].append({"host": m.group(1), "ranks": m.group(2),
                                      "gpus": int(m.group(3)), "libtorch": m.group(4)})
    return out


def read_timeline(path):
    out = {"d": [], "sync_ms": [], "anchor_changes": 0}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tl = json.load(f)
    for e in tl.get("events", []):
        k = e.get("k")
        if k == "div" and "d" in e:
            out["d"].append(float(e["d"]))
        elif k == "sync_end" and "ms" in e:
            out["sync_ms"].append(float(e["ms"]))
        elif k == "anchor":
            out["anchor_changes"] += 1
    return out


def band_final_evals(band_root):
    """{seed: final eval} over the band's completed cells."""
    evals = {}
    for log in sorted(glob.glob(os.path.join(band_root, "s*", "olmo-graph", "cpu-async-diloco",
                                              "training.log"))):
        seed_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(log))))
        try:
            seed = int(seed_dir[1:])
        except ValueError:
            continue
        info = read_training_log(log)
        if info["final_eval"] is not None and info["total_s"] is not None:
            evals[seed] = info["final_eval"]
    return evals


def fmt(v, spec="{:.4f}", dash="-"):
    return dash if v is None else spec.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", help="…/olmo-graph/cpu-async-diloco directory of the cell")
    ap.add_argument("--seed", type=int, default=None, help="the cell's seed, for pairing")
    ap.add_argument("--band", default=None, help="band root holding s<seed>/… cells")
    ap.add_argument("--floor", type=float, default=0.3, help="divergence guard floor")
    ap.add_argument("--warn-sd", type=float, default=2.0,
                    help="an arm cell beyond this many band SDs is flagged, still GO")
    ap.add_argument("--stop-sd", type=float, default=3.0,
                    help="an arm cell beyond this many band SDs is a STOP")
    ap.add_argument("--min-band", type=int, default=3, help="band members needed to judge an arm")
    args = ap.parse_args()

    tlog = os.path.join(args.cell, "training.log")
    if not os.path.exists(tlog):
        print(f"UNREADABLE: no training.log in {args.cell}")
        return 2
    train = read_training_log(tlog)
    ctrl = read_controller_log(os.path.join(args.cell, "controller.log"))
    tl = read_timeline(os.path.join(args.cell, "timeline.json"))

    reasons = []
    notes = []

    # Clean run, or not.
    if train["total_s"] is None:
        reasons.append("training.log has no `# total:` footer: the run did not finish")
    if ctrl is None:
        notes.append("no controller.log beside the cell: syncs, budgets and eval steps unavailable")
    else:
        if ctrl["done"] is None:
            reasons.append("controller.log has no `done:` line")
        if ctrl["warns"]:
            reasons.append(f"{ctrl['warns']} libtorch [W] line(s) in controller.log")
        if ctrl["degraded"]:
            reasons.append("cohort finished DEGRADED or hit a device-side assert")

    # Divergence against the guard floor.
    d_max = max(tl["d"]) if tl and tl["d"] else None
    d_end = tl["d"][-1] if tl and tl["d"] else None
    d_med = statistics.median(tl["d"]) if tl and tl["d"] else None
    if d_max is not None and d_max >= args.floor:
        reasons.append(f"d_raw peaked at {d_max:.3f}, at or above the {args.floor} guard floor")

    # Band placement. A cell that is itself a band member is placed among
    # the OTHER members and never judged: the band is what defines the
    # spread, and a partial band's SD says nothing yet (three tight seeds
    # once read a fourth, ordinary one as +7.6 SD and stopped the ladder).
    # Arm cells are judged: beyond --warn-sd is flagged, beyond --stop-sd
    # is a STOP, and the paired delta against the same seed is printed
    # beside the z because it cancels the seed's own share of the spread.
    band = band_final_evals(args.band) if args.band else {}
    # Membership is a path-component test, not a substring one: a cohort
    # named `band-repeat` must not read as a member of `band`.
    in_band = bool(args.band) and (
        os.path.abspath(args.cell) + os.sep
    ).startswith(os.path.abspath(args.band) + os.sep)
    others = {s: v for s, v in band.items() if not (in_band and s == args.seed)}
    z = None
    paired = None
    mean = sd = None
    if train["final_eval"] is not None and len(others) >= 2:
        mean = statistics.mean(others.values())
        sd = statistics.stdev(others.values())
        if sd > 0:
            z = (train["final_eval"] - mean) / sd
        if not in_band and args.seed in band:
            paired = train["final_eval"] - band[args.seed]
        if in_band:
            notes.append("band member: placed among the other members, not judged")
        elif len(others) < args.min_band:
            notes.append(f"band has {len(others)} completed cell(s); eval reported, not judged "
                         f"(needs {args.min_band})")
        elif z is not None and abs(z) > args.stop_sd:
            reasons.append(f"final eval {train['final_eval']:.4f} is {z:+.1f} SD from the band "
                           f"mean {mean:.4f} (SD {sd:.4f}, n={len(others)})")
        elif z is not None and abs(z) > args.warn_sd:
            notes.append(f"final eval is {z:+.1f} SD from the band mean: outside the "
                         f"{args.warn_sd} SD comfort zone, inside the {args.stop_sd} SD stop")
    elif args.band:
        notes.append(f"band has {len(others)} completed cell(s); eval reported, not judged "
                     f"(needs {args.min_band})")

    # The card.
    print(f"cell:        {args.cell}")
    if train["roster"]:
        for r in train["roster"]:
            print(f"rank:        {r}")
    if ctrl and ctrl["roster"]:
        for r in ctrl["roster"]:
            print(f"admitted:    {r['host']} ranks [{r['ranks']}] libtorch {r['libtorch']}")
    print(f"eval set:    {train['eval_set'] or 'default (out-of-domain C4-en)'}")
    print(f"epochs:      {len(train['epochs'])} split(s), wall {fmt(train['total_s'], '{:.0f}s')}")
    if ctrl and ctrl["done"]:
        print(f"reduces:     {ctrl['done']['syncs']}")
    if tl and tl["sync_ms"]:
        print(f"reduce ms:   mean {statistics.mean(tl['sync_ms']):.0f}, "
              f"max {max(tl['sync_ms']):.0f}, n={len(tl['sync_ms'])}")
    if d_max is not None:
        print(f"d_raw:       peak {d_max:.3f}, median {d_med:.3f}, end {d_end:.3f} "
              f"(guard floor {args.floor})")
    if ctrl and ctrl["budgets"]:
        first, last = ctrl["budgets"][0], ctrl["budgets"][-1]
        print(f"overshoot:   first {first[0]} @ {first[1]}ms reduce, last {last[0]} @ {last[1]}ms")
    if train["shares"]:
        print(f"shares:      {train['shares']}")
    print(f"final eval:  {fmt(train['final_eval'])}")
    if others:
        b_mean = statistics.mean(others.values())
        b_sd = statistics.stdev(others.values()) if len(others) >= 2 else None
        print(f"band:        mean {b_mean:.4f}, SD {fmt(b_sd)}, n={len(others)}, "
              f"z {fmt(z, '{:+.2f}')}, paired delta {fmt(paired, '{:+.4f}')}")
    # The curve, by step. Consecutive equal values are one measurement.
    if ctrl and ctrl["eval_points"]:
        print("eval curve:  tag  rank  step      eval      (ms)")
        prev = None
        for p in ctrl["eval_points"]:
            same = "  = previous (one measurement, two slots)" if prev is not None and p["metric"] == prev else ""
            print(f"             {p['tag']:>3}  {p['rank']:>4}  {p['step']:>8}  {fmt(p['metric'])}  "
                  f"({p['ms']}){same}")
            prev = p["metric"]
    elif train["evals"]:
        print("eval by tag: " + ", ".join(f"{k}:{v:.4f}" for k, v in sorted(train["evals"].items())))
    for n in notes:
        print(f"note:        {n}")
    for r in reasons:
        print(f"stop:        {r}")
    verdict = "STOP" if reasons else "GO"
    print(f"verdict:     {verdict}")
    return 3 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
