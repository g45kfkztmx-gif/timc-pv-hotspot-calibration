# -*- coding: utf-8 -*-
"""
rev6j_wait_report.py
====================
后台等待器：轮询产物文件，出现后立即汇总并打印（用于把长训练的收尾自动接回对话）。

用法:
  python rev6j_wait_report.py delta     # 等 A3(r1) 的 5 个 seed -> 逐 seed Δ=A5−A3
  python rev6j_wait_report.py detectors # 等 rev6i_detector_summary.json -> 检测器表
"""
import json
import os
import statistics as st
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(_HERE, "rev6_g42_r1_seed%d.json")
R2 = os.path.join(_HERE, "rev6_g42_r2_seed%d.json")
DET = os.path.join(_HERE, "rev6i_detector_summary.json")
OUT = os.path.join(_HERE, "rev6_phase3_delta.json")
POLL = 45
MAXW = 6 * 3600


def load(p):
    if not os.path.exists(p) or os.path.getsize(p) < 10:
        return None
    with open(p, encoding="utf8") as f:
        return json.load(f)


def wait_all():
    t0 = time.time()
    while time.time() - t0 < MAXW:
        if all(load(R1 % s) for s in range(5)) and all(load(R2 % s) for s in range(5)):
            return True
        time.sleep(POLL)
    return False


def delta():
    a3 = {s: load(R1 % s) for s in range(5)}
    a5 = {s: load(R2 % s) for s in range(5)}
    rows = []
    for s in range(5):
        d = a5[s]["test_map50"] - a3[s]["test_map50"]
        rows.append({"seed": s, "A3": a3[s]["test_map50"], "A5": a5[s]["test_map50"],
                     "delta": round(d, 4)})
    m3 = st.mean(r["A3"] for r in rows)
    m5 = st.mean(r["A5"] for r in rows)
    dd = [r["delta"] for r in rows]
    rep = {"A3": {"mean": round(m3, 4), "std": round(st.stdev([r["A3"] for r in rows]), 4)},
           "A5": {"mean": round(m5, 4), "std": round(st.stdev([r["A5"] for r in rows]), 4)},
           "delta": {"per_seed": dd, "mean": round(st.mean(dd), 4),
                     "std": round(st.stdev(dd), 4), "min": min(dd), "max": max(dd),
                     "n_positive": sum(1 for x in dd if x > 0),
                     "paired_verdict": ("refinement helps" if st.mean(dd) > 0.01 and min(dd) > 0
                                        else "no consistent gain" if abs(st.mean(dd)) <= 0.01
                                        else "net negative / seed-dependent")},
           "runs": rows}
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("=" * 70)
    print("PAIRED FIVE-SEED COMPARISON AT g=0.42  (A5 = with refinement, A3 = without)")
    print("=" * 70)
    print("seed |   A3 (no refine) |   A5 (refine) |  Δ = A5 − A3")
    for r in rows:
        print("  %d  |      %.4f       |    %.4f     |   %+0.4f" % (r["seed"], r["A3"], r["A5"], r["delta"]))
    print("-" * 70)
    print("A3  mean %.4f ± %.4f" % (rep["A3"]["mean"], rep["A3"]["std"]))
    print("A5  mean %.4f ± %.4f" % (rep["A5"]["mean"], rep["A5"]["std"]))
    print("Δ   mean %+0.4f ± %.4f   (min %+0.4f, max %+0.4f, positive %d/5)"
          % (rep["delta"]["mean"], rep["delta"]["std"], rep["delta"]["min"],
             rep["delta"]["max"], rep["delta"]["n_positive"]))
    print("VERDICT:", rep["delta"]["paired_verdict"])
    print("saved ->", OUT)


def detectors():
    t0 = time.time()
    while time.time() - t0 < MAXW:
        d = load(DET)
        if d:
            break
        time.sleep(POLL)
    else:
        print("[detectors] timeout"); return
    print("=" * 78)
    print("DETECTOR RETRAIN AT g=0.42 (yolo_rev6_g42_r2 pseudo-labels)")
    print("=" * 78)
    best = max(r["test_map50"] for r in d["results"])
    print("%-16s %7s %8s %8s %8s %8s %8s %8s" % ("detector", "params", "GFLOPs", "train_s", "mAP50", "mAP50-95", "P", "R"))
    for r in sorted(d["results"], key=lambda x: -x["test_map50"]):
        print("%-16s %7.2f %8s %8d %8.4f %8.4f %8.4f %8.4f"
              % (r["tag"], r["params_M"], ("%.1f" % r["gflops"]) if r.get("gflops") else "-",
                 r["train_seconds"], r["test_map50"], r["test_map50_95"],
                 r["test_precision"], r["test_recall"]))
    print("best:", max(d["results"], key=lambda x: x["test_map50"])["tag"], "%.4f" % best)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "delta"
    if mode == "delta":
        if wait_all():
            delta()
        else:
            print("[delta] timeout waiting for the five A3 seeds")
    else:
        detectors()
