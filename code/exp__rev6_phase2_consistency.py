# -*- coding: utf-8 -*-
"""
rev6_phase2_consistency.py
==========================
等 Phase 1（`rev6_phase1_summary.json` 落盘）后，补齐操作点切换带来的两处一致性缺口：

  (1) A4 = 用**过宽**的 adopt 门限 (0.50) 在**新门限 g=0.42** 的基座上重跑精炼臂
      —— 论文 Table VI 里 A4 那一行必须与新基座可比（旧值来自 g=0.45 基座）。

  (2) Table IX（adopt 门限校准表）在**新基座 + 新 R1 权重**上重算
      —— 基座伪标签从 757 框变成 800 框，采纳语义随之改变。
      复用 `rev5_b_tauadopt.py` 的实现（monkey-patch 常量），不复制代码。

顺序很重要：必须先等 Phase 1 结束（GPU 只有一块，且 R1 权重是关键输入）。
等待用**产物文件存在性**轮询，不用锁文件（本环境 os.remove 会被 safe-delete 护栏拦，
锁文件可能残留导致死等 —— 之前 rev5_queue2 就踩过）。

输出: rev6_phase2_summary.json
"""
import json
import os
import sys
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

REV5_RUNS = os.path.join(BASE, "rev5_runs")
DS_TAG = "yolo_fin42"                 # effective gate 0.42
LBL_ROOT = os.path.join(BASE, DS_TAG, "labels")
TAG_R1 = "rev6_g42_r1"
TAG_A4 = "rev6_g42_a4"                # adopt@0.50 (post-hoc sensitivity arm)
PHASE1 = os.path.join(_HERE, "rev6_phase1_summary.json")
OUT = os.path.join(_HERE, "rev6_phase2_summary.json")
WAIT_MAX_S = 4 * 3600


def wait_for_phase1():
    t0 = time.time()
    while not (os.path.exists(PHASE1) and os.path.getsize(PHASE1) > 10):
        if time.time() - t0 > WAIT_MAX_S:
            raise SystemExit("[abort] phase 1 not finished within %dh" % (WAIT_MAX_S // 3600))
        print("[wait] phase 1 summary not yet present ...", flush=True)
        time.sleep(60)
    with open(PHASE1, encoding="utf8") as f:
        return json.load(f)


def main():
    from rev5_common import out_json, result_exists, train_and_eval
    import run_ablation as AB

    print("=" * 78)
    print("PHASE 2 | consistency gaps after the operating-point change")
    print("=" * 78, flush=True)
    p1 = wait_for_phase1()
    print("[phase1] r1=%s" % json.dumps(p1.get("r1_calibration_guided")))
    print("[phase1] r2=%s" % json.dumps(p1.get("r2_with_refinement")), flush=True)

    r1_w = os.path.join(REV5_RUNS, "pseudo_%s_seed0" % TAG_R1, "weights", "best.pt")
    if not os.path.exists(r1_w):
        raise SystemExit("[abort] R1 weights missing: %s" % r1_w)

    # ---------- (1) A4: adopt@0.50 on the g=0.42 base ----------
    a4 = None
    if result_exists(TAG_A4, 0):
        print("[A4] exists, skip", flush=True)
    else:
        yaml4 = AB.build_refined_dataset(TAG_A4, 0.50, base_lbl_dir=LBL_ROOT,
                                         r1_weights=r1_w)
        train_and_eval(TAG_A4, yaml4, seed=0, project=REV5_RUNS)
    p = out_json(TAG_A4, 0)
    if os.path.exists(p) and os.path.getsize(p) > 10:
        with open(p, encoding="utf8") as f:
            r = json.load(f)
        a4 = {"map50": r["test_map50"], "map50_95": r["test_map50_95"],
              "P": r["test_precision"], "R": r["test_recall"]}
        print("[A4] adopt@0.50 -> %s" % json.dumps(a4), flush=True)

    # ---------- (2) Table IX on the new base ----------
    import rev5_b_tauadopt as TB
    TB.R1_WEIGHTS = r1_w
    TB.TAU_QUALITY = 0.42                      # effective gate of the new base
    TB.OUT = os.path.join(_HERE, "tauadopt_calibration_g42.json")
    print("\n[Table IX] re-running adoption-gate calibration on the g=0.42 base",
          flush=True)
    TB.main()

    with open(TB.OUT, encoding="utf8") as f:
        t9 = json.load(f)

    summary = {
        "phase1": p1,
        "a4_adopt050": a4,
        "tableIX_new_base": {"best_tau_adopt_by_F2": t9["best_tau_adopt_by_F2"],
                             "rows": t9["rows"]},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
