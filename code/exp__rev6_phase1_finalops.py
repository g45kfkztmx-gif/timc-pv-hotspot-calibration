# -*- coding: utf-8 -*-
"""
rev6_phase1_finalops.py
=======================
Phase 1: 用**最终操作点**重跑正式 pipeline, 先 seed 0。

最终操作点 (2026-09-10 用户拍板, 依据 rev6b_effective_gate_sweep.json):
  effective gate  g = max(score_threshold, tau) = max(0.38, 0.42) = 0.42
      -> 取整 GT 口径下校准 F2 的内部 argmax (0.6264, 平台 g in [0.36,0.44])
      -> 与预算曲线 N=197 独立重校准选出的 st 一致
  tau              = 0.42   (原 0.45, 落在下降支 F2=0.6176)
  local_min_delta  = 4.0    (改为**先验结构常量**: 校准 F2 对其不敏感, 2.0->4.5 只差 0.0005)
  其余不变: k=2.3, k_local=2.2, win=31, use_local_bg=True, bmf=0.40

Phase 1 内容:
  1) R1 = 只用校准引导的伪标签训练 detector  (对应旧 A3 的角色)
  2) R2 = 再加难样本挖掘 + adopt(0.60) 精炼     (对应旧 A5/full 的角色)
两个都在冻结的 350 张人工 GT test 上评估一次。

输出: rev5_experiments/rev6_phase1_summary.json
"""
import json
import os
import sys
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

from rev5_common import out_json, result_exists, train_and_eval  # noqa: E402
import run_ablation as AB  # noqa: E402

REV5_RUNS = os.path.join(BASE, "rev5_runs")
DS_TAG = "yolo_fin42"          # effective gate 0.42, local_min_delta 4.0
LBL_ROOT = os.path.join(BASE, DS_TAG, "labels")
CONF_ADOPT = 0.60
TAG_R1 = "rev6_g42_r1"
TAG_R2 = "rev6_g42_r2"
LOCK = os.path.join(_HERE, "rev6_phase1.lock")

FINAL_OPS = {
    "effective_gate": 0.42,
    "tau": 0.42,
    "score_threshold": 0.38,
    "local_min_delta": 4.0,
    "k_sigma": 2.3,
    "k_local": 2.2,
    "local_win": 31,
    "use_local_bg": True,
    "border_min_factor": 0.40,
    "tau_adopt": CONF_ADOPT,
    "note": ("local_min_delta reclassified as an a-priori structural constant: "
             "calibration F2 varies by <=0.0005 over 2.0-4.5"),
}
# 旧操作点 (effective gate 0.45 = max(0.38,0.45)) 的参考值, 仅用于对比
OLD_REF = {"A5_seed0_map50": 0.849, "A5_5seed_mean": 0.8129, "A5_5seed_std": 0.029,
           "A3_map50": 0.8142, "budgetcurve_N197_full": 0.8003,
           "budgetcurve_N197_r1": 0.8020}


def acquire_lock():
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("[lock] %s exists -> already running, exit." % LOCK)
        return False
    with os.fdopen(fd, "w") as f:
        f.write("pid=%d\n" % os.getpid())
    return True


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def read_result(tag):
    p = out_json(tag, 0)
    if not (os.path.exists(p) and os.path.getsize(p) > 10):
        return None
    with open(p, encoding="utf8") as f:
        r = json.load(f)
    return {"map50": r["test_map50"], "map50_95": r["test_map50_95"],
            "P": r["test_precision"], "R": r["test_recall"],
            "train_seconds": r.get("train_seconds")}


def weights(tag):
    return os.path.join(REV5_RUNS, "pseudo_%s_seed0" % tag, "weights", "best.pt")


def main():
    yaml_path = os.path.join(BASE, DS_TAG, "dataset.yaml")
    if not os.path.exists(yaml_path):
        raise SystemExit("[abort] dataset 不存在: %s (先跑 gen_weak_dataset.py "
                         "--tag %s --tau 0.42)" % (yaml_path, DS_TAG))
    if not acquire_lock():
        return
    try:
        t0 = time.time()
        print("=" * 78)
        print("PHASE 1 | final operating point | effective gate 0.42")
        print(json.dumps(FINAL_OPS, ensure_ascii=False, indent=2))
        print("=" * 78, flush=True)

        # ---- R1: 校准引导伪标签 (无精炼) ----
        if result_exists(TAG_R1, 0):
            print("[R1] exists, skip", flush=True)
        else:
            train_and_eval(TAG_R1, yaml_path, seed=0, project=REV5_RUNS)
        r1_w = weights(TAG_R1)
        if not os.path.exists(r1_w):
            raise SystemExit("[abort] R1 weights 缺失: %s" % r1_w)

        # ---- R2: 难样本挖掘 + adopt 精炼 ----
        yaml2 = AB.build_refined_dataset(TAG_R2, CONF_ADOPT,
                                         base_lbl_dir=LBL_ROOT,
                                         r1_weights=r1_w)
        if not result_exists(TAG_R2, 0):
            train_and_eval(TAG_R2, yaml2, seed=0, project=REV5_RUNS)

        r1 = read_result(TAG_R1)
        r2 = read_result(TAG_R2)
        summary = {"final_operating_point": FINAL_OPS,
                   "dataset": DS_TAG,
                   "r1_calibration_guided": r1,
                   "r2_with_refinement": r2,
                   "old_operating_point_reference": OLD_REF,
                   "elapsed_min": round((time.time() - t0) / 60.0, 1)}
        out = os.path.join(_HERE, "rev6_phase1_summary.json")
        with open(out, "w", encoding="utf8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 78)
        print("PHASE 1 RESULT (test, frozen 350-image set)")
        print("=" * 78)
        for nm, r in (("R1 (calibration-guided)", r1), ("R2 (+ refinement)", r2)):
            if r:
                print("%-24s mAP@50=%.4f  mAP@50-95=%.4f  P=%.4f  R=%.4f (%.0fs)"
                      % (nm, r["map50"], r["map50_95"], r["P"], r["R"],
                         r["train_seconds"] or 0))
            else:
                print("%-24s (pending)" % nm)
        print("\nold operating point (gate 0.45) for reference: "
              "A5 seed0=%.4f, A5 5-seed=%.4f±%.4f, A3=%.4f"
              % (OLD_REF["A5_seed0_map50"], OLD_REF["A5_5seed_mean"],
                 OLD_REF["A5_5seed_std"], OLD_REF["A3_map50"]))
        print("\nsaved ->", out, flush=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
