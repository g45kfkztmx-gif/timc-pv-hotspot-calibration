# -*- coding: utf-8 -*-
"""
lopo_resume_ctrl.py —— 第 8 折 (20231121) ctrl 对照臂定向补跑
======================================================================
背景: 主任务 fxTJQj 在第 8 折 ctrl 臂训练 epoch 23/60 时因
PermissionError (results.csv 瞬态文件锁) 崩溃。该折 R1 与 final
两臂的 JSON + 权重均完好, 只需重跑 ctrl 一臂。

本脚本复刻 run_fold 的第 1/3/4 步 (校准/tau_adopt/精炼标签, 均有
确定性缓存语义) 后, 仅执行第 5 步 ctrl 臂, 并补齐:
  - lopo_20231121_ctrl_seed0.json
  - lopo_20231121_summary.json
  - lopo_summary.json (追加第 8 折)

用法: python lopo_resume_ctrl.py
"""
import json
import os
import random
import time

import rev5_h_lopo as L
import rev5_f_budget as F

DAY = "20231121"
SEED = 0
TAG = "lopo_%s" % DAY


def main():
    t0 = time.time()
    L.log("=" * 72)
    L.log("resume ctrl arm for %s (R1/final cached)" % DAY)
    L.log("=" * 72)

    # ---- 1. 校准 (确定性重跑, ~3 min) ----
    files = L.calib_files(DAY)
    d = L.calibrate_files(files, TAG)
    params = F.calib_to_params(d)
    tau = d["tau"]
    L.log("[resume] calib: n=%d k=%s st=%s tau=%s F2=%s"
          % (d["n_calib"], d["k"], d["st"], d["tau"],
             round(d["calib_F2"], 4)))

    # ---- 2. R1 标签 (确定性重生成) + 日过滤选择 ----
    label_r1 = L.gen_r1_labels(params, tau, TAG)
    train_all = L.pool_images("train")
    val_all = L.pool_images("val")
    train_sel = [(n, s) for (n, s) in train_all if L.day_of(n) != DAY]
    val_sel = [(n, s) for (n, s) in val_all if L.day_of(n) != DAY]
    r1_tag = "%s_r1" % TAG
    r1_w = os.path.join(L.REV5_RUNS, "pseudo_%s_seed%d" % (r1_tag, SEED),
                        "weights", "best.pt")
    assert os.path.exists(r1_w), "R1 weights missing: %s" % r1_w

    # ---- 3. tau_adopt ----
    cache = L.load_cache(files)
    tau_adopt, ta_rows = L.select_tau_adopt_files(cache, params, tau, r1_w)

    # ---- 4. Stage II 精炼标签 ----
    label_ref, diff = L.gen_ref_labels(params, tau, tau_adopt, r1_w, TAG,
                                       label_r1)

    # ---- 5. ctrl 臂 (与 run_fold 逐字节同逻辑) ----
    ctrl_tag = "%s_ctrl" % TAG
    ctrl_json = os.path.join(L.REV5, "%s_seed%d.json" % (ctrl_tag, SEED))
    if os.path.exists(ctrl_json) and os.path.getsize(ctrl_json) > 10:
        L.log("[resume] ctrl cached, nothing to do")
        return
    rng = random.Random(int(DAY))
    ctrl_sel = rng.sample(train_all, len(train_sel))
    ctrl_yaml, _ = L.assemble(ctrl_tag, ctrl_sel, val_sel, DAY,
                              label_ref, diff)
    res_c = F.train_and_eval(ctrl_tag, ctrl_yaml, seed=SEED,
                             project=L.REV5_RUNS)
    L.log("[resume] ctrl done: mAP50=%.4f (%.0fs)"
          % (res_c["test_map50"], res_c.get("train_seconds", 0)))

    # ---- 6. 补齐 fold summary (复刻 run_fold 结构) ----
    r1_json = os.path.join(L.REV5, "%s_seed%d.json" % (r1_tag, SEED))
    fin_json = os.path.join(L.REV5, "%s_seed%d.json" % (TAG, SEED))
    res_r1 = json.load(open(r1_json, encoding="utf8"))
    res_fin = json.load(open(fin_json, encoding="utf8"))

    summary = {
        "day": DAY,
        "seed": SEED,
        "calib": {k: d[k] for k in
                  ("n_calib", "k", "st", "bmf", "kl", "win", "lmd",
                   "use_local_bg", "tau", "calib_F2", "calib_P",
                   "calib_R")},
        "fold_train_images": len(train_sel),
        "fold_val_images": len(val_sel),
        "r1": {k: res_r1[k] for k in
               ("test_map50", "test_map50_95", "test_precision",
                "test_recall", "train_seconds")},
        "tau_adopt": tau_adopt,
        "tau_adopt_rows": ta_rows,
        "final": {k: res_fin[k] for k in
                  ("test_map50", "test_map50_95", "test_precision",
                   "test_recall", "train_seconds")},
        "ctrl": {k: res_c[k] for k in
                 ("test_map50", "test_map50_95", "test_precision",
                  "test_recall", "train_seconds")},
        "ctrl_train_images": len(train_sel),
        "fold_seconds": round(time.time() - t0, 1),
        "resumed": True,
    }
    with open(os.path.join(L.REV5, "%s_summary.json" % TAG), "w",
              encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 7. 更新 lopo_summary.json ----
    sp = os.path.join(L.REV5, "lopo_summary.json")
    agg = json.load(open(sp, encoding="utf8"))
    folds = [x for x in agg.get("folds", []) if x.get("day") != DAY]
    folds.append(summary)
    agg["folds"] = folds
    with open(sp, "w", encoding="utf8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    L.log("[resume] summaries updated (%d folds)" % len(folds))
    L.log("===== resume finished (%.0fs) =====" % (time.time() - t0))


if __name__ == "__main__":
    main()
