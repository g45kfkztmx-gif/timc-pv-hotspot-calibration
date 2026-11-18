# -*- coding: utf-8 -*-
"""
rev6_a6_proposal_level.py —— A6: local rescue 的 proposal-level 证据 (CPU)
=========================================================================
动机 (TIM 审稿清单 A6): A1 -> A2 的**最终 detector** recall 从 0.781 降到 0.758,
因此不能仅凭 detector 结果说 "local rescue 提升 recall"。必须在**候选框层面**
(calibration set, 197 张, 与 train/test 不相交) 给出证据, 证明它提升的是
source-level candidate recall。

四档递进 (与论文 A1..A5 的模块顺序一致):
  1. global       : 只用全局统计阈值, 不做救援 / 不滤质量 / 不罚边界
  2. +rescue      : use_local_rescue=True
  3. +quality     : score_threshold=tau*
  4. +border      : border_min_factor=f_min

输出: rev5_experiments/a6_proposal_level.json
"""
import json
import os
import sys

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE))
sys.path.insert(0, HERE)

import rev5_f_budget as B            # noqa: E402
from auto_pseudo_diag import detect_diag   # noqa: E402

TAU = 0.42  # effective gate g* (plateau-centre selection, 2026-09-10)
F_MIN = 0.40


def fbeta(p, r, beta=2.0):
    b2 = beta * beta
    return (1 + b2) * p * r / max(b2 * p + r, 1e-8)


def evaluate(cache, params):
    TP = FP = FN = 0
    n_box = 0
    for name, e in cache.items():
        gray = e["gray"]
        cands, _ = detect_diag(gray, params)
        acc = [(c["score"], c["box"]) for c in cands
               if c["status"] == "accepted"]
        n_box += len(acc)
        used = B.greedy_match(acc, e["gt"])
        tp = sum(used)
        TP += tp
        FN += len(e["gt"]) - tp
        FP += len(acc) - tp
    p = TP / max(TP + FP, 1)
    r = TP / max(TP + FN, 1)
    return {"P": round(p, 4), "R": round(r, 4), "F2": round(fbeta(p, r), 4),
            "TP": TP, "FP": FP, "FN": FN, "boxes": n_box,
            "boxes_per_image": round(n_box / max(len(cache), 1), 2)}


def main():
    print("loading 197-image calibration cache (leak-free, never in train/test)")
    cache = B.load_calib_cache(197)
    print("calibration images:", len(cache))
    n_gt = sum(len(e["gt"]) for e in cache.values())
    print("GT instances:", n_gt)

    # NOTE (2026-09-10): the local-rescue parameters must be the CALIBRATED ones
    # (k_local=2.2, window=31, local_min_delta=4.0, local background on).  An
    # earlier version of this script let them fall back to the DiagParams
    # defaults (2.0 / 25 / 3.0, local background off), so its rows did not
    # correspond to the generator that actually produced the training labels.
    RESCUE = dict(use_local_rescue=True, k_local=2.2, local_win=31,
                  local_min_delta=4.0, use_local_bg=True)
    stages = [
        ("global", dict(use_local_rescue=False, score_threshold=0.0,
                        border_min_factor=1.0)),
        ("+local rescue", dict(score_threshold=0.0, border_min_factor=1.0,
                               **RESCUE)),
        ("+quality filter", dict(score_threshold=TAU, border_min_factor=1.0,
                                 **RESCUE)),
        ("+border penalty", dict(score_threshold=TAU, border_min_factor=F_MIN,
                                 **RESCUE)),
    ]
    rows = []
    print("\n%-18s %8s %8s %8s %8s %10s" % ("stage", "P", "R", "F2", "boxes",
                                            "box/img"))
    for label, over in stages:
        p = B.base_params(**over)
        r = evaluate(cache, p)
        r["stage"] = label
        rows.append(r)
        print("%-18s %8.4f %8.4f %8.4f %8d %10.2f"
              % (label, r["P"], r["R"], r["F2"], r["boxes"],
                 r["boxes_per_image"]))

    out = {"calibration_images": len(cache), "gt_instances": n_gt,
           "tau": TAU, "f_min": F_MIN,
           "rescue_params": {"k_local": 2.2, "local_win": 31,
                             "local_min_delta": 4.0, "use_local_bg": True},
           "rows": rows}
    p = os.path.join(HERE, "a6_proposal_level.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
