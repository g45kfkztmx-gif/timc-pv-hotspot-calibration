# -*- coding: utf-8 -*-
"""
rev6_frozen_param_audit.py
==========================
审计: 论文 FROZEN_PARAMS (Delta_min=4.0, tau=0.45) 与预算曲线在 N=197 上
重新校准得到的参数 (Delta_min=3.0, tau=0.30) 不一致。

问题: 论文 Table II / L107 声称 tau 与 Delta_min 由校准集 F2 最大化选出,
但同一条流水线在 N=197 上重跑网格, argmax 落在 (st=0.42, lmd=3.0, tau=0.30)。
若两者不可调和, 就会出现 "论文方法参数 != 主实验参数" 的硬伤。

假设 H: FROZEN_PARAMS 是在 **GT 保留浮点** 口径下选出的; 而预算曲线脚本走
auto_pseudo.load_labels -> GT 取整。7x6 像素的热斑框上取整会让 IoU 跌破 0.5,
足以改变网格 argmax。

本脚本在两种 GT 口径下重算同一小网格, 判定 H 是否成立:
  k=2.3, kl=2.2, win=31, use_local_bg=True, bmf=0.40 固定
  st in {0.38, 0.42} x lmd in {3.0, 4.0} x tau in TAU_GRID

输出: rev6_frozen_param_audit.json
"""
import json
import os
import sys
import itertools

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

import rev5_f_budget as FB  # noqa: E402
from auto_pseudo import load_labels  # noqa: E402
from rev5_g_tauadopt_gtcheck import read_gt_float  # noqa: E402

N = 197
ST_GRID = [0.38, 0.42]
LMD_GRID = [3.0, 4.0]
BMF = 0.40
K, KL, WIN, LBG = 2.3, 2.2, 31, True


def build_cache(loader):
    cache = {}
    for name in FB.calib_names(N):
        p = os.path.join(FB.BASELINE_HOT_DIR, name)
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        h, w = gray.shape
        gt = loader(os.path.join(FB.BASELINE_LABEL_DIR,
                                 os.path.splitext(name)[0] + ".txt"), w, h)
        cache[name] = {"gt": gt, "gray": gray, "H": h, "W": w}
    return cache


def main():
    proto = FB.base_params()
    out = {"N": N, "grid": "frozen-param audit", "conventions": {}}

    for conv, loader in (("gt_int", load_labels), ("gt_float", read_gt_float)):
        cache = build_cache(loader)
        n_gt = sum(len(e["gt"]) for e in cache.values())
        rows = []
        for lmd in LMD_GRID:
            per_img = FB.detect_cache_for(
                cache, k_sigma=K, use_local_rescue=True, k_local=KL,
                local_win=WIN, local_min_delta=lmd, use_local_bg=LBG)
            for st in ST_GRID:
                for tau in FB.TAU_GRID:
                    TP, FP, FN, p, r, f1, f2 = FB.eval_cached(
                        per_img, proto, st, BMF, tau)
                    rows.append({"st": st, "lmd": lmd, "tau": tau,
                                 "TP": TP, "FP": FP, "FN": FN,
                                 "P": round(p, 4), "R": round(r, 4),
                                 "F1": round(f1, 4), "F2": round(f2, 4)})
            print("[%s] lmd=%.1f done" % (conv, lmd), flush=True)

        rows.sort(key=lambda x: -x["F2"])
        best = rows[0]
        frozen = [x for x in rows if x["st"] == 0.38 and x["lmd"] == 4.0
                  and x["tau"] == 0.45]
        budget = [x for x in rows if x["st"] == 0.42 and x["lmd"] == 3.0
                  and x["tau"] == 0.30]
        out["conventions"][conv] = {
            "n_gt_boxes": n_gt, "rows": rows, "argmax": best,
            "frozen_tuple": frozen[0] if frozen else None,
            "budget_tuple": budget[0] if budget else None,
        }

        print("\n===== %s (GT boxes=%d) =====" % (conv, n_gt))
        print("  argmax      : st=%.2f lmd=%.1f tau=%.2f  F2=%.4f (P=%.3f R=%.3f)"
              % (best["st"], best["lmd"], best["tau"], best["F2"],
                 best["P"], best["R"]))
        if frozen:
            f = frozen[0]
            rank = rows.index(f) + 1
            print("  FROZEN tuple: st=0.38 lmd=4.0 tau=0.45  F2=%.4f "
                  "(rank %d/%d, P=%.3f R=%.3f)"
                  % (f["F2"], rank, len(rows), f["P"], f["R"]))
        if budget:
            b = budget[0]
            print("  BUDGET tuple: st=0.42 lmd=3.0 tau=0.30  F2=%.4f (P=%.3f R=%.3f)"
                  % (b["F2"], b["P"], b["R"]))
        # 每种 tau 的最优 F2 (用于看 tau 的单调性)
        by_tau = {}
        for x in rows:
            by_tau.setdefault(x["tau"], 0.0)
            by_tau[x["tau"]] = max(by_tau[x["tau"]], x["F2"])
        print("  best F2 by tau:", {k_: by_tau[k_] for k_ in sorted(by_tau)})
        print()

    p = os.path.join(_HERE, "rev6_frozen_param_audit.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved ->", p)


if __name__ == "__main__":
    main()
