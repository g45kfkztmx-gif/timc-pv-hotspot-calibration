# -*- coding: utf-8 -*-
"""
rev6b_effective_gate_sweep.py
=============================
诊断 + 网格扩展: 校准准则到底退不退化?

关键观察 (先验诊断)
------------------
rev5_f_budget.score_candidates 的判据是:
    if sc < st or sc < tau: continue
=> 有效门限是 max(st, tau), **tau 在 tau <= st 时完全不起作用**。

证据(已有落盘结果):
  N=20  st=0.45: tau=0.30/0.35/0.40/0.45 四行 P/R/F2 **完全相同** (0.6552/0.5758/0.5901)
  N=50  st=0.42: tau=0.30/0.35/0.40 三行完全相同 (0.6792)
  N=197 st=0.42: tau=0.30/0.35/0.40 相同 (0.6264), tau=0.45 才变化

=> 所谓 "F2 随 tau 放宽单调上升到网格下界" 很可能是**参数冗余造成的假象**:
   真正被选择的量是有效门限 g = max(st, tau), 而不是 tau 本身。

因此本脚本不再扫 tau, 直接扫**有效门限** g 与 Delta_min:
    g      in 0.20 .. 0.60 (step 0.02, 21 点)
    lmd    in {2.0, 2.5, 3.0, 3.5, 4.0, 4.5}
其余固定为已校准值 k=2.3, k_local=2.2, win=31, use_local_bg=True, bmf=0.40。
实现方式: st=g, tau=0.0 (门限即 g)。

回答两个问题:
  Q1 calibration-set F2 在 g 上是否存在**内部**最大值? (还是单调)
  Q2 若单调, 预先规定的假警报预算 lambda (FP/image) 下, recall 最优的 g 是多少?

两种 GT 口径 (取整 / 浮点) 都算 (候选缓存与 GT 无关, 只需重算 eval)。
输出: rev6b_effective_gate_sweep.json
"""
import json
import os
import sys

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

import rev5_f_budget as FB  # noqa: E402
from auto_pseudo import load_labels  # noqa: E402
from rev5_g_tauadopt_gtcheck import read_gt_float  # noqa: E402

N = 197
K, KL, WIN, LBG, BMF = 2.3, 2.2, 31, True, 0.40
G_GRID = [round(0.20 + 0.02 * i, 2) for i in range(21)]     # 0.20 .. 0.60
LMD_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


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
        cache[name] = {
            "gt": loader(os.path.join(FB.BASELINE_LABEL_DIR,
                                      os.path.splitext(name)[0] + ".txt"), w, h),
            "gray": gray, "H": h, "W": w}
    return cache


def main():
    proto = FB.base_params()
    c_int = build_cache(load_labels)
    n_img = len(c_int)
    out = {"N": N, "n_images": n_img, "g_grid": G_GRID, "lmd_grid": LMD_GRID,
           "fixed": {"k": K, "k_local": KL, "win": WIN, "use_local_bg": LBG,
                     "bmf": BMF},
           "note": "st=g, tau=0 -> effective gate = max(st,tau) = g",
           "conventions": {}}

    # 缓存候选 (与 GT 无关): 每个 lmd 一次 detect
    per_by_lmd = {}
    for lmd in LMD_GRID:
        per_by_lmd[lmd] = FB.detect_cache_for(
            c_int, k_sigma=K, use_local_rescue=True, k_local=KL,
            local_win=WIN, local_min_delta=lmd, use_local_bg=LBG)
        print("[detect] lmd=%.1f done" % lmd, flush=True)

    for conv, loader in (("gt_int", None), ("gt_float", read_gt_float)):
        cache = c_int if conv == "gt_int" else build_cache(loader)
        rows = []
        for lmd in LMD_GRID:
            per = per_by_lmd[lmd]
            if conv == "gt_float":
                per = {k_: dict(v, gt=cache[k_]["gt"]) for k_, v in per.items()}
            for g in G_GRID:
                TP, FP, FN, p, r, f1, f2 = FB.eval_cached(per, proto, g, BMF, 0.0)
                rows.append({"g": g, "lmd": lmd, "P": round(p, 4),
                             "R": round(r, 4), "F2": round(f2, 4),
                             "TP": TP, "FP": FP, "FN": FN,
                             "FP_per_image": round(FP / max(n_img, 1), 3)})

        best = max(rows, key=lambda x: x["F2"])
        # 每个 lmd 下 F2 随 g 的形状
        print("\n===== %s (images=%d) =====" % (conv, n_img))
        print("g      " + " ".join("%6.2f" % x["g"] for x in rows
                                   if x["lmd"] == best["lmd"]))
        print("F2     " + " ".join("%6.4f" % x["F2"] for x in rows
                                   if x["lmd"] == best["lmd"]))
        print("R      " + " ".join("%6.3f" % x["R"] for x in rows
                                   if x["lmd"] == best["lmd"]))
        print("FP/img " + " ".join("%6.3f" % x["FP_per_image"] for x in rows
                                   if x["lmd"] == best["lmd"]))
        print("  argmax: g=%.2f lmd=%.1f F2=%.4f (P=%.3f R=%.3f)"
              % (best["g"], best["lmd"], best["F2"], best["P"], best["R"]))

        # g 上的单调性判定 (对 argmax 的 lmd)
        seq = [x for x in rows if x["lmd"] == best["lmd"]]
        seq.sort(key=lambda x: x["g"])
        f2s = [x["F2"] for x in seq]
        interior = max(range(len(f2s)), key=lambda i: f2s[i])
        is_boundary = interior in (0, len(f2s) - 1)
        print("  F2 over g: interior_max=%s (at index %d) monotone_increasing=%s"
              % (not is_boundary, interior,
                 all(f2s[i] <= f2s[i + 1] + 1e-9 for i in range(len(f2s) - 1))))

        # lambda 约束下的 recall 最优 g (每个 lmd)
        lam_rows = {}
        for lam in (0.25, 0.50, 1.00):
            cands = [x for x in rows if x["FP_per_image"] <= lam]
            lam_rows[str(lam)] = (max(cands, key=lambda x: x["R"])
                                  if cands else None)
        for lam, v in lam_rows.items():
            if v:
                print("  lambda<=%s FP/img: g=%.2f lmd=%.1f R=%.3f P=%.3f F2=%.4f"
                      % (lam, v["g"], v["lmd"], v["R"], v["P"], v["F2"]))

        out["conventions"][conv] = {
            "rows": rows, "argmax": best,
            "f2_by_g_at_argmax_lmd": seq,
            "interior_max": not is_boundary,
            "monotone_increasing": all(f2s[i] <= f2s[i + 1] + 1e-9
                                       for i in range(len(f2s) - 1)),
            "lambda_constrained": {k_: v for k_, v in lam_rows.items()},
        }

    p = os.path.join(_HERE, "rev6b_effective_gate_sweep.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
