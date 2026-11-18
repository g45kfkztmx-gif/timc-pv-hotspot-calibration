# -*- coding: utf-8 -*-
"""
rev6d_perN_gate.py
==================
**每个标注预算 N 的有效门限响应曲线** —— 验证预算曲线在"新校准规则"
(plateau-centre selection) 下是否自洽。

背景
----
预算曲线 (budget_curve.json) 的每个点都是"用该 N 的校准子集独立重跑校准 →
生成伪标签 → 训练"。旧规则下校准扫描 st ∈ {0.38,0.40,0.42,0.45} 并对 tau
再扫一遍 (tau 冗余)。实测各 N 的有效门限 g = max(st, tau):

    N=20  -> 0.50     (st=0.45, tau=0.50)
    N=50  -> 0.42
    N=100 -> 0.42
    N=150 -> 0.42
    N=197 -> 0.42

新规则把"有效门限 g"当作唯一被识别的自由度, 并用 plateau-centre 选点。
因此需要核对: **各 N 的 F2(g) 响应是否也有内部平台, 平台中心是否也落在 0.42?**

做法
----
对每个 N, 冻结该 N 自己校准出的 (k, k_local, win, Delta_min, local_bg, f_min),
只扫有效门限 g (st=g, tau=0), 画 F2(g)。输出每个 N 的:
  - argmax g 与 F2, plateau 区间 (F2 >= max - 0.005), 平台中心
  - 与曲线实际使用的门限是否一致

输出: rev6d_perN_gate.json
"""
import json
import os
import sys
import time

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

import rev5_f_budget as FB  # noqa: E402
from auto_pseudo import load_labels  # noqa: E402

NS = [20, 50, 100, 150, 197]
GATES = [0.24, 0.28, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
         0.50, 0.52, 0.55, 0.60]
PLATEAU_TOL = 0.005           # F2 在峰值下方 0.005 以内视为同一平台


def build_cache(N):
    cache = {}
    for name in FB.calib_names(N):
        p = os.path.join(FB.BASELINE_HOT_DIR, name)
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        lp = os.path.join(FB.BASELINE_LABEL_DIR, os.path.splitext(name)[0] + ".txt")
        cache[name] = {"gt": load_labels(lp, w, h), "gray": gray, "H": h, "W": w}
    return cache


def main():
    proto = FB.base_params()
    out = {"gates": GATES, "plateau_tol": PLATEAU_TOL, "results": {}}
    print("=" * 78)
    print("PER-N EFFECTIVE-GATE RESPONSE  (new calibration rule)")
    print("=" * 78, flush=True)

    for N in NS:
        cp = os.path.join(_HERE, "budget_calib_N%d.json" % N)
        if not os.path.exists(cp):
            print("[N=%d] calibration file missing, skip" % N, flush=True)
            continue
        with open(cp, encoding="utf8") as f:
            c = json.load(f)
        old_g = max(c["st"], c["tau"])

        t0 = time.time()
        cache = build_cache(N)
        n_img = len(cache)
        kw = dict(k_sigma=c["k"])
        if c.get("kl") is not None:
            kw.update(use_local_rescue=True, k_local=c["kl"], local_win=c["win"],
                      local_min_delta=c["lmd"], use_local_bg=c["use_local_bg"])
        per = FB.detect_cache_for(cache, **kw)

        rows = []
        for g in GATES:
            TP, FP, FN, p, r, f1, f2 = FB.eval_cached(per, proto, g, c["bmf"], 0.0)
            rows.append({"g": g, "P": round(p, 4), "R": round(r, 4),
                         "F2": round(f2, 4), "TP": TP, "FP": FP, "FN": FN,
                         "FP_per_image": round(FP / max(n_img, 1), 3)})

        f2s = [x["F2"] for x in rows]
        gmax = max(f2s)
        best = max(rows, key=lambda x: x["F2"])
        plateau = [x["g"] for x in rows if x["F2"] >= gmax - PLATEAU_TOL]
        centre = round((min(plateau) + max(plateau)) / 2.0, 3)
        # 离 0.42 最近的平台成员 (用于判定新规则选点)
        pick = min(plateau, key=lambda g: abs(g - 0.42))
        f2_at_042 = [x["F2"] for x in rows if abs(x["g"] - 0.42) < 1e-9]
        f2_at_old = [x["F2"] for x in rows if abs(x["g"] - old_g) < 1e-9]

        out["results"][N] = {
            "n_images": n_img,
            "calibrated": {k: c[k] for k in
                           ("k", "st", "bmf", "kl", "win", "lmd", "use_local_bg", "tau")},
            "old_effective_gate": old_g,
            "rows": rows,
            "f2_argmax_g": best["g"], "f2_argmax": best["F2"],
            "plateau": plateau, "plateau_centre": centre,
            "new_rule_pick": pick,
            "f2_at_0.42": f2_at_042[0] if f2_at_042 else None,
            "f2_at_old_gate": f2_at_old[0] if f2_at_old else None,
            "gate_changed": abs(pick - old_g) > 1e-9,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print("[N=%3d] imgs=%3d  argmax g=%.2f (F2=%.4f)  plateau=%s centre=%.2f  "
              "new_pick=%.2f  old_gate=%.2f  changed=%s  (%.0fs)"
              % (N, n_img, best["g"], best["F2"], plateau, centre, pick, old_g,
                 abs(pick - old_g) > 1e-9, time.time() - t0), flush=True)
        print("        F2(g): " + " ".join("%.2f:%.4f" % (x["g"], x["F2"]) for x in rows),
              flush=True)

    p = os.path.join(_HERE, "rev6d_perN_gate.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", p, flush=True)


if __name__ == "__main__":
    main()
