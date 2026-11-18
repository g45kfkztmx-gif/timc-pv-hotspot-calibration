# -*- coding: utf-8 -*-
"""
rev6c_param_sensitivity.py
==========================
一次性 (one-factor-at-a-time) 参数敏感性分析, 为论文 Table II 的
"calibration-derived vs structural prior" 两分法提供证据。

问题背景
--------
rev6b 已证明: 真正被校准准则识别的自由度是**有效门限** g = max(st, tau),
且 F2(g) 在 g in [0.36,0.44] 上是一个**宽平台**(非尖峰)。
但论文 Table II 曾把 k / k_local / win / Delta_min / f_min / tau 全部写成
"F2-calibrated", 这需要逐个核对: 哪些参数的 F2 响应真的有结构, 哪些平坦
到应归为先验常量 (structural prior)。

方法
----
在**最终操作点** g=0.42, bmf=0.40, lmd=4.0, k=2.3, k_local=2.2, win=31,
local_bg=True 附近, 对每个参数做一维扫描, 其余参数冻结。
- 需要重跑 detect 的 (mask 相关): k, k_local, win, lmd, use_local_bg
- 只需重算 eval 的 (仅门限相关): g, bmf
判据: 该参数在扫描范围内的 F2 峰谷差 dF2。dF2 小 (<0.01 ~ 约5个框/272)
则视为准则不识别 -> structural prior。

输出: rev6c_param_sensitivity.json
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

# ---- 最终操作点 (2026-09-10 定案) ----
FINAL = dict(k=2.3, k_local=2.2, win=31, lmd=4.0, lbg=True, bmf=0.40, g=0.42)

SWEEPS = {
    "g":        [0.30, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.50, 0.55],
    "k":        [1.5, 1.9, 2.3, 2.7, 3.1, 3.5],
    "k_local":  [1.4, 1.8, 2.2, 2.6, 3.0],
    "win":      [9, 15, 21, 31, 41, 61],
    "lmd":      [1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "bmf":      [0.20, 0.30, 0.40, 0.50, 0.60],
    "use_local_bg": [True, False],
}
# 只影响门限的参数 -> 无需重新 detect
EVAL_ONLY = {"g", "bmf"}


def build_cache(loader=None):
    cache = {}
    for name in FB.calib_names(N):
        p = os.path.join(FB.BASELINE_HOT_DIR, name)
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        lp = os.path.join(FB.BASELINE_LABEL_DIR, os.path.splitext(name)[0] + ".txt")
        cache[name] = {"gt": loader(lp, w, h) if loader else load_labels(lp, w, h),
                       "gray": gray, "H": h, "W": w}
    return cache


def detect_for(cache, k, k_local, win, lmd, lbg):
    return FB.detect_cache_for(
        cache, k_sigma=k, use_local_rescue=True, k_local=k_local,
        local_win=win, local_min_delta=lmd, use_local_bg=lbg)


def main():
    proto = FB.base_params()
    cache_int = build_cache(None)
    proto_f = FB.base_params()
    n_img = len(cache_int)
    out = {"N": N, "n_images": n_img, "final_operating_point": FINAL,
           "sweeps": SWEEPS, "results": {}, "summary": {}}
    print("[calib] images=%d" % n_img, flush=True)

    # 预计算: 默认操作点的 detect (供 eval-only 扫描复用)
    per_default = detect_for(cache_int, FINAL["k"], FINAL["k_local"],
                             FINAL["win"], FINAL["lmd"], FINAL["lbg"])
    per_by_key = {}

    for pname, values in SWEEPS.items():
        rows = []
        for v in values:
            if pname in EVAL_ONLY:
                per = per_default
            else:
                key = (pname, v)
                if key not in per_by_key:
                    kw = dict(k=FINAL["k"], k_local=FINAL["k_local"],
                              win=FINAL["win"], lmd=FINAL["lmd"],
                              lbg=FINAL["lbg"])
                    kw[{"k": "k", "k_local": "k_local", "win": "win",
                        "lmd": "lmd", "use_local_bg": "lbg"}[pname]] = v
                    per_by_key[key] = detect_for(cache_int, **kw)
                    print("  [detect] %s=%s" % (pname, v), flush=True)
                per = per_by_key[key]

            g = v if pname == "g" else FINAL["g"]
            bmf = v if pname == "bmf" else FINAL["bmf"]
            TP, FP, FN, p, r, f1, f2 = FB.eval_cached(per, proto, g, bmf, 0.0)
            rows.append({"value": v, "P": round(p, 4), "R": round(r, 4),
                         "F2": round(f2, 4), "TP": TP, "FP": FP, "FN": FN,
                         "FP_per_image": round(FP / max(n_img, 1), 3)})
        f2s = [x["F2"] for x in rows]
        d = max(f2s) - min(f2s)
        best = max(rows, key=lambda x: x["F2"])
        out["results"][pname] = rows
        out["summary"][pname] = {
            "f2_min": min(f2s), "f2_max": max(f2s), "f2_range": round(d, 4),
            "argmax_value": best["value"],
            "verdict": ("structural prior (insensitive)" if d < 0.010
                        else "calibration-derived"),
        }
        print("  %-14s F2 range %.4f (min %.4f max %.4f) argmax=%s -> %s"
              % (pname, d, min(f2s), max(f2s), best["value"],
                 out["summary"][pname]["verdict"]), flush=True)

    p = os.path.join(_HERE, "rev6c_param_sensitivity.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
