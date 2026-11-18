# -*- coding: utf-8 -*-
"""
rev6h4_dimension_attribution.py
===============================
Section V-B 的**无边界标签**失败分析口径。

不引入任何"贴近图像边界"的判据（本数据集的注释规范把越界框裁到图像边界，
导致 ~90% 的 GT 与 96% 的正确检出都落在边界带内，边界标志没有区分度）。
改以**质量分自身的五个证据维度**归因：

  对每个残差误报 / 漏检实例，在
      contrast / peak_z / edge_density / fill / area
  五个维度上，与"被正确处理的同类对象"（正确检出的检测框 / 命中的 GT）比较，
  用稳健 z（基于中位数与 IQR）度量偏离，取偏离最大的维度作为该错误的**主责维度**。

同时报告:
  * 每个维度的"低于正确对象 P10"的比例（null 期望 = 10%）；
  * 每个错误在多少个维度上同时低于 P10（严重度）。

输出: rev6h4_dimension_attribution.json
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("WEIGHTS_TAG", "rev6_g42_r1")

DIMS = ["area", "contrast", "peak_z"]          # 只有这三个维度的"低"是明确有害且 P10 有判别力
PROFILE_DIMS = ["area", "contrast", "peak_z", "edge_density", "aspect"]
LABEL = {
    "area": "sub-threshold size",
    "contrast": "low thermal contrast",
    "peak_z": "no distinct core",
}


def robust_z(x, med, iqr):
    # 半 IQR 当尺度; 值越低 -> 偏离越负
    s = max(iqr / 1.349, 1e-6)
    return (x - med) / s


def main():
    with open(os.path.join(_HERE, "rev6h3_boxes_%s.json" % TAG), encoding="utf8") as f:
        recs = json.load(f)["records"]

    def grp(k):
        return [r for r in recs if r["kind"] == k]

    tp, fp, ml = grp("pred_tp"), grp("pred_fp"), grp("pred_misloc")
    gm, gx = grp("gt_matched"), grp("gt_missed")

    def arr(rows, d):
        v = np.array([r[d] for r in rows if d in r], dtype=float)
        return v[np.isfinite(v)]

    ref = {  # 参考分布: 检测框用 pred_tp, GT 用 gt_matched
        "pred": {d: (np.median(arr(tp, d)), np.percentile(arr(tp, d), 75) - np.percentile(arr(tp, d), 25),
                     np.percentile(arr(tp, d), 10)) for d in DIMS},
        "gt": {d: (np.median(arr(gm, d)), np.percentile(arr(gm, d), 75) - np.percentile(arr(gm, d), 25),
                   np.percentile(arr(gm, d), 10)) for d in DIMS},
    }

    def analyse(rows, key, refkey):
        out = {"n": len(rows), "below_p10_rate": {}, "n_failing_dims_hist": {},
               "primary_dim": {}, "median": {}, "profile_median": {}}
        for d in DIMS:
            p10 = ref[refkey][d][2]
            out["below_p10_rate"][d] = round(float(np.mean(arr(rows, d) < p10)), 4)
            out["median"][d] = round(float(np.median(arr(rows, d))), 4)
        for d in PROFILE_DIMS:
            out["profile_median"][d] = round(float(np.median(arr(rows, d))), 4)
        hist, prim = {}, {}
        for r in rows:
            n_fail, worst, worst_dev = 0, None, 0.0
            for d in DIMS:
                med, iqr, p10 = ref[refkey][d]
                v = r[d]
                if v < p10:
                    n_fail += 1
                z = robust_z(v, med, iqr)
                if z < worst_dev:
                    worst_dev, worst = z, d
            hist[n_fail] = hist.get(n_fail, 0) + 1
            k = LABEL[worst] if (worst is not None and worst_dev <= -0.5) \
                else "no single dominant deficit"
            prim[k] = prim.get(k, 0) + 1
        out["n_failing_dims_hist"] = {str(k): hist[k] for k in sorted(hist)}
        out["primary_dim"] = prim
        return out

    res = {
        "reference": {k: {d: {"median": round(v[0], 4), "iqr": round(v[1], 4), "p10": round(v[2], 4)}
                          for d, v in dd.items()} for k, dd in ref.items()},
        "residual_false_positives": analyse(fp, "pred_fp", "pred"),
        "mislocalized": analyse(ml, "pred_ml", "pred"),
        "missed_instances": analyse(gx, "gt_missed", "gt"),
        "counts": {"detections": len(tp) + len(fp) + len(ml), "matched": len(tp),
                   "false_positive": len(fp), "mislocalized": len(ml),
                   "gt_total": len(gm) + len(gx), "missed": len(gx)},
    }

    p = os.path.join(_HERE, "rev6h4_dimension_attribution_%s.json" % TAG)
    with open(p, "w", encoding="utf8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    for k in ("residual_false_positives", "mislocalized", "missed_instances"):
        r = res[k]
        print("\n=== %s  (n=%d)" % (k, r["n"]))
        print("  below-P10 rate:", json.dumps(r["below_p10_rate"]))
        print("  failing-dim histogram:", json.dumps(r["n_failing_dims_hist"]))
        print("  primary deficit:", json.dumps(r["primary_dim"], ensure_ascii=False))
        print("  medians:", json.dumps(r["median"]))
    print("\ncounts:", json.dumps(res["counts"]))
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
