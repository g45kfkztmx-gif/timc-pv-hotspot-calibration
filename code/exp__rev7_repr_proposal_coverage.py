# -*- coding: utf-8 -*-
"""
rev7_repr_proposal_coverage.py
==============================
Thermal representation sensitivity —— **提议层**的测试集侧测量
(与训练/下游 mAP 相互独立, 只用冻结生成器, 不训练)。

对四种标量强度映射 (luma / maxch / meanch / pnorm, 定义见
rev7_representation_sensitivity.py), 在冻结的 200 张 test/hot 图上跑冻结生成器
(FROZEN_PARAMS + 有效门限 g = 0.42), 与人工 GT (279 个实例) 在 IoU 0.5 下贪心匹配:

  * 提议层 recall / precision
  * 按 GT 实例的"相对热对比"分箱的 recall (与 Table I 同样的分箱)
  * 每图候选框数

GT 实例属性 (含 contrast) 取自 rev6h3_boxes_rev6_g42_r1.json 的 gt_matched/gt_missed 记录,
不重新计算, 保证与 Table I / Figure 7 完全同一口径。

输出: rev5_experiments/rev7_repr_proposal_coverage.json
"""
import os
import sys
import json

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split")

from rev7_representation_sensitivity import (          # noqa: E402
    FROZEN_PARAMS, TAU, EFFECTIVE_GATE, REPS, _to_scalar, iou)

from auto_pseudo_diag import DiagParams, detect_diag     # noqa: E402

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
BOXES = os.path.join(HERE, "rev6h3_boxes_rev6_g42_r1.json")
OUT = os.path.join(HERE, "rev7_repr_proposal_coverage.json")

CONTRAST_BINS = [(0, 20), (20, 40), (40, 60), (60, 1e9)]
BIN_LABELS = ["<20", "20-40", "40-60", ">=60"]


def load_gt():
    d = json.load(open(BOXES, encoding="utf8"))
    gt = [r for r in d["records"] if r["kind"] in ("gt_matched", "gt_missed")]
    by_img = {}
    for r in gt:
        by_img.setdefault(r["image"], []).append(r)
    return by_img, gt


def proposals(path, rep, params):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    g = _to_scalar(bgr, rep)
    H, W = g.shape
    cands, _ = detect_diag(g, params)
    out = []
    for c in cands:
        if c["status"] != "accepted" or c["score"] < TAU:
            continue
        x, y, w, h = c["box"]
        out.append((x, y, x + w, y + h))
    return out


def main():
    params = DiagParams(**FROZEN_PARAMS)
    by_img, gt_all = load_gt()
    print("[cfg] gate=%.2f  GT instances=%d  images=%d"
          % (EFFECTIVE_GATE, len(gt_all), len(by_img)), flush=True)

    res = {}
    for rep in REPS:
        tp = fp = fn = 0
        nprop = 0
        per_bin = {lab: [0, 0] for lab in BIN_LABELS}   # [matched, total]
        for img, recs in by_img.items():
            p = os.path.join(BASE, "test", "hot", img)
            props = proposals(p, rep, params)
            if props is None:
                continue
            nprop += len(props)
            used = [False] * len(props)
            for r in recs:
                x1, y1, x2, y2 = r["box"]
                g = (x1, y1, x2, y2)
                best, bi = 0.0, -1
                for j, pb in enumerate(props):
                    if used[j]:
                        continue
                    v = iou(g, pb)
                    if v > best:
                        best, bi = v, j
                hit = best >= 0.5
                if hit:
                    used[bi] = True
                    tp += 1
                else:
                    fn += 1
                c = float(r["contrast"])
                for lab, (lo, hi) in zip(BIN_LABELS, CONTRAST_BINS):
                    if lo <= c < hi:
                        per_bin[lab][1] += 1
                        per_bin[lab][0] += int(hit)
                        break
            fp += len(props) - sum(used)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        res[rep] = dict(
            proposer_recall=round(rec, 4), proposer_precision=round(prec, 4),
            proposals_total=nprop, proposals_per_image=round(nprop / len(by_img), 3),
            by_contrast={lab: dict(matched=v[0], total=v[1],
                                   recall=round(v[0] / v[1], 3) if v[1] else None)
                         for lab, v in per_bin.items()})
        print("[%s] proposal P=%.3f R=%.3f  %d proposals (%.1f/img)"
              % (rep, prec, rec, nprop, nprop / len(by_img)), flush=True)
        for lab in BIN_LABELS:
            v = res[rep]["by_contrast"][lab]
            print("     contrast %-6s n=%3d recall=%s" % (lab, v["total"], v["recall"]),
                  flush=True)

    blob = dict(frozen_params=FROZEN_PARAMS, tau=TAU, effective_gate=EFFECTIVE_GATE,
                gt_instances=len(gt_all), images=len(by_img),
                match_iou=0.5, source="rev6h3_boxes_rev6_g42_r1.json (gt records)",
                results=res)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("[done] ->", OUT, flush=True)


if __name__ == "__main__":
    main()
