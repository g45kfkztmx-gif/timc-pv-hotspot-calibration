# -*- coding: utf-8 -*-
"""
rev7_repr_ordering_control.py
=============================
IPT Round 18 —— 表示敏感性实验的**核心补强**: 破坏"强度排序"的对照臂。

动机 (审稿人最可能的反击)
-------------------------
Round 16 的 Table X 里, 唯一崩掉的臂是 `peak channel` = max(R,G,B)。
审稿人可以说: "那不是因为排序被破坏, 而是因为这块调色板的红通道在热区饱和 ——
一个调色板特有的偶然事故, 不能证明方法依赖 ordering"。

本脚本构造两个**与调色板无关**的破坏排序对照:

  * `fold`    = luma 的三角折叠映射  out = 2*v (v<128) else 2*(255-v)
                —— 分段线性、局部仍然平滑, 唯一被动的是**单调性**。
  * `lutshuf` = luma 经一个固定随机置换 LUT (seed 0)
                —— **直方图逐灰度完全相同**, 只有像素间的排序被打乱。

两者都不依赖任何通道权重的假设, 所以如果它们同样崩,
"方法依赖的是保留顺序的标量代理, 而不是某个具体颜色映射" 才成立。

同时它们回答用户提的 "channel shuffled" 那一行: 破坏颜色/强度顺序 ⇒ 掉。
(注: 通道置换对 mean/max 是恒等的, 所以真正可测的"打乱顺序"是强度级 LUT 置换。)

流程与 Round 16 完全一致, 不引入任何新变量
------------------------------------------------
  1) 用冻结生成器 + 冻结门限 g = 0.42 在 1,301 张训练图上重建标签集
  2) 与 luma 参考标签集比对 (F1 / 逐图全同比例)
  3) 提议层测试集覆盖 (无训练, 与 Table X 的 "Proposal recall" 同口径)
  4) 同一检测器 / 同一 seed / 同一协议训练 + 在冻结 350 图测试集评估

可断点续跑: 数据集有 `.done`, 训练结果有 rev7_repr_<rep>_seed0.json。

用法: python rev7_repr_ordering_control.py [gen|train|all]
输出: rev5_experiments/rev7_repr_ordering.json
"""
import json
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rev7_representation_sensitivity as R          # noqa: E402
from auto_pseudo_diag import DiagParams, detect_diag  # noqa: E402

BASE = R.BASE
OUT = os.path.join(HERE, "rev7_repr_ordering.json")
NEW_REPS = ["fold", "lutshuf"]

CONTRAST_BINS = [(0, 20), (20, 40), (40, 60), (60, 1e9)]
BIN_LABELS = ["<20", "20-40", "40-60", ">=60"]

_LUT_PERM = np.random.RandomState(0).permutation(256).astype(np.uint8)


# ---------------------------------------------------------------- 映射
def _to_scalar_ext(bgr, rep):
    """在 R._to_scalar 基础上补两个破坏排序的映射。"""
    if rep == "fold":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        out = np.where(g < 128, 2 * g, 2 * (255 - g))
        return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
    if rep == "lutshuf":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(_LUT_PERM[g])
    return R._to_scalar(bgr, rep)


# ---------------------------------------------------- 提议层覆盖 (无训练)
def load_gt():
    d = json.load(open(os.path.join(HERE, "rev6h3_boxes_rev6_g42_r1.json"), encoding="utf8"))
    gt = [r for r in d["records"] if r["kind"] in ("gt_matched", "gt_missed")]
    by_img = {}
    for r in gt:
        by_img.setdefault(r["image"], []).append(r)
    return by_img


def proposal_coverage(rep, params, by_img):
    tp = fp = fn = 0
    nprop = 0
    per_bin = {lab: [0, 0] for lab in BIN_LABELS}
    for img, recs in by_img.items():
        p = os.path.join(BASE, "test", "hot", img)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        g = _to_scalar_ext(bgr, rep)
        cands, _ = detect_diag(g, params)
        props = [(c["box"][0], c["box"][1], c["box"][0] + c["box"][2], c["box"][1] + c["box"][3])
                 for c in cands if c["status"] == "accepted" and c["score"] >= R.TAU]
        nprop += len(props)
        used = [False] * len(props)
        for r in recs:
            gbox = tuple(r["box"])
            best, bi = 0.0, -1
            for j, pb in enumerate(props):
                if used[j]:
                    continue
                v = R.iou(gbox, pb)
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
    return dict(proposer_recall=round(rec, 4), proposer_precision=round(prec, 4),
                proposals_total=nprop,
                proposals_per_image=round(nprop / len(by_img), 3),
                by_contrast={lab: dict(matched=v[0], total=v[1],
                                       recall=round(v[0] / v[1], 3) if v[1] else None)
                             for lab, v in per_bin.items()})


# ------------------------------------------------------------------ main
def main():
    stages = sys.argv[1] if len(sys.argv) > 1 else "all"
    R._to_scalar = _to_scalar_ext          # 让 build_dataset / pseudo_lines 认得新 rep
    params = DiagParams(**R.FROZEN_PARAMS)
    print("[cfg] gate=%.2f  new reps=%s" % (R.EFFECTIVE_GATE, NEW_REPS), flush=True)

    prev = {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf8"))
        except Exception:
            prev = {}

    roots, agr = {}, dict(prev.get("agreement", {}))
    for rep in NEW_REPS:
        roots[rep] = R.build_dataset(rep, params)
        agr[rep + "_vs_luma"] = dict(
            train=R.agreement(os.path.join(BASE, "yolo_rep_luma"), roots[rep], "train"))
        print("[agree] %s vs luma: %s" % (rep, json.dumps(agr[rep + "_vs_luma"]["train"],
                                                        ensure_ascii=False)), flush=True)

    cov = dict(prev.get("coverage", {}))
    by_img = load_gt()
    for rep in NEW_REPS:
        cov[rep] = proposal_coverage(rep, params, by_img)
        print("[cover] %s proposal R=%.3f (%d proposals, %.1f/img)"
              % (rep, cov[rep]["proposer_recall"], cov[rep]["proposals_total"],
                 cov[rep]["proposals_per_image"]), flush=True)

    downstream = dict(prev.get("downstream", {}))
    if stages in ("all", "train"):
        for rep in NEW_REPS:
            downstream[rep] = R.train_eval(rep, os.path.join(roots[rep], "dataset.yaml"))

    blob = dict(cfg=dict(frozen_params=R.FROZEN_PARAMS, tau=R.TAU,
                         effective_gate=R.EFFECTIVE_GATE, seed=0,
                         epochs=60, imgsz=320, batch=32),
                folds=dict(fold="out = 2*v if v<128 else 2*(255-v) on BT.601 luma",
                           lutshuf="BT.601 luma indexed through a fixed random "
                                   "permutation of the 256 intensity levels (seed 0)"),
                reference="luma arm of rev7_representation_sensitivity.py (BT.601)",
                agreement=agr, coverage=cov, downstream=downstream)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("[done] ->", OUT, flush=True)
    print(json.dumps(dict(label_F1={k: v["train"]["f1"] for k, v in agr.items()},
                          proposal_recall={k: v["proposer_recall"] for k, v in cov.items()},
                          map50={k: v["test_map50"] for k, v in downstream.items()}),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
