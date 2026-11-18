# -*- coding: utf-8 -*-
"""
rev5_b_tauadopt.py  (实验B: tau_adopt 在 calibration 上独立选择)  v2 忠实版
============================================================================
忠实还原 A5 的采纳语义:
  R1(A3 模型) 对每张 calibration 图预测 -> 若 conf>=tau 且与"该图统计
  伪标签(baseline pseudo, 即 A3 标签来源)" IoU<0.5, 则采纳为新框。

评价(仅 calibration 197 GT, 不接触 test):
  对每个 tau, 把 [baseline pseudo + 采纳框] 合并成最终伪标签集, 与 GT
  计算总体 P/R/F2 (IoU>=0.5 匹配)。选 F2 最大(且 precision 足够)的 tau。

这与原论文 A5 数据集构造一致: 采纳框是"模型补充统计管线的漏检框"。
"""
import os
import sys
import json
import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from rev5_common import BASE

from auto_pseudo import IoU
from auto_pseudo_diag import DiagParams, detect_diag
from ultralytics import YOLO

R1_WEIGHTS = os.path.join(BASE, "runs", "pseudo_weak_r1", "weights", "best.pt")
HOT_SRC = os.path.join(BASE, "baseline", "hot")
HOT_LBL = os.path.join(BASE, "baseline", "hot_labels")
OUT = os.path.join(BASE, "rev5_experiments", "tauadopt_calibration.json")

IOU_MATCH = 0.5
CONF_SCAN = [0.50, 0.60, 0.70, 0.80, 0.90]

# 与 A3/weak_r1 数据集完全一致的统计参数 (run_ablation.py FROZEN_PARAMS)
FROZEN_PARAMS = dict(
    k_sigma=2.3, blur_size=3, expand_ratio=0.0,
    border_min_factor=0.40, score_threshold=0.38,
    use_close=True, fit_to_core=True, core_expand_ratio=0.0,
    core_delta_mult=1.0, expand_min_px=1,
    use_local_rescue=True, k_local=2.2, local_win=31,
    local_min_delta=4.0, use_local_bg=True,
)
TAU_QUALITY = 0.45


def detect_pseudo(img_path):
    """统计管线在 calibration 图上生成的候选(与 A3 标签同源)。"""
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    H, W = gray.shape
    cands, _ = detect_diag(gray, DiagParams(**FROZEN_PARAMS))
    boxes = []
    for c in cands:
        if c["status"] != "accepted":
            continue
        if c["score"] < TAU_QUALITY:
            continue
        x, y, w, h = c["box"]
        boxes.append((x, y, w, h))
    return boxes


def read_gt(path, W, H):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            _, xc, yc, nw, nh = p[:5]
            xc, yc, nw, nh = float(xc), float(yc), float(nw), float(nh)
            boxes.append(((xc - nw / 2) * W, (yc - nh / 2) * H,
                          nw * W, nh * H))
    return boxes


def main():
    print("loading R1:", R1_WEIGHTS)
    model = YOLO(R1_WEIGHTS)

    imgs = sorted(f for f in os.listdir(HOT_SRC)
                  if f.lower().endswith((".png", ".jpg")))
    imgs = [f for f in imgs
            if os.path.exists(os.path.join(
                HOT_LBL, os.path.splitext(f)[0] + ".txt"))]
    print("calibration GT images:", len(imgs))

    data = []  # per-image: gt, baseline_pseudo, preds(conf,box)

    for name in imgs:
        img = cv2.imread(os.path.join(HOT_SRC, name))
        if img is None:
            continue
        stem = os.path.splitext(name)[0]
        H, W = img.shape[:2]

        gts = read_gt(os.path.join(HOT_LBL, stem + ".txt"), W, H)
        pseudo = detect_pseudo(os.path.join(HOT_SRC, name))
        res = model.predict(img, conf=0.20, iou=0.5, imgsz=320,
                            verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                preds.append(((x1, y1, x2 - x1, y2 - y1), conf))
        data.append({"gt": gts, "pseudo": pseudo, "preds": preds})

    n_gt = sum(len(d["gt"]) for d in data)
    n_pseudo = sum(len(d["pseudo"]) for d in data)
    print("total GT:", n_gt, "| total baseline pseudo:", n_pseudo)

    rows = []

    for tau in CONF_SCAN:
        tp = fp = fn = 0
        adopted_total = 0
        for d in data:
            gts = d["gt"]
            pseudo = list(d["pseudo"])
            adopted = []
            for (db, conf) in d["preds"]:
                if conf < tau:
                    continue
                if any(IoU(db, b) >= 0.5 for b in pseudo + adopted):
                    continue
                adopted.append(db)
            adopted_total += len(adopted)
            final = pseudo + adopted

            gt_used = [False] * len(gts)
            for fb in final:
                hit_i, best = None, 0.0
                for i, g in enumerate(gts):
                    if gt_used[i]:
                        continue
                    v = IoU(fb, g)
                    if v > best:
                        best, hit_i = v, i
                if best >= IOU_MATCH:
                    tp += 1
                    gt_used[hit_i] = True
                else:
                    fp += 1
            fn += sum(1 for u in gt_used if not u)

        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F2 = 5 * P * R / (4 * P + R) if (4 * P + R) else 0.0
        rows.append({
            "tau_adopt": tau,
            "adopted_boxes": adopted_total,
            "precision": round(P, 4),
            "recall": round(R, 4),
            "F2": round(F2, 4),
        })
        print("tau=%.2f adopted=%d P=%.4f R=%.4f F2=%.4f"
              % (tau, adopted_total, P, R, F2))

    # 选择规则: F2 最大且 precision 不低于 baseline pseudo 的 P 一个阈值
    # 这里直接报 max F2, 同时给 precision 数据供论文约束选择
    best = max(rows, key=lambda r: r["F2"])
    result = {
        "protocol": ("calibration-only (197 baseline GT); adoption = R1 "
                     "preds conf>=tau with IoU<0.5 to statistical pseudo"),
        "best_tau_adopt_by_F2": best["tau_adopt"],
        "rows": rows,
        "note": ("test-set adopt@0.5/0.7 comparison demoted to post-hoc "
                 "sensitivity; the calibrated tau_adopt is used in the "
                 "final model"),
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("best tau_adopt by calibration F2:", best["tau_adopt"])
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
