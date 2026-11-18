# -*- coding: utf-8 -*-
"""
rev6h2_border_eta_sweep.py
==========================
问题: rev6h 用 eta=0.15 (= 0.15*min(W,H) ~ 77 px) 作为 "boundary-adjacent" 的
二值判据时, 连**正确检出**的框也有 94% 落入该区间 —— 判据没有区分度。
(0.15 是 border penalty 的**归一化常数**, 不是分类阈值; 两者不该混用。)

本脚本扫描一批 eta, 给出富集表, 以便选一个有区分度的判据:

    eta        FP_near/FP    TP_near/TP     missed_near/missed  matched_near/matched
    0.02  ...
    ...

同时给出两种边界距离口径: 框边 (edge) / 框心 (center)。

输出: rev6h2_border_eta_sweep.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

CONF_OP = 0.25
NMS_IOU = 0.70
IOU_TP = 0.50
IOU_MISLOC = 0.30
IMGSZ = 320
ETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
WEIGHTS = r"rev5_runs\pseudo_rev6_g42_r2_seed0\weights\best.pt"
DATA = "yolo_rev6_g42_r2"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def dnorm(box, W, H, mode):
    x1, y1, x2, y2 = box
    if mode == "edge":
        d = min(x1, y1, W - x2, H - y2)
    else:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = min(cx, cy, W - cx, H - cy)
    return d / float(min(W, H))


def main():
    from ultralytics import YOLO

    root = os.path.join(BASE, DATA)
    img_dir = os.path.join(root, "images", "test")
    lbl_dir = os.path.join(root, "labels", "test")
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(exts))

    model = YOLO(os.path.join(BASE, WEIGHTS))
    det = model.predict(source=img_dir, imgsz=IMGSZ, conf=CONF_OP, iou=NMS_IOU,
                        device="cpu", stream=True, verbose=False)

    rec = {"edge": {"tp": [], "fp": [], "gtm": [], "gtx": []},
           "center": {"tp": [], "fp": [], "gtm": [], "gtx": []}}

    for res, name in zip(det, names):
        H, W = res.orig_shape
        gts = []
        lp = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                gts.append(((cx - bw / 2) * W, (cy - bh / 2) * H,
                            (cx + bw / 2) * W, (cy + bh / 2) * H))
        preds = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            for i in range(len(xyxy)):
                preds.append(tuple(map(float, xyxy[i])))

        pairs = []
        for pi, pb in enumerate(preds):
            for gi, gb in enumerate(gts):
                v = iou(pb, gb)
                if v >= IOU_TP:
                    pairs.append((v, pi, gi))
        pairs.sort(reverse=True)
        up, ug = set(), set()
        for v, pi, gi in pairs:
            if pi in up or gi in ug:
                continue
            up.add(pi)
            ug.add(gi)

        for mode in ("edge", "center"):
            for pi, pb in enumerate(preds):
                if len(gts) == 0:
                    continue
                d = dnorm(pb, W, H, mode)
                # 仅统计"非 mislocalization"的预测, 与 rev6h 的 FP 定义一致
                best = max((iou(pb, gb) for gb in gts), default=0.0)
                if pi in up:
                    rec[mode]["tp"].append(d)
                elif best < IOU_MISLOC:
                    rec[mode]["fp"].append(d)
            for gi, gb in enumerate(gts):
                d = dnorm(gb, W, H, mode)
                (rec[mode]["gtm"] if gi in ug else rec[mode]["gtx"]).append(d)

    out = {"etas": ETAS, "weights": WEIGHTS, "conf_op": CONF_OP, "results": {}}
    for mode in ("edge", "center"):
        r = rec[mode]
        tp, fp = np.array(r["tp"]), np.array(r["fp"])
        gtm, gtx = np.array(r["gtm"]), np.array(r["gtx"])
        rows = []
        print("\n=== mode = %s  (nTP=%d nFP=%d nMISS=%d nMATCH=%d)"
              % (mode, len(tp), len(fp), len(gtx), len(gtm)))
        print("%-8s %-14s %-14s %-16s %-16s %s" % (
            "eta", "FP_near/FP", "TP_near/TP", "miss_near/miss",
            "match_near/match", "lift(FP/TP)"))
        for e in ETAS:
            f = float((fp < e).mean()) if len(fp) else 0.0
            t = float((tp < e).mean()) if len(tp) else 0.0
            m = float((gtx < e).mean()) if len(gtx) else 0.0
            g = float((gtm < e).mean()) if len(gtm) else 0.0
            lift = (f / t) if t > 0 else float("nan")
            rows.append({"eta": e, "fp_near_frac": round(f, 4),
                         "tp_near_frac": round(t, 4),
                         "miss_near_frac": round(m, 4),
                         "match_near_frac": round(g, 4),
                         "lift_fp_over_tp": round(lift, 3)})
            print("%-8.3f %-14s %-14s %-16s %-16s %.2f" % (
                e, "%.3f (%d)" % (f, int((fp < e).sum())),
                "%.3f (%d)" % (t, int((tp < e).sum())),
                "%.3f (%d)" % (m, int((gtx < e).sum())),
                "%.3f (%d)" % (g, int((gtm < e).sum())), lift))
        out["results"][mode] = rows

    p = os.path.join(_HERE, "rev6h2_border_eta_sweep.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
