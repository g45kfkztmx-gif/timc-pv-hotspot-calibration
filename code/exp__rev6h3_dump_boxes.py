# -*- coding: utf-8 -*-
"""
rev6h3_dump_boxes.py
====================
把冻结模型 (g=0.42, A5 = rev6_g42_r2_seed0) 在 350 张测试图上的
**逐框记录**导出为 JSON, 便于对 Section V-B 的失败模式做任意口径的再分析。

每条记录包含:
  kind      : 'pred_tp' | 'pred_fp' | 'pred_misloc' | 'gt_matched' | 'gt_missed'
  box       : (x1,y1,x2,y2) 像素
  W, H
  d_edge    : min(x1,y1,W-x2,H-y2)                (px)
  d_center  : min(cx,cy,W-cx,H-cy)                (px)
  area, aspect, contrast(mu_R-mu_B), peak_z, fill, edge_density

输出: rev6h3_boxes.json
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
TAG = os.environ.get("WEIGHTS_TAG", "rev6_g42_r1")
# 主配置切到无精炼后，误差分析须在 r1 模型上进行（r1 的伪标签基座是 yolo_fin42）
DATA = "yolo_fin42" if TAG.endswith("_r1") else "yolo_rev6_g42_r2"
WEIGHTS = "rev5_runs\\pseudo_%s_seed0\\weights\\best.pt" % TAG


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def stats(gray, box):
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return {}
    roi = gray[y1:y2, x1:x2].astype(np.float64)
    mu_i, sd_i = float(gray.mean()), float(gray.std()) + 1e-6
    pad = int(round(0.5 * max(x2 - x1, y2 - y1)))
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(W, x2 + pad), min(H, y2 + pad)
    ring = gray[by1:by2, bx1:bx2].astype(np.float64)
    m = np.ones_like(ring, dtype=bool)
    m[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1] = False
    mu_b = float(ring[m].mean()) if m.any() else float(gray.mean())
    ed = -1.0
    try:
        import cv2
        e = cv2.Canny(np.ascontiguousarray(gray[y1:y2, x1:x2]), 50, 150)
        ed = float((e > 0).mean())
    except Exception:
        pass
    return {"area": float((x2 - x1) * (y2 - y1)),
            "aspect": float(max(x2 - x1, y2 - y1) / max(min(x2 - x1, y2 - y1), 1)),
            "contrast": round(mu_r_mu_b(mu_b, roi), 3),
            "peak_z": round((float(roi.max()) - mu_i) / sd_i, 3),
            "fill": round(float((roi > (mu_i + sd_i)).mean()), 4),
            "edge_density": round(ed, 5)}


def mu_r_mu_b(mu_b, roi):
    return float(roi.mean()) - mu_b


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

    recs = []
    for res, name in zip(det, names):
        H, W = res.orig_shape
        gray = np.asarray(Image.open(os.path.join(img_dir, name)).convert("L"))
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
            cnf = res.boxes.conf.cpu().numpy()
            preds = [(tuple(map(float, xyxy[i])), float(cnf[i])) for i in range(len(xyxy))]

        pairs = []
        for pi, (pb, _) in enumerate(preds):
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

        def base(b):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return {"box": [round(v, 2) for v in b], "W": W, "H": H,
                    "d_edge": round(min(x1, y1, W - x2, H - y2), 2),
                    "d_center": round(min(cx, cy, W - cx, H - cy), 2),
                    "d_edge_n": round(min(x1, y1, W - x2, H - y2) / min(W, H), 5),
                    "d_center_n": round(min(cx, cy, W - cx, H - cy) / min(W, H), 5)}

        for pi, (pb, pc) in enumerate(preds):
            if len(gts) == 0:
                continue
            best = max((iou(pb, gb) for gb in gts), default=0.0)
            kind = "pred_tp" if pi in up else ("pred_misloc" if best >= IOU_MISLOC else "pred_fp")
            r = base(pb)
            r.update({"kind": kind, "image": name, "conf": round(pc, 4),
                      "iou_best_gt": round(best, 4)})
            r.update(stats(gray, pb))
            recs.append(r)
        for gi, gb in enumerate(gts):
            r = base(gb)
            r.update({"kind": "gt_matched" if gi in ug else "gt_missed",
                      "image": name,
                      "iou_best_pred": round(max((iou(pb, gb) for pb, _ in preds), default=0.0), 4)})
            r.update(stats(gray, gb))
            recs.append(r)

    p = os.path.join(_HERE, "rev6h3_boxes_%s.json" % TAG)
    with open(p, "w", encoding="utf8") as f:
        json.dump({"weights": WEIGHTS, "data": DATA, "conf_op": CONF_OP,
                   "records": recs}, f, indent=1, ensure_ascii=False)
    print("records: %d -> %s" % (len(recs), p), flush=True)


if __name__ == "__main__":
    main()
