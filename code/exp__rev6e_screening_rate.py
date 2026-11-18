# -*- coding: utf-8 -*-
"""
rev6e_screening_rate.py
======================
在新操作点 (g=0.42) 的冻结模型上, 重新计算论文里的**运行筛选指标**:

  - standard detection recall (mAP/IoU-0.5 口径) 由 rev5_common 给出;
  - **screening hit rate**: 在 conf=0.25 / IoU=0.45 的较宽松匹配下, 一个
    实例只要被"近似框住"就算命中 —— 足以把缺陷板推给人工复核。
    论文此前报 242/279 = 86.7%, 那是旧操作点 (g=0.45) 的模型, 必须重算。
  - 同时统计 false alarms (未匹配到任何 GT 的预测框数) 与每图误报数。

对同一张图的每个 GT 实例, 匹配规则:
  预测框与该实例 IoU >= IOU_MATCH 且置信度 >= CONF_OP, 取 IoU 最大者;
  一个预测框只能匹配一个实例 (贪心, 先高 IoU)。

输出: rev6e_screening_rate.json
"""
import json
import os
import sys

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

CONF_OP = 0.25
IOU_MATCH = 0.45
IMGSZ = 320

MODELS = {
    "A5_full_g42": r"rev5_runs\pseudo_rev6_g42_r2_seed0\weights\best.pt",
    "A3_r1_g42":   r"rev5_runs\pseudo_rev6_g42_r1_seed0\weights\best.pt",
}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    from ultralytics import YOLO

    root = os.path.join(BASE, "yolo_rev6_g42_r2")
    img_dir = os.path.join(root, "images", "test")
    lbl_dir = os.path.join(root, "labels", "test")
    names = sorted(os.listdir(img_dir))
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    names = [n for n in names if n.lower().endswith(exts)]
    print("test images: %d" % len(names), flush=True)

    out = {"conf_op": CONF_OP, "iou_match": IOU_MATCH, "n_test_images": len(names),
           "models": {}}

    for tag, rel in MODELS.items():
        w = os.path.join(BASE, rel)
        if not os.path.exists(w):
            print("[%s] weights missing: %s" % (tag, w), flush=True)
            continue
        model = YOLO(w)
        n_inst = n_hit = n_fp = n_pred = 0
        det = model.predict(source=img_dir, imgsz=IMGSZ, conf=CONF_OP, iou=0.7,
                            device="cpu", stream=True, verbose=False)
        for res, name in zip(det, names):
            h, w_ = res.orig_shape
            gts = []
            lp = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lp):
                for line in open(lp):
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cx, cy, bw, bh = (float(v) for v in p[1:5])
                    gts.append(((cx - bw / 2) * w_, (cy - bh / 2) * h,
                                (cx + bw / 2) * w_, (cy + bh / 2) * h))
            preds = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                preds = [(tuple(map(float, xyxy[i])), float(conf[i]))
                         for i in range(len(xyxy))]
            n_inst += len(gts)
            n_pred += len(preds)

            # 贪心匹配: 先取 IoU 最高的 (pred, gt) 对
            pairs = []
            for pi, (pb, pc) in enumerate(preds):
                for gi, gb in enumerate(gts):
                    v = iou(pb, gb)
                    if v >= IOU_MATCH:
                        pairs.append((v, pi, gi))
            pairs.sort(reverse=True)
            used_p, used_g = set(), set()
            for v, pi, gi in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
            n_hit += len(used_g)
            n_fp += len(preds) - len(used_p)

        r = {"n_instances": n_inst, "hit": n_hit, "n_pred": n_pred, "false_alarms": n_fp,
             "hit_rate": round(n_hit / max(n_inst, 1), 4),
             "fp_per_image": round(n_fp / max(len(names), 1), 3)}
        out["models"][tag] = r
        print("[%s] hit %d/%d = %.1f%%   preds=%d  FP=%d (%.2f/img)"
              % (tag, n_hit, n_inst, 100.0 * r["hit_rate"], n_pred, n_fp,
                 r["fp_per_image"]), flush=True)

    p = os.path.join(_HERE, "rev6e_screening_rate.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", p, flush=True)


if __name__ == "__main__":
    main()
