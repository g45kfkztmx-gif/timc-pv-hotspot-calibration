# -*- coding: utf-8 -*-
"""
rev6h_failure_analysis.py
=========================
Section V-B 的**失败模式分类**在新操作点 (g=0.42) 上重跑, 并把
"boundary-adjacent" 判据从**框心**改成**框边距离**, 与 Section III-B
的 border penalty 定义完全闭环。

判据 (统一口径)
----------------
令预测/真值框 B=(x1,y1,x2,y2), 图像尺寸 (W,H):

    d_b  = min(x1, y1, W - x2, H - y2)          # 框边到最近图像边的距离
    d_b* = d_b / min(W, H)                      # 归一化
    boundary-adjacent  <=>  d_b* < ETA          # ETA = 0.15

ETA 取 0.15, 与 Section III-B 中 border penalty 的归一化常数一致。
**不用框心** —— 大框的框心可以离边界很远, 但框本身已经贴边, 与
border artifact 的物理含义不符。

评估口径
--------
conf = 0.25, NMS IoU = 0.7, imgsz = 320 (与筛选口径一致), device = cpu
(避免与正在进行的 Phase 3 训练抢 GPU)。
贪心匹配 IoU >= 0.5 记为 TP;
未匹配且与任一 GT 的 max IoU ∈ [0.3, 0.5) -> mislocalization;
其余未匹配预测 -> FP, 再按下述优先级归入 5 类。

FP 分类优先级 (先命中先归类)
  1. border-adjacent false response   d_b*(pred) < ETA
  2. small artifact                   area < 30 px^2  (与 Q_e 的降级阈值一致)
  3. reflective structure             peak z-score >= 2.0 但 fill ratio < 0.5
                                      (局部极亮、面积支撑不足 -> 镜面/反光)
  4. low contrast background          mu_R - mu_B < 20 gray levels
                                      (与 Q_c = sigmoid((mu_R-mu_B)/20) 的尺度一致)
  5. annotation ambiguity / unlabelled 其余

FN 分类优先级
  1. occlusion                       与该 GT 的 IoU >= 0.25 的另一 GT 已被检出
  2. boundary truncation              d_b*(gt) < ETA
  3. small / diffuse hotspot         area < 30 px^2, 或边缘密度 < 0.02
  4. weak contrast hotspot           mu_R - mu_B < 20 gray levels
  5. other / ambiguous ground truth  其余

输出: rev6h_failure_analysis.json  (+ 可选重建 figure 8 的数据)
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
NMS_IOU = float(os.environ.get("NMS_IOU", "0.70"))
IOU_TP = 0.50
IOU_MISLOC = 0.30
IMGSZ = 320
ETA = 0.15            # border-adjacent 归一化阈值 (= border penalty 常数)
BORDER_MODE = os.environ.get("BORDER_MODE", "edge")   # 'edge' | 'center'
SMALL_AREA = 30.0     # px^2
CONTRAST_MIN = 20.0   # gray levels
PEAK_Z_HIGH = 2.0
FILL_LOW = 0.5
DIFFUSE_EDGE = 0.02

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


def border_dist(box, W, H, mode=None):
    """mode='edge' -> 框边距离; mode='center' -> 框心距离 (= 现版 border penalty)。"""
    mode = mode or BORDER_MODE
    x1, y1, x2, y2 = box
    if mode == "edge":
        d = min(x1, y1, W - x2, H - y2)
    else:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = min(cx, cy, W - cx, H - cy)
    return d / float(min(W, H))


def edge_density(gray, box):
    """Canny 边缘占比 (hysteresis 50/150, 与 Q_e 一致)。"""
    try:
        import cv2
    except Exception:
        return -1.0
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return -1.0
    roi = np.ascontiguousarray(gray[y1:y2, x1:x2])
    e = cv2.Canny(roi, 50, 150)
    return float((e > 0).mean())


def box_stats(gray, box):
    """返回 (contrast mu_R-mu_B, peak z-score, fill ratio)。"""
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    roi = gray[y1:y2, x1:x2].astype(np.float64)
    mu_r = float(roi.mean())
    mu_i, sd_i = float(gray.mean()), float(gray.std()) + 1e-6
    peak_z = (float(roi.max()) - mu_i) / sd_i
    # 背景 = 以 R 为中心、外扩 0.5*max(w,h) 的环带
    pad = int(round(0.5 * max(x2 - x1, y2 - y1)))
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(W, x2 + pad), min(H, y2 + pad)
    ring = gray[by1:by2, bx1:bx2].astype(np.float64)
    mask = np.ones_like(ring, dtype=bool)
    mask[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1] = False
    mu_b = float(ring[mask].mean()) if mask.any() else float(gray.mean())
    # fill ratio: 高于 (mu_i + sd_i) 的像素占比 / 0.8
    fill = float((roi > (mu_i + sd_i)).mean()) / 0.8
    return mu_r - mu_b, peak_z, min(fill, 1.0)


def main():
    from ultralytics import YOLO

    w_path = os.path.join(BASE, WEIGHTS)
    assert os.path.exists(w_path), "weights missing: %s" % w_path
    root = os.path.join(BASE, DATA)
    img_dir = os.path.join(root, "images", "test")
    lbl_dir = os.path.join(root, "labels", "test")
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(exts))
    print("test images: %d | weights: %s" % (len(names), WEIGHTS), flush=True)

    model = YOLO(w_path)
    det = model.predict(source=img_dir, imgsz=IMGSZ, conf=CONF_OP, iou=NMS_IOU,
                        device="cpu", stream=True, verbose=False)

    fp_cat = {}
    fn_cat = {}
    n_tp = n_fp = n_fn = n_mis = n_gt = n_pred = 0
    fp_border_uncertain = 0
    examples = {"missed": None, "false_alarm": None}
    # 富集统计: 每类对象的 d_b* 分布
    hist = {"gt_matched_near": 0, "gt_matched": 0,
            "gt_missed_near": 0, "gt_missed": 0,
            "pred_tp_near": 0, "pred_tp": 0,
            "pred_fp_near": 0, "pred_fp": 0}

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

        n_gt += len(gts)
        n_pred += len(preds)

        # --- 贪心匹配 IoU >= 0.5 ---
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
        n_tp += len(up)

        for gi, gb in enumerate(gts):
            near = border_dist(gb, W, H) < ETA
            if gi in ug:
                hist["gt_matched"] += 1
                hist["gt_matched_near"] += int(near)
            else:
                hist["gt_missed"] += 1
                hist["gt_missed_near"] += int(near)
        for pi, (pb, _) in enumerate(preds):
            near = border_dist(pb, W, H) < ETA
            if pi in up:
                hist["pred_tp"] += 1
                hist["pred_tp_near"] += int(near)
            elif len(gts) > 0:
                hist["pred_fp"] += 1
                hist["pred_fp_near"] += int(near)

        # --- FN ---
        for gi, gb in enumerate(gts):
            if gi in ug:
                continue
            n_fn += 1
            # 1. occlusion: 另一个已被检出的 GT 与它重叠
            occ = False
            for gj, gb2 in enumerate(gts):
                if gj == gi or gj not in ug:
                    continue
                if iou(gb, gb2) >= 0.25:
                    occ = True
                    break
            if occ:
                k = "occlusion"
            elif border_dist(gb, W, H) < ETA:
                k = "boundary truncation"
            else:
                area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                ed = edge_density(gray, gb)
                if area < SMALL_AREA or (ed >= 0 and ed < DIFFUSE_EDGE):
                    k = "small / diffuse hotspot"
                else:
                    c, _, _ = box_stats(gray, gb)
                    k = "weak contrast hotspot" if c < CONTRAST_MIN else "other / ambiguous ground truth"
            fn_cat[k] = fn_cat.get(k, 0) + 1
            if examples["missed"] is None:
                examples["missed"] = {"image": name, "box": gb, "cat": k,
                                      "conf_max": 0.0}

        # --- FP / mislocalization (仅统计测试集上真正未匹配的预测) ---
        # NOTE: 论文的 FP 统计只在 200 张含缺陷图上做, 避免把纯背景图的
        #       预测计入; 这里保持与旧口径一致。
        is_hot = len(gts) > 0
        for pi, (pb, pc) in enumerate(preds):
            if pi in up:
                continue
            if not is_hot:
                continue
            n_fp += 1
            best = max((iou(pb, gb) for gb in gts), default=0.0)
            if best >= IOU_MISLOC:
                n_mis += 1
                continue
            if border_dist(pb, W, H) < ETA:
                k = "border-adjacent false response"
            else:
                area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                c, pz, fill = box_stats(gray, pb)
                if area < SMALL_AREA:
                    k = "small artifact"
                elif pz >= PEAK_Z_HIGH and fill < FILL_LOW:
                    k = "reflective structure"
                elif c < CONTRAST_MIN:
                    k = "low-contrast background"
                else:
                    k = "annotation ambiguity / unlabelled"
            fp_cat[k] = fp_cat.get(k, 0) + 1
            if examples["false_alarm"] is None:
                examples["false_alarm"] = {"image": name, "box": pb, "cat": k,
                                           "conf": pc}

    out = {
        "model": WEIGHTS,
        "operating_point": {"effective_gate": 0.42, "conf_op": CONF_OP,
                            "nms_iou": NMS_IOU, "imgsz": IMGSZ,
                            "iou_tp": IOU_TP, "iou_misloc": IOU_MISLOC},
        "border_criterion": {"mode": BORDER_MODE,
                             "definition": ("d_b = min(x1,y1,W-x2,H-y2)  [edge]"
                                            if BORDER_MODE == "edge" else
                                            "d_b = min(cx,cy,W-cx,H-cy)  [center]"),
                             "normalisation": "d_b* = d_b / min(W,H)",
                             "eta": ETA,
                             "note": "eta = 0.15 matches the border-penalty normalisation constant"},
        "constants": {"small_area_px2": SMALL_AREA, "contrast_min_gray": CONTRAST_MIN,
                      "peak_z_high": PEAK_Z_HIGH, "fill_low": FILL_LOW,
                      "diffuse_edge_density": DIFFUSE_EDGE},
        "n_test_images": len(names),
        "n_ground_truth": n_gt,
        "n_predictions": n_pred,
        "n_true_positive": n_tp,
        "n_false_positive": n_fp,
        "n_false_negative": n_fn,
        "n_mislocalized": n_mis,
        "fp_categories": fp_cat,
        "fn_categories": fn_cat,
        "border_enrichment": {
            "gt_matched_near_frac": round(hist["gt_matched_near"] / max(hist["gt_matched"], 1), 4),
            "gt_missed_near_frac": round(hist["gt_missed_near"] / max(hist["gt_missed"], 1), 4),
            "pred_tp_near_frac": round(hist["pred_tp_near"] / max(hist["pred_tp"], 1), 4),
            "pred_fp_near_frac": round(hist["pred_fp_near"] / max(hist["pred_fp"], 1), 4),
            "raw": hist,
        },
        "fp_examples": examples,
        "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    p = os.path.join(_HERE, "rev6h_failure_analysis_%s.json" % BORDER_MODE)
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nGT=%d  preds=%d" % (n_gt, n_pred), flush=True)
    print("TP=%d  FP=%d  FN=%d  mislocalized=%d" % (n_tp, n_fp, n_fn, n_mis), flush=True)
    print("FP categories: %s" % json.dumps(fp_cat, ensure_ascii=False), flush=True)
    print("FN categories: %s" % json.dumps(fn_cat, ensure_ascii=False), flush=True)
    print("saved ->", p, flush=True)


if __name__ == "__main__":
    main()
