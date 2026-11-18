# -*- coding: utf-8 -*-
"""
tim_eval.py —— TIM 修订新增实验共用的 COCO 风格评估器
======================================================================
独立实现 mAP@50 / mAP@50-95 (101 点插值, 单类, 贪心匹配), 以及
部署口径指标 (FP/image 预算下的 P/R/F2 与 IoU 0.45 筛查命中率)。
用法: 与 ultralytics val 的输出对拍 (见 tim_exp3 的校验步)。
"""
import os


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    sa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    sb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, sa + sb - inter)


def load_gt(lbl_dir, W=60, H=110):
    """YOLO 归一化标签 -> {stem: [xyxy px]}"""
    gt = {}
    if not os.path.isdir(lbl_dir):
        return gt
    for f in sorted(os.listdir(lbl_dir)):
        if not f.endswith(".txt"):
            continue
        stem = os.path.splitext(f)[0]
        boxes = []
        for line in open(os.path.join(lbl_dir, f), encoding="utf8"):
            parts = line.split()
            if len(parts) < 5:
                continue
            _c, xc, yc, w, h = [float(x) for x in parts[:5]]
            x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
            boxes.append((x1, y1, x1 + w * W, y1 + h * H))
        gt[stem] = boxes
    return gt


def _match(preds, gt, iou_thr):
    """preds: [(stem, xyxy, conf)] 已按 conf 降序 -> tp/fp 序列"""
    matched = {s: [False] * len(b) for s, b in gt.items()}
    tp, fp = [], []
    for stem, box, _c in preds:
        gts = gt.get(stem, [])
        hit = False
        for j, g in enumerate(gts):
            if matched[stem][j]:
                continue
            if iou_xyxy(box, g) >= iou_thr:
                matched[stem][j] = True
                hit = True
                break
        tp.append(1 if hit else 0)
        fp.append(0 if hit else 1)
    return tp, fp


def ap_at(preds, gt, iou_thr):
    """COCO 风格 AP (101 点插值)"""
    if not any(gt.values()):
        return 0.0
    if not preds:
        return 0.0
    tp, fp = _match(preds, gt, iou_thr)
    n_gt = sum(len(v) for v in gt.values())
    ctp, cfp = 0, 0
    recs, precs = [], []
    for t, f in zip(tp, fp):
        ctp += t
        cfp += f
        recs.append(ctp / n_gt)
        precs.append(ctp / max(1e-9, ctp + cfp))
    # 101 点插值: 每个 recall 层级取 >= 该层级的最大 precision
    ap = 0.0
    for r in [i / 100.0 for i in range(101)]:
        p = 0.0
        for rec, prec in zip(recs, precs):
            if rec >= r:
                p = max(p, prec)
        ap += p / 101.0
    return ap


def map5095(preds, gt):
    return sum(ap_at(preds, gt, t / 100.0)
               for t in range(50, 100, 5)) / 10.0


def prf_at_fp_budget(preds, gt, n_img, fp_per_img=0.25, iou_thr=0.5):
    """按 conf 降序取前缀, 使 FP/image <= 预算; 报 P/R/F2 与 IoU 0.45 命中率"""
    budget = fp_per_img * n_img
    tp, fp = _match(preds, gt, iou_thr)
    ctp, cfp, cut = 0, 0, 0
    for t, f in zip(tp, fp):
        if cfp + f > budget + 1e-9:
            break
        ctp += t
        cfp += f
        cut += 1
    prec = ctp / max(1e-9, ctp + cfp)
    rec = ctp / max(1, sum(len(v) for v in gt.values()))
    f2 = 5 * prec * rec / max(1e-9, 4 * prec + rec)
    # 筛查命中率: 同一前缀下以 IoU 0.45 重新匹配
    tp45, _ = _match(preds[:cut], gt, 0.45)
    n_gt = sum(len(v) for v in gt.values())
    return {"P": round(prec, 4), "R": round(rec, 4), "F2": round(f2, 4),
            "hit45": round(sum(tp45) / max(1, n_gt), 4),
            "fp_per_img": round(cfp / max(1, n_img), 4),
            "n_det": cut}


def read_pred_txt(pred_dir, W=60, H=110):
    """ultralytics save_txt 输出 -> [(stem, xyxy, conf)]"""
    out = []
    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".txt"):
            continue
        stem = os.path.splitext(f)[0]
        for line in open(os.path.join(pred_dir, f), encoding="utf8"):
            parts = line.split()
            if len(parts) < 5:
                continue
            _c, xc, yc, w, h = [float(x) for x in parts[:5]]
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
            out.append((stem, (x1, y1, x1 + w * W, y1 + h * H), conf))
    out.sort(key=lambda t: -t[2])
    return out
