# -*- coding: utf-8 -*-
"""
rev6g_quality_ablation.py
=========================
Review-checklist item A7: component-level ablation of the five-dimensional
quality score, measured at the PSEUDO-LABEL level on the 197-image
calibration set (no detector training required).

The composite score is  Q = sum_i w_i * m_i  with
    w = (contrast .30, peak .25, stability .20, edge .15, fill .10)
followed by the multiplicative border factor.  Each variant removes one
evidence term and RENORMALISES the remaining weights to unit sum, so the
ablation isolates the information contributed by that term rather than
merely rescaling the score.

For tiny ROIs (w*h < small_roi_px) the implementation already drops `edge`
and renormalises over the remaining four terms; that behaviour is replicated.

The border-penalty row instead disables the multiplicative border factor
(border_min_factor = 1), which is the only way to ablate that term.

Output: rev6g_quality_ablation.json
"""
import json
import os
import sys
import time

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_pseudo import load_labels, BASELINE_HOT_DIR, BASELINE_LABEL_DIR
from auto_pseudo_diag import DiagParams, detect_diag
from rev5_f_budget import calib_names, greedy_match, fbeta

GATE = 0.42
SMALL_ROI_PX = 30
EXEMPT_SCORE = 0.90
EXEMPT_FACTOR = 0.80

W_FULL = {"contrast": 0.30, "peak": 0.25, "stability": 0.20,
          "edge": 0.15, "fill": 0.10}

FROZEN_PARAMS = dict(
    k_sigma=2.3, blur_size=3, expand_ratio=0.0, border_min_factor=0.40,
    score_threshold=0.38, use_close=True, fit_to_core=True,
    core_expand_ratio=0.0, core_delta_mult=1.0, expand_min_px=1,
    use_local_rescue=True, k_local=2.2, local_win=31,
    local_min_delta=4.0, use_local_bg=True,
)


def border_factor(bp, base_score, border_min_factor):
    f = border_min_factor + (1.0 - border_min_factor) * bp
    if base_score >= EXEMPT_SCORE:
        f = max(f, EXEMPT_FACTOR)
    return f


def composite(metrics, roi_px, drop=None, weights=None):
    """Recompute the composite base score, renormalising over active terms."""
    w = dict(W_FULL if weights is None else weights)
    if drop is not None:
        w.pop(drop, None)
    active = list(w.keys())
    if roi_px < SMALL_ROI_PX:
        active = [k for k in active if k != "edge"]
    tot = sum(w[k] for k in active)
    if tot <= 0:
        return 0.0
    return sum(w[k] * metrics[k] for k in active) / tot


def main():
    names = calib_names(197)
    print("calibration images:", len(names), flush=True)

    data = []
    for i, name in enumerate(names):
        src = os.path.join(BASELINE_HOT_DIR, name)
        bgr = cv2.imread(src)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        stem = os.path.splitext(name)[0]
        gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if gray.ndim == 3:
            gray = gray[:, :, 0]

        gt = load_labels(os.path.join(BASELINE_LABEL_DIR, stem + ".txt"), W, H)

        cands, _ = detect_diag(gray, DiagParams(**FROZEN_PARAMS))
        keep = []
        for c in cands:
            if c["status"] != "accepted":
                continue
            m = c["metrics"]
            if not m or "fill" not in m:
                continue
            keep.append({"box": c["box"], "metrics": m,
                         "bp": c["border_penalty"],
                         "roi_px": c["box"][2] * c["box"][3]})
        data.append({"gt": gt, "cands": keep})
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(names)), flush=True)

    n_gt = sum(len(d["gt"]) for d in data)
    n_cand = sum(len(d["cands"]) for d in data)
    print("GT instances=%d  raw accepted candidates=%d" % (n_gt, n_cand), flush=True)

    variants = [
        ("full", dict(drop=None, bmf=0.40)),
        ("w/o border penalty", dict(drop=None, bmf=1.00)),
        ("w/o contrast", dict(drop="contrast", bmf=0.40)),
        ("w/o peak", dict(drop="peak", bmf=0.40)),
        ("w/o stability", dict(drop="stability", bmf=0.40)),
        ("w/o edge", dict(drop="edge", bmf=0.40)),
        ("w/o fill", dict(drop="fill", bmf=0.40)),
    ]

    rows = []
    for label, cfg in variants:
        TP = FP = FN = 0
        kept = 0
        for d in data:
            sel = []
            for c in d["cands"]:
                b = composite(c["metrics"], c["roi_px"], drop=cfg["drop"])
                s = b * border_factor(c["bp"], b, cfg["bmf"])
                if s >= GATE:
                    sel.append(c["box"])
            kept += len(sel)
            acc = [(1.0, b) for b in sel]
            used = greedy_match(acc, d["gt"])
            tp = sum(used)
            TP += tp
            FN += len(d["gt"]) - tp
            FP += len(acc) - tp
        p = TP / max(TP + FP, 1)
        r = TP / max(TP + FN, 1)
        rows.append({"variant": label, "box_count": kept,
                     "boxes_per_image": round(kept / max(len(data), 1), 2),
                     "TP": TP, "FP": FP, "FN": FN,
                     "P": round(p, 4), "R": round(r, 4),
                     "F2": round(fbeta(p, r), 4)})
        print("%-20s boxes=%4d (%.2f/img) TP=%3d FP=%3d FN=%3d "
              "P=%.4f R=%.4f F2=%.4f"
              % (label, kept, kept / max(len(data), 1), TP, FP, FN, p, r,
                 fbeta(p, r)), flush=True)

    out = os.path.join(BASE, "rev5_experiments", "rev6g_quality_ablation.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump({"protocol": ("197-image calibration set; candidates from the "
                                "frozen generator; composite score recomputed "
                                "with the listed term removed and remaining "
                                "weights renormalised; accepted at g=0.42; "
                                "matched to rounded-integer GT at IoU>=0.5"),
                   "gate": GATE, "weights": W_FULL,
                   "n_images": len(data), "n_gt": n_gt, "rows": rows,
                   "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, indent=2, ensure_ascii=False)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
