# -*- coding: utf-8 -*-
"""
rev6f_tauadopt_g42.py   (derived from rev5_g_tauadopt_gtcheck.py)
Same dual-convention adoption-gate table, but at the FINAL operating point:
R1 weights = rev6_g42_r1_seed0, effective quality gate = 0.42.
"""
_orig_doc = """
rev5_g_tauadopt_gtcheck.py
==========================
发现的口径不一致问题:
  - 论文的 tau 选择用 sweep_baseline.py -> auto_pseudo.load_labels -> GT **取整**
  - tau_adopt 选择用 rev5_b_tauadopt.py -> 自写 read_gt -> GT **保留浮点**
在 7x6 像素的小框上, 取整会让 IoU 跌破 0.5, 导致同一批框的 TP 从 195 掉到 168。

本脚本用同一套 R1 预测 + 同一套统计伪标签, 分别在两种 GT 口径下重算
tau_adopt 选择表, 判断"选择的 tau_adopt 是否稳健"。

输出: rev5_experiments/tauadopt_gt_convention.json
"""
import os
import sys
import json

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rev5_common import list_images
from rev5_f_budget import (fbeta, greedy_match, calib_names, CONF_PREDICT,
                           TAU_ADOPT_GRID, BASE as _B)
from auto_pseudo import load_labels, IoU, BASELINE_HOT_DIR, BASELINE_LABEL_DIR
from auto_pseudo_diag import DiagParams, detect_diag

R1_WEIGHTS = os.path.join(BASE, "rev5_runs", "pseudo_rev6_g42_r1_seed0", "weights", "best.pt")
IOU_MATCH = 0.5

FROZEN_PARAMS = dict(
    k_sigma=2.3, blur_size=3, expand_ratio=0.0, border_min_factor=0.40,
    score_threshold=0.38, use_close=True, fit_to_core=True,
    core_expand_ratio=0.0, core_delta_mult=1.0, expand_min_px=1,
    use_local_rescue=True, k_local=2.2, local_win=31,
    local_min_delta=4.0, use_local_bg=True,
)
TAU_QUALITY = 0.42   # effective gate of the final operating point


def read_gt_float(path, W, H):
    """rev5_b 口径: 保留浮点 (与标注文件一致, 不取整)"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            _, xc, yc, nw, nh = p[:5]
            xc, yc, nw, nh = float(xc), float(yc), float(nw), float(nh)
            out.append(((xc - nw / 2) * W, (yc - nh / 2) * H, nw * W, nh * H))
    return out


def main():
    from ultralytics import YOLO

    names = calib_names(197)
    print("calibration images:", len(names), flush=True)
    model = YOLO(R1_WEIGHTS)

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

        cands, _ = detect_diag(gray, DiagParams(**FROZEN_PARAMS))
        pseudo = [c["box"] for c in cands
                  if c["status"] == "accepted" and c["score"] >= TAU_QUALITY]

        gt_int = load_labels(os.path.join(BASELINE_LABEL_DIR, stem + ".txt"), W, H)
        gt_flt = read_gt_float(os.path.join(BASELINE_LABEL_DIR, stem + ".txt"), W, H)

        res = model.predict(bgr, conf=CONF_PREDICT, iou=0.5, imgsz=320,
                            verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds.append(((x1, y1, x2 - x1, y2 - y1), float(b.conf[0])))

        data.append({"gt_int": gt_int, "gt_flt": gt_flt,
                     "pseudo": pseudo, "preds": preds})
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(names)), flush=True)

    n_gt_int = sum(len(d["gt_int"]) for d in data)
    n_gt_flt = sum(len(d["gt_flt"]) for d in data)
    n_pseudo = sum(len(d["pseudo"]) for d in data)
    print("GT(int)=%d GT(float)=%d pseudo boxes=%d"
          % (n_gt_int, n_gt_flt, n_pseudo), flush=True)

    result = {"n_gt_int": n_gt_int, "n_gt_float": n_gt_flt,
              "n_pseudo": n_pseudo, "conventions": {}}

    for conv in ("gt_int", "gt_flt"):
        rows = []
        for tau in TAU_ADOPT_GRID:
            TP = FP = FN = 0
            adopted_total = 0
            for d in data:
                pseudo = d["pseudo"]
                adopted = []
                for (db, conf) in d["preds"]:
                    if conf < tau:
                        continue
                    if any(IoU(db, b) >= 0.5 for b in pseudo + adopted):
                        continue
                    adopted.append(db)
                adopted_total += len(adopted)
                acc = [(1.0, b) for b in pseudo + adopted]
                used = greedy_match(acc, d[conv])
                tp = sum(used)
                TP += tp
                FN += len(d[conv]) - tp
                FP += len(acc) - tp
            p = TP / max(TP + FP, 1)
            r = TP / max(TP + FN, 1)
            rows.append({"tau_adopt": tau, "adopted_boxes": adopted_total,
                         "TP": TP, "FP": FP, "FN": FN,
                         "P": round(p, 4), "R": round(r, 4),
                         "F2": round(fbeta(p, r), 4)})
            print("[%s] tau=%.2f adopted=%2d TP=%3d FP=%3d FN=%3d "
                  "P=%.4f R=%.4f F2=%.4f"
                  % (conv, tau, adopted_total, TP, FP, FN, p, r,
                     fbeta(p, r)), flush=True)
        best = max(rows, key=lambda x: x["F2"])
        result["conventions"][conv] = {"rows": rows,
                                       "best_tau_adopt": best["tau_adopt"],
                                       "best_F2": best["F2"]}
        print("[%s] BEST tau_adopt=%.2f (F2=%.4f)"
              % (conv, best["tau_adopt"], best["F2"]), flush=True)

    out = os.path.join(BASE, "rev5_experiments", "rev6f_tauadopt_g42.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
