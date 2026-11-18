# -*- coding: utf-8 -*-
"""
tim_exp3_baselines.py —— TIM 审稿意见 #6: 经典热成像检测 baseline
======================================================================
在 350 图测试集 (279 实例, yolo_weak test GT) 上按同一评估器比较:
  C1  全局固定阈值:        gray > mean + 3*std           (教科书方法)
  C2  局部背景减除:        gray - medianBlur(gray,31) > mean + 3*std
  C3a 本文统计前端(校准 k=2.3), 关闭质量门与评分          (= 前端本身)
  C3b 本文统计前端(经典 k=3.0), 关闭质量门与评分          (= 未校准前端)
  Ours 报告配置 (校准前端 + 质量门 g*=0.42 + YOLOv8n)
指标: mAP@50 / mAP@50-95, 以及 FP=0.25/img 预算下的 P/R/F2/hit45。
同时用 ultralytics val 对拍, 校验自研评估器 (应复现 0.802)。
输出: tim_baselines_summary.json
"""
import os
import sys
import json
import shutil
import numpy as np
import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.join(BASE, "rev5_experiments")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

import rev5_f_budget as F                                  # noqa: E402
from auto_pseudo_diag import DiagParams, detect_diag       # noqa: E402
import tim_eval as E                                       # noqa: E402

TEST_IMG = os.path.join(BASE, "yolo_weak", "images", "test")
TEST_LBL = os.path.join(BASE, "yolo_weak", "labels", "test")
REF_CALIB = os.path.join(HERE, "budget_calib_N197.json")
MAIN_W = os.path.join(BASE, "rev5_runs",
                      "pseudo_rev5_bud_ours197_r1_seed0", "weights", "best.pt")
OUT = os.path.join(HERE, "tim_baselines_summary.json")


def components_to_boxes(mask, score_img, min_area=2):
    """连通域 -> [(xyxy, score)]; score = 域内 score_img 均值 (用于 AP 排序)"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        sel = lab == i
        out.append(((float(x), float(y), float(x + w), float(y + h)),
                    float(score_img[sel].mean())))
    return out


def c1_global_threshold(gray):
    thr = gray.mean() + 3 * gray.std()
    mask = gray > thr
    return components_to_boxes(mask, gray.astype(np.float32))


def c2_local_bg_sub(gray):
    bg = cv2.medianBlur(gray, 31)
    r = gray.astype(np.float32) - bg.astype(np.float32)
    thr = r.mean() + 3 * r.std()
    mask = r > thr
    return components_to_boxes(mask, r)


def c3_frontend(gray, params):
    """本文统计前端 (阈值+局部救援+形态学+连通域+框扩展), 无质量门/评分"""
    cands, _ = detect_diag(gray, params)
    out = []
    for c in cands:
        x, y, w, h = c["box"]
        out.append(((float(x), float(y), float(x + w), float(y + h)),
                    float(c["score"])))
    return out


def eval_arms(arms, gt, n_img):
    """arms: {name: {stem: [(xyxy, score)]}} -> 指标表"""
    rows = {}
    for name, per_img in arms.items():
        preds = []
        for stem, dets in per_img.items():
            for box, sc in dets:
                preds.append((stem, box, sc))
        preds.sort(key=lambda t: -t[2])
        rows[name] = {
            "mAP50": round(E.ap_at(preds, gt, 0.5), 4),
            "mAP50_95": round(E.map5095(preds, gt), 4),
            **E.prf_at_fp_budget(preds, gt, n_img, 0.25, 0.5),
        }
    return rows


def main():
    ref = json.load(open(REF_CALIB, encoding="utf8"))
    gt = E.load_gt(TEST_LBL)
    n_img = len(os.listdir(TEST_IMG))
    n_inst = sum(len(v) for v in gt.values())
    print("[setup] test images=%d instances=%d" % (n_img, n_inst), flush=True)

    names = sorted(f for f in os.listdir(TEST_IMG)
                   if f.lower().endswith(".png"))
    stems = [os.path.splitext(f)[0] for f in names]

    arms = {"C1_global_thr3sigma": {}, "C2_local_bg_sub": {},
            "C3a_frontend_calibrated": {}, "C3b_frontend_k3": {}}
    params_cal = F.calib_to_params(ref)
    p_k3 = F.calib_to_params(ref)
    p_k3.k_sigma = 3.0
    for p in (params_cal, p_k3):
        p.score_threshold = -1.0          # 关闭质量门: 全部候选都留下

    for i, (f, stem) in enumerate(zip(names, stems)):
        gray = cv2.imread(os.path.join(TEST_IMG, f), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.ndim == 3:
            gray = gray[:, :, 0]
        arms["C1_global_thr3sigma"][stem] = c1_global_threshold(gray)
        arms["C2_local_bg_sub"][stem] = c2_local_bg_sub(gray)
        arms["C3a_frontend_calibrated"][stem] = c3_frontend(gray, params_cal)
        arms["C3b_frontend_k3"][stem] = c3_frontend(gray, p_k3)
        if (i + 1) % 100 == 0:
            print("  [classical] %d/%d" % (i + 1, len(names)), flush=True)

    rows = eval_arms(arms, gt, n_img)
    print(json.dumps(rows, indent=1), flush=True)

    # ---------- Ours: 报告配置权重, 自研评估器 ----------
    from ultralytics import YOLO
    model = YOLO(MAIN_W)
    proj = os.path.join(HERE, "tim_pred_ours")
    if os.path.isdir(proj):
        shutil.rmtree(proj, ignore_errors=True)
    model.predict(TEST_IMG, conf=0.001, iou=0.7, imgsz=320, device=0,
                  save_txt=True, save_conf=True, project=proj,
                  name="labels", exist_ok=True, verbose=False)
    pred_dir = os.path.join(proj, "labels", "labels")
    preds = E.read_pred_txt(pred_dir)
    ours = {
        "mAP50": round(E.ap_at(preds, gt, 0.5), 4),
        "mAP50_95": round(E.map5095(preds, gt), 4),
        **E.prf_at_fp_budget(preds, gt, n_img, 0.25, 0.5),
    }
    print("[Ours via own evaluator]", ours, flush=True)

    # ---------- 对拍: ultralytics val ----------
    yaml_p = os.path.join(HERE, "tim_eval_weak.yaml")
    with open(yaml_p, "w", encoding="utf8") as f:
        f.write("path: %s\n" % os.path.join(BASE, "yolo_weak").replace("\\", "/"))
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("nc: 1\nnames: ['hotspot']\n")
    m = model.val(data=yaml_p, split="test", imgsz=320, batch=32,
                  device=0, verbose=False)
    val_map50 = round(float(m.box.map50), 4)
    print("[ultralytics val] map50=%.4f map=%.4f" % (val_map50, m.box.map),
          flush=True)

    out = {
        "test_images": n_img, "test_instances": n_inst,
        "evaluator_check": {"own_mAP50": ours["mAP50"],
                            "ultralytics_val_map50": val_map50,
                            "match": abs(ours["mAP50"] - val_map50) < 0.01},
        "reference_calib": {k: ref.get(k) for k in
                            ("k", "st", "bmf", "kl", "win", "lmd",
                             "use_local_bg", "tau", "calib_F2")},
        "rows": rows, "Ours": ours,
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
