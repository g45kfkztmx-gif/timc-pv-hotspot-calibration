# -*- coding: utf-8 -*-
"""
tim_exp4_attenuation.py —— TIM 审稿意见 #2/#3: 负载衰减敏感性
======================================================================
审稿人问: "低辐照度下对比度降低, 阈值 k=2.2 是否仍有效?"
源发布不记录辐照度/负载, 因此用**受控衰减实验**代替:
  对每幅测试图做局部对比度衰减 I' = bg + s*(I - bg), bg = medianBlur(I,31),
  s ∈ {1.0, 0.8, 0.6, 0.4} —— 模拟故障温差的相对 excess 随负载下降而成比例
  缩小 (故障电阻固定时耗散功率 ∝ 电流, 故温差 ∝ 负载), 渲染映射固定。
  同时测两端:
    (a) 冻结提案生成器 (校准参数 + tau) 的实例覆盖率与门控通过数
    (b) 已训练检测器在衰减图上的 mAP@50 / hit45
输出: tim_attenuation_summary.json
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
from auto_pseudo import IoU                                # noqa: E402
import tim_eval as E                                       # noqa: E402

TEST_IMG = os.path.join(BASE, "yolo_weak", "images", "test")
TEST_LBL = os.path.join(BASE, "yolo_weak", "labels", "test")
REF_CALIB = os.path.join(HERE, "budget_calib_N197.json")
MAIN_W = os.path.join(BASE, "rev5_runs",
                      "pseudo_rev5_bud_ours197_r1_seed0", "weights", "best.pt")
OUT = os.path.join(HERE, "tim_attenuation_summary.json")

SCALES = [1.0, 0.8, 0.6, 0.4]


def attenuate(bgr, s):
    """逐通道局部对比度衰减 (保持调色板结构, 只缩小相对 excess)"""
    out = np.empty_like(bgr)
    for c in range(bgr.shape[2]):
        ch = bgr[:, :, c]
        bg = cv2.medianBlur(ch, 31).astype(np.float32)
        out[:, :, c] = np.clip(bg + s * (ch.astype(np.float32) - bg),
                               0, 255).astype(np.uint8)
    return out


def main():
    ref = json.load(open(REF_CALIB, encoding="utf8"))
    gt = E.load_gt(TEST_LBL)
    n_img = len(os.listdir(TEST_IMG))
    n_inst = sum(len(v) for v in gt.values())
    params = F.calib_to_params(ref)
    tau = ref["tau"]
    names = sorted(f for f in os.listdir(TEST_IMG)
                   if f.lower().endswith(".png"))
    print("[setup] images=%d instances=%d tau=%.2f k=%.1f"
          % (n_img, n_inst, tau, ref["k"]), flush=True)

    rows = []
    for s in SCALES:
        d = os.path.join(BASE, "tim_atten_s%02d" % int(s * 100))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

        # (a) 冻结生成器
        prop_hit = 0
        prop_boxes = 0
        for f in names:
            stem = os.path.splitext(f)[0]
            bgr = cv2.imread(os.path.join(TEST_IMG, f))
            if s != 1.0:
                bgr = attenuate(bgr, s)
            cv2.imwrite(os.path.join(d, f), bgr)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cands, _ = detect_diag(gray, params)
            kept = []
            for c in cands:
                if c["status"] != "accepted":
                    continue
                if c["score"] < tau:
                    continue
                x, y, w, h = c["box"]
                kept.append((x, y, w, h))
            prop_boxes += len(kept)
            used = [False] * len(kept)
            for g in gt.get(stem, []):
                for j, k in enumerate(kept):
                    if used[j]:
                        continue
                    if IoU(g, k) >= 0.5:
                        used[j] = True
                        prop_hit += 1
                        break
        # (b) 检测器
        from ultralytics import YOLO
        model = YOLO(MAIN_W)
        proj = os.path.join(HERE, "tim_pred_atten_s%02d" % int(s * 100))
        if os.path.isdir(proj):
            shutil.rmtree(proj, ignore_errors=True)
        model.predict(d, conf=0.001, iou=0.7, imgsz=320, device=0,
                      save_txt=True, save_conf=True, project=proj,
                      name="labels", exist_ok=True, verbose=False)
        preds = E.read_pred_txt(os.path.join(proj, "labels", "labels"))
        row = {
            "scale": s,
            "proposal_recall": round(prop_hit / n_inst, 4),
            "proposal_boxes": prop_boxes,
            "det_mAP50": round(E.ap_at(preds, gt, 0.5), 4),
            "det_mAP50_95": round(E.map5095(preds, gt), 4),
            **E.prf_at_fp_budget(preds, gt, n_img, 0.25, 0.5),
        }
        rows.append(row)
        print("[s=%.1f] %s" % (s, json.dumps(row)), flush=True)

    with open(OUT, "w", encoding="utf8") as f:
        json.dump({"n_img": n_img, "n_inst": n_inst, "rows": rows},
                  f, indent=1, ensure_ascii=False)
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
