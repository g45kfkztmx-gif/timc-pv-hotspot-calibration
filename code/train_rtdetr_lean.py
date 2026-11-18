# -*- coding: utf-8 -*-
"""
train_rtdetr_lean.py
RT-DETR-l 低资源版训练: batch=4, cache=False(禁 RAM 缓存), workers=0
协议其余项与 run_detectors.py 一致: 60 epochs, imgsz=320, amp=False, seed=0,
A5 数据集, 测试集一次性评估。
用法: python train_rtdetr_lean.py
"""
import os
import json
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
ROOT = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset"
DATA = os.path.join(BASE, "yolo_abl5_adopt070", "dataset.yaml")

res_json = os.path.join(BASE, "train_results_det_rtdetr_l.json")

if os.path.exists(res_json):
    print("[skip] results exist")
    raise SystemExit

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

model = YOLO(os.path.join(ROOT, "rtdetr-l.pt"))
n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
try:
    gflops = round(float(get_flops(model.model, 320)), 1)
except Exception:
    gflops = None

print("TRAINING det_rtdetr_l batch=4 cache=False workers=0", flush=True)
t0 = time.time()
model.train(
    data=DATA, epochs=60, imgsz=320, batch=4, workers=0,
    device=0, amp=False, cache=False,
    project=os.path.join(BASE, "runs"), name="pseudo_det_rtdetr_l",
    exist_ok=True, verbose=False, seed=0,
)
train_sec = time.time() - t0

print(">>> Evaluating on TEST split ...", flush=True)
metrics = model.val(data=DATA, split="test", imgsz=320, batch=16,
                    workers=0, device=0, verbose=False)

result = {
    "tag": "det_rtdetr_l",
    "weights": "rtdetr-l.pt",
    "ultralytics": "8.4.0",
    "params_M": round(n_params, 2),
    "gflops": gflops,
    "batch": 4,
    "train_seconds": round(train_sec, 1),
    "test_map50": round(float(metrics.box.map50), 4),
    "test_map50_95": round(float(metrics.box.map), 4),
    "test_precision": round(float(metrics.box.mp), 4),
    "test_recall": round(float(metrics.box.mr), 4),
}
print("[det_rtdetr_l] test mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f (%.0fs)"
      % (result["test_map50"], result["test_map50_95"],
         result["test_precision"], result["test_recall"], train_sec), flush=True)

with open(res_json, "w", encoding="utf8") as f:
    json.dump([result], f, indent=2, ensure_ascii=False)
print("[train] saved ->", res_json, flush=True)
