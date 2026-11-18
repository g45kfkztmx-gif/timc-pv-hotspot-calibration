# -*- coding: utf-8 -*-
"""measure_gflops640.py — 测三个新检测器在 640 分辨率下的 GFLOPs(官方口径)"""
import os
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset"

for name in ["yolov10n.pt", "yolo11n.pt", "rtdetr-l.pt"]:
    try:
        m = YOLO(os.path.join(ROOT, name))
        g = round(float(get_flops(m.model, 640)), 1)
        p = round(sum(x.numel() for x in m.model.parameters()) / 1e6, 2)
        print("%-14s params=%.2fM  GFLOPs@640=%.1f" % (name, p, g), flush=True)
    except Exception as e:
        print("%-14s FAILED: %s" % (name, e), flush=True)
