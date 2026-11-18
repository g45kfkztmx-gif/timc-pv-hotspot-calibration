# -*- coding: utf-8 -*-
"""
rev7_repr_control_a3.py
=======================
Representation-sensitivity 实验的**对照跑**:

用与 rev7_representation_sensitivity.py **完全相同**的训练/评估设置
(60 epochs, imgsz 320, batch 32, amp False, seed 0, 同一 yolov8n.pt),
去训练仓库里**已存在的、论文所报告的 A3 数据集** `yolo_fin42`
(728 train + 72 val = 800 框, 与 Table VI 的 A3 一致)。

目的: 判定 luma 重建臂单跑 0.758 与论文头条 0.802 的差距来自
      (a) 标签/数据集差异, 还是 (b) 训练环境/单跑波动。
      luma 重建臂与 yolo_fin42 的标签在 97.9% 的图像上完全一致
      (box-level F1 = 0.975), 所以若本对照也落在 0.75–0.78,
      则差距属于 (b), 与标签无关。

用法: python rev7_repr_control_a3.py
输出: rev5_experiments/rev7_repr_a3control_seed0.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rev7_representation_sensitivity import train_eval  # noqa: E402

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"


def main():
    yaml_path = os.path.join(BASE, "yolo_fin42", "dataset.yaml")
    if not os.path.exists(yaml_path):
        raise SystemExit("missing %s" % yaml_path)
    out = train_eval("a3control", yaml_path, seed=0)
    print("[a3control]", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
