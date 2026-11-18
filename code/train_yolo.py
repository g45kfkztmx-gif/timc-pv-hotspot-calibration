# -*- coding: utf-8 -*-
"""
train_yolo.py
用两套伪标签数据集分别训练标准 YOLOv8n, 并在测试集(GT标注)上评估。

数据集:
  split/yolo_f1/dataset.yaml      (Best F1 配置伪标签)
  split/yolo_recall/dataset.yaml  (召回优先配置伪标签)

训练环境: GTX 1650 4GB, imgsz=320, amp=False(防NaN), workers=0(Win兼容)
"""

import os
import sys
import json
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

WEIGHTS = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\yolov8n.pt"

CONFIGS = [
    ("weak_r1", os.path.join(BASE, "yolo_weak", "dataset.yaml")),
]


def train_and_eval(tag, yaml_path):

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("TRAINING:", tag)
    print("=" * 60)

    model = YOLO(WEIGHTS)

    t0 = time.time()

    model.train(
        data=yaml_path,
        epochs=60,
        imgsz=320,
        batch=32,
        workers=2,
        device=0,
        amp=False,          # GTX1650 防 NaN
        cache="ram",        # 图片总共仅17MB, 全部缓存进内存
        project=os.path.join(BASE, "runs"),
        name="pseudo_" + tag,
        exist_ok=True,
        verbose=False,
        seed=0,
    )

    train_sec = time.time() - t0

    print("\n>>> Evaluating on TEST split ...")

    metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=320,
        batch=32,
        workers=4,
        device=0,
        verbose=False,
    )

    result = {
        "tag": tag,
        "train_seconds": round(train_sec, 1),
        "test_map50": round(float(metrics.box.map50), 4),
        "test_map50_95": round(float(metrics.box.map), 4),
        "test_precision": round(float(metrics.box.mp), 4),
        "test_recall": round(float(metrics.box.mr), 4),
    }

    print(
        "[%s] test mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f (%.0fs)"
        % (
            tag,
            result["test_map50"],
            result["test_map50_95"],
            result["test_precision"],
            result["test_recall"],
            train_sec,
        )
    )

    return result


def main():

    configs = CONFIGS

    if len(sys.argv) >= 3:
        configs = [(sys.argv[1], sys.argv[2])]

    results = []

    for tag, yaml_path in configs:

        if not os.path.exists(yaml_path):
            print("missing:", yaml_path)
            continue

        results.append(train_and_eval(tag, yaml_path))

    print("\n" + "=" * 60)
    print("FINAL COMPARISON (test set, GT labels)")
    print("=" * 60)

    print(
        "%-10s %10s %12s %10s %10s"
        % ("config", "mAP50", "mAP50-95", "P", "R")
    )

    for r in results:

        print(
            "%-10s %10.4f %12.4f %10.4f %10.4f"
            % (
                r["tag"],
                r["test_map50"],
                r["test_map50_95"],
                r["test_precision"],
                r["test_recall"],
            )
        )

    out_json = os.path.join(
        BASE, "train_results_%s.json" % results[0]["tag"]
    )

    with open(out_json, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nsaved ->", out_json)


if __name__ == "__main__":
    main()
