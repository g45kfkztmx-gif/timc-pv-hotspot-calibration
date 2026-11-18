# -*- coding: utf-8 -*-
"""
run_detectors.py
检测器对比实验: YOLOv5n / YOLOv5s / YOLOv8s vs YOLOv8n(本文 A5)

协议:
  - 全部检测器在同一数据集上训练: yolo_abl5_adopt070 (A5 最终方法数据集,
    质量过滤伪标签 + 难样本挖掘 + adopt@0.7 自训练精修)
  - 训练配置与主方法完全一致: 60 epochs, imgsz=320, batch=32,
    workers=2, amp=False, cache=ram, seed=0
  - 4GB 显存不足时自动降 batch 至 16 重试(记录实际 batch)
  - 测试集(GT)一次性评估, 无泄露

说明: ultralytics 加载官方 yolov5n/s.pt 时会自动转换为
YOLOv5u(anchor-free head)结构, 训练评估流程统一。

输出:
  runs/pseudo_det_yolov5n|det_yolov5s|det_yolov8s/weights/best.pt
  train_results_det_*.json
  detector_comparison.json
"""

import os
import sys
import json
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
ROOT = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset"

sys.path.insert(0, BASE)

DATA = os.path.join(BASE, "yolo_abl5_adopt070", "dataset.yaml")

JOBS = [
    ("det_yolov5n", os.path.join(ROOT, "yolov5n.pt")),
    ("det_yolov5s", os.path.join(ROOT, "yolov5s.pt")),
    ("det_yolov8s", os.path.join(ROOT, "yolov8s.pt")),
]


def train_and_eval(tag, weights, batch=32):

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("TRAINING: %s (batch=%d)" % (tag, batch))
    print("=" * 60)

    model = YOLO(weights)

    n_params = sum(
        p.numel() for p in model.model.parameters()
    ) / 1e6

    gflops = None

    try:
        from ultralytics.utils.torch_utils import get_flops
        # 必须传内部 nn.Module: 传 YOLO 包装对象会被 thop 调到
        # YOLO.train() 触发默认参数训练
        gflops = round(float(get_flops(model.model, 320)), 1)
    except Exception:
        pass

    t0 = time.time()

    model.train(
        data=DATA,
        epochs=60,
        imgsz=320,
        batch=batch,
        workers=2,
        device=0,
        amp=False,
        cache="ram",
        project=os.path.join(BASE, "runs"),
        name="pseudo_" + tag,
        exist_ok=True,
        verbose=False,
        seed=0,
    )

    train_sec = time.time() - t0

    print("\n>>> Evaluating on TEST split ...")

    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=320,
        batch=32,
        workers=4,
        device=0,
        verbose=False,
    )

    result = {
        "tag": tag,
        "weights": os.path.basename(weights),
        "params_M": round(n_params, 2),
        "gflops": gflops,
        "batch": batch,
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

    import torch

    results = []

    for tag, weights in JOBS:

        res_json = os.path.join(
            BASE, "train_results_%s.json" % tag
        )

        if os.path.exists(res_json):
            print("[skip] %s results exist" % tag)
            with open(res_json, "r", encoding="utf8") as f:
                results.append(json.load(f)[0])
            continue

        try:
            r = train_and_eval(tag, weights, batch=32)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            msg = str(e)
            if "out of memory" not in msg.lower():
                raise
            print("[OOM] %s batch=32 failed, retry batch=16" % tag)
            torch.cuda.empty_cache()
            r = train_and_eval(tag, weights, batch=16)

        with open(res_json, "w", encoding="utf8") as f:
            json.dump([r], f, indent=2, ensure_ascii=False)

        print("[train] saved ->", res_json)

        results.append(r)

        torch.cuda.empty_cache()

    # ---------- 汇总对比 ----------

    ref_path = os.path.join(
        BASE, "train_results_abl5_adopt070.json"
    )

    ref = None

    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf8") as f:
            ref = json.load(f)[0]

    table = []

    if ref is not None:
        table.append({
            "detector": "YOLOv8n (本文)",
            "tag": ref["tag"],
            "params_M": 3.2,
            "gflops": 8.7,
            "batch": 32,
            "mAP50": ref["test_map50"],
            "mAP50-95": ref["test_map50_95"],
            "P": ref["test_precision"],
            "R": ref["test_recall"],
        })

    names = {
        "det_yolov5n": "YOLOv5n",
        "det_yolov5s": "YOLOv5s",
        "det_yolov8s": "YOLOv8s",
    }

    for r in results:
        table.append({
            "detector": names[r["tag"]],
            "tag": r["tag"],
            "params_M": r["params_M"],
            "gflops": r["gflops"],
            "batch": r["batch"],
            "mAP50": r["test_map50"],
            "mAP50-95": r["test_map50_95"],
            "P": r["test_precision"],
            "R": r["test_recall"],
        })

    print("\n" + "=" * 74)
    print("DETECTOR COMPARISON (A5 dataset, test set GT)")
    print("=" * 74)
    print("%-16s %8s %8s %6s %8s %10s %8s %8s" % (
        "detector", "params", "GFLOPs", "batch",
        "mAP50", "mAP50-95", "P", "R"
    ))

    for t in table:
        print("%-16s %8s %8s %6s %8.4f %10.4f %8.4f %8.4f" % (
            t["detector"], t["params_M"], t["gflops"], t["batch"],
            t["mAP50"], t["mAP50-95"], t["P"], t["R"]
        ))

    summary = {
        "dataset": "yolo_abl5_adopt070 (A5 最终方法)",
        "protocol": (
            "epochs=60, imgsz=320, workers=2, amp=False, "
            "cache=ram, seed=0; 测试集一次性评估"
        ),
        "note": (
            "ultralytics 加载官方 yolov5n/s.pt 自动转换为 "
            "YOLOv5u(anchor-free) 结构; YOLOv8n 参考结果取自 "
            "A5 单次训练(seed=0), 其三次独立训练均值为 "
            "mAP50=0.8130±0.0390"
        ),
        "results": table,
    }

    out = os.path.join(BASE, "detector_comparison.json")

    with open(out, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nsaved ->", out)
    print("DETECTORS DONE")


if __name__ == "__main__":
    main()
