# -*- coding: utf-8 -*-
"""
train_det_one.py
检测器补充实验(单任务版): yolov10n / yolo11n / rtdetr-l on A5 dataset
协议与 run_detectors.py 完全一致: 60 epochs, imgsz=320, batch=32,
workers=2, amp=False, cache=ram, seed=0, 测试集一次性评估。
显存不足自动降 batch=16 重试(记录实际 batch)。

用法: python train_det_one.py <tag> <weights>
环境: E:\\PV DATASET\\.venv (ultralytics 8.4.0; 8.1.0 不支持 v10/v11)
"""

import os
import sys
import json
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
ROOT = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset"
DATA = os.path.join(BASE, "yolo_abl5_adopt070", "dataset.yaml")


def train_and_eval(tag, weights, batch=32):

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("TRAINING: %s (batch=%d)" % (tag, batch))
    print("=" * 60, flush=True)

    model = YOLO(weights)

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = round(float(get_flops(model.model, 320)), 1)
    except Exception as e:
        print("[warn] get_flops failed:", e)

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

    print("\n>>> Evaluating on TEST split ...", flush=True)

    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=320,
        batch=32,
        workers=2,
        device=0,
        verbose=False,
    )

    result = {
        "tag": tag,
        "weights": os.path.basename(weights),
        "ultralytics": "8.4.0",
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
        % (tag, result["test_map50"], result["test_map50_95"],
           result["test_precision"], result["test_recall"], train_sec),
        flush=True,
    )

    return result


def main():

    tag = sys.argv[1]
    weights = sys.argv[2]
    batch0 = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    batch_fb = max(4, batch0 // 2)
    if not os.path.isabs(weights):
        weights = os.path.join(ROOT, weights)

    res_json = os.path.join(BASE, "train_results_%s.json" % tag)

    if os.path.exists(res_json):
        print("[skip] %s results exist" % tag)
        return

    import torch

    try:
        r = train_and_eval(tag, weights, batch=batch0)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" not in str(e).lower():
            raise
        print("[OOM] %s batch=%d failed, retry batch=%d" % (tag, batch0, batch_fb), flush=True)
        torch.cuda.empty_cache()
        r = train_and_eval(tag, weights, batch=batch_fb)

    with open(res_json, "w", encoding="utf8") as f:
        json.dump([r], f, indent=2, ensure_ascii=False)

    print("[train] saved ->", res_json, flush=True)


if __name__ == "__main__":
    main()
