# -*- coding: utf-8 -*-
"""
train_a5_seed.py
A5 可重复性补跑(单种子版): seed=3 -> run4, seed=4 -> run5
协议与 run_a5_repeat.py 完全一致(conda yolo 环境, ultralytics 8.1.0)。

用法: python train_a5_seed.py <seed> <run_tag>
"""

import os
import sys
import json
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
WEIGHTS = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\yolov8n.pt"
DATA = os.path.join(BASE, "yolo_abl5_adopt070", "dataset.yaml")


def main():

    seed = int(sys.argv[1])
    tag = sys.argv[2]

    res_json = os.path.join(BASE, "train_results_%s.json" % tag)

    if os.path.exists(res_json):
        print("[skip] %s results exist" % tag)
        return

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("TRAINING: %s (seed=%d)" % (tag, seed))
    print("=" * 60, flush=True)

    model = YOLO(WEIGHTS)

    t0 = time.time()

    model.train(
        data=DATA,
        epochs=60,
        imgsz=320,
        batch=32,
        workers=2,
        device=0,
        amp=False,
        cache="ram",
        project=os.path.join(BASE, "runs"),
        name="pseudo_" + tag,
        exist_ok=True,
        verbose=False,
        seed=seed,
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
        "seed": seed,
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

    with open(res_json, "w", encoding="utf8") as f:
        json.dump([result], f, indent=2, ensure_ascii=False)

    print("[train] saved ->", res_json, flush=True)


if __name__ == "__main__":
    main()
