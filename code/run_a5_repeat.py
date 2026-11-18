# -*- coding: utf-8 -*-
"""
run_a5_repeat.py
A5 配置可重复性实验: 重跑 2 次(不同随机种子), 与原始 run1 合并统计均值±标准差。

  run1 = 原始 A5 (seed=0, 已有结果 0.8494)
  run2 = seed=1
  run3 = seed=2

训练配置与 train_yolo.py 完全一致(epochs=60, imgsz=320, batch=32,
workers=2, cache=ram, amp=False), 仅随机种子不同。

输出:
  runs/pseudo_a5_run2|run3/weights/best.pt
  train_results_a5_run2|run3.json
  a5_reproducibility.json   三次结果 + 均值±标准差
"""

import os
import sys
import json
import time
import statistics

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

sys.path.insert(0, BASE)

WEIGHTS = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\yolov8n.pt"
DATA = os.path.join(BASE, "yolo_abl5_adopt070", "dataset.yaml")

RUNS = [
    ("a5_run2", 1),
    ("a5_run3", 2),
]


def train_and_eval(tag, seed):

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("TRAINING: %s (seed=%d)" % (tag, seed))
    print("=" * 60)

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

    print("\n>>> Evaluating on TEST split ...")

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


def summarize():
    """合并 run1(原始A5) + run2 + run3, 输出均值±标准差"""

    paths = [
        os.path.join(BASE, "train_results_abl5_adopt070.json"),
        os.path.join(BASE, "train_results_a5_run2.json"),
        os.path.join(BASE, "train_results_a5_run3.json"),
    ]

    all_r = []

    for p in paths:

        if not os.path.exists(p):
            print("missing:", p)
            continue

        with open(p, "r", encoding="utf8") as f:
            all_r.append(json.load(f)[0])

    if len(all_r) < 2:
        print("not enough runs to summarize yet")
        return

    keys = [
        ("test_map50", "mAP50"),
        ("test_map50_95", "mAP50-95"),
        ("test_precision", "P"),
        ("test_recall", "R"),
    ]

    summary = {"runs": all_r, "statistics": {}}

    print("\n" + "=" * 66)
    print("A5 REPRODUCIBILITY (%d runs)" % len(all_r))
    print("=" * 66)

    print("%-12s" % "run", end="")

    for _, label in keys:
        print("%12s" % label, end="")

    print()

    for r in all_r:

        print("%-12s" % r["tag"], end="")

        for k, _ in keys:
            print("%12.4f" % r[k], end="")

        print()

    for k, label in keys:

        vals = [r[k] for r in all_r]

        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0

        summary["statistics"][label] = {
            "mean": round(m, 4),
            "std": round(s, 4),
            "values": vals,
            "text": "%.4f +/- %.4f" % (m, s),
        }

    print()

    for k, label in keys:
        print(
            "%-10s %s"
            % (label, summary["statistics"][label]["text"])
        )

    out = os.path.join(BASE, "a5_reproducibility.json")

    with open(out, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nsaved ->", out)


def main():

    for tag, seed in RUNS:

        res_json = os.path.join(BASE, "train_results_%s.json" % tag)

        if os.path.exists(res_json):
            print("[train] %s results exist, skip" % tag)
            continue

        r = train_and_eval(tag, seed)

        with open(res_json, "w", encoding="utf8") as f:
            json.dump([r], f, indent=2, ensure_ascii=False)

        print("[train] saved ->", res_json)

    summarize()

    print("A5 REPEAT DONE")


if __name__ == "__main__":
    main()
