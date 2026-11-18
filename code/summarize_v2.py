# -*- coding: utf-8 -*-
"""
summarize_v2.py
汇总补充实验:
1) 检测器对比 7 模型 (v5n/v5s/v8s/v8n + v10n/v11n/rtdetr-l) -> detector_comparison_v2.json
2) A5 五种子统计 (seed 0/1/2/3/4) -> a5_reproducibility_5seed.json
"""
import os
import json
import statistics

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

DETS = [
    ("YOLOv5n",      "train_results_det_yolov5n.json"),
    ("YOLOv5s",      "train_results_det_yolov5s.json"),
    ("YOLOv8s",      "train_results_det_yolov8s.json"),
    ("YOLOv10n",     "train_results_det_yolov10n.json"),
    ("YOLO11n",      "train_results_det_yolo11n.json"),
    ("RT-DETR-l",    "train_results_det_rtdetr_l.json"),
]

# YOLOv8n 参考值取自 A5 单次训练 (seed=0)
V8N = {"detector": "YOLOv8n (ours)", "params_M": 3.2, "gflops": 8.7,
       "train_s": 779, "mAP50": 0.8494, "mAP50_95": 0.2780,
       "P": 0.8245, "R": 0.8029, "batch": 32}


def load(fn):
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print("MISSING:", fn)
        return None
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)[0]


def main():
    # ---------- detector table ----------
    rows = [V8N]
    for name, fn in DETS:
        r = load(fn)
        if r is None:
            continue
        rows.append({
            "detector": name,
            "params_M": r["params_M"],
            "gflops": r.get("gflops"),
            "train_s": round(r.get("train_seconds", 0)),
            "mAP50": r["test_map50"],
            "mAP50_95": r["test_map50_95"],
            "P": r["test_precision"],
            "R": r["test_recall"],
            "batch": r.get("batch", 32),
        })

    rows.sort(key=lambda x: x["mAP50"])

    print("\nDETECTOR COMPARISON (A5 pseudo-labels, seed=0)")
    print("%-18s %7s %7s %7s %8s %10s %7s %7s" %
          ("detector", "params", "GFLOPs", "train_s", "mAP50", "mAP50-95", "P", "R"))
    for t in rows:
        print("%-18s %7s %7s %7s %8.4f %10.4f %7.4f %7.4f" % (
            t["detector"], t["params_M"], t["gflops"], t["train_s"],
            t["mAP50"], t["mAP50_95"], t["P"], t["R"]))

    with open(os.path.join(BASE, "detector_comparison_v2.json"), "w", encoding="utf8") as f:
        json.dump({"protocol": "60ep imgsz320 batch32 amp=False seed=0; new dets on ultralytics 8.4.0",
                   "results": rows}, f, indent=2, ensure_ascii=False)

    # ---------- 5-seed statistics ----------
    seed_files = [
        ("seed0", "train_results_abl5_adopt070.json"),
        ("seed1", "train_results_a5_run2.json"),
        ("seed2", "train_results_a5_run3.json"),
        ("seed3", "train_results_a5_run4.json"),
        ("seed4", "train_results_a5_run5.json"),
    ]
    runs = []
    for tag, fn in seed_files:
        r = load(fn)
        if r:
            r["run"] = tag
            runs.append(r)

    keys = [("test_map50", "mAP50"), ("test_map50_95", "mAP50-95"),
            ("test_precision", "P"), ("test_recall", "R")]
    stats = {}
    print("\nA5 FIVE-SEED STATISTICS (n=%d)" % len(runs))
    for k, label in keys:
        vals = [r[k] for r in runs]
        m = statistics.mean(vals)
        s = statistics.stdev(vals)
        stats[label] = {"mean": round(m, 4), "std": round(s, 4), "values": vals}
        print("%-10s %.4f +/- %.4f  vals=%s" % (label, m, s, vals))

    with open(os.path.join(BASE, "a5_reproducibility_5seed.json"), "w", encoding="utf8") as f:
        json.dump({"runs": runs, "statistics": stats}, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY DONE")


if __name__ == "__main__":
    main()
