# -*- coding: utf-8 -*-
"""
lopo_reference_eval.py — 参考臂: 用主实验 R1 权重 (= 论文报告配置 A3, 0.802 mAP@50)
在各留出日测试子集上评估, 作为 Table XIII 的乐观参考列。
子集复用各折 assemble 构建的 yolo_lopo_<day>/dataset.yaml (test split 仅含该日图)。
用法: python lopo_reference_eval.py [--days 20230222 ...]
"""
import os
import sys
import json

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
REV5 = os.path.join(BASE, "rev5_experiments")
W = os.path.join(BASE, "rev5_runs", "pseudo_rev5_bud_ours197_r1_seed0",
                 "weights", "best.pt")
DAYS = ["20230222", "20230223", "20230227", "20230228",
        "20230302", "20230308", "20231114", "20231121"]


def main():
    days = DAYS
    if "--days" in sys.argv:
        i = sys.argv.index("--days")
        days = sys.argv[i + 1].split(",")

    from ultralytics import YOLO
    model = YOLO(W)

    summary = {}
    for day in days:
        root = os.path.join(BASE, "yolo_lopo_%s" % day)
        yaml_path = os.path.join(root, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print("[skip] %s: no dataset.yaml" % day)
            continue
        out_p = os.path.join(REV5, "lopo_ref_%s_seed0.json" % day)
        if os.path.exists(out_p) and os.path.getsize(out_p) > 10:
            summary[day] = json.load(open(out_p, encoding="utf8"))
            print("[%s] cached mAP50=%.4f" % (day, summary[day]["test_map50"]))
            continue
        metrics = model.val(
            data=yaml_path,
            split="test",
            imgsz=320,
            batch=32,
            workers=2,
            device=0,
            verbose=False,
        )
        res = {
            "tag": "lopo_ref_%s" % day,
            "day": day,
            "test_map50": round(float(metrics.box.map50), 4),
            "test_map50_95": round(float(metrics.box.map), 4),
            "test_precision": round(float(metrics.box.mp), 4),
            "test_recall": round(float(metrics.box.mr), 4),
        }
        with open(out_p, "w", encoding="utf8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        summary[day] = res
        print("[%s] ref mAP50=%.4f P=%.4f R=%.4f" % (
            day, res["test_map50"], res["test_precision"], res["test_recall"]))

    out_all = os.path.join(REV5, "lopo_reference_summary.json")
    with open(out_all, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    vals = [v["test_map50"] for v in summary.values()]
    if vals:
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        print("REF mean=%.4f sd=%.4f n=%d" % (m, sd, len(vals)))


if __name__ == "__main__":
    main()
