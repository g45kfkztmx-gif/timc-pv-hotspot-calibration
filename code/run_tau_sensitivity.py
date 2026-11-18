# -*- coding: utf-8 -*-
"""
run_tau_sensitivity.py
质量过滤阈值 tau 敏感性实验:

  tau=0.40 / 0.45(=weak_r1 已有) / 0.50
  其余配置与 A3 一致 (局部救援开, 冻结检测参数)

输出:
  yolo_tau040/, yolo_tau050/          数据集
  runs/pseudo_tau040|tau050/weights/  模型
  train_results_tau040|tau050.json    结果
  tau_sensitivity.json                敏感性汇总表
"""

import os
import sys
import json

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

sys.path.insert(0, BASE)

from run_ablation import build_pseudo_dataset


def main():

    from train_yolo import train_and_eval

    jobs = [
        ("tau040", 0.40),
        ("tau050", 0.50),
    ]

    for tag, tau in jobs:

        yaml_path = build_pseudo_dataset(tag, use_rescue=True, tau=tau)

        res_json = os.path.join(BASE, "train_results_%s.json" % tag)

        if os.path.exists(res_json):
            print("[train] %s results exist, skip" % tag)
            continue

        r = train_and_eval(tag, yaml_path)

        with open(res_json, "w", encoding="utf8") as f:
            json.dump([r], f, indent=2, ensure_ascii=False)

        print("[train] saved ->", res_json)

    # ---------- 敏感性汇总 ----------

    table = []

    rows = [
        ("tau=0.40", "tau040"),
        ("tau=0.45", "weak_r1"),
        ("tau=0.50", "tau050"),
    ]

    for label, tag in rows:

        if tag == "weak_r1":
            path = os.path.join(BASE, "train_results.json")
        else:
            path = os.path.join(BASE, "train_results_%s.json" % tag)

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf8") as f:
            r = json.load(f)[0]

        table.append({
            "tau": label,
            "tag": tag,
            "mAP50": r["test_map50"],
            "mAP50-95": r["test_map50_95"],
            "P": r["test_precision"],
            "R": r["test_recall"],
        })

    print("\n" + "=" * 60)
    print("TAU SENSITIVITY (test set, GT labels)")
    print("=" * 60)
    print("%-10s %8s %10s %8s %8s" % (
        "tau", "mAP50", "mAP50-95", "P", "R"
    ))

    for t in table:
        print("%-10s %8.4f %10.4f %8.4f %8.4f" % (
            t["tau"], t["mAP50"], t["mAP50-95"], t["P"], t["R"]
        ))

    out = os.path.join(BASE, "tau_sensitivity.json")

    with open(out, "w", encoding="utf8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    print("\nsaved ->", out)
    print("TAU SENSITIVITY DONE")


if __name__ == "__main__":
    main()
