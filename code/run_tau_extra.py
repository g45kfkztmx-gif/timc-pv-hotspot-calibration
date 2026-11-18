# -*- coding: utf-8 -*-
"""
run_tau_extra.py
tau 敏感性曲线补点: tau=0.35 / 0.55
协议与 run_tau_sensitivity.py 完全一致 (use_rescue=True, 冻结检测参数,
60ep / imgsz320 / batch32 / seed0), 每个配置在测试集上仅评估一次。
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
        ("tau035", 0.35),
        ("tau055", 0.55),
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

    print("TAU EXTRA DONE")


if __name__ == "__main__":
    main()
