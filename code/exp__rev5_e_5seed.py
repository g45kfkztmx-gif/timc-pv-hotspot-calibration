# -*- coding: utf-8 -*-
"""
rev5_e_5seed.py  (实验E: 关键对比的 5-seed 复现, 修改建议第 5 项)
====================================================================
动机: 单 seed 的模块间差异 (A1..A5, 2x2 拆分, tau 档位) 全部落在已有
      A5 5-seed 标准差 sigma=0.029 的 +-2sigma 内, 无法支撑"某模块有效"
      的结论。必须给出 5-seed 均值 +- 标准差。

本脚本只补"支撑核心主张"的配置 (A5@0.7 已有 5-seed, 见
a5_reproducibility_5seed.json, 不重跑):

  tag             数据集                    含义
  ---------------------------------------------------------------
  rev5_a3         yolo_weak                 A3: 弱标签+质量过滤, 无 mining
  rev5_a5_tau060  yolo_rev5_a5_tau060       A5 但 tau_adopt=0.6(校准集选择)
  rev5_fact_os    yolo_rev5_fact_os         仅 hard oversampling
  rev5_fact_ad    yolo_rev5_fact_ad         仅 adoption(不过采样)

协议与主实验完全一致 (epochs=60, imgsz=320, batch=32, workers=2,
amp=False, cache=ram, test 评估一次)。每 seed 结果落盘
rev5_experiments/<tag>_seed<n>.json, 已存在自动跳过, 可反复启动续跑。

用法:
  python rev5_e_5seed.py              # 全部
  python rev5_e_5seed.py rev5_a3      # 只跑某一配置
"""
import os
import sys

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, os.path.dirname(__file__))

from rev5_common import train_and_eval, result_exists  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]

JOBS = [
    ("rev5_a1",        "yolo_abl1_norescue_nofilter"),   # A1: 纯全局阈值
    ("rev5_a2",        "yolo_abl2_rescue_nofilter"),     # A2: +局部救援
    ("rev5_a3",        "yolo_weak"),
    ("rev5_a5_tau060", "yolo_rev5_a5_tau060"),
    ("rev5_fact_os",   "yolo_rev5_fact_os"),
    ("rev5_fact_ad",   "yolo_rev5_fact_ad"),
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        jobs = [j for j in JOBS if j[0] == only]
        if not jobs:
            print("unknown tag: %s" % only)
            return
    else:
        jobs = JOBS

    for tag, ds in jobs:
        yaml_path = os.path.join(BASE, ds, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print("[%s] MISSING dataset.yaml -> %s" % (tag, yaml_path))
            continue
        print("\n########## %s (%s) ##########" % (tag, ds))
        for seed in SEEDS:
            if result_exists(tag, seed):
                print("[%s|seed%d] exists, skip" % (tag, seed))
                continue
            train_and_eval(tag, yaml_path, seed=seed,
                           project=os.path.join(BASE, "rev5_runs"))

    print("\n===== rev5_e_5seed finished =====")


if __name__ == "__main__":
    main()
