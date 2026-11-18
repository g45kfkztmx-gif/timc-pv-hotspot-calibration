# -*- coding: utf-8 -*-
"""
rev6_phase3_5seed.py
====================
Phase 3: 在新操作点 (g=0.42) 上补 A5 / A3 的多 seed 重复, 以支撑
摘要与结论里的 "five-seed mean ± std" 绝对性能声明。

协议要点 (与主实验一致, 且与 rev5_e_5seed.py 的做法相同):
  **伪标签数据集固定不变**, 只改变训练随机种子 —— 因此这里度量的是
  训练方差, 不是整条 pipeline 的方差。seed 0 已有 (Phase 1), 故补 1..4。

  tag               数据集                   含义
  ----------------------------------------------------------------
  rev6_g42_r2       yolo_rev6_g42_r2         A5 = 完整方法 (校准引导 + 精炼 adopt@0.60)
  rev6_g42_r1       yolo_fin42               A3 = 纯校准引导 (无精炼)

顺序: 等 Phase 2 收尾 (GPU 只有一块) -> A5 先跑 (绝对声明需要) -> A3 次之。
已存在的 seed 自动跳过, 可随时中断后重跑。

输出: rev6_g42_r2_seed{0..4}.json / rev6_g42_r1_seed{0..4}.json
      rev6_phase3_summary.json
"""
import json
import os
import sys
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

PHASE2 = os.path.join(_HERE, "rev6_phase2_summary.json")
OUT = os.path.join(_HERE, "rev6_phase3_summary.json")
SEEDS = [0, 1, 2, 3, 4]
WAIT_MAX_S = 6 * 3600

JOBS = [
    ("rev6_g42_r2", "yolo_rev6_g42_r2"),     # A5 first
    ("rev6_g42_r1", "yolo_fin42"),           # A3 second
]


def wait_for_phase2():
    t0 = time.time()
    while not (os.path.exists(PHASE2) and os.path.getsize(PHASE2) > 10):
        if time.time() - t0 > WAIT_MAX_S:
            raise SystemExit("[abort] phase 2 not finished within %dh" % (WAIT_MAX_S // 3600))
        print("[wait] phase 2 summary not yet present ...", flush=True)
        time.sleep(60)


def main():
    from rev5_common import train_and_eval, result_exists, out_json

    print("=" * 78)
    print("PHASE 3 | five-seed replication at the final operating point (g=0.42)")
    print("=" * 78, flush=True)
    wait_for_phase2()
    print("[phase2] done -> starting seed replication", flush=True)

    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": {}}
    for tag, ds in JOBS:
        yaml_path = os.path.join(BASE, ds, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print("[%s] MISSING %s" % (tag, yaml_path), flush=True)
            continue
        print("\n########## %s (%s) ##########" % (tag, ds), flush=True)
        for seed in SEEDS:
            if result_exists(tag, seed):
                print("[%s|seed%d] exists, skip" % (tag, seed), flush=True)
                continue
            t0 = time.time()
            train_and_eval(tag, yaml_path, seed=seed,
                           project=os.path.join(BASE, "rev5_runs"))
            print("[%s|seed%d] done (%.0fs)" % (tag, seed, time.time() - t0), flush=True)

        vals = []
        for seed in SEEDS:
            p = out_json(tag, seed)
            if os.path.exists(p) and os.path.getsize(p) > 10:
                with open(p, encoding="utf8") as f:
                    r = json.load(f)
                vals.append({"seed": seed, "map50": r["test_map50"],
                             "map50_95": r["test_map50_95"],
                             "P": r["test_precision"], "R": r["test_recall"]})
        if vals:
            n = len(vals)
            m = sum(v["map50"] for v in vals) / n
            sd = (sum((v["map50"] - m) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
            summary["arms"][tag] = {"runs": vals, "n_seeds": n,
                                    "map50_mean": round(m, 4), "map50_std": round(sd, 4),
                                    "map50_95_mean": round(sum(v["map50_95"] for v in vals) / n, 4),
                                    "P_mean": round(sum(v["P"] for v in vals) / n, 4),
                                    "R_mean": round(sum(v["R"] for v in vals) / n, 4)}
            with open(OUT, "w", encoding="utf8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print("[%s] n=%d  mAP@50 %.4f ± %.4f" % (tag, n, m, sd), flush=True)

    with open(OUT, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
