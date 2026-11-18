# -*- coding: utf-8 -*-
"""
rev5_h_supervised_curve.py  (第三条曲线: supervised-N)
======================================================
"label-efficient" 的论证需要一个最朴素的对照臂:

  如果把 N 张人工标注**直接**拿去监督训练 detector, 能得到多少 mAP@50?

三条曲线共用同一套嵌套子集 (seed=42 洗牌后的前 N 张, 见 calib_names):

  supervised-N : N 张 GT  -> 直接监督 detector           (本脚本)
  ts-N         : N 张 GT  -> 训 teacher -> 伪标签 -> student
  ours-N       : N 张 GT  -> 校准统计生成器(冻结) -> 伪标签 -> detector

数据集构造与 rev5_a_s197.py 完全一致 (N=197 时即该实验, 直接复用):

  train = N 张 GT 的 80% + 320 张 normal(空标签背景, 不计入标注预算)
  val   = N 张 GT 的 20%  (仅用于 best checkpoint 选择)
  test  = 冻结的 weak test (人工 GT, 只评估一次)

训练协议与原论文一致: epochs=60, imgsz=320, batch=32, workers=2,
cache=ram, amp=False。

用法:
  python rev5_h_supervised_curve.py                 # 跑 BUDGETS 中缺失的点
  python rev5_h_supervised_curve.py --n 20,50       # 只跑指定预算
  python rev5_h_supervised_curve.py --merge-only    # 不训练, 只汇总曲线
"""
import os
import sys
import json
import random
import shutil

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # split/
sys.path.insert(0, _HERE)                       # rev5_experiments/

from rev5_common import (out_json, result_exists, train_and_eval,
                         write_yaml, copy_test_from_weak, list_images)
from rev5_f_budget import BUDGETS, SHUFFLE_SEED, calib_names, REV5_RUNS

TRAIN_HOT_RATIO = 0.80
LOCK_PATH = os.path.join(_HERE, "rev5_h.lock")

# N=197 时 supervised-N 与已有的 rev5_s197 是同一实验 -> 直接复用, 不重训
REUSE = {197: ("rev5_s197", 0)}


# ==========================================================
# 单实例锁 (防止 waiter 与手工启动重复训练)
# ==========================================================

def acquire_lock():
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("[lock] %s exists -> another instance is running, exit."
              % LOCK_PATH)
        return False
    with os.fdopen(fd, "w") as f:
        f.write("pid=%d\n" % os.getpid())
    return True


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


# ==========================================================
# 数据集构造
# ==========================================================

def build_sup_dataset(tag, N):
    """N 张人工 GT 直接监督: train=N*0.8 + normal, val=N*0.2, test=冻结。"""
    root = os.path.join(BASE, "yolo_" + tag)
    yaml_path = os.path.join(root, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("[gen] %s exists, skip" % root, flush=True)
        return yaml_path

    hot_src = os.path.join(BASE, "baseline", "hot")
    hot_lbl = os.path.join(BASE, "baseline", "hot_labels")
    normal_src = os.path.join(BASE, "baseline", "normal")

    names = list(calib_names(N))
    if len(names) < N:
        raise RuntimeError("calib_names(%d) returned only %d names" % (N, len(names)))

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * (1 - TRAIN_HOT_RATIO))))
    val_names = names[:n_val]
    train_names = names[n_val:]

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    def put(img, txt, split):
        shutil.copy2(img, os.path.join(root, "images", split,
                                      os.path.basename(img)))
        if txt and os.path.exists(txt):
            shutil.copy2(txt, os.path.join(root, "labels", split,
                                           os.path.basename(txt)))
        else:
            stem = os.path.splitext(os.path.basename(img))[0]
            open(os.path.join(root, "labels", split, stem + ".txt"),
                 "w").close()

    for f in train_names:
        stem = os.path.splitext(f)[0]
        put(os.path.join(hot_src, f),
            os.path.join(hot_lbl, stem + ".txt"), "train")

    for f in list_images(normal_src):
        put(os.path.join(normal_src, f), None, "train")

    for f in val_names:
        stem = os.path.splitext(f)[0]
        put(os.path.join(hot_src, f),
            os.path.join(hot_lbl, stem + ".txt"), "val")

    copy_test_from_weak(root)
    yaml_path = write_yaml(root)

    n_img = {s: len(os.listdir(os.path.join(root, "images", s)))
             for s in ["train", "val", "test"]}
    n_lbl = {s: sum(1 for f in os.listdir(os.path.join(root, "labels", s))
                    if os.path.getsize(os.path.join(root, "labels", s, f)) > 0)
             for s in ["train", "val", "test"]}
    print("[gen] sup N=%d: images=%s non-empty labels=%s"
          % (N, n_img, n_lbl), flush=True)
    return yaml_path


# ==========================================================
# 主流程
# ==========================================================

def run_sup(N, seed=0):
    tag = "rev5_sup%d" % N
    reuse = REUSE.get(N)
    if reuse and seed == 0 and result_exists(*reuse):
        print("[sup N=%d] reuse %s" % (N, reuse[0]), flush=True)
        return
    if result_exists(tag, seed):
        print("[sup N=%d seed=%d] exists, skip" % (N, seed), flush=True)
        return

    yaml_path = build_sup_dataset(tag, N)     # 数据集与 seed 无关
    train_and_eval(tag, yaml_path, seed=seed, project=REV5_RUNS)


def collect_sup(seeds=(0, 1, 2, 3, 4)):
    """汇总 supervised-N 曲线 (多 seed 均值±std), 并合并进 budget_curve.json。"""
    def _val(N, tg, sd):
        p = out_json(tg, sd)
        if not (os.path.exists(p) and os.path.getsize(p) > 10):
            return None
        with open(p, encoding="utf8") as f:
            r = json.load(f)
        return {"tag": tg, "seed": sd, "mAP50": r["test_map50"],
                "mAP50_95": r["test_map50_95"],
                "P": r["test_precision"], "R": r["test_recall"]}

    out = {"N": list(BUDGETS), "sup": []}
    for N in BUDGETS:
        runs = []
        for tg in ["rev5_sup%d" % N, (REUSE.get(N) or (None,))[0]]:
            if not tg:
                continue
            for sd in seeds:
                v = _val(N, tg, sd)
                if v:
                    runs.append(v)
        if not runs:
            out["sup"].append(None)
            continue

        def agg(key):
            vals = [x[key] for x in runs]
            m = sum(vals) / len(vals)
            sd_ = 0.0
            if len(vals) > 1:
                sd_ = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            return round(m, 4), round(sd_, 4)

        m50, s50 = agg("mAP50")
        r, sr = agg("R")
        p_, sp = agg("P")
        m95, _ = agg("mAP50_95")
        out["sup"].append({"tag": runs[0]["tag"], "mAP50": m50,
                           "std_mAP50": s50, "mAP50_95": m95, "P": p_,
                           "std_P": sp, "R": r, "std_R": sr,
                           "n_seeds": len(runs),
                           "seeds": [x["seed"] for x in runs], "runs": runs})

    p = os.path.join(_HERE, "sup_curve.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 合并进 budget_curve.json (若已存在), 让画图脚本只读一个文件
    bp = os.path.join(_HERE, "budget_curve.json")
    if os.path.exists(bp) and os.path.getsize(bp) > 10:
        with open(bp, encoding="utf8") as f:
            curve = json.load(f)
        curve["sup"] = out["sup"]
        with open(bp, "w", encoding="utf8") as f:
            json.dump(curve, f, indent=2, ensure_ascii=False)
        print("[sup] merged 'sup' into", bp)
    else:
        print("[sup] budget_curve.json not found yet; wrote sup_curve.json")

    print("\n" + "=" * 72)
    print("SUPERVISED-N CURVE (test mAP@50)")
    print("=" * 72)
    for i, N in enumerate(BUDGETS):
        v = out["sup"][i]
        print("%-8d %-12s %-10s %-10s %-10s" % (
            N, "%.4f" % v["mAP50"] if v else "pending",
            "%.4f" % v["R"] if v else "-",
            "%.4f" % v["P"] if v else "-",
            v["tag"] if v else "-"))
    print("\nsaved ->", p)
    return out


def main():
    args = sys.argv[1:]
    merge_only = "--merge-only" in args
    budgets = BUDGETS
    seeds = [0]
    for i, a in enumerate(args):
        if a == "--n" and i + 1 < len(args):
            budgets = [int(x) for x in args[i + 1].split(",")]
        if a == "--seeds" and i + 1 < len(args):
            seeds = [int(x) for x in args[i + 1].split(",")]

    if merge_only:
        collect_sup()
        return

    if not acquire_lock():
        return
    try:
        print("=" * 72)
        print("rev5_h_supervised_curve.py | budgets=%s | seeds=%s"
              % (budgets, seeds))
        print("=" * 72, flush=True)
        for N in budgets:
            print("\n########## SUPERVISED arm | N=%d ##########" % N, flush=True)
            for s in seeds:
                run_sup(N, seed=s)
        collect_sup()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
