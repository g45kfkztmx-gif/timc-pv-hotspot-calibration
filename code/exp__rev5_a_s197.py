# -*- coding: utf-8 -*-
"""
rev5_a_s197.py  (实验A: S-197 全监督 baseline)
================================================
构造 yolo_rev5_s197 数据集并训练 YOLOv8n:

  train = baseline/hot 中有人工GT的 197 张(80% 训练, 20% 从预算内留作 val)
          + baseline/normal 320 张空标签负样本(背景, 不计标注预算)
  val   = 197 张人工GT中 hold-out 的 20% (仅用于 best checkpoint 选择)
  test  = 冻结的 yolo_weak test (350 张人工GT, 评估一次)

协议与原论文完全一致: epochs=60, imgsz=320, batch=32, workers=2,
cache=ram, amp=False; seed 可指定 (默认 0, 也用于 5-seed 补跑)。

论文叙述口径: "197 manual labels (80/20 train/validation split inside
the calibration budget); test set evaluated exactly once."
"""
import os
import sys
import json
import random
import shutil

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # split/
sys.path.insert(0, os.path.dirname(__file__))  # rev5_experiments/

from rev5_common import (BASE, out_json, result_exists, train_and_eval,
                         write_yaml, copy_test_from_weak, list_images)


ROOT = os.path.join(BASE, "yolo_rev5_s197")
TRAIN_HOT_RATIO = 0.8


def build_dataset():
    """一次性构建 yolo_rev5_s197。已存在则跳过。"""

    yaml_path = os.path.join(ROOT, "dataset.yaml")

    if os.path.exists(yaml_path):
        print("[s197] dataset exists ->", yaml_path)
        return yaml_path

    hot_src = os.path.join(BASE, "baseline", "hot")
    hot_lbl = os.path.join(BASE, "baseline", "hot_labels")
    normal_src = os.path.join(BASE, "baseline", "normal")

    hot_names = []
    for f in list_images(hot_src):
        stem = os.path.splitext(f)[0]
        if os.path.exists(os.path.join(hot_lbl, stem + ".txt")):
            hot_names.append(f)

    # 确认 197
    print("[s197] GT-hot images:", len(hot_names))

    random.seed(42)
    random.shuffle(hot_names)
    n_val = max(1, int(len(hot_names) * (1 - TRAIN_HOT_RATIO)))
    val_names = hot_names[:n_val]
    train_names = hot_names[n_val:]

    print("[s197] train hot:", len(train_names), "| val hot:", len(val_names))

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(ROOT, "labels", split), exist_ok=True)

    def put(img, txt, split):
        shutil.copy2(img, os.path.join(ROOT, "images", split,
                                      os.path.basename(img)))
        if txt and os.path.exists(txt):
            shutil.copy2(txt, os.path.join(ROOT, "labels", split,
                                           os.path.basename(txt)))
        else:
            stem = os.path.splitext(os.path.basename(img))[0]
            open(os.path.join(ROOT, "labels", split, stem + ".txt"),
                 "w").close()

    # train: GT hot 80% + normal(空标签)
    for f in train_names:
        stem = os.path.splitext(f)[0]
        put(os.path.join(hot_src, f),
            os.path.join(hot_lbl, stem + ".txt"), "train")

    for f in list_images(normal_src):
        put(os.path.join(normal_src, f), None, "train")

    # val: GT hot 20% hold-out (模型选择)
    for f in val_names:
        stem = os.path.splitext(f)[0]
        put(os.path.join(hot_src, f),
            os.path.join(hot_lbl, stem + ".txt"), "val")

    # test: 冻结
    copy_test_from_weak(ROOT)

    yaml_path = write_yaml(ROOT)

    n_img = {
        s: len(os.listdir(os.path.join(ROOT, "images", s)))
        for s in ["train", "val", "test"]
    }
    n_lbl = {
        s: sum(1 for f in os.listdir(os.path.join(ROOT, "labels", s))
               if os.path.getsize(os.path.join(ROOT, "labels", s, f)) > 0)
        for s in ["train", "val", "test"]
    }
    print("[s197] images:", n_img, "| non-empty labels:", n_lbl)
    print("[s197] DONE ->", yaml_path)
    return yaml_path


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tag = "rev5_s197"

    if result_exists(tag, seed):
        print("[s197] seed %d result exists, skip" % seed)
        return

    yaml_path = build_dataset()

    train_and_eval(tag, yaml_path, seed=seed,
                   project=os.path.join(BASE, "rev5_runs"))


if __name__ == "__main__":
    main()
