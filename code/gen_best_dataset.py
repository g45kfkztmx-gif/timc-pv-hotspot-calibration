# -*- coding: utf-8 -*-
"""
gen_best_dataset.py
构建冲刺 mAP50>0.7 的混合数据集 yolo_best:

  train = train/hot(伪标签,Best-F1配置) + train/normal(空) + baseline/hot(人工GT)
  val   = val/hot(伪标签) + val/normal(空)
  test  = test/hot(人工GT) + test/normal(空)

baseline 的 197 张人工标注是高质量监督信号, 与 test 无重叠。
"""

import os
import sys
import shutil

import cv2

sys.path.insert(
    0,
    r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
)

from auto_pseudo import list_images
from auto_pseudo_diag import DiagParams, detect_diag


BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

# Best F1 配置
CFG_F1 = dict(
    k_sigma=2.6, blur_size=3, expand_ratio=0.0,
    border_min_factor=0.5, score_threshold=0.45,
    use_close=True, fit_to_core=True, core_expand_ratio=0.0,
    core_delta_mult=1.0, expand_min_px=1,
)


def pseudo_label_image(img_path, params):

    gray = cv2.imread(img_path, 0)

    if gray is None:
        return []

    H, W = gray.shape

    cands, _ = detect_diag(gray, params)

    lines = []

    for c in cands:

        if c["status"] != "accepted":
            continue

        x, y, w, h = c["box"]

        lines.append(
            "0 %.6f %.6f %.6f %.6f"
            % ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
        )

    return lines


def copy(img_src, img_dir, lbl_dir, label_lines=None, label_src=None):

    name = os.path.basename(img_src)
    stem = os.path.splitext(name)[0]

    shutil.copy2(img_src, os.path.join(img_dir, name))

    dst_lbl = os.path.join(lbl_dir, stem + ".txt")

    if label_src is not None and os.path.exists(label_src):
        shutil.copy2(label_src, dst_lbl)
        return "gt"

    with open(dst_lbl, "w") as f:
        if label_lines:
            f.write("\n".join(label_lines) + "\n")

    return "pseudo" if label_lines else "empty"


def main():

    params = DiagParams(**CFG_F1)

    root = os.path.join(BASE, "yolo_best")

    if os.path.exists(root):
        shutil.rmtree(root)

    counts = {}

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    # ---------- train: train/hot 伪标签 ----------

    train_hot = os.path.join(BASE, "train", "hot")
    img_dir = os.path.join(root, "images", "train")
    lbl_dir = os.path.join(root, "labels", "train")

    names = list_images(train_hot)

    for i, name in enumerate(names):

        lines = pseudo_label_image(os.path.join(train_hot, name), params)

        copy(os.path.join(train_hot, name), img_dir, lbl_dir,
             label_lines=lines)

        if (i + 1) % 100 == 0:
            print("  train/hot pseudo %d/%d" % (i + 1, len(names)))

    counts["train_hot_pseudo"] = len(names)

    # ---------- train: train/normal 空标注 ----------

    train_normal = os.path.join(BASE, "train", "normal")

    for name in list_images(train_normal):
        copy(os.path.join(train_normal, name), img_dir, lbl_dir)

    counts["train_normal"] = len(list_images(train_normal))

    # ---------- train: baseline/hot 人工GT ----------

    bl_hot = os.path.join(BASE, "baseline", "hot")
    bl_lbl = os.path.join(BASE, "baseline", "hot_labels")

    added = 0

    for name in list_images(bl_hot):

        stem = os.path.splitext(name)[0]
        gt = os.path.join(bl_lbl, stem + ".txt")

        if not os.path.exists(gt):
            continue

        copy(os.path.join(bl_hot, name), img_dir, lbl_dir, label_src=gt)
        added += 1

    counts["baseline_gt"] = added

    # ---------- val ----------

    img_dir = os.path.join(root, "images", "val")
    lbl_dir = os.path.join(root, "labels", "val")

    val_hot = os.path.join(BASE, "val", "hot")

    for name in list_images(val_hot):
        lines = pseudo_label_image(os.path.join(val_hot, name), params)
        copy(os.path.join(val_hot, name), img_dir, lbl_dir,
             label_lines=lines)

    counts["val_hot"] = len(list_images(val_hot))

    val_normal = os.path.join(BASE, "val", "normal")

    for name in list_images(val_normal):
        copy(os.path.join(val_normal, name), img_dir, lbl_dir)

    counts["val_normal"] = len(list_images(val_normal))

    # ---------- test ----------

    img_dir = os.path.join(root, "images", "test")
    lbl_dir = os.path.join(root, "labels", "test")

    test_hot = os.path.join(BASE, "test", "hot")
    test_lbl = os.path.join(BASE, "test", "hot_labels")

    for name in list_images(test_hot):
        stem = os.path.splitext(name)[0]
        copy(os.path.join(test_hot, name), img_dir, lbl_dir,
             label_src=os.path.join(test_lbl, stem + ".txt"))

    counts["test_hot"] = len(list_images(test_hot))

    test_normal = os.path.join(BASE, "test", "normal")

    for name in list_images(test_normal):
        copy(os.path.join(test_normal, name), img_dir, lbl_dir)

    counts["test_normal"] = len(list_images(test_normal))

    # ---------- yaml ----------

    with open(os.path.join(root, "dataset.yaml"), "w", encoding="utf8") as f:
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n  0: hotspot\n")

    print("\ncounts:", counts)
    print("yaml:", os.path.join(root, "dataset.yaml"))
    print("DONE")


if __name__ == "__main__":
    main()
