# -*- coding: utf-8 -*-
"""
gen_pseudo_labels.py
用诊断增强版伪标签算法(auto_pseudo_diag)为 train/val 生成 YOLO 标注,
构建标准 YOLO 数据集。

生成两套数据集用于对比:
  yolo_f1      —— Best F1 配置 (k=2.6 st=0.45 bmf=0.50, 无local rescue)
  yolo_recall  —— 召回优先配置 (k=2.6 st=0.38 bmf=0.35 + local rescue)

规则:
  - train/hot, val/hot   : 跑检测算法生成伪标注
  - train/normal, val/normal : 空txt(负样本)
  - test/hot  : 复制人工GT标注; test/normal : 空txt
"""

import os
import sys
import json
import shutil

import cv2

sys.path.insert(
    0,
    r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
)

from auto_pseudo import list_images
from auto_pseudo_diag import DiagParams, detect_diag


BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

SPLITS = {
    "train": os.path.join(BASE, "train"),
    "val": os.path.join(BASE, "val"),
    "test": os.path.join(BASE, "test"),
}

# Best F1 配置 (全测试集 P=0.664 R=0.617 F1=0.639)
CFG_F1 = dict(
    k_sigma=2.6,
    blur_size=3,
    expand_ratio=0.0,
    border_min_factor=0.5,
    score_threshold=0.45,
    use_close=True,
    fit_to_core=True,
    core_expand_ratio=0.0,
    core_delta_mult=1.0,
    expand_min_px=1,
)

# 召回优先配置 (全测试集 P=0.511 R=0.656, local rescue 救回no_signal)
CFG_RECALL = dict(
    k_sigma=2.6,
    blur_size=3,
    expand_ratio=0.0,
    border_min_factor=0.35,
    score_threshold=0.38,
    use_close=True,
    fit_to_core=True,
    core_expand_ratio=0.0,
    core_delta_mult=1.0,
    expand_min_px=1,
    use_local_rescue=True,
    k_local=2.0,
    local_win=31,
    local_min_delta=4.0,
    use_local_bg=False,
)


def pseudo_label_image(img_path, params):
    """返回YOLO格式行列表 ['0 xc yc w h', ...]"""

    gray = cv2.imread(img_path, 0)

    if gray is None:
        return None

    H, W = gray.shape

    cands, _ = detect_diag(gray, params)

    lines = []

    for c in cands:

        if c["status"] != "accepted":
            continue

        x, y, w, h = c["box"]

        xc = (x + w / 2) / W
        yc = (y + h / 2) / H
        nw = w / W
        nh = h / H

        lines.append(
            "0 %.6f %.6f %.6f %.6f" % (xc, yc, nw, nh)
        )

    return lines


def build_dataset(tag, params):

    root = os.path.join(BASE, "yolo_" + tag)

    if os.path.exists(root):
        shutil.rmtree(root)

    stats = {"train": 0, "val": 0, "test": 0, "boxes": 0, "empty": 0}

    for split, src in SPLITS.items():

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        hot_dir = os.path.join(src, "hot")
        normal_dir = os.path.join(src, "normal")

        # ---------- hot 图片 ----------

        for name in list_images(hot_dir):

            src_img = os.path.join(hot_dir, name)
            dst_img = os.path.join(img_dir, name)

            shutil.copy2(src_img, dst_img)

            stem = os.path.splitext(name)[0]
            dst_lbl = os.path.join(lbl_dir, stem + ".txt")

            if split == "test":

                # 测试集: 复制人工GT标注
                gt_lbl = os.path.join(
                    src, "hot_labels", stem + ".txt"
                )

                if os.path.exists(gt_lbl):
                    shutil.copy2(gt_lbl, dst_lbl)
                else:
                    open(dst_lbl, "w").close()

                stats[split] += 1
                continue

            # train/val: 算法生成伪标注
            lines = pseudo_label_image(src_img, params)

            if lines is None:
                lines = []

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            stats[split] += 1
            stats["boxes"] += len(lines)

            if not lines:
                stats["empty"] += 1

        # ---------- normal 图片 (负样本, 空标注) ----------

        for name in list_images(normal_dir):

            shutil.copy2(
                os.path.join(normal_dir, name),
                os.path.join(img_dir, name)
            )

            stem = os.path.splitext(name)[0]

            open(
                os.path.join(lbl_dir, stem + ".txt"), "w"
            ).close()

            stats[split] += 1

        print(
            "[%s] %s done: %d images" % (tag, split, stats[split])
        )

    # ---------- dataset.yaml ----------

    yaml_path = os.path.join(root, "dataset.yaml")

    with open(yaml_path, "w", encoding="utf8") as f:

        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        f.write("  0: hotspot\n")

    # ---------- 参数记录 ----------

    with open(
        os.path.join(root, "pseudo_params.json"), "w", encoding="utf8"
    ) as f:
        json.dump(params.__dict__ if hasattr(params, "__dict__")
                  else {}, f, indent=2)

    print(
        "[%s] boxes=%d empty_hot=%d yaml=%s"
        % (tag, stats["boxes"], stats["empty"], yaml_path)
    )

    return root


def main():

    print(">>> Building yolo_f1 (Best F1 config) ...")
    build_dataset("f1", DiagParams(**CFG_F1))

    print("\n>>> Building yolo_recall (Recall-first config) ...")
    build_dataset("recall", DiagParams(**CFG_RECALL))

    print("\nALL DONE")


if __name__ == "__main__":
    main()
