# -*- coding: utf-8 -*-
"""
refine_and_train.py
Round2: 难样本动态加权 + 自训练精修, 然后训练并在测试集评估。

流程:
  1. 用 Round1 模型对 train/val hot 图推理
  2. 难样本挖掘:
     missed  = 伪标签框没有被模型预测命中 (模型没学会 -> 难样本)
     new_det = 模型高置信预测但伪标签里没有 (疑似漏标 -> 自训练采纳)
  3. 精修标注 = 原伪标签 + 采纳的新检测 (去重)
  4. 难样本加权: difficulty 越高, 该图在训练集中复制份数越多
     (过采样 = 隐式提升该样本损失权重)
  5. 构建 yolo_weak_r2 -> 训练 -> test 评估
"""

import os
import sys
import json
import shutil

import cv2
import numpy as np

sys.path.insert(
    0,
    r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
)

from auto_pseudo import list_images, IoU

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

R1_WEIGHTS = os.path.join(
    BASE, "runs", "pseudo_weak_r1", "weights", "best.pt"
)

CONF_PREDICT = 0.20      # 推理低阈值收集全部预测
CONF_ADOPT = 0.50        # 自训练采纳阈值
IOU_MATCH = 0.5


def read_yolo_txt(path, W, H):
    """YOLO归一化 -> 像素框列表 [(x,y,w,h), ...]"""

    boxes = []

    if not os.path.exists(path):
        return boxes

    with open(path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            _, xc, yc, nw, nh = parts[:5]

            xc, yc, nw, nh = (
                float(xc), float(yc), float(nw), float(nh)
            )

            x = (xc - nw / 2) * W
            y = (yc - nh / 2) * H
            w = nw * W
            h = nh * H

            boxes.append((x, y, w, h))

    return boxes


def refine_split(model, split, src_img_dir, src_lbl_dir,
                 img_dir, lbl_dir):
    """对单个split做难样本挖掘+标注精修, 返回统计"""

    names = list_images(src_img_dir)

    stat = {
        "images": 0,
        "pseudo_boxes": 0,
        "adopted": 0,
        "missed": 0,
        "hard_copies": 0,
    }

    for name in names:

        src_img = os.path.join(src_img_dir, name)
        stem = os.path.splitext(name)[0]

        bgr = cv2.imread(src_img)

        if bgr is None:
            continue

        H, W = bgr.shape[:2]

        stat["images"] += 1

        pseudo = read_yolo_txt(
            os.path.join(src_lbl_dir, stem + ".txt"), W, H
        )

        stat["pseudo_boxes"] += len(pseudo)

        # ---------- 模型推理 ----------

        res = model.predict(
            src_img, conf=CONF_PREDICT, iou=0.5,
            imgsz=320, verbose=False
        )[0]

        preds = []

        if res.boxes is not None and len(res.boxes):

            for b in res.boxes:

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])

                preds.append(
                    ((x1, y1, x2 - x1, y2 - y1), conf)
                )

        # ---------- 匹配: 伪标签 vs 预测 ----------

        pred_used = [False] * len(preds)

        missed = 0

        for pb in pseudo:

            hit = False

            for j, (db, conf) in enumerate(preds):

                if pred_used[j]:
                    continue

                if IoU(pb, db) >= IOU_MATCH:
                    hit = True
                    pred_used[j] = True
                    break

            if not hit:
                missed += 1

        stat["missed"] += missed

        # ---------- 自训练: 采纳未匹配的高置信预测 ----------

        adopted = []

        for j, (db, conf) in enumerate(preds):

            if pred_used[j]:
                continue

            if conf >= CONF_ADOPT:

                # 与已有伪标签去重
                if all(IoU(db, pb) < 0.5 for pb in pseudo + adopted):
                    adopted.append(db)

        stat["adopted"] += len(adopted)

        # ---------- 写精修标注 ----------

        refined = pseudo + adopted

        lines = []

        for (x, y, w, h) in refined:

            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            nw = w / W
            nh = h / H

            lines.append(
                "0 %.6f %.6f %.6f %.6f" % (xc, yc, nw, nh)
            )

        shutil.copy2(src_img, os.path.join(img_dir, name))

        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        # ---------- 难样本过采样加权 ----------

        difficulty = missed + len(adopted)

        copies = 1

        if difficulty == 1:
            copies = 2
        elif difficulty >= 2:
            copies = 3

        for c in range(1, copies):

            dup_name = "dup%d_%s" % (c, name)
            dup_stem = os.path.splitext(dup_name)[0]

            shutil.copy2(
                src_img, os.path.join(img_dir, dup_name)
            )

            with open(
                os.path.join(lbl_dir, dup_stem + ".txt"), "w"
            ) as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            stat["hard_copies"] += 1

    return stat


def main():

    from ultralytics import YOLO

    if not os.path.exists(R1_WEIGHTS):
        print("ERROR: Round1 weights not found:", R1_WEIGHTS)
        return

    print("loading Round1 model:", R1_WEIGHTS)

    model = YOLO(R1_WEIGHTS)

    root = os.path.join(BASE, "yolo_weak_r2")

    if os.path.exists(root):
        shutil.rmtree(root)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    # ---------- train/val: 精修 + 难样本加权 ----------

    for split in ["train", "val"]:

        print("\n>>> refining %s ..." % split)

        stat = refine_split(
            model, split,
            os.path.join(BASE, split, "hot"),
            os.path.join(BASE, "yolo_weak", "labels", split),
            os.path.join(root, "images", split),
            os.path.join(root, "labels", split),
        )

        print("[%s] %s" % (split, json.dumps(stat)))

        # normal 负样本照抄
        normal_dir = os.path.join(BASE, split, "normal")

        for name in list_images(normal_dir):

            shutil.copy2(
                os.path.join(normal_dir, name),
                os.path.join(root, "images", split, name)
            )

            stem = os.path.splitext(name)[0]

            open(
                os.path.join(
                    root, "labels", split, stem + ".txt"
                ), "w"
            ).close()

    # ---------- test: 照抄 yolo_weak ----------

    for sub in ["images", "labels"]:

        src = os.path.join(BASE, "yolo_weak", sub, "test")
        dst = os.path.join(root, sub, "test")

        for name in os.listdir(src):
            shutil.copy2(os.path.join(src, name),
                         os.path.join(dst, name))

    # ---------- yaml ----------

    with open(os.path.join(root, "dataset.yaml"), "w",
              encoding="utf8") as f:
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n  0: hotspot\n")

    print("\nrefine DONE ->", root)


if __name__ == "__main__":
    main()
