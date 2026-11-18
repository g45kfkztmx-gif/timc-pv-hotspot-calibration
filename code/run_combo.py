# -*- coding: utf-8 -*-
"""
run_combo.py
组合最优配置: tau=0.40 质量过滤底座 + 难样本挖掘 + 自训练 adopt@0.7

流程 (对齐 A5, 仅把底座 tau 从 0.45 换成 0.40):
  底座数据集 = yolo_tau040          (tau=0.40, 局部救援开)
  底座模型   = runs/pseudo_tau040/weights/best.pt
  精修       = 难样本挖掘 + adopt@0.7 + 难样本过采样
  输出       = yolo_combo_t40_a07 -> 训练 -> test 评估

输出:
  yolo_combo_t40_a07/                 精修数据集
  runs/pseudo_combo_t40_a07/weights/  最终模型
  train_results_combo_t40_a07.json    测试集结果
"""

import os
import sys
import json
import shutil

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

sys.path.insert(0, BASE)

from auto_pseudo import list_images, IoU

TAG = "combo_t40_a07"
BASE_DATASET = "yolo_tau040"
BASE_WEIGHTS = os.path.join(
    BASE, "runs", "pseudo_tau040", "weights", "best.pt"
)

CONF_PREDICT = 0.20
CONF_ADOPT = 0.70
IOU_MATCH = 0.5


def read_yolo_txt(path, W, H):

    boxes = []

    if not os.path.exists(path):
        return boxes

    with open(path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            _, xc, yc, nw, nh = parts[:5]
            xc, yc, nw, nh = float(xc), float(yc), float(nw), float(nh)

            boxes.append((
                (xc - nw / 2) * W,
                (yc - nh / 2) * H,
                nw * W,
                nh * H,
            ))

    return boxes


def refine_split(model, split, src_lbl_dir, img_dir, lbl_dir):

    src_img_dir = os.path.join(BASE, split, "hot")

    stat = {"images": 0, "adopted": 0, "missed": 0, "hard_copies": 0}

    for name in list_images(src_img_dir):

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

        res = model.predict(
            src_img, conf=CONF_PREDICT, iou=0.5,
            imgsz=320, verbose=False
        )[0]

        preds = []

        if res.boxes is not None and len(res.boxes):

            for b in res.boxes:

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                preds.append(((x1, y1, x2 - x1, y2 - y1), conf))

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

        adopted = []

        for j, (db, conf) in enumerate(preds):

            if pred_used[j]:
                continue

            if conf >= CONF_ADOPT:

                if all(IoU(db, pb) < 0.5 for pb in pseudo + adopted):
                    adopted.append(db)

        stat["adopted"] += len(adopted)

        refined = pseudo + adopted

        lines = []

        for (x, y, w, h) in refined:

            lines.append(
                "0 %.6f %.6f %.6f %.6f"
                % ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
            )

        shutil.copy2(src_img, os.path.join(img_dir, name))

        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        difficulty = missed + len(adopted)

        copies = 1
        if difficulty == 1:
            copies = 2
        elif difficulty >= 2:
            copies = 3

        for c in range(1, copies):

            dup_name = "dup%d_%s" % (c, name)
            dup_stem = os.path.splitext(dup_name)[0]

            shutil.copy2(src_img, os.path.join(img_dir, dup_name))

            with open(os.path.join(lbl_dir, dup_stem + ".txt"),
                      "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            stat["hard_copies"] += 1

    return stat


def main():

    from ultralytics import YOLO
    from train_yolo import train_and_eval

    if not os.path.exists(BASE_WEIGHTS):
        print("ERROR: base weights missing:", BASE_WEIGHTS)
        return

    root = os.path.join(BASE, "yolo_" + TAG)
    res_json = os.path.join(BASE, "train_results_%s.json" % TAG)

    if os.path.exists(res_json):
        print("results exist, skip all:", res_json)
        return

    # ---------- 构建精修数据集 ----------

    if not os.path.exists(root):

        print("loading base model:", BASE_WEIGHTS)
        model = YOLO(BASE_WEIGHTS)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(root, "images", split),
                        exist_ok=True)
            os.makedirs(os.path.join(root, "labels", split),
                        exist_ok=True)

        for split in ["train", "val"]:

            print(">>> refining %s ..." % split)

            stat = refine_split(
                model, split,
                os.path.join(BASE, BASE_DATASET, "labels", split),
                os.path.join(root, "images", split),
                os.path.join(root, "labels", split),
            )

            print("[%s|%s] %s" % (TAG, split, json.dumps(stat)))

            normal_dir = os.path.join(BASE, split, "normal")

            for name in list_images(normal_dir):

                shutil.copy2(
                    os.path.join(normal_dir, name),
                    os.path.join(root, "images", split, name)
                )

                stem = os.path.splitext(name)[0]
                open(os.path.join(
                    root, "labels", split, stem + ".txt"), "w").close()

        for sub in ["images", "labels"]:

            src = os.path.join(BASE, BASE_DATASET, sub, "test")
            dst = os.path.join(root, sub, "test")

            for name in os.listdir(src):
                shutil.copy2(os.path.join(src, name),
                             os.path.join(dst, name))

        with open(os.path.join(root, "dataset.yaml"), "w",
                  encoding="utf8") as f:
            f.write("path: %s\n" % root.replace("\\", "/"))
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("test: images/test\n")
            f.write("names:\n  0: hotspot\n")

        print("[gen] DONE ->", root)

    # ---------- 训练 + 测试集评估 ----------

    yaml_path = os.path.join(root, "dataset.yaml")

    r = train_and_eval(TAG, yaml_path)

    with open(res_json, "w", encoding="utf8") as f:
        json.dump([r], f, indent=2, ensure_ascii=False)

    print("[train] saved ->", res_json)
    print("COMBO DONE")


if __name__ == "__main__":
    main()
