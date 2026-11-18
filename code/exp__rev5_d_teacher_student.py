# -*- coding: utf-8 -*-
"""
rev5_d_teacher_student.py  (实验D: 轻量 teacher-student SSOD baseline)
======================================================================
对应建议3.3: "197-label YOLO teacher -> 对 unlabeled pool 生成高置信伪标签
-> student 重训"。 用与 S-197 相同的 197 张人工 GT 训练 teacher,
对 pool(1301 train + 143 val 的 hot 图) 生成 conf>=0.5 的伪标签,
student 用 197 GT + teacher 伪标签重训 (同一协议)。

若 S-197 已有 best.pt 则直接复用作 teacher, 不重复训练。

输出: rev5_experiments/rev5_ts_student.json (student 在冻结 test 上评估)
"""
import os
import sys
import json
import shutil

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from rev5_common import (BASE, out_json, result_exists, train_and_eval,
                         write_yaml, copy_test_from_weak, list_images)

TEACHER_TAG = "rev5_s197"
TEACHER_W = os.path.join(BASE, "rev5_runs", "pseudo_" + TEACHER_TAG,
                         "weights", "best.pt")
STUDENT_TAG = "rev5_ts_student"
CONF_TEACHER = 0.50


def build_student_dataset():
    root = os.path.join(BASE, "yolo_" + STUDENT_TAG)
    yaml_path = os.path.join(root, "dataset.yaml")

    if os.path.exists(yaml_path):
        print("[ts] dataset exists ->", yaml_path)
        return yaml_path

    teacher_pt = TEACHER_W
    if not os.path.exists(teacher_pt):
        raise RuntimeError("teacher weights missing: %s" % teacher_pt)

    from ultralytics import YOLO
    print("[ts] loading teacher:", teacher_pt)
    teacher = YOLO(teacher_pt)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    # labeled: baseline 197 GT (80/20 与 S-197 同划分保持一致训练分布)
    hot_src = os.path.join(BASE, "baseline", "hot")
    hot_lbl = os.path.join(BASE, "baseline", "hot_labels")
    import random
    hot_names = [f for f in list_images(hot_src)
                 if os.path.exists(os.path.join(
                     hot_lbl, os.path.splitext(f)[0] + ".txt"))]
    rng = random.Random(42)
    rng.shuffle(hot_names)
    n_val = max(1, int(len(hot_names) * 0.2))
    val_names = set(hot_names[:n_val])
    train_labeled = [f for f in hot_names if f not in val_names]

    for split, names in [("train", train_labeled), ("val", list(val_names))]:
        for f in names:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(hot_src, f),
                         os.path.join(root, "images", split, f))
            shutil.copy2(os.path.join(hot_lbl, stem + ".txt"),
                         os.path.join(root, "labels", split, stem + ".txt"))

    # normal 作为负样本 (labeled 部分也配 baseline normal)
    normal_src = os.path.join(BASE, "baseline", "normal")
    for split, names in [("train", range(0, 260)),
                         ("val", range(260, 320))]:
        for i in names:
            fs = sorted(list_images(normal_src))
            if i >= len(fs):
                break
            f = fs[i]
            shutil.copy2(os.path.join(normal_src, f),
                         os.path.join(root, "images", split, f))
            stem = os.path.splitext(f)[0]
            open(os.path.join(root, "labels", split, stem + ".txt"),
                 "w").close()

    # unlabeled: teacher 对 split/train|val 的 hot 图生成伪标签
    pseudo_stat = {"train": 0, "val": 0}
    for split, d in [("train", os.path.join(BASE, "train", "hot")),
                     ("val", os.path.join(BASE, "val", "hot"))]:
        for name in list_images(d):
            src = os.path.join(d, name)
            bgr = cv2.imread(src)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            res = teacher.predict(bgr, conf=CONF_TEACHER, iou=0.5,
                                  imgsz=320, verbose=False)[0]
            lines = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    bw, bh = (x2 - x1) / W, (y2 - y1) / H
                    lines.append("0 %.6f %.6f %.6f %.6f"
                                 % (cx, cy, bw, bh))
            shutil.copy2(src, os.path.join(root, "images", split, name))
            stem = os.path.splitext(name)[0]
            with open(os.path.join(root, "labels", split, stem + ".txt"),
                      "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            pseudo_stat[split] += len(lines)
        print("[ts] teacher pseudo boxes in %s: %d"
              % (split, pseudo_stat[split]))

    # 注意: labeled GT 与 teacher 伪标签可能有文件名冲突(DJI 前缀不同, 已核验无重叠)
    copy_test_from_weak(root)
    yaml_path = write_yaml(root)

    for split in ["train", "val", "test"]:
        n_img = len(os.listdir(os.path.join(root, "images", split)))
        n_box = sum(
            1 for f in os.listdir(os.path.join(root, "labels", split))
            if os.path.getsize(os.path.join(root, "labels", split, f)) > 0)
        print("[ts] %s: %d imgs, %d non-empty" % (split, n_img, n_box))

    return yaml_path


def main():
    if result_exists(STUDENT_TAG, 0):
        print("[ts] student result exists, skip")
        return
    yaml_path = build_student_dataset()
    train_and_eval(STUDENT_TAG, yaml_path, seed=0,
                   project=os.path.join(BASE, "rev5_runs"))


if __name__ == "__main__":
    main()
