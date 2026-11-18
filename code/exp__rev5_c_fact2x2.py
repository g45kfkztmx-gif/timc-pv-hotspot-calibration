# -*- coding: utf-8 -*-
"""
rev5_c_fact2x2.py  (实验C: hard oversampling vs adoption 2x2 拆分)
====================================================================
回答建议4: A3->A5 的增益到底来自 oversampling 还是 self-training 新增
伪标签 (adoption)? 原 A3->A5 box 数仅 +4, 但 hard copies +95, mAP +3.5。

因子:
  A3                 : adopt=off, oversample=off   (已有, yolo_weak/weak_r1)
  A3 + oversample-only: adopt=off, oversample=on   (rev5_fact_os)
  A3 + adopt-only     : adopt=on , oversample=off  (rev5_fact_ad)
  Full (A5)          : adopt=on , oversample=on    (已有, yolo_abl5_adopt070)

另加 random-oversample control (rev5_fact_rand): 随机复制与 hard 相同
数量的图, 证明提升不是"重复训练样本"本身带来的。

只报告 seed=0 (与已有 A3/A5 的 seed0 可比); 如需可后续补 5-seed。
输出: rev5_experiments/rev5_fact_*.json
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

from auto_pseudo import list_images as _li, IoU

R1_WEIGHTS = os.path.join(BASE, "runs", "pseudo_weak_r1", "weights", "best.pt")
CONF_PREDICT = 0.20
IOU_MATCH = 0.5
CONF_ADOPT = 0.70


def read_yolo_txt(path, W, H):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            _, xc, yc, nw, nh = p[:5]
            xc, yc, nw, nh = float(xc), float(yc), float(nw), float(nh)
            boxes.append(((xc - nw / 2) * W, (yc - nh / 2) * H,
                          nw * W, nh * H))
    return boxes


def build_fact_dataset(tag, do_adopt, do_oversample, random_ctrl=False,
                       conf_adopt=CONF_ADOPT):
    root = os.path.join(BASE, "yolo_" + tag)
    yaml_path = os.path.join(root, "dataset.yaml")

    if os.path.exists(yaml_path):
        print("[%s] exists, skip" % tag)
        return yaml_path

    from ultralytics import YOLO
    print("[%s] loading R1 for refine..." % tag)
    model = YOLO(R1_WEIGHTS)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    rand_copies_total = None
    if random_ctrl:
        # 先跑一次 hard 流程统计总复制数 (只统计不落盘)
        rand_copies_total = count_hard_copies()
        print("[rand] will replicate %d random images" % rand_copies_total)

    for split in ["train", "val"]:
        src_img_dir = os.path.join(BASE, split, "hot")
        src_lbl_dir = os.path.join(BASE, "yolo_weak", "labels", split)
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        stat = {"images": 0, "adopted": 0, "missed": 0, "hard_copies": 0}

        # random ctrl: 需要随机选取的图名(在整个 split 内)
        all_names = list_images(src_img_dir)
        rand_pool = list(all_names)
        import random
        rng = random.Random(42)
        rng.shuffle(rand_pool)

        for name in all_names:
            src_img = os.path.join(src_img_dir, name)
            stem = os.path.splitext(name)[0]
            bgr = cv2.imread(src_img)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            stat["images"] += 1

            pseudo = read_yolo_txt(
                os.path.join(src_lbl_dir, stem + ".txt"), W, H)

            res = model.predict(src_img, conf=CONF_PREDICT, iou=0.5,
                                imgsz=320, verbose=False)[0]
            preds = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0])
                    preds.append(((x1, y1, x2 - x1, y2 - y1), conf))

            # ---- miss 计数 (与 A5 一致: pseudo 未被任何 pred 覆盖) ----
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

            # ---- adoption ----
            adopted = []
            if do_adopt:
                for j, (db, conf) in enumerate(preds):
                    if pred_used[j]:
                        continue
                    if conf >= conf_adopt:
                        if all(IoU(db, pb) < 0.5 for pb in pseudo + adopted):
                            adopted.append(db)
            stat["adopted"] += len(adopted)

            refined = pseudo + adopted

            lines = []
            for (x, y, w, h) in refined:
                lines.append("0 %.6f %.6f %.6f %.6f" %
                             ((x + w / 2) / W, (y + h / 2) / H,
                              w / W, h / H))

            # 落盘原图
            shutil.copy2(src_img, os.path.join(img_dir, name))
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            # ---- 复制规则 ----
            copies = 1
            if random_ctrl:
                copies = 1  # 由外部统一随机复制
            elif do_oversample:
                difficulty = missed + (len(adopted) if do_adopt else 0)
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

        # random ctrl: 在 split 完成后再补随机复制
        if random_ctrl and rand_copies_total:
            made = 0
            while made < rand_copies_total:
                for name in rand_pool:
                    if made >= rand_copies_total:
                        break
                    src_img = os.path.join(src_img_dir, name)
                    stem = os.path.splitext(name)[0]
                    bgr = cv2.imread(src_img)
                    if bgr is None:
                        continue
                    dup_name = "rdup%d_%s" % (made, name)
                    dup_stem = os.path.splitext(dup_name)[0]
                    shutil.copy2(src_img, os.path.join(img_dir, dup_name))
                    lbl_src = os.path.join(lbl_dir, stem + ".txt")
                    if os.path.exists(lbl_src):
                        shutil.copy2(lbl_src,
                                     os.path.join(lbl_dir,
                                                  dup_stem + ".txt"))
                    made += 1
            stat["random_copies"] = made

        print("[%s|%s] %s" % (tag, split, json.dumps(stat, ensure_ascii=False)))

        # normal: 空标签
        normal_dir = os.path.join(BASE, split, "normal")
        for name in list_images(normal_dir):
            shutil.copy2(os.path.join(normal_dir, name),
                         os.path.join(root, "images", split, name))
            stem = os.path.splitext(name)[0]
            open(os.path.join(root, "labels", split, stem + ".txt"),
                 "w").close()

    copy_test_from_weak(root)
    yaml_path = write_yaml(root)
    print("[%s] DONE -> %s" % (tag, yaml_path))
    return yaml_path


def count_hard_copies():
    """统计 A5 hard oversampling 复制的总张数 (用于 random control 对齐)。"""
    root = os.path.join(BASE, "yolo_abl5_adopt070", "images", "train")
    n = sum(1 for f in os.listdir(root) if f.startswith("dup"))
    return n


def main():
    jobs = [
        ("rev5_fact_os",   dict(do_adopt=False, do_oversample=True)),
        ("rev5_fact_ad",   dict(do_adopt=True,  do_oversample=False)),
        ("rev5_fact_rand", dict(do_adopt=False, do_oversample=False,
                                random_ctrl=True)),
    ]
    for tag, kw in jobs:
        if result_exists(tag, 0):
            print("[%s] seed0 exists, skip" % tag)
            continue
        yaml_path = build_fact_dataset(tag, **kw)
        train_and_eval(tag, yaml_path, seed=0,
                       project=os.path.join(BASE, "rev5_runs"))


if __name__ == "__main__":
    main()
