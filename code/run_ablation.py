# -*- coding: utf-8 -*-
"""
run_ablation.py
消融实验一键脚本 (纯弱监督, 无泄露):

  A1  abl1_norescue_nofilter : 纯全局阈值伪标签基线 (无局部救援, 无质量过滤)
  A2  abl2_rescue_nofilter   : + 局部自适应救援 (无质量过滤)
  A3  = weak_r1 (已训)       : + 多维质量评估噪声剔除 (tau=0.45)
  A4  = weak_r2 (已训)       : + 难样本挖掘 + 自训练精修 (adopt@0.5)
  A5  abl5_adopt070          : A4 但自训练采纳阈值 0.5->0.7 (敏感性)

设计原则:
  - 检测器冻结参数 FROZEN_PARAMS 全程不变 (baseline 标定, 与 test 无关)
  - 每组只在训练/验证集上生成伪标签, test 只用人工GT评估一次
  - 支持断点续跑: 已存在的数据集/结果自动跳过

输出:
  yolo_abl1/, yolo_abl2/, yolo_abl5/   消融数据集
  runs/pseudo_abl*/weights/best.pt     消融模型
  ablation_results.json                全消融汇总表
"""

import os
import sys
import json
import shutil

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

sys.path.insert(0, BASE)

from auto_pseudo import list_images, IoU
from auto_pseudo_diag import DiagParams, detect_diag

# ===== 冻结参数: 与 gen_weak_dataset.py 完全一致 =====
FROZEN_PARAMS = dict(
    k_sigma=2.3,
    blur_size=3,
    expand_ratio=0.0,
    border_min_factor=0.40,
    score_threshold=0.38,
    use_close=True,
    fit_to_core=True,
    core_expand_ratio=0.0,
    core_delta_mult=1.0,
    expand_min_px=1,
    use_local_rescue=True,
    k_local=2.2,
    local_win=31,
    local_min_delta=4.0,
    use_local_bg=True,
)

TAU_QUALITY = 0.45

R1_WEIGHTS = os.path.join(BASE, "runs", "pseudo_weak_r1", "weights", "best.pt")

CONF_PREDICT = 0.20
IOU_MATCH = 0.5


# ==========================================================
# 伪标签数据集生成 (参数化: 救援开关 + 质量过滤阈值)
# ==========================================================

def collect_candidates(img_path, params):

    gray = cv2.imread(img_path, 0)

    if gray is None:
        return None

    H, W = gray.shape

    cands, _ = detect_diag(gray, params)

    kept = []

    for c in cands:

        if c["status"] != "accepted":
            continue

        x, y, w, h = c["box"]

        kept.append((
            (x + w / 2) / W,
            (y + h / 2) / H,
            w / W,
            h / H,
            c["score"],
        ))

    return kept


def build_pseudo_dataset(tag, use_rescue, tau):
    """生成一套伪标签消融数据集 -> yolo_{tag}/"""

    root = os.path.join(BASE, "yolo_" + tag)

    if os.path.exists(root):
        print("[gen] %s exists, skip" % root)
        return os.path.join(root, "dataset.yaml")

    pd = dict(FROZEN_PARAMS)
    pd["use_local_rescue"] = use_rescue

    params = DiagParams(**pd)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    stats = {}

    for split in ["train", "val"]:

        hot_dir = os.path.join(BASE, split, "hot")
        normal_dir = os.path.join(BASE, split, "normal")

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        names = list_images(hot_dir)
        n_box = 0
        n_empty = 0

        for i, name in enumerate(names):

            src = os.path.join(hot_dir, name)
            stem = os.path.splitext(name)[0]

            shutil.copy2(src, os.path.join(img_dir, name))

            entries = collect_candidates(src, params) or []

            lines = []

            for (xc, yc, nw, nh, score) in entries:

                if score < tau:
                    continue

                lines.append(
                    "0 %.6f %.6f %.6f %.6f" % (xc, yc, nw, nh)
                )

            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            n_box += len(lines)
            n_empty += (len(lines) == 0)

            if (i + 1) % 100 == 0:
                print("  %s/hot %d/%d" % (split, i + 1, len(names)))

        for name in list_images(normal_dir):

            shutil.copy2(
                os.path.join(normal_dir, name),
                os.path.join(img_dir, name)
            )

            stem = os.path.splitext(name)[0]
            open(os.path.join(lbl_dir, stem + ".txt"), "w").close()

        stats[split] = {
            "hot": len(names),
            "normal": len(list_images(normal_dir)),
            "pseudo_boxes_kept": n_box,
            "empty_hot": n_empty,
        }

        print("[%s|%s] hot=%d kept=%d empty=%d" % (
            tag, split, len(names), n_box, n_empty
        ))

    # test 直接照抄 yolo_weak (人工GT)
    for sub in ["images", "labels"]:

        src = os.path.join(BASE, "yolo_weak", sub, "test")
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

    print("[gen] DONE -> %s | %s" % (root, json.dumps(stats, ensure_ascii=False)))

    return os.path.join(root, "dataset.yaml")


# ==========================================================
# A5: 难样本精修变体 (自训练采纳阈值可调)
# ==========================================================

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


def build_refined_dataset(tag, conf_adopt, base_lbl_dir=None, r1_weights=None):
    """基于 base 数据集 + R1 模型, 难样本挖掘 + 自训练 (adopt 阈值可调)。

    base_lbl_dir / r1_weights 可覆盖; 默认保持论文操作点 (yolo_weak + weak_r1)。
    """

    root = os.path.join(BASE, "yolo_" + tag)

    if os.path.exists(root):
        print("[gen] %s exists, skip" % root)
        return os.path.join(root, "dataset.yaml")

    from ultralytics import YOLO

    print("[gen] loading R1 model for refine ...")
    model = YOLO(r1_weights or R1_WEIGHTS)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split in ["train", "val"]:

        src_img_dir = os.path.join(BASE, split, "hot")
        lbl_root = base_lbl_dir or os.path.join(BASE, "yolo_weak", "labels")
        src_lbl_dir = os.path.join(lbl_root, split)

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

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

                if conf >= conf_adopt:

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

        print("[%s|%s] %s" % (tag, split, json.dumps(stat)))

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

        src = os.path.join(BASE, "yolo_weak", sub, "test")
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

    return os.path.join(root, "dataset.yaml")


# ==========================================================
# 主流程: 生成 -> 训练 -> 汇总
# ==========================================================

def main():

    from train_yolo import train_and_eval

    jobs = [
        # (tag, 结果json, 数据集构建函数)
        ("abl1_norescue_nofilter", None),
        ("abl2_rescue_nofilter", None),
        ("abl5_adopt070", None),
    ]

    # ---------- 数据集构建 ----------

    yamls = {}

    yamls["abl1_norescue_nofilter"] = build_pseudo_dataset(
        "abl1_norescue_nofilter", use_rescue=False, tau=0.0
    )

    yamls["abl2_rescue_nofilter"] = build_pseudo_dataset(
        "abl2_rescue_nofilter", use_rescue=True, tau=0.0
    )

    yamls["abl5_adopt070"] = build_refined_dataset(
        "abl5_adopt070", conf_adopt=0.70
    )

    # ---------- 训练 + 测试集评估 ----------

    for tag, _ in jobs:

        res_json = os.path.join(BASE, "train_results_%s.json" % tag)

        if os.path.exists(res_json):
            print("[train] %s results exist, skip" % tag)
            continue

        r = train_and_eval(tag, yamls[tag])

        with open(res_json, "w", encoding="utf8") as f:
            json.dump([r], f, indent=2, ensure_ascii=False)

        print("[train] saved ->", res_json)

    # ---------- 汇总消融表 ----------

    table = []

    rows = [
        ("A1", "abl1_norescue_nofilter", "纯全局阈值伪标签基线"),
        ("A2", "abl2_rescue_nofilter", "+局部自适应救援"),
        ("A3", "weak_r1", "+多维质量过滤tau=0.45 (Round1)"),
        ("A4", "weak_r2", "+难样本挖掘+自训练adopt@0.5 (Round2)"),
        ("A5", "abl5_adopt070", "A4但adopt@0.7"),
    ]

    for gid, tag, desc in rows:

        if tag == "weak_r1":
            path = os.path.join(BASE, "train_results.json")
        else:
            path = os.path.join(BASE, "train_results_%s.json" % tag)

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf8") as f:
            r = json.load(f)[0]

        table.append({
            "group": gid,
            "tag": tag,
            "desc": desc,
            "mAP50": r["test_map50"],
            "mAP50-95": r["test_map50_95"],
            "P": r["test_precision"],
            "R": r["test_recall"],
        })

    print("\n" + "=" * 72)
    print("ABLATION TABLE (test set, GT labels)")
    print("=" * 72)
    print("%-4s %-24s %8s %10s %8s %8s  %s" % (
        "grp", "tag", "mAP50", "mAP50-95", "P", "R", "desc"
    ))

    for t in table:
        print("%-4s %-24s %8.4f %10.4f %8.4f %8.4f  %s" % (
            t["group"], t["tag"], t["mAP50"], t["mAP50-95"],
            t["P"], t["R"], t["desc"]
        ))

    out = os.path.join(BASE, "ablation_results.json")

    with open(out, "w", encoding="utf8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    print("\nsaved ->", out)
    print("ALL ABLATIONS DONE")


if __name__ == "__main__":
    main()
