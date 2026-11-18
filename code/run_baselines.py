# -*- coding: utf-8 -*-
"""
run_baselines.py
传统方法对比 baseline: Otsu全局阈值 / 自适应阈值 (adaptiveThreshold)

协议与主方法完全一致 (无泄露):
  1. 在 baseline(197张人工标注) 上标定各自超参, F2 排序
     Otsu: 无阈值参数, 只标定 min_area
     自适应: blockSize x C x min_area 网格
  2. 冻结配置, 为无标注 train/val 生成伪标签
     (相同的模糊/闭运算/连通域后处理, 无质量评分/无局部救援 — 保持 baseline 纯粹)
  3. 同配置训练 YOLOv8n, 测试集一次性评估

输出:
  yolo_baseline_otsu/, yolo_baseline_adapt/
  runs/pseudo_baseline_otsu|adapt/weights/best.pt
  train_results_baseline_otsu|adapt.json
  baseline_comparison.json
"""

import os
import sys
import json
import time
import shutil
import itertools

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

sys.path.insert(0, BASE)

from auto_pseudo import (
    list_images, load_labels, IoU, extract_components,
    BASELINE_HOT_DIR, BASELINE_LABEL_DIR,
)

BLUR_SIZE = 3
CLOSE_K = 5


# ==========================================================
# 掩膜生成器
# ==========================================================

def otsu_mask(blur):

    _, m = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return m


def adapt_mask(blur, block, cval):

    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, cval
    )


def postprocess(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K)
    )

    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def mask_to_boxes(mask, min_area):

    boxes = []

    comps, _ = extract_components(mask)

    for c in comps:

        if c["area"] < min_area:
            continue

        boxes.append(c["box"])

    return boxes


# ==========================================================
# baseline 集评估 (F2)
# ==========================================================

def fbeta(p, r, beta=2.0):

    b2 = beta * beta

    return (1 + b2) * p * r / max(b2 * p + r, 1e-8)


def load_cache():

    cache = {}

    for name in list_images(BASELINE_HOT_DIR):

        gray = cv2.imread(
            os.path.join(BASELINE_HOT_DIR, name), 0
        )

        if gray is None:
            continue

        h, w = gray.shape

        label = os.path.join(
            BASELINE_LABEL_DIR,
            os.path.splitext(name)[0] + ".txt"
        )

        if not os.path.exists(label):
            continue

        cache[name] = (load_labels(label, w, h), gray)

    return cache


def match_greedy(boxes, gts, iou_thr=0.5):

    pairs = []

    for i, b in enumerate(boxes):

        for j, g in enumerate(gts):

            v = IoU(b, g)

            if v >= iou_thr:
                pairs.append((v, i, j))

    pairs.sort(reverse=True)

    used_b = set()
    used_g = set()
    tp = 0

    for v, i, j in pairs:

        if i in used_b or j in used_g:
            continue

        used_b.add(i)
        used_g.add(j)
        tp += 1

    return tp, len(boxes) - tp, len(gts) - tp


def evaluate_mask_fn(mask_fn, min_area, cache):

    TP = FP = FN = 0

    for name in cache:

        gts, gray = cache[name]

        blur = cv2.GaussianBlur(
            gray, (BLUR_SIZE, BLUR_SIZE), 0
        )

        mask = postprocess(mask_fn(blur))

        boxes = mask_to_boxes(mask, min_area)

        tp, fp, fn = match_greedy(boxes, gts)

        TP += tp
        FP += fp
        FN += fn

    p = TP / max(TP + FP, 1)
    r = TP / max(TP + FN, 1)

    return TP, FP, FN, p, r, fbeta(p, r)


def calibrate(cache):
    """在baseline上标定两种传统方法, 返回冻结配置"""

    # ---------- Otsu: 只标 min_area ----------

    best_otsu = None

    print("[calib] Otsu:")

    for ma in [8, 16, 32]:

        TP, FP, FN, p, r, f2 = evaluate_mask_fn(
            otsu_mask, ma, cache
        )

        print(
            "  min_area=%2d  TP=%3d FP=%4d FN=%3d  "
            "P=%.3f R=%.3f F2=%.4f" % (ma, TP, FP, FN, p, r, f2)
        )

        if best_otsu is None or f2 > best_otsu[0]:
            best_otsu = (f2, ma, p, r)

    print(
        "[calib] Otsu frozen: min_area=%d (F2=%.4f)"
        % (best_otsu[1], best_otsu[0])
    )

    # ---------- 自适应: blockSize x C x min_area ----------

    best_adapt = None

    grid = list(itertools.product(
        [21, 31, 51], [2, 3, 5, 8], [8, 16, 32]
    ))

    print("[calib] adaptive: %d combinations" % len(grid))

    for i, (block, cval, ma) in enumerate(grid):

        fn = lambda blur, b=block, c=cval: adapt_mask(blur, b, c)

        TP, FP, FN, p, r, f2 = evaluate_mask_fn(fn, ma, cache)

        print(
            "  [%2d/%d] block=%2d C=%d min_area=%2d  "
            "TP=%3d FP=%4d FN=%3d  P=%.3f R=%.3f F2=%.4f"
            % (i + 1, len(grid), block, cval, ma,
               TP, FP, FN, p, r, f2)
        )

        if best_adapt is None or f2 > best_adapt[0]:
            best_adapt = (f2, block, cval, ma, p, r)

    print(
        "[calib] adaptive frozen: block=%d C=%d min_area=%d "
        "(F2=%.4f)" % (
            best_adapt[1], best_adapt[2], best_adapt[3],
            best_adapt[0]
        )
    )

    return best_otsu, best_adapt


# ==========================================================
# 数据集生成
# ==========================================================

def build_dataset(tag, mask_fn, min_area):

    root = os.path.join(BASE, "yolo_" + tag)

    if os.path.exists(root):
        print("[gen] %s exists, skip" % root)
        return os.path.join(root, "dataset.yaml")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split),
                    exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split),
                    exist_ok=True)

    stats = {}

    for split in ["train", "val"]:

        hot_dir = os.path.join(BASE, split, "hot")
        normal_dir = os.path.join(BASE, split, "normal")

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        names = list_images(hot_dir)
        n_box = 0
        n_empty = 0

        for name in names:

            src = os.path.join(hot_dir, name)
            stem = os.path.splitext(name)[0]

            shutil.copy2(src, os.path.join(img_dir, name))

            gray = cv2.imread(src, 0)

            lines = []

            if gray is not None:

                H, W = gray.shape

                blur = cv2.GaussianBlur(
                    gray, (BLUR_SIZE, BLUR_SIZE), 0
                )

                mask = postprocess(mask_fn(blur))

                for (x, y, w, h) in mask_to_boxes(mask, min_area):

                    lines.append(
                        "0 %.6f %.6f %.6f %.6f" % (
                            (x + w / 2) / W,
                            (y + h / 2) / H,
                            w / W, h / H
                        )
                    )

            with open(os.path.join(lbl_dir, stem + ".txt"),
                      "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            n_box += len(lines)
            n_empty += (len(lines) == 0)

        for name in list_images(normal_dir):

            shutil.copy2(
                os.path.join(normal_dir, name),
                os.path.join(img_dir, name)
            )

            stem = os.path.splitext(name)[0]
            open(os.path.join(lbl_dir, stem + ".txt"),
                 "w").close()

        stats[split] = {
            "hot": len(names),
            "pseudo_boxes": n_box,
            "empty_hot": n_empty,
        }

        print("[%s|%s] hot=%d boxes=%d empty=%d" % (
            tag, split, len(names), n_box, n_empty
        ))

    # test 照抄 yolo_weak (人工GT)
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
# 主流程
# ==========================================================

def main():

    from train_yolo import train_and_eval

    cache = load_cache()

    print("loaded %d baseline images" % len(cache))

    t0 = time.time()

    best_otsu, best_adapt = calibrate(cache)

    print("[calib] done in %.0fs" % (time.time() - t0))

    _, otsu_ma, _, _ = best_otsu
    _, a_block, a_c, a_ma, _, _ = best_adapt

    # ---------- 数据集 ----------

    yaml_otsu = build_dataset("baseline_otsu", otsu_mask, otsu_ma)

    yaml_adapt = build_dataset(
        "baseline_adapt",
        lambda blur: adapt_mask(blur, a_block, a_c),
        a_ma
    )

    # ---------- 训练 + 评估 ----------

    for tag, yaml_path in [
        ("baseline_otsu", yaml_otsu),
        ("baseline_adapt", yaml_adapt),
    ]:

        res_json = os.path.join(
            BASE, "train_results_%s.json" % tag
        )

        if os.path.exists(res_json):
            print("[train] %s results exist, skip" % tag)
            continue

        r = train_and_eval(tag, yaml_path)

        with open(res_json, "w", encoding="utf8") as f:
            json.dump([r], f, indent=2, ensure_ascii=False)

        print("[train] saved ->", res_json)

    # ---------- 汇总对比 ----------

    table = []

    rows = [
        ("B1", "baseline_otsu", "Otsu全局阈值"),
        ("B2", "baseline_adapt", "自适应阈值"),
        ("A1", "abl1_norescue_nofilter",
         "本文统计阈值基线(mu+k sigma)"),
        ("A5", "abl5_adopt070", "本文完整方法"),
    ]

    for gid, tag, desc in rows:

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

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (test set, GT labels)")
    print("=" * 70)
    print("%-4s %-22s %8s %10s %8s %8s  %s" % (
        "grp", "tag", "mAP50", "mAP50-95", "P", "R", "desc"
    ))

    for t in table:
        print("%-4s %-22s %8.4f %10.4f %8.4f %8.4f  %s" % (
            t["group"], t["tag"], t["mAP50"], t["mAP50-95"],
            t["P"], t["R"], t["desc"]
        ))

    summary = {
        "calibration": {
            "otsu": {
                "min_area": otsu_ma,
                "baseline_F2": round(best_otsu[0], 4),
            },
            "adaptive": {
                "blockSize": a_block,
                "C": a_c,
                "min_area": a_ma,
                "baseline_F2": round(best_adapt[0], 4),
            },
        },
        "results": table,
    }

    out = os.path.join(BASE, "baseline_comparison.json")

    with open(out, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nsaved ->", out)
    print("BASELINES DONE")


if __name__ == "__main__":
    main()
