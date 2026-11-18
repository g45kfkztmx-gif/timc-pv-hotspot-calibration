# -*- coding: utf-8 -*-
"""
audit_split_leakage.py —— S8: 数据泄漏审计 (数据集已冻结, 只读不改)
====================================================================
回答 TIM 审稿清单 S8:
  1. image-level split (seed 42) 是否把**同一飞行架次/同一 PV 阵列**的相邻帧
     分到了不同 split? —— 用文件名里的 DJI <date>_<seq> 作为 group key。
  2. split 之间是否存在**近重复图像对**? —— aHash/dHash 汉明距离 + 降采样 NCC。

只输出报告, 不修改任何数据。
结果落盘: rev5_experiments/split_leakage_audit.json
"""
import json
import os
import re
import sys

import cv2
import numpy as np

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "split_leakage_audit.json")

SPLITS = {
    "train": ["train/hot", "train/normal"],
    "val": ["val/hot", "val/normal"],
    "test": ["test/hot", "test/normal"],
    "calib": ["baseline/hot"],
}

FLIGHT_RE = re.compile(r"^(DJI_\d{14}_\d{4})")
EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(d):
    if not os.path.isdir(d):
        return []
    return [f for f in sorted(os.listdir(d))
            if f.lower().endswith(EXTS)]


def flight_key(name):
    m = FLIGHT_RE.match(name)
    return m.group(1) if m else os.path.splitext(name)[0]


def ahash(gray, n=8):
    small = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small > small.mean()).flatten()


def dhash(gray, n=8):
    small = cv2.resize(gray, (n + 1, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small[:, 1:] > small[:, :-1]).flatten()


def collect():
    items = []          # (split, path, flight)
    for split, dirs in SPLITS.items():
        for d in dirs:
            for f in list_images(os.path.join(BASE, d)):
                items.append((split, os.path.join(BASE, d, f), flight_key(f)))
    return items


def main():
    items = collect()
    print("total images: %d" % len(items))

    # ---------- 1. group(flight) 重叠 ----------
    per_split = {}
    for split, _p, fl in items:
        per_split.setdefault(split, set()).add(fl)

    print("\n=== unique flight groups per split ===")
    for s in ("train", "val", "test", "calib"):
        print("  %-6s flights=%-5d images=%d"
              % (s, len(per_split.get(s, ())), sum(1 for x in items if x[0] == s)))

    names = ["train", "val", "test", "calib"]
    print("\n=== flight-group overlap matrix (shared groups) ===")
    print("        " + "".join("%-8s" % n for n in names))
    overlap = {}
    for a in names:
        row = []
        for b in names:
            k = len(per_split.get(a, set()) & per_split.get(b, set()))
            row.append(k)
            if a < b:
                overlap["%s|%s" % (a, b)] = k
        print("  %-6s" % a + "".join("%-8d" % v for v in row))

    # 每张 test 图有多少 train/calib 图共享其 flight
    tgt = {}
    for s in ("train", "calib"):
        for src in ("test",):
            k = len(per_split.get(src, set()) & per_split.get(s, set()))
            tgt["%s_%s_shared_flights" % (src, s)] = k

    n_test_img_in_shared = sum(
        1 for split, _p, fl in items
        if split == "test" and (fl in per_split.get("train", set())
                                or fl in per_split.get("calib", set())))
    tgt["test_images_in_flights_also_in_train_or_calib"] = n_test_img_in_shared

    # ---------- 2. 近重复图像对 ----------
    print("\n=== computing hashes for %d images ..." % len(items))
    feats = {}
    for i, (split, path, fl) in enumerate(items):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if g.ndim == 3:
            g = g[:, :, 0]
        g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
        feats[(split, path, fl)] = (ahash(g), dhash(g), g.astype(np.float32))
        if (i + 1) % 500 == 0:
            print("  %d/%d" % (i + 1, len(items)), flush=True)

    keys = list(feats.keys())
    A = np.array([feats[k][0] for k in keys])
    D = np.array([feats[k][1] for k in keys])
    sp = np.array([k[0] for k in keys])
    fl = np.array([k[2] for k in keys])

    # 组内/组间近重复: aHash 汉明 <= 6 或 dHash 汉明 <= 8, 且来自**不同 split**
    n = len(keys)
    pair_same_split = 0
    cross = {}
    near_dup_cross = []
    for i in range(n):
        ai = A[i]
        di = D[i]
        ha = np.count_nonzero(A[i + 1:] != ai, axis=1)
        hd = np.count_nonzero(D[i + 1:] != di, axis=1)
        hit = np.where((ha <= 6) | (hd <= 8))[0]
        for j in hit:
            jj = j + i + 1
            same = sp[i] == sp[jj]
            same_fl = fl[i] == fl[jj]
            if same:
                pair_same_split += 1
                continue
            key = "%s-%s" % tuple(sorted((sp[i], sp[jj])))
            cross[key] = cross.get(key, 0) + 1
            near_dup_cross.append({"a": keys[i][1], "b": keys[jj][1],
                                   "splits": sorted((sp[i], sp[jj])),
                                   "same_flight": bool(same_fl)})

    print("\n=== near-duplicate pairs (aHash<=6 or dHash<=8) ===")
    print("  within-split pairs : %d" % pair_same_split)
    for k, v in sorted(cross.items()):
        print("  cross %-14s : %d" % (k, v))
    n_cross_flightdiff = sum(1 for d in near_dup_cross if not d["same_flight"])
    print("  cross-split pairs in DIFFERENT flight groups: %d" % n_cross_flightdiff)

    report = {
        "n_images": len(items),
        "flights_per_split": {s: len(per_split.get(s, ())) for s in names},
        "images_per_split": {s: sum(1 for x in items if x[0] == s) for s in names},
        "flight_overlap_matrix": overlap,
        "test_specific": tgt,
        "near_dup": {
            "within_split_pairs": pair_same_split,
            "cross_split_pairs": cross,
            "cross_split_pairs_different_flight": n_cross_flightdiff,
            "examples": near_dup_cross[:25],
        },
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", OUT)
    return report


if __name__ == "__main__":
    main()
