# -*- coding: utf-8 -*-
"""
sweep_baseline.py
在 baseline(197张人工标注) 上重新调优伪标签超参数。

目的: 消除"在测试集上调参"的泄露。
baseline 仅用于校准伪标签生成器, 调完冻结, 之后:
  - 在无标注 train/val 上生成伪标签
  - test 只在最终评估时碰一次

排序指标: F2 (召回加权) + F1, 与之前测试集扫描同网格便于对照。
"""

import os
import cv2
import sys
import time
import itertools

sys.path.insert(
    0,
    r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
)

from auto_pseudo import (
    list_images, load_labels,
    BASELINE_HOT_DIR, BASELINE_LABEL_DIR,
)
from auto_pseudo_diag import DiagParams, detect_diag, match_diag


def fbeta(p, r, beta=2.0):
    b2 = beta * beta
    return (1 + b2) * p * r / max(b2 * p + r, 1e-8)


def evaluate(params, cache):

    TP = FP = FN = 0

    for name in cache:

        gts, gray = cache[name]

        cands, info = detect_diag(gray, params)

        matched = match_diag(cands, gts)

        TP += sum(matched)
        FN += len(gts) - sum(matched)

        FP += sum(
            1 for c in cands
            if c["status"] == "accepted" and c["matched_gt"] is None
        )

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    f2 = fbeta(precision, recall)

    return TP, FP, FN, precision, recall, f1, f2


def base_params(**over):

    kw = dict(
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

    kw.update(over)

    return DiagParams(**kw)


def load_cache():

    files = list_images(BASELINE_HOT_DIR)

    cache = {}

    for name in files:

        gray = cv2.imread(os.path.join(BASELINE_HOT_DIR, name), 0)

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


def stage1(cache):

    grid = list(itertools.product(
        [2.3, 2.4, 2.5, 2.6],         # k_sigma
        [0.38, 0.40, 0.42, 0.45],     # score_threshold
        [0.35, 0.40, 0.50],           # border_min_factor
    ))

    print("[stage1] combinations:", len(grid))

    results = []

    t0 = time.time()

    for i, (k, st, bmf) in enumerate(grid):

        params = base_params(
            k_sigma=k, score_threshold=st, border_min_factor=bmf
        )

        TP, FP, FN, p, r, f1, f2 = evaluate(params, cache)

        results.append((f2, r, p, f1, TP, FP, FN, k, st, bmf))

        print(
            "[%2d/%d] k=%.1f st=%.2f bmf=%.2f  TP=%3d FP=%3d FN=%3d  "
            "P=%.3f R=%.3f F1=%.3f F2=%.3f"
            % (i + 1, len(grid), k, st, bmf, TP, FP, FN, p, r, f1, f2)
        )

    results.sort(reverse=True)

    print("\n[stage1] TOP 8 by F2 (%.0fs):" % (time.time() - t0))

    for f2, r, p, f1, TP, FP, FN, k, st, bmf in results[:8]:
        print(
            "  k=%.1f st=%.2f bmf=%.2f  F2=%.4f R=%.3f P=%.3f F1=%.3f"
            % (k, st, bmf, f2, r, p, f1)
        )

    return results


def stage2(cache, top_bases):

    grid = list(itertools.product(
        [1.8, 2.0, 2.2],              # k_local
        [25, 31],                     # local_win
        [3.0, 4.0],                   # local_min_delta
        [False, True],                # use_local_bg
    ))

    print("\n[stage2] local rescue combinations per base:", len(grid))

    results = []

    t0 = time.time()

    n = 0
    total = len(top_bases) * len(grid)

    for (bk, bst, bbmf) in top_bases:

        for kl, win, lmd, lbg in grid:

            n += 1

            params = base_params(
                k_sigma=bk,
                score_threshold=bst,
                border_min_factor=bbmf,
                use_local_rescue=True,
                k_local=kl,
                local_win=win,
                local_min_delta=lmd,
                use_local_bg=lbg,
            )

            TP, FP, FN, p, r, f1, f2 = evaluate(params, cache)

            results.append(
                (f2, r, p, f1, TP, FP, FN,
                 bk, bst, bbmf, kl, win, lmd, lbg)
            )

            print(
                "[%3d/%d] base(k=%.1f st=%.2f bmf=%.2f) "
                "kl=%.1f win=%d lmd=%.0f lbg=%d  TP=%3d FP=%3d FN=%3d  "
                "P=%.3f R=%.3f F1=%.3f F2=%.3f"
                % (
                    n, total, bk, bst, bbmf, kl, win, lmd, int(lbg),
                    TP, FP, FN, p, r, f1, f2
                )
            )

    results.sort(reverse=True)

    print("\n[stage2] TOP 10 by F2 (%.0fs):" % (time.time() - t0))

    for (f2, r, p, f1, TP, FP, FN,
         bk, bst, bbmf, kl, win, lmd, lbg) in results[:10]:

        print(
            "  k=%.1f st=%.2f bmf=%.2f kl=%.1f win=%d lmd=%.0f lbg=%d  "
            "F2=%.4f R=%.3f P=%.3f F1=%.3f"
            % (bk, bst, bbmf, kl, win, lmd, int(lbg), f2, r, p, f1)
        )

    guarded = [x for x in results if x[2] >= 0.45]

    if guarded:

        guarded.sort(key=lambda x: -x[1])

        print("\n[stage2] TOP 5 by RECALL (P>=0.45):")

        for (f2, r, p, f1, TP, FP, FN,
             bk, bst, bbmf, kl, win, lmd, lbg) in guarded[:5]:

            print(
                "  k=%.1f st=%.2f bmf=%.2f kl=%.1f win=%d lmd=%.0f lbg=%d  "
                "R=%.3f P=%.3f F1=%.3f F2=%.4f"
                % (bk, bst, bbmf, kl, win, lmd, int(lbg), r, p, f1, f2)
            )

    return results


def main():

    cache = load_cache()

    print("loaded %d baseline images (with labels)" % len(cache))

    s1 = stage1(cache)

    top_bases = [(r[7], r[8], r[9]) for r in s1[:2]]

    print("\nstage2 bases:", top_bases)

    stage2(cache, top_bases)


if __name__ == "__main__":
    main()
