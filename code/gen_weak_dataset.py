# -*- coding: utf-8 -*-
"""
gen_weak_dataset.py
纯弱监督数据集构建 (无泄露版):

  - 伪标签超参数来自 baseline 校准结果(FROZEN_PARAMS), 与 test 无关
  - train/val hot: 算法生成伪标签(低阈值收集全候选 -> 质量分过滤)
  - train/val normal: 空标注(负样本)
  - test: 人工GT, 只在最终评估使用

创新点落地:
  多维质量评估 = quality_score_diag 五维指标
  置信度       = 候选综合分 score (质量分 x 边缘系数)
  边界重合度   = stability 指标 (多级掩膜重合率)
  噪声剔除     = score >= TAU_QUALITY 才写入标注

输出:
  yolo_weak/          按 TAU_QUALITY 过滤后的训练数据
  yolo_weak/candidates.json   全候选+质量分明细 (供消融实验改阈值)
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

# ===== 冻结参数: 来自 sweep_baseline.py 在 baseline 上的调优结果 =====
# 与 test 完全无关, 消除调参泄露
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

# 噪声剔除阈值: 综合分(置信度)低于该值的伪标签被剔除
TAU_QUALITY = 0.45


def collect_candidates(img_path, params):
    """收集单图全部accepted候选(含框/分数/质量指标)"""

    gray = cv2.imread(img_path, 0)

    if gray is None:
        return None, 0, 0

    H, W = gray.shape

    cands, _ = detect_diag(gray, params)

    kept = []

    for c in cands:

        if c["status"] != "accepted":
            continue

        x, y, w, h = c["box"]

        kept.append({
            "box": [
                round((x + w / 2) / W, 6),
                round((y + h / 2) / H, 6),
                round(w / W, 6),
                round(h / H, 6),
            ],
            "score": c["score"],
            "base_score": c["base_score"],
            "metrics": c["metrics"],
        })

    return kept, H, W


def write_labels(entries, dst_lbl, tau):

    lines = []

    for e in entries:

        if e["score"] < tau:
            continue

        xc, yc, nw, nh = e["box"]

        lines.append("0 %.6f %.6f %.6f %.6f" % (xc, yc, nw, nh))

    with open(dst_lbl, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return len(lines)


def parse_args(argv):
    """CLI 覆盖项。默认值 = 论文冻结操作点, 不传参时行为与原来完全一致。"""
    opt = {"tag": "yolo_weak", "tau": TAU_QUALITY,
           "local_min_delta": FROZEN_PARAMS["local_min_delta"],
           "score_threshold": FROZEN_PARAMS["score_threshold"],
           "overwrite": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--tag" and i + 1 < len(argv):
            opt["tag"] = argv[i + 1]
            i += 2
        elif a == "--tau" and i + 1 < len(argv):
            opt["tau"] = float(argv[i + 1])
            i += 2
        elif a == "--local-min-delta" and i + 1 < len(argv):
            opt["local_min_delta"] = float(argv[i + 1])
            i += 2
        elif a == "--score-threshold" and i + 1 < len(argv):
            opt["score_threshold"] = float(argv[i + 1])
            i += 2
        elif a == "--overwrite":
            opt["overwrite"] = True
            i += 1
        else:
            i += 1
    return opt


def main(argv=None):

    opt = parse_args(argv if argv is not None else sys.argv[1:])

    # 有效门限 g = max(score_threshold, tau): 候选接受由 score_threshold 控制,
    # 写标签再由 tau 过滤, 二者取大。必须显式打印, 避免冗余参数再次掩盖真实门限。
    pd = dict(FROZEN_PARAMS)
    pd["score_threshold"] = opt["score_threshold"]
    pd["local_min_delta"] = opt["local_min_delta"]
    tau = opt["tau"]
    print("[opt] tag=%s tau=%.2f st=%.2f -> effective_gate=%.2f "
          "local_min_delta=%.2f"
          % (opt["tag"], tau, pd["score_threshold"],
             max(pd["score_threshold"], tau), pd["local_min_delta"]),
          flush=True)

    params = DiagParams(**pd)

    root = os.path.join(BASE, opt["tag"])

    if os.path.exists(root):
        if not opt["overwrite"]:
            raise SystemExit("[abort] %s 已存在; 需要重建请显式加 --overwrite"
                             % root)
        shutil.rmtree(root)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    all_cands = {}
    stats = {}

    # ---------- train/val: 伪标签 ----------

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

            entries, H, W = collect_candidates(src, params)

            if entries is None:
                entries = []

            all_cands["%s/%s" % (split, name)] = entries

            n = write_labels(
                entries, os.path.join(lbl_dir, stem + ".txt"),
                tau
            )

            n_box += n
            n_empty += (n == 0)

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

        print("[%s] hot=%d normal=%d kept_boxes=%d empty_hot=%d" % (
            split, len(names), stats[split]["normal"], n_box, n_empty
        ))

    # ---------- test: 人工GT ----------

    img_dir = os.path.join(root, "images", "test")
    lbl_dir = os.path.join(root, "labels", "test")

    test_hot = os.path.join(BASE, "test", "hot")
    test_lbl = os.path.join(BASE, "test", "hot_labels")
    test_normal = os.path.join(BASE, "test", "normal")

    for name in list_images(test_hot):

        stem = os.path.splitext(name)[0]

        shutil.copy2(os.path.join(test_hot, name),
                     os.path.join(img_dir, name))

        gt = os.path.join(test_lbl, stem + ".txt")
        dst = os.path.join(lbl_dir, stem + ".txt")

        if os.path.exists(gt):
            shutil.copy2(gt, dst)
        else:
            open(dst, "w").close()

    for name in list_images(test_normal):

        shutil.copy2(os.path.join(test_normal, name),
                     os.path.join(img_dir, name))

        stem = os.path.splitext(name)[0]
        open(os.path.join(lbl_dir, stem + ".txt"), "w").close()

    stats["test"] = {
        "hot": len(list_images(test_hot)),
        "normal": len(list_images(test_normal)),
    }

    # ---------- yaml / 候选明细 ----------

    with open(os.path.join(root, "dataset.yaml"), "w", encoding="utf8") as f:
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n  0: hotspot\n")

    with open(os.path.join(root, "candidates.json"), "w",
              encoding="utf8") as f:
        json.dump(
            {
                "frozen_params": pd,
                "tau_quality": tau,
                "effective_gate": max(pd["score_threshold"], tau),
                "stats": stats,
                "candidates": all_cands,
            },
            f, ensure_ascii=False
        )

    print("\nstats:", json.dumps(stats, ensure_ascii=False))
    print("DONE ->", root)


if __name__ == "__main__":
    main()
