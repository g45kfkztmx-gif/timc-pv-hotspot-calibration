# -*- coding: utf-8 -*-
"""
rev7_repr_input_sensitivity.py
==============================
IPT Round 19 —— 检测器**输入端**的表示敏感性实验。

为什么还需要这一个实验 (与 R16/R18 的 Table X 的区别)
----------------------------------------------------
R16/R18 的 Table X 变的是 **生成器消费的标量映射** (`_to_scalar`), 而四臂写进
数据集的**图像始终是原始彩色裁剪** (build_dataset 里 `shutil.copy2(src, ...)`)。
所以 Table X 证明的是:
    "标签生成只依赖强度的排序, 不依赖颜色。"      -> 生成器侧
它**没有**回答 IPT 审稿人更可能问的那句:
    "你的检测器是不是其实在学习这块 RGB 伪彩调色板?" -> 检测器侧

本脚本补的就是这一问, 设计上刻意与 Table X 正交:

  * 标签集**逐字节固定**为参考臂 (yolo_rep_luma) 的标签 —— 只让检测器的输入图像变。
    这样两臂之间唯一的差别就是"喂给网络的像素", 标签位移不参与解释。

四臂 (2 通道面 x 2 排序面):
  - `rgb`  = 原始伪彩渲染 (直接复用 yolo_rep_luma, 不重训; 与本表参考行同一配置)
  - `gray` = BT.601 亮度 -> 复制到 3 通道。**色彩完全移除、排序保留**
             且 luma(gray) == gray, 所以生成器在这张图上会产出**逐字节相同**的标签
             (probe 可验证 F1 = 1.000) => 这是"仅换输入、其余全同"的最干净隔离。
  - `pstr` = 逐图 1st-99th 百分位拉伸的亮度 -> 3 通道。**增益/偏置移除、排序保留**
  - `luts` = 亮度过固定随机置换 LUT -> 3 通道。**直方图逐灰度不变、排序被破坏** (负控制)

三项无训练的 probe (便宜, 先跑):
  1. 生成器作用在**变换后**图像上的标签, 与参考标签集的 F1 / 逐图全同比例
     —— 量化"这个变换有多激进" (gray 应为 1.000 恒等)
  2. 变换后图像的直方图与参考的 L1 距离 —— 证明 luts 的直方图确实没变

可断点续跑: 数据集有 `.done`, 训练结果有 rev7_repr_<rep>_seed0.json。

用法: python rev7_repr_input_sensitivity.py [gen|train|all]
输出: rev5_experiments/rev7_repr_input_sensitivity.json
"""
import json
import os
import shutil
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rev7_representation_sensitivity as R          # noqa: E402
from auto_pseudo_diag import DiagParams, detect_diag  # noqa: E402

BASE = R.BASE
OUT = os.path.join(HERE, "rev7_repr_input_sensitivity.json")
REF_ROOT = os.path.join(BASE, "yolo_rep_luma")        # 参考臂 = Table X 的 luma 行
NEW_ARMS = ["gray", "pstr", "luts"]
ALL_ARMS = ["rgb"] + NEW_ARMS

_LUT_PERM = np.random.RandomState(0).permutation(256).astype(np.uint8)


# ------------------------------------------------------------ 输入端变换
def to_input(bgr, arm):
    """把原始伪彩裁剪映射成"喂给检测器的那张图" (始终 3 通道 uint8)。"""
    if arm == "rgb":
        return np.ascontiguousarray(bgr, dtype=np.uint8)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if arm == "gray":
        out = g
    elif arm == "pstr":
        lo, hi = np.percentile(g, [1.0, 99.0])
        out = g.copy() if hi - lo < 1e-6 else np.clip(
            (g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    elif arm == "luts":
        out = _LUT_PERM[g]
    else:
        raise ValueError(arm)
    return np.ascontiguousarray(np.repeat(out[:, :, None], 3, axis=2))


# ---------------------------------------------------------------- 构建
def build_arm(arm, params, overwrite=False):
    root = os.path.join(BASE, "yolo_inp_" + arm)
    done = os.path.join(root, ".done")
    if os.path.exists(done) and not overwrite:
        print("[skip] %s built" % root, flush=True)
        return root
    if os.path.exists(root):
        shutil.rmtree(root)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split, srcs in (("train", ("hot", "normal")), ("val", ("hot", "normal")),
                        ("test", ("hot", "normal"))):
        for sub in srcs:
            src_dir = os.path.join(BASE, split, sub)
            if not os.path.isdir(src_dir):
                continue
            for name in sorted(R.list_images(src_dir)):
                bgr = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                cv2.imwrite(os.path.join(root, "images", split, name),
                            to_input(bgr, arm))
                # 标签一律从参考臂逐字节复制 —— 本实验的唯变量是输入像素
                stem = os.path.splitext(name)[0]
                ref_lbl = os.path.join(REF_ROOT, "labels", split, stem + ".txt")
                dst = os.path.join(root, "labels", split, stem + ".txt")
                if os.path.exists(ref_lbl):
                    shutil.copy2(ref_lbl, dst)
                else:
                    open(dst, "w").close()

    with open(os.path.join(root, "dataset.yaml"), "w", encoding="utf8") as f:
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n  0: hotspot\n")

    nb = 0
    for split in ("train", "val"):
        p = os.path.join(root, "labels", split)
        nb += sum(len([l for l in open(os.path.join(p, f)) if l.strip()])
                  for f in os.listdir(p) if f.endswith(".txt"))
    with open(os.path.join(root, "inp_meta.json"), "w", encoding="utf8") as f:
        json.dump(dict(arm=arm, labels="copied verbatim from yolo_rep_luma",
                       train_val_boxes=nb,
                       transform=dict(
                           rgb="identity (original false-colour rendering)",
                           gray="BT.601 luma replicated to three channels",
                           pstr="per-image 1st-99th percentile stretch of luma, replicated",
                           luts="luma through a fixed random permutation of the 256 "
                                "intensity levels (seed 0), replicated")[arm]),
                  f, indent=2, ensure_ascii=False)
    open(done, "w").close()
    print("[%s] built -> %s (train+val boxes=%d)" % (arm, root, nb), flush=True)
    return root


# ------------------------------------------------- 无训练 probe: 变换的激进度
def probe_labels(arm, params):
    """生成器作用在变换后图像上会产出什么标签 (与参考标签集比)。"""
    probe_root = os.path.join(BASE, "_probe_inp_" + arm)
    lbl_dir = os.path.join(probe_root, "labels", "train")
    os.makedirs(lbl_dir, exist_ok=True)
    # agreement() 用 ref_root 的 images/train 列目录, 所以只要 labels 就够
    for name in sorted(R.list_images(os.path.join(BASE, "train", "hot"))):
        bgr = cv2.imread(os.path.join(BASE, "train", "hot", name), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        g = cv2.cvtColor(to_input(bgr, arm), cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        cands, _ = detect_diag(g, params)
        lines = []
        for c in cands:
            if c["status"] != "accepted" or c["score"] < R.TAU:
                continue
            x, y, w, h = c["box"]
            lines.append("0 %.6f %.6f %.6f %.6f" % ((x + w / 2) / W, (y + h / 2) / H,
                                                    w / W, h / H))
        with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
    agr = R.agreement(REF_ROOT, probe_root, "train")
    # 顺带量一下变换后图像的亮度直方图与参考的距离 (证明 luts 的直方图没变)
    hist_ref = np.zeros(256, dtype=np.int64)
    hist_new = np.zeros(256, dtype=np.int64)
    for name in sorted(R.list_images(os.path.join(BASE, "train", "hot")))[:300]:
        a = cv2.imread(os.path.join(BASE, "train", "hot", name), cv2.IMREAD_COLOR)
        if a is None:
            continue
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gn = cv2.cvtColor(to_input(a, arm), cv2.COLOR_BGR2GRAY)
        hist_ref += np.bincount(ga.ravel(), minlength=256)
        hist_new += np.bincount(gn.ravel(), minlength=256)
    l1 = float(np.abs(hist_ref - hist_new).sum() / max(hist_ref.sum(), 1))
    return dict(probe_label_f1=agr["f1"], probe_identical_frac=agr["identical_frac"],
                probe_ref_boxes=agr["ref_boxes"], probe_gen_boxes=agr["cmp_boxes"],
                luma_hist_l1_vs_reference=round(l1, 5))


# ------------------------------------------------------------------ main
def main():
    stages = sys.argv[1] if len(sys.argv) > 1 else "all"
    params = DiagParams(**R.FROZEN_PARAMS)
    print("[cfg] arms=%s  gate=%.2f  labels fixed at yolo_rep_luma" %
          (ALL_ARMS, R.EFFECTIVE_GATE), flush=True)

    prev = {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf8"))
        except Exception:
            prev = {}

    roots = {"rgb": REF_ROOT}
    for arm in NEW_ARMS:
        roots[arm] = build_arm(arm, params)

    probe = dict(prev.get("probe", {}))
    if stages in ("all", "gen"):
        for arm in ALL_ARMS:
            probe[arm] = probe_labels(arm, params)
            print("[probe] %-5s label_F1=%.4f identical=%.1f%% luma_hist_L1=%.5f"
                  % (arm, probe[arm]["probe_label_f1"],
                     100 * probe[arm]["probe_identical_frac"],
                     probe[arm]["luma_hist_l1_vs_reference"]), flush=True)

    downstream = dict(prev.get("downstream", {}))
    if stages in ("all", "train"):
        for arm in NEW_ARMS:
            downstream[arm] = R.train_eval("inp_" + arm,
                                           os.path.join(roots[arm], "dataset.yaml"))
    # 参考臂 = Table X 的 luma 行, 直接复用, 不重训
    ref_json = os.path.join(R.REV5, "rev7_repr_luma_seed0.json")
    if stages in ("all", "train") and os.path.exists(ref_json) and "rgb" not in downstream:
        downstream["rgb"] = json.load(open(ref_json, encoding="utf8"))[0]
    if ref_json and os.path.exists(ref_json):
        downstream.setdefault("rgb", json.load(open(ref_json, encoding="utf8"))[0])

    blob = dict(
        cfg=dict(frozen_params=R.FROZEN_PARAMS, tau=R.TAU,
                 effective_gate=R.EFFECTIVE_GATE, seed=0,
                 epochs=60, imgsz=320, batch=32,
                 design="labels held fixed at the reference (luma) label set; "
                        "only the detector input representation varies"),
        reference={"arm": "rgb", "root": "yolo_rep_luma",
                   "note": "identical configuration to the luma row of the "
                           "generator-side table (Section IV-G / Table X); reused, not retrained"},
        probe=probe, downstream=downstream)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("[done] ->", OUT, flush=True)
    print(json.dumps(dict(probe_label_f1={k: v["probe_label_f1"] for k, v in probe.items()},
                          map50={k: v.get("test_map50") for k, v in downstream.items()}),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
