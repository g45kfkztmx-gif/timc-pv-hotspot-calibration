# -*- coding: utf-8 -*-
"""
rev7_representation_sensitivity.py
==================================
红外热表示敏感性分析 (Thermal representation sensitivity analysis)

问题: 源图为伪彩渲染 (rainbow palette), 生成器读入的是单一标量强度图。
      cv2.imread(path, 0) = BT.601 亮度, 只是"调色板 -> 标量"的一种映射。
      论文必须证明: 方法依赖的是局部相对结构, 而不是某一种具体映射。

做法: 冻结生成器 (FROZEN_PARAMS + tau=0.60, 与 A5 操作点完全一致) 不变,
      只替换标量强度的推导方式, 共 4 种:

  luma   : cv2 BT.601 亮度 (现有流水线, 参考)
  maxch  : max(R,G,B)          —— 峰值通道, 对调色板饱和不敏感
  meanch : mean(R,G,B)
  pnorm  : 逐图 1–99 百分位拉伸后的 luma —— 抹掉全局增益/偏置

输出:
  yolo_rep_<rep>/           4 套数据集 (同一 test GT)
  rev7_repr_sensitivity.json  标签集一致性 + 下游 mAP@50

协议: epochs=60, imgsz=320, batch=32, workers=2, cache=ram, amp=False,
      device=0, seed=0 (与主实验一致)。
断点续跑: 每步检查产物是否存在, 可反复重启。
"""
import os
import sys
import json
import time
import shutil
import random

import numpy as np
import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
REV5 = os.path.join(BASE, "rev5_experiments")
WEIGHTS = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\yolov8n.pt"

sys.path.insert(0, BASE)

from auto_pseudo import list_images              # noqa: E402
from auto_pseudo_diag import DiagParams, detect_diag   # noqa: E402

# ---- 冻结操作点: 与 gen_weak_dataset.py FROZEN_PARAMS + tau=0.60 完全一致 ----
FROZEN_PARAMS = dict(
    k_sigma=2.3,
    blur_size=3,
    expand_ratio=0.0,
    border_min_factor=0.40,
    score_threshold=0.42,
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
TAU = 0.42          # 论文报告操作点: g = max(0.42, 0.42) = 0.42
EFFECTIVE_GATE = max(FROZEN_PARAMS["score_threshold"], TAU)

OUT_JSON = os.path.join(REV5, "rev7_repr_sensitivity.json")


# ============================ 4 种标量强度映射 ============================

def _to_scalar(bgr, rep):
    """bgr: HxWx3 uint8 (OpenCV BGR). 返回 HxW uint8 标量强度图。"""
    if rep == "luma":
        # 与 cv2.imread(path, 0) 等价
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif rep == "maxch":
        out = bgr.max(axis=2)
    elif rep == "meanch":
        out = bgr.mean(axis=2).round().astype(np.uint8)
    elif rep == "pnorm":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lo, hi = np.percentile(g, [1.0, 99.0])
        if hi - lo < 1e-6:
            out = g.copy()
        else:
            out = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    else:
        raise ValueError(rep)
    return np.ascontiguousarray(out, dtype=np.uint8)


REPS = ["luma", "maxch", "meanch", "pnorm"]


# ============================ 数据集构建 ============================

def scalar_image(path, rep):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return _to_scalar(bgr, rep)


def pseudo_lines(path, rep, params):
    g = scalar_image(path, rep)
    if g is None:
        return None
    H, W = g.shape
    cands, _ = detect_diag(g, params)
    lines = []
    for c in cands:
        if c["status"] != "accepted":
            continue
        if c["score"] < TAU:          # 有效门限 g = max(0.38, 0.60) = 0.60
            continue
        x, y, w, h = c["box"]
        lines.append("0 %.6f %.6f %.6f %.6f" % (
            (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    return lines


def build_dataset(rep, params, overwrite=False):
    root = os.path.join(BASE, "yolo_rep_" + rep)
    done_flag = os.path.join(root, ".done")

    if os.path.exists(done_flag) and not overwrite:
        print("[skip] %s already built" % root, flush=True)
        return root

    if os.path.exists(root):
        shutil.rmtree(root)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    stats = {}

    for split in ["train", "val"]:
        hot = os.path.join(BASE, split, "hot")
        nrm = os.path.join(BASE, split, "normal")
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        names = list_images(hot)
        n_box, n_empty = 0, 0
        for i, name in enumerate(names):
            src = os.path.join(hot, name)
            shutil.copy2(src, os.path.join(img_dir, name))
            lines = pseudo_lines(src, rep, params)
            if lines is None:
                lines = []
            dst = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            with open(dst, "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            n_box += len(lines)
            n_empty += (len(lines) == 0)
            if (i + 1) % 200 == 0:
                print("   [%s] %s/hot %d/%d boxes=%d" % (rep, split, i + 1, len(names), n_box),
                      flush=True)

        for name in list_images(nrm):
            shutil.copy2(os.path.join(nrm, name), os.path.join(img_dir, name))
            open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w").close()

        stats[split] = dict(hot=len(names), normal=len(list_images(nrm)),
                            boxes=n_box, empty_hot=n_empty)
        print("[%s] %s hot=%d normal=%d boxes=%d empty=%d"
              % (rep, split, len(names), stats[split]["normal"], n_box, n_empty), flush=True)

    # test: 人工 GT (四个表示共用同一冻结测试集)
    for split_src, lbl_src, empty in ((os.path.join(BASE, "test", "hot"),
                                       os.path.join(BASE, "test", "hot_labels"), False),
                                      (os.path.join(BASE, "test", "normal"), None, True)):
        for name in list_images(split_src):
            stem = os.path.splitext(name)[0]
            shutil.copy2(os.path.join(split_src, name),
                         os.path.join(root, "images", "test", name))
            dst = os.path.join(root, "labels", "test", stem + ".txt")
            if empty:
                open(dst, "w").close()
            elif os.path.exists(os.path.join(lbl_src, stem + ".txt")):
                shutil.copy2(os.path.join(lbl_src, stem + ".txt"), dst)
            else:
                open(dst, "w").close()

    stats["test"] = dict(hot=len(list_images(os.path.join(BASE, "test", "hot"))),
                         normal=len(list_images(os.path.join(BASE, "test", "normal"))))

    with open(os.path.join(root, "dataset.yaml"), "w", encoding="utf8") as f:
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n  0: hotspot\n")

    with open(os.path.join(root, "repr_meta.json"), "w", encoding="utf8") as f:
        json.dump(dict(rep=rep, frozen_params=params.__dict__ if hasattr(params, "__dict__") else {},
                       tau_quality=TAU, effective_gate=EFFECTIVE_GATE, stats=stats),
                  f, indent=2, ensure_ascii=False)

    open(done_flag, "w").close()
    print("[%s] built -> %s" % (rep, root), flush=True)
    return root


# ============================ 标签集一致性 ============================

def read_labels(path, W, H):
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        xc, yc, w, h = [float(v) for v in p[1:5]]
        out.append((xc * W, yc * H, w * W, h * H))
    return out


def iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def agreement(ref_root, cmp_root, split, W=60, H=110, thr=0.5):
    """贪心 IoU 匹配, 返回 tag/box 级一致性与逐图完全一致比例。"""
    ref_dir = os.path.join(ref_root, "labels", split)
    cmp_dir = os.path.join(cmp_root, "labels", split)
    imgs = sorted(os.listdir(os.path.join(ref_root, "images", split)))
    tp = fp = fn = 0
    identical = 0
    n = 0
    for name in imgs:
        stem = os.path.splitext(name)[0]
        a = read_labels(os.path.join(ref_dir, stem + ".txt"), W, H)
        b = read_labels(os.path.join(cmp_dir, stem + ".txt"), W, H)
        n += 1
        used = [False] * len(b)
        matches = 0
        for x in a:
            best, bi = 0.0, -1
            for j, y in enumerate(b):
                if used[j]:
                    continue
                v = iou(x, y)
                if v > best:
                    best, bi = v, j
            if best >= thr:
                used[bi] = True
                matches += 1
        tp += matches
        fn += len(a) - matches
        fp += len(b) - matches
        if len(a) == len(b) and matches == len(a):
            identical += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n_images=n, ref_boxes=tp + fn, cmp_boxes=tp + fp, matched=tp,
                precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                identical_images=identical,
                identical_frac=round(identical / n, 4) if n else 0.0)


# ============================ 训练 + 评估 ============================

def train_eval(rep, yaml_path, seed=0):
    from ultralytics import YOLO

    res_json = os.path.join(REV5, "rev7_repr_%s_seed%d.json" % (rep, seed))
    if os.path.exists(res_json) and os.path.getsize(res_json) > 10:
        print("[skip] %s trained" % rep, flush=True)
        return json.load(open(res_json, encoding="utf8"))[0]

    print("\n" + "=" * 64)
    print("TRAIN repr=%s seed=%d" % (rep, seed))
    print("=" * 64, flush=True)

    model = YOLO(WEIGHTS)
    t0 = time.time()
    model.train(data=yaml_path, epochs=60, imgsz=320, batch=32, workers=2,
                device=0, amp=False, cache="ram",
                project=os.path.join(BASE, "runs"),
                name="pseudo_repr_%s" % rep, exist_ok=True, verbose=False, seed=seed)
    train_sec = time.time() - t0

    print(">>> evaluating on frozen test split ...", flush=True)
    m = model.val(data=yaml_path, split="test", imgsz=320, batch=32,
                  workers=2, device=0, verbose=False)

    out = dict(rep=rep, seed=seed, train_seconds=round(train_sec, 1),
               test_map50=round(float(m.box.map50), 4),
               test_map50_95=round(float(m.box.map), 4),
               test_precision=round(float(m.box.mp), 4),
               test_recall=round(float(m.box.mr), 4))
    print("[%s] mAP50=%.4f P=%.4f R=%.4f (%.0fs)"
          % (rep, out["test_map50"], out["test_precision"], out["test_recall"], train_sec),
          flush=True)
    with open(res_json, "w", encoding="utf8") as f:
        json.dump([out], f, indent=2, ensure_ascii=False)
    return out


# ============================ main ============================

def main():
    stages = sys.argv[1] if len(sys.argv) > 1 else "all"
    params = DiagParams(**FROZEN_PARAMS)
    print("[cfg] effective_gate=%.2f tau=%.2f k_sigma=%.2f k_local=%.2f "
          "local_min_delta=%.1f" % (EFFECTIVE_GATE, TAU, FROZEN_PARAMS["k_sigma"],
                                    FROZEN_PARAMS["k_local"],
                                    FROZEN_PARAMS["local_min_delta"]), flush=True)

    roots = {}
    t0 = time.time()
    for rep in REPS:
        roots[rep] = build_dataset(rep, params)
    print("[gen] done in %.0fs" % (time.time() - t0), flush=True)

    ref = roots["luma"]
    agr = {}
    for rep in REPS:
        if rep == "luma":
            continue
        agr[rep + "_vs_luma"] = dict(
            train=agreement(ref, roots[rep], "train"),
            val=agreement(ref, roots[rep], "val"))
        print("[agree] %s vs luma:" % rep, json.dumps(agr[rep + "_vs_luma"]["train"],
                                                     ensure_ascii=False), flush=True)

    # 与仓库中已有参考数据集 (yolo_weak, 论文操作点下的原始构建) 对照
    legacy = os.path.join(BASE, "yolo_weak")
    if os.path.isdir(os.path.join(legacy, "labels", "train")):
        agr["luma_vs_yolo_weak"] = dict(train=agreement(legacy, ref, "train"))
        print("[agree] luma vs yolo_weak (reference build):",
              json.dumps(agr["luma_vs_yolo_weak"]["train"], ensure_ascii=False), flush=True)

    results = {}
    if stages in ("all", "train"):
        for rep in REPS:
            results[rep] = train_eval(rep, os.path.join(roots[rep], "dataset.yaml"))

    prev = {}
    if os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON, encoding="utf8"))
        except Exception:
            prev = {}
    blob = dict(cfg=dict(frozen_params=FROZEN_PARAMS, tau=TAU, effective_gate=EFFECTIVE_GATE,
                         seed=0, epochs=60, imgsz=320, batch=32),
                agreement=agr,
                downstream=results or prev.get("downstream", {}),
                reference=f"paper operating point: {FROZEN_PARAMS['score_threshold']} + tau={TAU}")
    with open(OUT_JSON, "w", encoding="utf8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("[done] ->", OUT_JSON, flush=True)
    print(json.dumps(dict(agreement={k: v["train"]["f1"] for k, v in agr.items()},
                          downstream={k: v["test_map50"] for k, v in (results or {}).items()}),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
