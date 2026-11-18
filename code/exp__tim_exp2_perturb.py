# -*- coding: utf-8 -*-
"""
tim_exp2_perturb.py —— TIM 审稿意见 #5: 标注不确定度 -> 校准参数的传播
======================================================================
审稿人要求 (无法双标注时的替代方案): "人为扰动边界框 (±10%), 观察校准结果变化"。
做法:
  对 197 张校准图的全部 GT 框做 ±10% 随机扰动 (宽高缩放 + 中心平移,
  各 4 个独立 U(-0.1,0.1) 分量), 在扰动标签上重跑**完整校准**
  (stage1 网格 + stage2 + select_tau, 与主实验同一函数), 报告:
    - 各校准参数 (k, g*=st, bmf, kl, win, lmd, lbg, tau) 的漂移
    - 校准 F2/P/R 的漂移
    - 冻结生成器产出的**门控标签集**与未扰动参考集的逐框重合度
        (匹配数/Jaccard), 即不确定度如何传到训练标签
输出: tim_perturb_summary.json
"""
import os
import sys
import json
import random

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.join(BASE, "rev5_experiments")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

import rev5_h_lopo as L                                    # noqa: E402
import rev5_f_budget as F                                  # noqa: E402
from auto_pseudo import BASELINE_LABEL_DIR, IoU            # noqa: E402

N_SEED = 6
REF_CALIB = os.path.join(HERE, "budget_calib_N197.json")
OUT = os.path.join(HERE, "tim_perturb_summary.json")


def read_boxes(p):
    out = []
    with open(p, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                out.append([float(x) for x in parts[:5]])
    return out


def perturb(boxes, rng):
    out = []
    for cls, xc, yc, w, h in boxes:
        u = [rng.uniform(-0.10, 0.10) for _ in range(4)]
        w2 = min(1.0, max(0.01, w * (1 + u[0])))
        h2 = min(1.0, max(0.01, h * (1 + u[1])))
        xc2 = min(max(xc + u[2] * w2 / 2, w2 / 2), 1 - w2 / 2)
        yc2 = min(max(yc + u[3] * h2 / 2, h2 / 2), 1 - h2 / 2)
        out.append([cls, xc2, yc2, w2, h2])
    return out


def gen_labels(params, tau, tag):
    """冻结生成器对 train/val hot 全池产门控标签 -> 返回 {stem: [box px xywh]}"""
    root = os.path.join(L.STAGE, "r1_%s" % tag)
    os.makedirs(root, exist_ok=True)
    out = {}
    for split in ("train", "val"):
        d = os.path.join(root, split)
        os.makedirs(d, exist_ok=True)
        for name in F.list_images(os.path.join(BASE, split, "hot")):
            stem = os.path.splitext(name)[0]
            dst = os.path.join(d, stem + ".txt")
            if not os.path.exists(dst):
                entries = L.collect_entries(
                    os.path.join(BASE, split, "hot", name), params, tau) or []
                L.write_label(dst, entries)
            H, W = 110, 60
            boxes = []
            with open(dst, encoding="utf8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        _c, xc, yc, w, h = [float(x) for x in parts[:5]]
                        boxes.append(((xc - w / 2) * W, (yc - h / 2) * H,
                                      w * W, h * H))
            out[stem] = boxes
    return out


def set_stats(a, b):
    """两个标签集的逐框重合: 匹配数(IoU>=0.5) / 集合 Jaccard"""
    inter = 0
    na = sum(len(v) for v in a.values())
    nb = sum(len(v) for v in b.values())
    for stem in set(a) | set(b):
        ba = a.get(stem, [])
        bb = b.get(stem, [])
        used = [False] * len(bb)
        for x in ba:
            for j, y in enumerate(bb):
                if used[j]:
                    continue
                if IoU(x, y) >= 0.5:
                    used[j] = True
                    break
    # 重新统计匹配数
    inter = 0
    for stem in set(a) | set(b):
        ba = a.get(stem, [])
        bb = b.get(stem, [])
        used = [False] * len(bb)
        for x in ba:
            for j, y in enumerate(bb):
                if used[j]:
                    continue
                if IoU(x, y) >= 0.5:
                    used[j] = True
                    inter += 1
                    break
    union = na + nb - inter
    return {"matched": inter, "n_ref": na, "n_pert": nb,
            "recall_of_ref": round(inter / max(1, na), 4),
            "jaccard": round(inter / max(1, union), 4)}


def main():
    ref = json.load(open(REF_CALIB, encoding="utf8"))
    files = L.calib_files("")
    print("[setup] calibration files = %d (expect 197)" % len(files), flush=True)
    assert len(files) == 197, len(files)

    ref_params = F.calib_to_params(ref)
    ref_labels = gen_labels(ref_params, ref["tau"], "ref197")
    print("[ref labels] boxes=%d" % sum(len(v) for v in ref_labels.values()),
          flush=True)

    rows = []
    for seed in range(1, N_SEED + 1):
        rng = random.Random(1000 + seed)
        pdir = os.path.join(BASE, "tim_perturb_labels_s%d" % seed)
        os.makedirs(pdir, exist_ok=True)
        for name in files:
            stem = os.path.splitext(name)[0]
            src = os.path.join(BASELINE_LABEL_DIR, stem + ".txt")
            dst = os.path.join(pdir, stem + ".txt")
            if not os.path.exists(dst):
                boxes = perturb(read_boxes(src), rng)
                with open(dst, "w", encoding="utf8") as f:
                    for b in boxes:
                        f.write("0 %.6f %.6f %.6f %.6f\n"
                                % (b[1], b[2], b[3], b[4]))
        L.BASELINE_LABEL_DIR = pdir            # load_cache 改读扰动标签
        d = L.calibrate_files(files, "pert_s%d" % seed)
        L.BASELINE_LABEL_DIR = BASELINE_LABEL_DIR

        p = F.calib_to_params(d)
        lab = gen_labels(p, d["tau"], "pert_s%d" % seed)
        row = {
            "seed": seed,
            "k": d["k"], "st": d["st"], "bmf": d["bmf"], "kl": d["kl"],
            "win": d["win"], "lmd": d["lmd"], "lbg": d["use_local_bg"],
            "tau": d["tau"], "calib_F2": d["calib_F2"],
            "calib_P": d["calib_P"], "calib_R": d["calib_R"],
            "label_set": set_stats(ref_labels, lab),
        }
        rows.append(row)
        print("[perturb s%d] k=%.1f g*=%.2f tau=%.2f F2=%.4f | labels %s"
              % (seed, row["k"], row["st"], row["tau"], row["calib_F2"],
                 row["label_set"]), flush=True)

    summary = {
        "reference": {k: ref.get(k) for k in
                      ("k", "st", "bmf", "kl", "win", "lmd", "use_local_bg",
                       "tau", "calib_F2", "calib_P", "calib_R")},
        "n_seed": N_SEED,
        "perturbation": "per-box: w,h x (1+U(-0.1,0.1)); centre shifted by "
                        "U(-0.1,0.1) x new box size (clipped to image)",
        "rows": rows,
    }
    for key in ("k", "st", "bmf", "kl", "win", "lmd", "tau", "calib_F2"):
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            summary.setdefault("drift", {})[key] = {
                "min": min(vals), "max": max(vals),
                "mean": round(sum(vals) / len(vals), 4),
                "sd": round((sum((v - sum(vals) / len(vals)) ** 2
                                for v in vals) / max(1, len(vals) - 1)) ** 0.5, 4)
                if len(vals) > 1 else 0.0}
    ls = [r["label_set"] for r in rows]
    summary["label_set_overall"] = {
        "recall_of_ref_mean": round(sum(x["recall_of_ref"] for x in ls) / len(ls), 4),
        "jaccard_mean": round(sum(x["jaccard"] for x in ls) / len(ls), 4),
        "jaccard_min": min(x["jaccard"] for x in ls),
        "n_pert_boxes_mean": round(sum(x["n_pert"] for x in ls) / len(ls), 1),
        "n_ref_boxes": ls[0]["n_ref"],
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("\nWROTE", OUT)
    print(json.dumps(summary["drift"], indent=1))
    print(json.dumps(summary["label_set_overall"], indent=1))


if __name__ == "__main__":
    main()
