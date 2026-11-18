# -*- coding: utf-8 -*-
"""
rev5_f_budget.py  (实验F: 标注预算曲线 / calibration-budget curve)
======================================================================
动机 (导师/审稿人视角):
  论文主张 "label-efficient", 但 Ours 与 Teacher-Student 都恰好用了 197 张
  人工标注图。审稿人必然质疑:
      "Both methods require exactly 197 manually annotated images.
       Why is the proposed method more label-efficient?"
  因此必须证明的不是"初始标注更少", 而是:
      人工标注的**角色不同** + 部署时的**边际标注成本不同**:
        T-S : 197 GT -> teacher 训练 -> 伪标签 -> student (标注直接监督检测器)
        Ours: 197 GT -> 校准统计生成器 -> 冻结 -> 无标注数据 -> 伪标签
              -> 检测器 (标注只用于配置, 不进入检测器参数优化)
  核心假设: 一旦校准完成, 统计伪标签生成器能否在标注预算极小时仍可用?
      -> 标注预算曲线 mAP@50 vs N (N = 人工标注图数量)

做法 (两臂, N = 20/50/100/150/197, 嵌套子集 seed=42):
  Arm OURS (Calibration-Guided):
      N 张人工框 -> 在 N 上重跑 2 阶段 F2 网格搜索校准统计生成器
                 -> 在 N 上选质量阈值 tau (F2)
                 -> 冻结 -> 对无标注 pool 生成伪标签 -> 训 R1
                 -> 用 R1 在 N 上选 tau_adopt (F2)
                 -> 难样本挖掘 + 自训练 -> 最终检测器
  Arm TS (Teacher-Student):
      N 张人工框 -> 训 teacher -> 对无标注 pool 生成 conf>=0.5 伪标签
                 -> student 用 N GT + teacher 伪标签重训
                 构造与 rev5_d_teacher_student.py 完全一致 (N=197 即 0.9325)

预期结论 (无论结果如何都可写):
  - 若出现 crossover (小 N 时 Ours 更高, 大 N 时 TS 更高), 则论文叙事升级为
    "annotation-budget / supervision-performance trade-off", 直接给工程用户
    一个预算决策规则;
  - 若 Ours 在全 N 上更平坦, 则证明"校准引导方法对标注预算低敏感"。

输出:
  rev5_experiments/budget_calib_N{N}.json   各 N 的校准结果 (参数/tau/F2)
  rev5_experiments/rev5_bud_ours{N}_*.json  各臂训练结果
  rev5_experiments/budget_curve.json        汇总曲线

用法:
  python rev5_f_budget.py                # 全部 (可反复启动续跑)
  python rev5_f_budget.py --only ours    # 只跑 OURS 臂
  python rev5_f_budget.py --n 20,50      # 只跑指定预算
"""
import os
import sys
import json
import time
import random
import shutil
import itertools

import cv2
import numpy as np

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

from rev5_common import (out_json, result_exists, train_and_eval,  # noqa: E402
                         write_yaml, copy_test_from_weak, list_images, REV5)

from auto_pseudo import (list_images as ap_list_images, load_labels,  # noqa: E402
                         IoU, BASELINE_HOT_DIR, BASELINE_LABEL_DIR)
from auto_pseudo_diag import DiagParams, detect_diag  # noqa: E402

# ==========================================================
# 配置
# ==========================================================

BUDGETS = [20, 50, 100, 150, 197]
SHUFFLE_SEED = 42

# 校准网格: 与论文原校准程序 sweep_baseline.py 完全一致 (2 阶段 F2 网格)。
# 对每个 N 使用同一网格, 因此预算曲线内部完全可比。
K_GRID = [2.3, 2.4, 2.5, 2.6]           # stage1: 全局 z 阈值
ST_GRID = [0.38, 0.40, 0.42, 0.45]      # stage1: 候选质量门 (score_threshold)
BMF_GRID = [0.35, 0.40, 0.50]           # stage1: 边界罚最小因子
KL_GRID = [1.8, 2.0, 2.2]               # stage2: 局部救援 z 阈值
WIN_GRID = [25, 31]                     # stage2: 局部窗口
LMD_GRID = [3.0, 4.0]                   # stage2: local_min_delta
LBG_GRID = [False, True]                # stage2: 局部环形背景
N_BASES = 2                             # stage2 取 stage1 前 2 名作为 base
TAU_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]          # 写标签质量阈值
TAU_ADOPT_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]

# 采纳门的**预注册**网格: 论文 Section IV-G 声明的就是这一段。
# 预算曲线在**每个 N 上独立**于该网格内选择 tau_adopt, 绝不使用超出该预算的
# 信息 (审稿清单 S4)。判据确定性: (1) 校准 F2 最大; (2) F2 相差 < EPS 时取
# precision 更高者; (3) 仍相同则取更大的阈值。
# 注: 若把网格放宽到 0.30-0.40, 校准 F2 会继续单调上升 (见 rev5_g 实测), 那属于
# 网格外推而非预注册协议, 论文只声明预注册网格内的最大值。
TAU_ADOPT_PREREG = [0.50, 0.60, 0.70]
TAU_ADOPT_EPS = 0.005
TAU_ADOPT_FIXED = 0.60                  # 兼容旧单点运行 (yolo_weak 口径)

CONF_PREDICT = 0.20                     # R1 推理下限 (与 run_ablation 一致)
CONF_TEACHER = 0.50                     # teacher 伪标签置信度
IOU_MATCH = 0.5

REV5_RUNS = os.path.join(BASE, "rev5_runs")

# N=197 的 TS 臂复用已有结果 (构造完全一致, 避免重复训练)。
# OURS 臂不复用: 预算曲线在每个 N 上重新校准生成器, N=197 的重校准参数
# 与原 FROZEN_PARAMS 不同 (见 budget_calib_N197.json), 必须重训才自洽。
REUSE = {
    ("ts", 197): ("rev5_ts_student", 0),
}


# ==========================================================
# 工具
# ==========================================================

def fbeta(p, r, beta=2.0):
    b2 = beta * beta
    return (1 + b2) * p * r / max(b2 * p + r, 1e-8)


def calib_names(N):
    """嵌套校准子集: 固定 shuffle(seed=42) 后的前 N 张 (含标签) 人工标注图。"""
    files = []
    for name in list_images(BASELINE_HOT_DIR):
        stem = os.path.splitext(name)[0]
        if os.path.exists(os.path.join(BASELINE_LABEL_DIR, stem + ".txt")):
            files.append(name)
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(files)
    return files[:N]


def load_calib_cache(N):
    cache = {}
    for name in calib_names(N):
        p = os.path.join(BASELINE_HOT_DIR, name)
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if gray.ndim == 3:                      # ultralytics 会 patch cv2.imread
            gray = gray[:, :, 0]
        h, w = gray.shape
        gt = load_labels(
            os.path.join(BASELINE_LABEL_DIR,
                         os.path.splitext(name)[0] + ".txt"), w, h)
        cache[name] = {"gt": gt, "gray": gray, "H": h, "W": w}
    return cache


def base_params(**over):
    kw = dict(
        k_sigma=2.3, blur_size=3, expand_ratio=0.0,
        border_min_factor=0.40, score_threshold=0.38,
        use_close=True, fit_to_core=True, core_expand_ratio=0.0,
        core_delta_mult=1.0, expand_min_px=1,
    )
    kw.update(over)
    return DiagParams(**kw)


def greedy_match(acc, gt, thr=IOU_MATCH):
    """acc: [(score, box)] -> matched flags (与 match_diag 同语义)。"""
    acc = sorted(acc, key=lambda x: -x[0])
    used = [False] * len(gt)
    for _sc, box in acc:
        best, idx = 0.0, -1
        for i, g in enumerate(gt):
            if used[i]:
                continue
            v = IoU(box, g)
            if v > best:
                best, idx = v, i
        if best >= thr and idx >= 0:
            used[idx] = True
    return used


def score_candidates(cands, H, W, proto, st, bmf, tau):
    """
    从"宽松参数"下缓存的候选里, 精确重算 (st, bmf, tau) 下的 accepted 集合。
    依赖: expand_ratio == core_expand_ratio == 0 -> box 不受阈值影响;
          score_threshold / border_min_factor 只作用于最终 score 的门限。
    """
    min_area = H * W * proto.min_area_ratio
    acc = []
    for c in cands:
        if c["origin"] == "merged":          # 被核心拆分替代的合并框, 恒拒绝
            continue
        ma = min_area * 0.5 if c["origin"] == "core" else min_area
        if c["area"] < ma:
            continue
        if c["aspect"] > proto.max_aspect:
            continue
        bs = c["base_score"]
        bp = c["border_penalty"]
        factor = bmf + (1 - bmf) * bp
        if bs >= proto.border_exempt_score:
            factor = max(factor, proto.border_exempt_factor)
        sc = bs * factor
        if sc < st or sc < tau:
            continue
        acc.append((sc, c["box"]))
    return acc


def eval_cached(per_img, proto, st, bmf, tau):
    TP = FP = FN = 0
    for name, e in per_img.items():
        acc = score_candidates(e["cands"], e["H"], e["W"], proto, st, bmf, tau)
        used = greedy_match(acc, e["gt"])
        tp = sum(used)
        TP += tp
        FN += len(e["gt"]) - tp
        FP += len(acc) - tp
    p = TP / max(TP + FP, 1)
    r = TP / max(TP + FN, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return TP, FP, FN, p, r, f1, fbeta(p, r)


def detect_cache_for(cache, **over):
    """对给定 (mask 相关) 参数跑一次 detect_diag, 缓存候选 (宽松门限)。"""
    params = base_params(score_threshold=0.0, border_min_factor=0.0, **over)
    out = {}
    for name, e in cache.items():
        cands, _ = detect_diag(e["gray"], params)
        out[name] = {"cands": cands, "gt": e["gt"], "H": e["H"], "W": e["W"]}
    return out


# ==========================================================
# 校准: 2 阶段 F2 网格 + tau 选择
# ==========================================================

def stage1(cache, proto):
    """返回按 F2 降序的前 N_BASES 个 base 配置 (cfg dict)。"""
    print("  [stage1] k x (st, bmf) 网格 ...", flush=True)
    rows = []
    t0 = time.time()
    for k in K_GRID:
        per_img = detect_cache_for(cache, k_sigma=k)
        for (st, bmf) in itertools.product(ST_GRID, BMF_GRID):
            res = eval_cached(per_img, proto, st, bmf, 0.0)
            rows.append({"res": res, "k": k, "st": st, "bmf": bmf,
                         "kl": None, "win": None, "lmd": None, "lbg": None})
        print("    k=%.1f done (%.0fs)" % (k, time.time() - t0), flush=True)
    rows.sort(key=lambda x: -x["res"][6])
    b = rows[0]
    print("  [stage1] best k=%.1f st=%.2f bmf=%.2f  P=%.3f R=%.3f F2=%.4f"
          % (b["k"], b["st"], b["bmf"], b["res"][3], b["res"][4], b["res"][6]),
          flush=True)
    return rows[:N_BASES]


def stage2(cache, proto, bases):
    """在 top-N bases 上做局部救援网格搜索, 返回最优 cfg dict。"""
    best = min(bases, key=lambda c: -c["res"][6])
    print("  [stage2] 局部救援网格 (bases=%s) ..."
          % [(c["k"], c["st"], c["bmf"]) for c in bases], flush=True)
    t0 = time.time()
    for base in bases:
        bk, bst, bbmf = base["k"], base["st"], base["bmf"]
        for (kl, win, lmd, lbg) in itertools.product(
                KL_GRID, WIN_GRID, LMD_GRID, LBG_GRID):
            per_img = detect_cache_for(
                cache, k_sigma=bk, use_local_rescue=True, k_local=kl,
                local_win=win, local_min_delta=lmd, use_local_bg=lbg)
            for (st, bmf) in itertools.product(ST_GRID, BMF_GRID):
                res = eval_cached(per_img, proto, st, bmf, 0.0)
                if res[6] > best["res"][6]:
                    best = {"res": res, "k": bk, "st": st, "bmf": bmf,
                            "kl": kl, "win": win, "lmd": lmd, "lbg": lbg}
        print("    base k=%.1f st=%.2f bmf=%.2f done (%.0fs)"
              % (bk, bst, bbmf, time.time() - t0), flush=True)
    r = best["res"]
    print("  [stage2] best k=%.1f st=%.2f bmf=%.2f kl=%s win=%s lmd=%s lbg=%s  "
          "P=%.3f R=%.3f F2=%.4f"
          % (best["k"], best["st"], best["bmf"], best["kl"], best["win"],
             best["lmd"], best["lbg"], r[3], r[4], r[6]), flush=True)
    return best


def detect_over(cfg):
    """由 cfg 构造 detect 侧参数字典 (mask 相关)。"""
    over = dict(k_sigma=cfg["k"])
    if cfg.get("kl") is not None:
        over.update(use_local_rescue=True, k_local=cfg["kl"],
                    local_win=cfg["win"], local_min_delta=cfg["lmd"],
                    use_local_bg=cfg["lbg"])
    return over


def select_tau(cache, proto, cfg):
    per_img = detect_cache_for(cache, **detect_over(cfg))
    rows = []
    for tau in TAU_GRID:
        res = eval_cached(per_img, proto, cfg["st"], cfg["bmf"], tau)
        rows.append({"tau": tau, "P": round(res[3], 4), "R": round(res[4], 4),
                     "F1": round(res[5], 4), "F2": round(res[6], 4),
                     "TP": res[0], "FP": res[1], "FN": res[2]})
    best = max(rows, key=lambda r: r["F2"])
    print("  [tau] " + " | ".join("t=%.2f F2=%.4f" % (r["tau"], r["F2"])
                                  for r in rows), flush=True)
    print("  [tau] best tau=%.2f (F2=%.4f)" % (best["tau"], best["F2"]),
          flush=True)
    return best, rows


def calibrate(N):
    """在 N 张校准图上重跑完整校准, 返回参数字典。结果落盘可续跑。"""
    out = os.path.join(REV5, "budget_calib_N%d.json" % N)
    if os.path.exists(out) and os.path.getsize(out) > 10:
        with open(out, encoding="utf8") as f:
            d = json.load(f)
        if d.get("grid") == "full":          # 仅复用完整网格的结果
            print("[calib N=%d] cached: k=%.1f st=%.2f bmf=%.2f kl=%s win=%s "
                  "tau=%.2f" % (N, d["k"], d["st"], d["bmf"], d["kl"],
                                d["win"], d["tau"]), flush=True)
            return d
        print("[calib N=%d] 旧网格结果, 重新校准" % N, flush=True)

    print("\n" + "-" * 66)
    print("[calib N=%d] 在 %d 张人工标注图上重新校准生成器" % (N, N))
    print("-" * 66, flush=True)

    proto = DiagParams()
    cache = load_calib_cache(N)
    print("  校准图: %d 张 (含标签)" % len(cache), flush=True)

    t0 = time.time()
    bases = stage1(cache, proto)
    cfg = stage2(cache, proto, bases)
    tau_best, tau_rows = select_tau(cache, proto, cfg)

    d = {
        "N": N, "grid": "full",
        "k": cfg["k"], "st": cfg["st"], "bmf": cfg["bmf"],
        "kl": cfg["kl"], "win": cfg["win"], "lmd": cfg["lmd"],
        "use_local_bg": cfg["lbg"],
        "tau": tau_best["tau"],
        "calib_F2": tau_best["F2"], "calib_P": tau_best["P"],
        "calib_R": tau_best["R"],
        "tau_rows": tau_rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out, "w", encoding="utf8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print("[calib N=%d] DONE k=%.1f st=%.2f bmf=%.2f kl=%s win=%s tau=%.2f "
          "F2=%.4f (%.0fs)" % (N, d["k"], d["st"], d["bmf"], d["kl"],
                               d["win"], d["tau"], d["calib_F2"],
                               d["elapsed_sec"]), flush=True)
    return d


def calib_to_params(d):
    over = dict(
        k_sigma=d["k"], score_threshold=d["st"], border_min_factor=d["bmf"],
    )
    if d.get("kl") is not None:
        over.update(
            use_local_rescue=True, k_local=d["kl"], local_win=d["win"],
            local_min_delta=d["lmd"], use_local_bg=d["use_local_bg"],
        )
    return base_params(**over)


# ==========================================================
# OURS 臂: 伪标签数据集构造 (参数化)
# ==========================================================

def collect_candidates(img_path, params):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    H, W = gray.shape
    cands, _ = detect_diag(gray, params)
    kept = []
    for c in cands:
        if c["status"] != "accepted":
            continue
        x, y, w, h = c["box"]
        kept.append(((x + w / 2) / W, (y + h / 2) / H, w / W, h / H,
                     c["score"]))
    return kept


def build_pseudo_dataset(tag, params, tau):
    root = os.path.join(BASE, "yolo_" + tag)
    yaml_path = os.path.join(root, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("[gen] %s exists, skip" % root, flush=True)
        return yaml_path

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split in ["train", "val"]:
        hot_dir = os.path.join(BASE, split, "hot")
        normal_dir = os.path.join(BASE, split, "normal")
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        names = list_images(hot_dir)
        n_box = n_empty = 0
        for i, name in enumerate(names):
            src = os.path.join(hot_dir, name)
            stem = os.path.splitext(name)[0]
            shutil.copy2(src, os.path.join(img_dir, name))
            entries = collect_candidates(src, params) or []
            lines = ["0 %.6f %.6f %.6f %.6f" % e[:4]
                     for e in entries if e[4] >= tau]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            n_box += len(lines)
            n_empty += (len(lines) == 0)
            if (i + 1) % 200 == 0:
                print("  %s/hot %d/%d" % (split, i + 1, len(names)), flush=True)

        for name in list_images(normal_dir):
            shutil.copy2(os.path.join(normal_dir, name),
                         os.path.join(img_dir, name))
            open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"),
                 "w").close()

        print("[%s|%s] hot=%d kept=%d empty=%d"
              % (tag, split, len(names), n_box, n_empty), flush=True)

    copy_test_from_weak(root)
    return write_yaml(root)


def read_yolo_txt(path, W, H):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, nw, nh = parts[:5]
            xc, yc, nw, nh = float(xc), float(yc), float(nw), float(nh)
            boxes.append(((xc - nw / 2) * W, (yc - nh / 2) * H, nw * W, nh * H))
    return boxes


def pick_tau_adopt(rows, grid):
    """确定性 tie-break: 先取 F2 最大, 次取 precision 高, 末取阈值大。"""
    ok = [r for r in rows if r["tau_adopt"] in grid]
    assert ok, "no tau_adopt rows inside the pre-registered grid"
    best_f2 = max(r["F2"] for r in ok)
    near = [r for r in ok if r["F2"] >= best_f2 - TAU_ADOPT_EPS]
    best_p = max(r["P"] for r in near)
    near = [r for r in near if r["P"] >= best_p - 1e-9]
    return max(near, key=lambda r: r["tau_adopt"])


def select_tau_adopt(N, params, tau, r1_weights, grid=None):
    """在 N 张校准图上用 F2 选择自训练采纳阈值 tau_adopt。

    只用这 N 张校准图 (load_calib_cache(N)), 不触碰任何更大预算的信息,
    因此预算曲线在低 N 时严格成立 (审稿清单 S4)。
    """
    grid = grid if grid is not None else TAU_ADOPT_PREREG
    from ultralytics import YOLO
    print("  [tau_adopt] 在 N=%d 校准图上选阈值 ..." % N, flush=True)
    model = YOLO(r1_weights)

    cache = load_calib_cache(N)
    rows = []
    for name, e in cache.items():
        src = os.path.join(BASELINE_HOT_DIR, name)
        bgr = cv2.imread(src)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        # 统计伪标签 (冻结生成器)
        cands, _ = detect_diag(e["gray"], params)
        pseudo = [c["box"] for c in cands
                  if c["status"] == "accepted" and c["score"] >= tau]
        # R1 预测
        res = model.predict(src, conf=CONF_PREDICT, iou=0.5, imgsz=320,
                            verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds.append(((x1, y1, x2 - x1, y2 - y1), float(b.conf[0])))
        e["_pseudo"] = pseudo
        e["_preds"] = preds

    for t in TAU_ADOPT_GRID:
        TP = FP = FN = 0
        n_adopted = 0
        for name, e in cache.items():
            if "_pseudo" not in e:
                continue
            pseudo = e["_pseudo"]
            pred_used = [False] * len(e["_preds"])
            for pb in pseudo:
                for j, (db, _c) in enumerate(e["_preds"]):
                    if pred_used[j]:
                        continue
                    if IoU(pb, db) >= IOU_MATCH:
                        pred_used[j] = True
                        break
            adopted = []
            for j, (db, conf) in enumerate(e["_preds"]):
                if pred_used[j]:
                    continue
                if conf >= t and all(IoU(db, pb) < 0.5
                                     for pb in pseudo + adopted):
                    adopted.append(db)
            acc = [(1.0, b) for b in pseudo + adopted]
            n_adopted += len(adopted)
            used = greedy_match(acc, e["gt"])
            tp = sum(used)
            TP += tp
            FN += len(e["gt"]) - tp
            FP += len(acc) - tp
        p = TP / max(TP + FP, 1)
        r = TP / max(TP + FN, 1)
        rows.append({"tau_adopt": t, "adopted": n_adopted,
                     "P": round(p, 4), "R": round(r, 4),
                     "F2": round(fbeta(p, r), 4)})
        print("    tau_adopt=%.2f P=%.4f R=%.4f F2=%.4f"
              % (t, p, r, fbeta(p, r)), flush=True)

    best = pick_tau_adopt(rows, grid)
    print("  [tau_adopt] N=%d selected=%.2f (F2=%.4f, pre-registered grid %s; "
          "tie-break: max F2 -> max P -> max tau)"
          % (N, best["tau_adopt"], best["F2"], grid), flush=True)
    return best["tau_adopt"], rows


def build_refined_dataset(tag, conf_adopt, r1_weights, base_lbl_dir=None):
    """Stage II 精炼数据集。

    base_lbl_dir 必须是**该 N 自己**的统计伪标签目录
    (yolo_<r1_tag>/labels), 而不是全 197 张口径的 yolo_weak ——
    否则低预算点会用到超出预算的校准信息 (审稿清单 S4)。
    默认 None 时回落到 yolo_weak 仅为兼容旧的单点运行。
    """
    root = os.path.join(BASE, "yolo_" + tag)
    yaml_path = os.path.join(root, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("[gen] %s exists, skip" % root, flush=True)
        return yaml_path

    from ultralytics import YOLO
    print("[gen] loading R1 model for refine:", r1_weights, flush=True)
    model = YOLO(r1_weights)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split in ["train", "val"]:
        src_img_dir = os.path.join(BASE, split, "hot")
        base = base_lbl_dir or os.path.join(BASE, "yolo_weak", "labels")
        src_lbl_dir = os.path.join(base, split)
        print("[gen] %s/%s base pseudo-labels <- %s"
              % (tag, split, src_lbl_dir), flush=True)
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
                os.path.join(src_lbl_dir, stem + ".txt"), W, H)
            res = model.predict(src_img, conf=CONF_PREDICT, iou=0.5,
                                imgsz=320, verbose=False)[0]
            preds = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    preds.append(((x1, y1, x2 - x1, y2 - y1),
                                  float(b.conf[0])))

            pred_used = [False] * len(preds)
            missed = 0
            for pb in pseudo:
                hit = False
                for j, (db, _c) in enumerate(preds):
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
                if conf >= conf_adopt and all(IoU(db, pb) < 0.5
                                              for pb in pseudo + adopted):
                    adopted.append(db)
            stat["adopted"] += len(adopted)

            lines = ["0 %.6f %.6f %.6f %.6f"
                     % ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
                     for (x, y, w, h) in pseudo + adopted]

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
                with open(os.path.join(lbl_dir, dup_stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                    if lines:
                        f.write("\n")
                stat["hard_copies"] += 1

        print("[%s|%s] %s" % (tag, split, json.dumps(stat)), flush=True)

        for name in list_images(os.path.join(BASE, split, "normal")):
            shutil.copy2(os.path.join(BASE, split, "normal", name),
                         os.path.join(root, "images", split, name))
            open(os.path.join(root, "labels", split,
                              os.path.splitext(name)[0] + ".txt"), "w").close()

    copy_test_from_weak(root)
    print("[gen] DONE ->", root, flush=True)
    return write_yaml(root)


# ==========================================================
# TS 臂: teacher -> student (构造与 rev5_d 一致, 按 N 参数化)
# ==========================================================

def build_teacher_dataset(tag, N):
    root = os.path.join(BASE, "yolo_" + tag)
    yaml_path = os.path.join(root, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("[gen] %s exists, skip" % root, flush=True)
        return yaml_path

    hot_src = os.path.join(BASE, "baseline", "hot")
    hot_lbl = os.path.join(BASE, "baseline", "hot_labels")
    names = calib_names(N)

    rng = random.Random(SHUFFLE_SEED)
    names = list(names)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * 0.2)))
    val_names = names[:n_val]
    train_names = names[n_val:]

    for split in ["train", "val"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split, subset in [("train", train_names), ("val", val_names)]:
        for f in subset:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(hot_src, f),
                         os.path.join(root, "images", split, f))
            shutil.copy2(os.path.join(hot_lbl, stem + ".txt"),
                         os.path.join(root, "labels", split, stem + ".txt"))

    normal_src = os.path.join(BASE, "baseline", "normal")
    normals = list_images(normal_src)
    for split, idxs in [("train", range(0, 260)), ("val", range(260, 320))]:
        for i in idxs:
            if i >= len(normals):
                break
            f = normals[i]
            shutil.copy2(os.path.join(normal_src, f),
                         os.path.join(root, "images", split, f))
            open(os.path.join(root, "labels", split,
                              os.path.splitext(f)[0] + ".txt"), "w").close()

    copy_test_from_weak(root)
    print("[gen] teacher N=%d: train=%d val=%d (+normal)"
          % (N, len(train_names), len(val_names)), flush=True)
    return write_yaml(root)


def build_student_dataset(tag, N, teacher_weights):
    root = os.path.join(BASE, "yolo_" + tag)
    yaml_path = os.path.join(root, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("[gen] %s exists, skip" % root, flush=True)
        return yaml_path

    from ultralytics import YOLO
    print("[gen] student: teacher =", teacher_weights, flush=True)
    teacher = YOLO(teacher_weights)

    hot_src = os.path.join(BASE, "baseline", "hot")
    hot_lbl = os.path.join(BASE, "baseline", "hot_labels")
    names = list(calib_names(N))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * 0.2)))
    val_names = names[:n_val]
    train_names = names[n_val:]

    for split in ["train", "val"]:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split, subset in [("train", train_names), ("val", val_names)]:
        for f in subset:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(hot_src, f),
                         os.path.join(root, "images", split, f))
            shutil.copy2(os.path.join(hot_lbl, stem + ".txt"),
                         os.path.join(root, "labels", split, stem + ".txt"))

    normal_src = os.path.join(BASE, "baseline", "normal")
    normals = list_images(normal_src)
    for split, idxs in [("train", range(0, 260)), ("val", range(260, 320))]:
        for i in idxs:
            if i >= len(normals):
                break
            f = normals[i]
            shutil.copy2(os.path.join(normal_src, f),
                         os.path.join(root, "images", split, f))
            open(os.path.join(root, "labels", split,
                              os.path.splitext(f)[0] + ".txt"), "w").close()

    pseudo_stat = {"train": 0, "val": 0}
    for split, d in [("train", os.path.join(BASE, "train", "hot")),
                     ("val", os.path.join(BASE, "val", "hot"))]:
        for name in list_images(d):
            src = os.path.join(d, name)
            bgr = cv2.imread(src)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            res = teacher.predict(bgr, conf=CONF_TEACHER, iou=0.5, imgsz=320,
                                  verbose=False)[0]
            lines = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    lines.append("0 %.6f %.6f %.6f %.6f"
                                 % ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                    (x2 - x1) / W, (y2 - y1) / H))
            shutil.copy2(src, os.path.join(root, "images", split, name))
            with open(os.path.join(root, "labels", split,
                                   os.path.splitext(name)[0] + ".txt"),
                      "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            pseudo_stat[split] += len(lines)
        print("[ts N=%d] teacher pseudo boxes in %s: %d"
              % (N, split, pseudo_stat[split]), flush=True)

    copy_test_from_weak(root)
    return write_yaml(root)


# ==========================================================
# 主流程
# ==========================================================

def run_ours(N, seed=0):
    fin_tag = "rev5_bud_ours%d" % N
    reuse = REUSE.get(("ours", N))
    if reuse and seed == 0 and result_exists(*reuse):
        print("[ours N=%d] reuse %s" % (N, reuse[0]), flush=True)
        return
    if result_exists(fin_tag, seed):
        print("[ours N=%d seed=%d] exists, skip" % (N, seed), flush=True)
        return

    d = calibrate(N)
    params = calib_to_params(d)
    tau = d["tau"]

    r1_tag = "rev5_bud_ours%d_r1" % N
    r1_yaml = build_pseudo_dataset(r1_tag, params, tau)
    if not result_exists(r1_tag, 0):
        train_and_eval(r1_tag, r1_yaml, seed=0, project=REV5_RUNS)
    r1_w = os.path.join(REV5_RUNS, "pseudo_%s_seed0" % r1_tag,
                        "weights", "best.pt")
    if not os.path.exists(r1_w):
        raise RuntimeError("R1 weights missing: %s" % r1_w)

    # tau_adopt: 只在这 N 张校准图上选择 (预注册网格 + 确定性 tie-break),
    # 选完冻结并用于该 N 的所有 seed —— 校准是确定性的协议步骤, seed 只影响训练。
    if "tau_adopt" in d and d.get("tau_adopt_grid") == "prereg" \
            and result_exists(fin_tag, 0):
        tau_adopt = d["tau_adopt"]
        print("[ours N=%d] reuse tau_adopt=%.2f from cache" % (N, tau_adopt),
              flush=True)
    else:
        tau_adopt, ta_rows = select_tau_adopt(N, params, tau, r1_w)
        d["tau_adopt"] = tau_adopt
        d["tau_adopt_grid"] = "prereg"
        d["tau_adopt_prereg"] = TAU_ADOPT_PREREG
        d["tau_adopt_rows"] = ta_rows
        with open(os.path.join(REV5, "budget_calib_N%d.json" % N), "w",
                  encoding="utf8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    fin_yaml = build_refined_dataset(
        fin_tag, tau_adopt, r1_w,
        base_lbl_dir=os.path.join(BASE, "yolo_" + r1_tag, "labels"))
    train_and_eval(fin_tag, fin_yaml, seed=seed, project=REV5_RUNS)


def run_ts(N, seed=0):
    fin_tag = "rev5_bud_ts%d" % N
    reuse = REUSE.get(("ts", N))
    if reuse and result_exists(*reuse):
        print("[ts N=%d] reuse %s" % (N, reuse[0]), flush=True)
        return
    if result_exists(fin_tag, seed):
        print("[ts N=%d seed=%d] exists, skip" % (N, seed), flush=True)
        return

    t_tag = "rev5_bud_ts%d_teacher" % N
    t_yaml = build_teacher_dataset(t_tag, N)     # 数据集与 seed 无关
    if not result_exists(t_tag, seed):
        train_and_eval(t_tag, t_yaml, seed=seed, project=REV5_RUNS)
    t_w = os.path.join(REV5_RUNS, "pseudo_%s_seed%d" % (t_tag, seed),
                       "weights", "best.pt")
    if not os.path.exists(t_w):
        raise RuntimeError("teacher weights missing: %s" % t_w)

    # student 数据集由 teacher 伪标签决定, 故必须按 seed 分开落盘
    s_ds = fin_tag if seed == 0 else "%s_s%d" % (fin_tag, seed)
    s_yaml = build_student_dataset(s_ds, N, t_w)
    train_and_eval(fin_tag, s_yaml, seed=seed, project=REV5_RUNS)


def _arm_value(tags, seeds=(0, 1, 2, 3, 4)):
    """收集某臂在多个 tag/seed 下的结果, 返回 {mean,std,n,runs} 或 None。"""
    runs = []
    for tg in tags:
        if not tg:
            continue
        for s in seeds:
            p = out_json(tg, s)
            if os.path.exists(p) and os.path.getsize(p) > 10:
                with open(p, encoding="utf8") as f:
                    r = json.load(f)
                runs.append({"tag": tg, "seed": s,
                             "mAP50": r["test_map50"],
                             "mAP50_95": r["test_map50_95"],
                             "P": r["test_precision"], "R": r["test_recall"]})
    if not runs:
        return None

    def _agg(key):
        vals = [x[key] for x in runs]
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            sd = var ** 0.5
        else:
            sd = 0.0
        return round(m, 4), round(sd, 4)

    m50, s50 = _agg("mAP50")
    r, sr = _agg("R")
    p_, sp = _agg("P")
    m95, s95 = _agg("mAP50_95")
    return {"tag": runs[0]["tag"], "mAP50": m50, "std_mAP50": s50,
            "mAP50_95": m95, "P": p_, "std_P": sp, "R": r, "std_R": sr,
            "n_seeds": len(runs), "seeds": [x["seed"] for x in runs],
            "runs": runs}


def collect_curve(seeds=(0, 1, 2, 3, 4)):
    """汇总曲线 -> budget_curve.json

    三臂:
      ours    = 完整流程 (校准 -> 统计伪标签 -> R1 -> Stage II 精炼 -> detector)
      ours_r1 = 仅校准引导伪标签 (不含 Stage II; 与 tau_adopt 无关)
      ts      = teacher-student (teacher 在 N 张 GT 上训 -> 伪标签 -> student)

    每个 N 的 tau_adopt 取 budget_calib_N{N}.json 中在该 N 上选出的值。
    """
    curve = {"N": [], "tau_adopt": [], "calib": [],
             "ours": [], "ours_r1": [], "ts": []}
    for N in BUDGETS:
        curve["N"].append(N)

        calib = {}
        cpath = os.path.join(REV5, "budget_calib_N%d.json" % N)
        if os.path.exists(cpath):
            with open(cpath, encoding="utf8") as f:
                calib = json.load(f)
        curve["tau_adopt"].append(calib.get("tau_adopt"))
        curve["calib"].append({"tau": calib.get("tau"),
                               "calib_F2": calib.get("calib_F2"),
                               "calib_P": calib.get("calib_P"),
                               "calib_R": calib.get("calib_R")})

        curve["ours"].append(_arm_value(
            ["rev5_bud_ours%d" % N,
             REUSE.get(("ours", N), (None,))[0]], seeds))
        curve["ours_r1"].append(_arm_value(["rev5_bud_ours%d_r1" % N], seeds))
        curve["ts"].append(_arm_value(
            ["rev5_bud_ts%d" % N, REUSE.get(("ts", N), (None,))[0]], seeds))

    out = os.path.join(REV5, "budget_curve.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump(curve, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("CALIBRATION-BUDGET CURVE (test mAP@50, mean over available seeds)")
    print("=" * 78)
    print("%-6s %-6s %-20s %-20s %-20s" % ("N", "t_ad", "Ours(full)",
                                           "Ours(R1 only)", "Teacher-Student"))

    def fmt(v):
        if not v:
            return "pending"
        if v["n_seeds"] > 1:
            return "%.4f+-%.4f(%d)" % (v["mAP50"], v["std_mAP50"],
                                       v["n_seeds"])
        return "%.4f(1)" % v["mAP50"]

    for i, N in enumerate(BUDGETS):
        ta = curve["tau_adopt"][i]
        print("%-6d %-6s %-20s %-20s %-20s"
              % (N, "%.2f" % ta if ta else "-", fmt(curve["ours"][i]),
                 fmt(curve["ours_r1"][i]), fmt(curve["ts"][i])))
    print("\nsaved ->", out)
    return curve


def main():
    only = None
    budgets = BUDGETS
    seeds = [0]
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--only" and i + 1 < len(args):
            only = args[i + 1]
        if a == "--n" and i + 1 < len(args):
            budgets = [int(x) for x in args[i + 1].split(",")]
        if a == "--seeds" and i + 1 < len(args):
            seeds = [int(x) for x in args[i + 1].split(",")]

    print("=" * 72)
    print("rev5_f_budget.py | budgets=%s | arms=%s | seeds=%s"
          % (budgets, only or "ours+ts", seeds))
    print("=" * 72, flush=True)

    if "--collect-only" in args:
        collect_curve()
        print("\n===== rev5_f_budget (collect only) done =====", flush=True)
        return

    for N in budgets:
        if only in (None, "ours"):
            print("\n########## OURS arm | N=%d ##########" % N, flush=True)
            for s in seeds:
                run_ours(N, seed=s)
        if only in (None, "ts"):
            print("\n########## TS arm | N=%d ##########" % N, flush=True)
            for s in seeds:
                run_ts(N, seed=s)

    collect_curve()
    print("\n===== rev5_f_budget finished =====", flush=True)


if __name__ == "__main__":
    main()
