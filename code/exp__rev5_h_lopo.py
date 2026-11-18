# -*- coding: utf-8 -*-
"""
rev5_h_lopo.py —— leave-one-acquisition-day-out (LODO) 实验
======================================================================
动机 (IPT 审稿报告 P0 / 论文 V-C 局限性段承诺的"必要下一步实验"):
  论文主张"标定协议在同 fleet 重复巡检中可迁移", 但缺少**语料内**的
  跨场次方差估计。外部压力测试 (PV-HSD-2025 -> mAP 0.10) 测的是跨语料,
  5-seed (SD=0.016) 测的是训练种子方差, 都不能回答:
      "标定在 A 场次采集的图上完成, 冻结后用到 B 场次, 会掉多少?"

设计:
  源发布 (PVF-10) **不提供电站标识**, 只有采集时间戳。因此以"采集日"
  作为最细的场次级代理分组 (全语料 18 个采集日, 每个采集日对应一段
  连续航拍视频)。取测试图最多的 8 个采集日做 8 折:
      20230222 / 20230223 / 20230227 / 20230228 / 20230302 /
      20230308 / 20231114 / 20231121
  8 折合计覆盖 331/350 测试图、275/279 实例。

  每折 (held-out day = d):
    fold 臂  : 标定图 = 非 d 日的人工标注图; 训练池 = 非 d 日图像
               (统计伪标签 + Stage II 精炼, 与主实验协议逐帧一致);
               评估 = d 日测试子集 (人工 GT)。
    ctrl 臂  : 同标定、同标签、同 val、同 test; 唯一差别是训练图像
               改为**等量的日混合随机抽样** (含 d 日图像) ——
               匹配数据量对照, 用于把"场次迁移损失"与"数据量差异"分离。

  每折 3 次训练 (R1 / final / ctrl), 协议与 rev5_common 完全一致:
    epochs=60, imgsz=320, batch=32, workers=2, amp=False, seed=0,
    COCO 预训练权重, test 评估一次。

用法:
  python rev5_h_lopo.py                 # 全部 8 折 (可反复启动续跑)
  python rev5_h_lopo.py --only 20231114 # 只跑指定日
  python rev5_h_lopo.py --skip-ctrl     # 不跑对照臂
输出:
  rev5_experiments/lopo_calib_<day>.json    每折校准结果
  rev5_experiments/lopo_<day>_r1_seed0.json R1 结果
  rev5_experiments/lopo_<day>_seed0.json    fold 臂最终结果
  rev5_experiments/lopo_<day>_ctrl_seed0.json 对照臂结果
  rev5_experiments/lopo_summary.json        汇总
"""
import os
import sys
import json
import time
import random
import shutil

import cv2

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

import rev5_f_budget as F                                    # noqa: E402
from auto_pseudo import (list_images as ap_list_images,     # noqa: E402
                         load_labels, IoU,
                         BASELINE_HOT_DIR, BASELINE_LABEL_DIR)
from auto_pseudo_diag import DiagParams, detect_diag        # noqa: E402

REV5 = F.REV5
REV5_RUNS = F.REV5_RUNS
STAGE = os.path.join(BASE, "lopo_stage")

# 8 折: 测试图最多的 8 个采集日 (331/350 图, 275/279 实例)
DAYS = ["20230222", "20230223", "20230227", "20230228",
        "20230302", "20230308", "20231114", "20231121"]

TEST_IMG_SRC = os.path.join(BASE, "yolo_weak", "images", "test")
TEST_LBL_SRC = os.path.join(BASE, "yolo_weak", "labels", "test")

CONF_PREDICT = F.CONF_PREDICT
IOU_MATCH = F.IOU_MATCH


def log(*a):
    print(*a, flush=True)


def day_of(name):
    return name[4:12]


def pool_images(split):
    """YOLO split ('train'/'val') -> [(name, sub)] 覆盖 hot+normal。"""
    out = []
    for sub in ("hot", "normal"):
        d = os.path.join(BASE, split, sub)
        for f in F.list_images(d):
            out.append((f, sub))
    return out


def src_path(name, sub, split):
    return os.path.join(BASE, split, sub, name)


# ==========================================================
# 校准 (日过滤版, 复用 rev5_f_budget 的 stage1/stage2/select_tau)
# ==========================================================

def calib_files(excl_day):
    """非 excl_day 的、带标签文件的人工标注图。"""
    files = []
    for name in F.list_images(BASELINE_HOT_DIR):
        if day_of(name) == excl_day:
            continue
        stem = os.path.splitext(name)[0]
        if os.path.exists(os.path.join(BASELINE_LABEL_DIR, stem + ".txt")):
            files.append(name)
    return sorted(files)


def load_cache(files):
    cache = {}
    for name in files:
        p = os.path.join(BASELINE_HOT_DIR, name)
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if gray.ndim == 3:                       # ultralytics 会 patch cv2.imread
            gray = gray[:, :, 0]
        h, w = gray.shape
        gt = load_labels(os.path.join(BASELINE_LABEL_DIR,
                                      os.path.splitext(name)[0] + ".txt"), w, h)
        cache[name] = {"gt": gt, "gray": gray, "H": h, "W": w}
    return cache


def calibrate_files(files, tag):
    out = os.path.join(REV5, "lopo_calib_%s.json" % tag)
    if os.path.exists(out) and os.path.getsize(out) > 10:
        with open(out, encoding="utf8") as f:
            d = json.load(f)
        log("[calib %s] cached: k=%.1f st=%.2f bmf=%.2f kl=%s tau=%.2f F2=%.4f"
            % (tag, d["k"], d["st"], d["bmf"], d["kl"], d["tau"], d["calib_F2"]))
        return d

    log("\n" + "-" * 66)
    log("[calib %s] 在 %d 张非本折人工标注图上重跑完整校准" % (tag, len(files)))
    log("-" * 66)
    proto = DiagParams()
    cache = load_cache(files)
    log("  校准图: %d 张 (含标签)" % len(cache))
    t0 = time.time()
    bases = F.stage1(cache, proto)
    cfg = F.stage2(cache, proto, bases)
    tau_best, tau_rows = F.select_tau(cache, proto, cfg)
    d = {
        "tag": tag, "n_calib": len(cache), "grid": "full",
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
    log("[calib %s] DONE k=%.1f st=%.2f bmf=%.2f kl=%s win=%s tau=%.2f "
        "F2=%.4f (%.0fs)" % (tag, d["k"], d["st"], d["bmf"], d["kl"],
                             d["win"], d["tau"], d["calib_F2"], d["elapsed_sec"]))
    return d


# ==========================================================
# 标签生成 (全池, 日过滤只在组装数据集时做)
# ==========================================================

def collect_entries(img_path, params, tau):
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
        kept.append(((x + w / 2) / W, (y + h / 2) / H, w / W, h / H, c["score"]))
    return [e for e in kept if e[4] >= tau]


def write_label(path, entries):
    lines = ["0 %.6f %.6f %.6f %.6f" % e[:4] for e in entries]
    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def gen_r1_labels(params, tau, tag):
    """冻结生成器对**全部** train/val hot 图生成统计伪标签 -> STAGE/r1_<tag>。

    注意: 每折的 (params, tau) 由该折自己的标定决定, 因此暂存目录必须
    按折隔离, 不能跨折复用。
    """
    root = os.path.join(STAGE, "r1_%s" % tag)
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, split), exist_ok=True)
    n_box = 0
    for split in ("train", "val"):
        names = F.list_images(os.path.join(BASE, split, "hot"))
        for i, name in enumerate(names):
            stem = os.path.splitext(name)[0]
            dst = os.path.join(root, split, stem + ".txt")
            if os.path.exists(dst):
                continue
            entries = collect_entries(
                os.path.join(BASE, split, "hot", name), params, tau) or []
            write_label(dst, entries)
            n_box += len(entries)
            if (i + 1) % 300 == 0:
                log("  [r1 labels] %s %d/%d" % (split, i + 1, len(names)))
    # normal: 空标注
    for split in ("train", "val"):
        for name in F.list_images(os.path.join(BASE, split, "normal")):
            stem = os.path.splitext(name)[0]
            p = os.path.join(root, split, stem + ".txt")
            if not os.path.exists(p):
                open(p, "w").close()
    log("[r1 labels] DONE total pseudo boxes=%d -> %s" % (n_box, root))
    return root


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
            boxes.append(((xc - nw / 2) * W, (yc - nw / 2) * H, nw * W, nh * H))
    return boxes


def gen_ref_labels(params, tau, conf_adopt, r1_weights, tag, r1_label_root):
    """Stage II 精炼标签 (伪标签 + R1 采纳框), 全池 -> STAGE/ref_<tag>。

    同时记录每图 difficulty (missed + adopted) 供组装时做难样本复制。
    r1_label_root: 本折自己的 R1 统计伪标签目录 (STAGE/r1_<tag>)。
    """
    from ultralytics import YOLO
    root = os.path.join(STAGE, "ref_%s" % tag)
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, split), exist_ok=True)
    diff_path = os.path.join(root, "difficulty.json")
    if os.path.exists(diff_path) and os.path.getsize(diff_path) > 10:
        with open(diff_path, encoding="utf8") as f:
            diff = json.load(f)
        log("[ref labels %s] cached (%d imgs)" % (tag, len(diff)))
        return root, diff

    model = YOLO(r1_weights)
    diff = {}
    stat = {"images": 0, "adopted": 0, "missed": 0}
    for split in ("train", "val"):
        names = F.list_images(os.path.join(BASE, split, "hot"))
        for i, name in enumerate(names):
            stem = os.path.splitext(name)[0]
            src = os.path.join(BASE, split, "hot", name)
            bgr = cv2.imread(src)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            stat["images"] += 1
            pseudo = read_yolo_txt(os.path.join(r1_label_root, split,
                                                stem + ".txt"), W, H)
            res = model.predict(src, conf=CONF_PREDICT, iou=0.5, imgsz=320,
                                verbose=False)[0]
            preds = []
            if res.boxes is not None and len(res.boxes):
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    preds.append(((x1, y1, x2 - x1, y2 - y1), float(b.conf[0])))
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
            adopted = []
            for j, (db, conf) in enumerate(preds):
                if pred_used[j]:
                    continue
                if conf >= conf_adopt and all(IoU(db, pb) < 0.5
                                              for pb in pseudo + adopted):
                    adopted.append(db)
            stat["missed"] += missed
            stat["adopted"] += len(adopted)
            lines = ["0 %.6f %.6f %.6f %.6f"
                     % ((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
                     for (x, y, w, h) in pseudo + adopted]
            with open(os.path.join(root, split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")
            diff[name] = missed + len(adopted)
            if (i + 1) % 300 == 0:
                log("  [ref labels] %s %d/%d" % (split, i + 1, len(names)))
    # normal: 空标注
    for split in ("train", "val"):
        for name in F.list_images(os.path.join(BASE, split, "normal")):
            stem = os.path.splitext(name)[0]
            p = os.path.join(root, split, stem + ".txt")
            if not os.path.exists(p):
                open(p, "w").close()
            diff[name] = 0
    with open(diff_path, "w", encoding="utf8") as f:
        json.dump(diff, f, indent=2, ensure_ascii=False)
    log("[ref labels %s] DONE %s -> %s" % (tag, json.dumps(stat), root))
    return root, diff


# ==========================================================
# 数据集组装
# ==========================================================

def build_test_subset(root, day):
    """held-out day 的测试子集 (人工 GT, 来自 yolo_weak)。"""
    img_dir = os.path.join(root, "images", "test")
    lbl_dir = os.path.join(root, "labels", "test")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n = 0
    for f in F.list_images(TEST_IMG_SRC):
        if day_of(f) != day:
            continue
        stem = os.path.splitext(f)[0]
        shutil.copy2(os.path.join(TEST_IMG_SRC, f), os.path.join(img_dir, f))
        src = os.path.join(TEST_LBL_SRC, stem + ".txt")
        dst = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            open(dst, "w").close()
        n += 1
    return n


def assemble(tag, train_sel, val_sel, day, label_root, diff):
    """train_sel/val_sel: [(name, sub)]; 按 difficulty 做难样本复制。"""
    root = os.path.join(BASE, "yolo_" + tag)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
    stats = {}
    for split, sel in (("train", train_sel), ("val", val_sel)):
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        n_img = n_box = n_dup = 0
        for name, sub in sel:
            stem = os.path.splitext(name)[0]
            shutil.copy2(src_path(name, sub, split),
                         os.path.join(img_dir, name))
            lines = open(os.path.join(label_root, split, stem + ".txt")).read()
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write(lines)
            n_img += 1
            n_box += sum(1 for l in lines.splitlines() if l.strip())
            if split == "train":
                d = diff.get(name, 0)
                copies = 1 if d == 0 else (2 if d == 1 else 3)
                for c in range(1, copies):
                    dup = "dup%d_%s" % (c, name)
                    shutil.copy2(src_path(name, sub, split),
                                 os.path.join(img_dir, dup))
                    with open(os.path.join(lbl_dir,
                                           os.path.splitext(dup)[0] + ".txt"),
                              "w") as f:
                        f.write(lines)
                    n_dup += 1
        stats[split] = {"images": n_img, "boxes": n_box, "hard_dups": n_dup}
    n_test = build_test_subset(root, day)
    stats["test"] = {"images": n_test}
    yaml_path = F.write_yaml(root)
    log("[assemble %s] %s test=%d" % (tag, json.dumps(stats), n_test))
    return yaml_path, stats


# ==========================================================
# tau_adopt 选择 (日过滤版)
# ==========================================================

def select_tau_adopt_files(cache, params, tau, r1_weights,
                           grid=F.TAU_ADOPT_PREREG):
    from ultralytics import YOLO
    model = YOLO(r1_weights)
    rows = []
    for name, e in cache.items():
        src = os.path.join(BASELINE_HOT_DIR, name)
        bgr = cv2.imread(src)
        if bgr is None:
            continue
        cands, _ = detect_diag(e["gray"], params)
        pseudo = [c["box"] for c in cands
                  if c["status"] == "accepted" and c["score"] >= tau]
        res = model.predict(src, conf=CONF_PREDICT, iou=0.5, imgsz=320,
                            verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                preds.append(((x1, y1, x2 - x1, y2 - y1), float(b.conf[0])))
        e["_pseudo"] = pseudo
        e["_preds"] = preds

    for t in F.TAU_ADOPT_GRID:
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
            used = F.greedy_match(acc, e["gt"])
            tp = sum(used)
            TP += tp
            FN += len(e["gt"]) - tp
            FP += len(acc) - tp
        p = TP / max(TP + FP, 1)
        r = TP / max(TP + FN, 1)
        rows.append({"tau_adopt": t, "adopted": n_adopted,
                     "P": round(p, 4), "R": round(r, 4),
                     "F2": round(F.fbeta(p, r), 4)})
        log("    tau_adopt=%.2f P=%.4f R=%.4f F2=%.4f"
            % (t, p, r, F.fbeta(p, r)))
    best = F.pick_tau_adopt(rows, grid)
    log("  [tau_adopt %s] selected=%.2f (F2=%.4f, grid %s)"
        % (cache and "day", best["tau_adopt"], best["F2"], grid))
    return best["tau_adopt"], rows


# ==========================================================
# 单折流程
# ==========================================================

def run_fold(day, seed=0, do_ctrl=True):
    tag = "lopo_%s" % day
    fin_json = os.path.join(REV5, "%s_seed%d.json" % (tag, seed))
    if os.path.exists(fin_json) and os.path.getsize(fin_json) > 10:
        log("[%s] final exists, skip fold" % tag)
        return json.load(open(fin_json, encoding="utf8"))

    t_fold = time.time()
    summary = {"day": day, "seed": seed}

    # ---- 1. 校准 (非本折标注图) ----
    files = calib_files(day)
    d = calibrate_files(files, tag)
    params = F.calib_to_params(d)
    tau = d["tau"]
    summary["calib"] = {k: d[k] for k in
                        ("n_calib", "k", "st", "bmf", "kl", "win", "lmd",
                         "use_local_bg", "tau", "calib_F2", "calib_P",
                         "calib_R")}

    # ---- 2. R1: 全池统计伪标签 + 日过滤组装 + 训练 ----
    label_r1 = gen_r1_labels(params, tau, tag)
    train_all = pool_images("train")
    val_all = pool_images("val")
    train_sel = [(n, s) for (n, s) in train_all if day_of(n) != day]
    val_sel = [(n, s) for (n, s) in val_all if day_of(n) != day]
    summary["fold_train_images"] = len(train_sel)
    summary["fold_val_images"] = len(val_sel)

    r1_tag = "%s_r1" % tag
    r1_json = os.path.join(REV5, "%s_seed%d.json" % (r1_tag, seed))
    r1_w = os.path.join(REV5_RUNS, "pseudo_%s_seed%d" % (r1_tag, seed),
                        "weights", "best.pt")
    if not (os.path.exists(r1_json) and os.path.getsize(r1_json) > 10
            and os.path.exists(r1_w)):
        r1_yaml, r1_stats = assemble(r1_tag, train_sel, val_sel, day,
                                     label_r1, {})
        res_r1 = F.train_and_eval(r1_tag, r1_yaml, seed=seed,
                                  project=REV5_RUNS)
    else:
        log("[%s] R1 cached" % r1_tag)
        res_r1 = json.load(open(r1_json, encoding="utf8"))
    summary["r1"] = {k: res_r1[k] for k in
                     ("test_map50", "test_map50_95", "test_precision",
                      "test_recall", "train_seconds")}

    # ---- 3. tau_adopt (非本折校准图) ----
    cache = load_cache(files)
    tau_adopt, ta_rows = select_tau_adopt_files(cache, params, tau, r1_w)
    summary["tau_adopt"] = tau_adopt
    summary["tau_adopt_rows"] = ta_rows

    # ---- 4. Stage II 精炼标签 (全池) + 日过滤组装 + 训练 ----
    label_ref, diff = gen_ref_labels(params, tau, tau_adopt, r1_w, tag,
                                     label_r1)
    if not (os.path.exists(fin_json) and os.path.getsize(fin_json) > 10):
        fin_yaml, fin_stats = assemble(tag, train_sel, val_sel, day,
                                       label_ref, diff)
        res = F.train_and_eval(tag, fin_yaml, seed=seed, project=REV5_RUNS)
    else:
        res = json.load(open(fin_json, encoding="utf8"))
    summary["final"] = {k: res[k] for k in
                        ("test_map50", "test_map50_95", "test_precision",
                         "test_recall", "train_seconds")}

    # ---- 5. 对照臂: 等量日混合随机抽样 (同标签/同 val/同 test) ----
    if do_ctrl:
        ctrl_tag = "%s_ctrl" % tag
        ctrl_json = os.path.join(REV5, "%s_seed%d.json" % (ctrl_tag, seed))
        if not (os.path.exists(ctrl_json) and os.path.getsize(ctrl_json) > 10):
            rng = random.Random(int(day))
            ctrl_sel = rng.sample(train_all, len(train_sel))
            ctrl_yaml, _ = assemble(ctrl_tag, ctrl_sel, val_sel, day,
                                    label_ref, diff)
            res_c = F.train_and_eval(ctrl_tag, ctrl_yaml, seed=seed,
                                     project=REV5_RUNS)
        else:
            log("[%s] ctrl cached" % ctrl_tag)
            res_c = json.load(open(ctrl_json, encoding="utf8"))
        summary["ctrl"] = {k: res_c[k] for k in
                           ("test_map50", "test_map50_95",
                            "test_precision", "test_recall",
                            "train_seconds")}
        summary["ctrl_train_images"] = len(train_sel)

    summary["fold_seconds"] = round(time.time() - t_fold, 1)
    with open(os.path.join(REV5, "%s_summary.json" % tag), "w",
              encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log("[%s] FOLD DONE final mAP50=%.4f ctrl mAP50=%s (%.0fs)"
        % (tag, summary["final"]["test_map50"],
           ("%.4f" % summary["ctrl"]["test_map50"]) if "ctrl" in summary
           else "-", summary["fold_seconds"]))
    return summary


def main():
    args = sys.argv[1:]
    only = None
    do_ctrl = True
    for i, a in enumerate(args):
        if a == "--only" and i + 1 < len(args):
            only = args[i + 1]
        if a == "--skip-ctrl":
            do_ctrl = False
    days = [only] if only else DAYS
    log("=" * 72)
    log("rev5_h_lopo.py | folds=%s | ctrl=%s" % (days, do_ctrl))
    log("=" * 72)
    summaries = []
    for day in days:
        s = run_fold(day, seed=0, do_ctrl=do_ctrl)
        summaries.append(s)
        with open(os.path.join(REV5, "lopo_summary.json"), "w",
                  encoding="utf8") as f:
            json.dump({"days": days, "folds": summaries}, f, indent=2,
                      ensure_ascii=False)
    log("\n===== rev5_h_lopo finished =====")


if __name__ == "__main__":
    main()
