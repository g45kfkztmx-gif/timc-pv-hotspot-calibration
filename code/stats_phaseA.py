# -*- coding: utf-8 -*-
"""Phase A: 数值复算 + 不确定度/统计检验（仅用复现包内数据）。"""
import os, sys, json, math, random
from collections import defaultdict

PKG = r"C:\Users\YJP\AppData\Local\Qianwen\User Data\qwen-agent-workspaces\runs\3612b721-61de-458e-89c4-b16c3f3594e1\2026-11-17T09-00-25-475Z\build\repro\TIMC_repro_package"
CODE = os.path.join(PKG, "06_环境与代码", "code")
sys.path.insert(0, CODE)
import importlib.util
spec = importlib.util.spec_from_file_location("tim_eval", os.path.join(CODE, "exp__tim_eval.py"))
TE = importlib.util.module_from_spec(spec); spec.loader.exec_module(TE)

import numpy as np
from scipy import stats as st

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
R = {}

def jload(p):
    with open(p, encoding="utf8") as f: return json.load(f)

# ---------- 1. 评测器对拍 ----------
gt = TE.load_gt(os.path.join(PKG, "05_训练数据", "test", "hot_labels"))
preds = TE.read_pred_txt(os.path.join(PKG, "03_预测文件", "ours"))
n_img = 350
own = TE.prf_at_fp_budget(preds, gt, n_img)
own["mAP50"] = round(TE.ap_at(preds, gt, 0.5), 4)
own["mAP50_95"] = round(TE.map5095(preds, gt), 4)
ref = jload(os.path.join(PKG, "01_数值溯源", "exp__tim_baselines_summary.json"))["Ours"]
R["evaluator_check"] = {"recomputed": own, "package_Ours": ref}

# 筛查口径（conf>=0.25, IoU0.45）
p25 = [p for p in preds if p[2] >= 0.25]
tp45, fp45 = TE._match(p25, gt, 0.45)
n_gt = sum(len(v) for v in gt.values())
R["screening_conf025"] = {"hit": int(sum(tp45)), "n_gt": n_gt,
                          "hit_rate": round(sum(tp45)/n_gt, 4),
                          "fp": int(sum(fp45)), "fp_per_img": round(sum(fp45)/n_img, 4)}

# ---------- 2. 逐图 bootstrap CI（mAP50 / hit@0.45 / fp_per_img） ----------
def per_image_stats(pred_list, gt):
    by_img = defaultdict(list)
    for stem, box, c in pred_list: by_img[stem].append((box, c))
    stems = sorted(gt.keys())
    rows = []
    for s in stems:
        pl = sorted(by_img.get(s, []), key=lambda t: -t[1])
        pl = [(s, b, c) for b, c in pl]
        g = {s: gt[s]}
        tp, fp = TE._match(pl, g, 0.5)
        pl25 = [p for p in pl if p[2] >= 0.25]
        tp45, fp45 = TE._match(pl25, g, 0.45)
        rows.append({"stem": s, "tp": tp, "fp": fp,
                     "n_gt": len(gt[s]), "hit45": sum(tp45), "fp45": sum(fp45),
                     "n_pred25": len(pl25)})
    return rows

rows = per_image_stats(preds, gt)
rng = np.random.default_rng(42)
B = 2000
n = len(rows)
boot_map, boot_hit, boot_fp = [], [], []
# per-image pred lists for correct AP under resampling
by_img = defaultdict(list)
for stem, box, c in preds: by_img[stem].append((stem, box, c))
img_preds = [sorted(by_img.get(s, []), key=lambda t: -t[2]) for s in sorted(gt.keys())]
gts_list = [gt[s] for s in sorted(gt.keys())]
def ap_from_concat(sel):
    pl = []
    for i in sel: pl += img_preds[i]
    pl.sort(key=lambda t: -t[2])
    g = {}
    for i in sel:
        s = sorted(gt.keys())[i]
        g[s] = g.get(s, []) + gt[s]
    return TE.ap_at(pl, g, 0.5)
keys = sorted(gt.keys())
all_stems = sorted(set(list(gt.keys()) + [p[0] for p in preds]))
rows_all = per_image_stats(preds, {s: gt.get(s, []) for s in all_stems})
n_all = len(rows_all)
for b in range(B):
    idx = rng.integers(0, n, n)
    boot_map.append(ap_from_concat(idx))
    ng = sum(rows[i]["n_gt"] for i in idx)
    boot_hit.append(sum(rows[i]["hit45"] for i in idx)/max(1, ng))
    idx_all = rng.integers(0, n_all, n_all)
    boot_fp.append(sum(rows_all[i]["fp45"] for i in idx_all)/n_all)
def ci(a):
    a = np.array(a); return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
R["bootstrap_test_CI"] = {"B": B, "mAP50_point": own["mAP50"], "mAP50_CI95": ci(boot_map),
                          "hit45_point": R["screening_conf025"]["hit_rate"], "hit45_CI95": ci(boot_hit),
                          "fp_per_img_point": own["fp_per_img"], "fp_CI95": ci(boot_fp)}

# ---------- 3. 衰减配对 McNemar（s=1.0 vs 0.4，GT 级 hit@0.45） ----------
def gt_hits(pred_dir):
    pl = TE.read_pred_txt(pred_dir)
    pl = [p for p in pl if p[2] >= 0.25]
    hits = {}
    for s in gt:
        g = {s: gt[s]}
        tp, _ = TE._match([p for p in pl if p[0] == s], g, 0.45)
        hits[s] = tp
    return hits
h100 = gt_hits(os.path.join(PKG, "03_预测文件", "atten_s100"))
h40 = gt_hits(os.path.join(PKG, "03_预测文件", "atten_s40"))
b = c = 0
for s in gt:
    for j in range(len(gt[s])):
        a1 = h100[s][j] if j < len(h100[s]) else 0
        a4 = h40[s][j] if j < len(h40[s]) else 0
        if a1 and not a4: b += 1
        if a4 and not a1: c += 1
if b + c > 0:
    p_mcn = st.binomtest(b, b+c, 0.5).pvalue
else:
    p_mcn = 1.0
R["mcnemar_atten_s100_vs_s40"] = {"hit100": int(sum(sum(v) for v in h100.values())),
                                 "hit40": int(sum(sum(v) for v in h40.values())),
                                 "lost_b": b, "gained_c": c, "p_exact_two_sided": round(float(p_mcn), 5)}

# ---------- 4. 五种子 t 区间 + A3 vs A5 配对 t ----------
ph3 = jload(os.path.join(PKG, "01_数值溯源", "exp__rev6_phase3_summary.json"))
r1 = [r["map50"] for r in ph3["arms"]["rev6_g42_r1"]["runs"]]
r2 = [r["map50"] for r in ph3["arms"]["rev6_g42_r2"]["runs"]]
m, sd = np.mean(r1), np.std(r1, ddof=1)
tcrit = st.t.ppf(0.975, 4)
R["seed_stats_A3"] = {"values": r1, "mean": round(float(m), 4), "sd": round(float(sd), 4),
                      "t_CI95": [round(float(m - tcrit*sd/math.sqrt(5)), 4), round(float(m + tcrit*sd/math.sqrt(5)), 4)]}
d = np.array(r1) - np.array(r2)
t_p = st.ttest_rel(r1, r2)
R["paired_t_A3_vs_A5"] = {"diff_mean": round(float(d.mean()), 4), "diff_sd": round(float(d.std(ddof=1)), 4),
                          "t": round(float(t_p.statistic), 3), "df": 4, "p": round(float(t_p.pvalue), 4)}

# ---------- 5. LODO 配对检验 ----------
lopo = jload(os.path.join(PKG, "01_数值溯源", "exp__lopo_summary.json"))
lopo_ref = jload(os.path.join(PKG, "01_数值溯源", "exp__lopo_reference_summary.json"))
ours_v, ctrl_v, ref_v = [], [], []
for f in lopo["folds"]:
    ours_v.append(f["r1"]["test_map50"]); ctrl_v.append(f["ctrl"]["test_map50"])
    ref_v.append(lopo_ref[f["day"]]["test_map50"])
w1 = st.wilcoxon(ours_v, ctrl_v)
R["wilcoxon_LODO_ours_vs_ctrl"] = {"ours": ours_v, "ctrl": ctrl_v,
                                   "stat": round(float(w1.statistic), 2), "p": round(float(w1.pvalue), 5), "n": len(ours_v)}
if len(ref_v) == len(ours_v):
    w2 = st.wilcoxon(ours_v, ref_v)
    R["wilcoxon_LODO_ours_vs_inpool_ref"] = {"stat": round(float(w2.statistic), 2), "p": round(float(w2.pvalue), 5)}

# ---------- 6. 飞行不重叠子集复算 + CI ----------
fd = jload(os.path.join(PKG, "01_数值溯源", "exp__test_flight_disjoint.json"))
fd_stems = {os.path.splitext(x)[0] for x in fd["disjoint_files"]}
gt_fd = {s: gt.get(s, []) for s in fd_stems}
pred_fd = [p for p in preds if p[0] in fd_stems]
ap_fd = TE.ap_at(pred_fd, gt_fd, 0.5)
pr_fd = TE.prf_at_fp_budget(pred_fd, gt_fd, len(fd_stems))
R["flight_disjoint_recomputed"] = {"n_img": len(fd_stems), "n_gt": sum(len(v) for v in gt_fd.values()),
                                   "mAP50": round(ap_fd, 4), "budget_metrics": pr_fd}

# ---------- 7. 长宽比边界口径 ----------
recs = jload(os.path.join(PKG, "03_预测文件", "per_instance", "rev6h3_boxes_rev6_g42_r1.json"))["records"]
gt_recs = [r for r in recs if r["kind"].startswith("gt")]
kinds = sorted({r["kind"] for r in recs})
asp = [r["aspect"] for r in gt_recs]
R["aspect_boundary"] = {"kinds": kinds, "n_gt_recs": len(gt_recs),
                        "frac_gt_1p6": round(sum(a > 1.6 for a in asp)/len(asp), 4),
                        "frac_ge_1p6": round(sum(a >= 1.6 for a in asp)/len(asp), 4),
                        "n_gt_1p6": sum(a > 1.6 for a in asp), "n_ge_1p6": sum(a >= 1.6 for a in asp)}

# ---------- 8. 标定元组与扰动漂移描述 ----------
pert = jload(os.path.join(PKG, "01_数值溯源", "exp__tim_perturb_summary.json"))
R["perturb_summary"] = pert["drift"]

with open(os.path.join(OUT, "phaseA_stats.json"), "w", encoding="utf8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
print(json.dumps(R, indent=2, ensure_ascii=False))
