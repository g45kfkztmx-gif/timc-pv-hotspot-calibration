# -*- coding: utf-8 -*-
"""Phase B: 标定重跑验证 + 校准集 bootstrap + 门控平台扫描 + 扰动副本重标定。"""
import os, sys, json, time, random, itertools
from collections import Counter

PKG = r"C:\Users\YJP\AppData\Local\Qianwen\User Data\qwen-agent-workspaces\runs\3612b721-61de-458e-89c4-b16c3f3594e1\2026-11-17T09-00-25-475Z\build\repro\TIMC_repro_package"
CODE = os.path.join(PKG, "06_环境与代码", "code")
WORK = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORK, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CODE)

import importlib.abc, importlib.util
class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("rev5_", "tim_")):
            p = os.path.join(CODE, "exp__" + fullname + ".py")
            if os.path.exists(p):
                return importlib.util.spec_from_file_location(fullname, p)
        return None
sys.meta_path.insert(0, _AliasFinder())

import cv2
import numpy as np

import exp__rev5_f_budget as FB
import auto_pseudo as AP
from auto_pseudo_diag import DiagParams

CAL = os.path.join(PKG, "04_校准集_图像与人工框")
FB.REV5 = os.path.join(WORK, "rev5_local")
os.makedirs(FB.REV5, exist_ok=True)
FB.BASELINE_HOT_DIR = os.path.join(CAL, "images")
FB.BASELINE_LABEL_DIR = os.path.join(CAL, "labels_A")

# cv2.imread 在 Windows 下不支持非 ASCII 路径 -> 字节解码
_orig_imread = cv2.imread
def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)
cv2.imread = _imread_unicode

# calib_names 在 import 时绑定了旧路径, 覆盖为包内路径
def _calib_names(N):
    files = []
    for name in AP.list_images(FB.BASELINE_HOT_DIR):
        stem = os.path.splitext(name)[0]
        if os.path.exists(os.path.join(FB.BASELINE_LABEL_DIR, stem + ".txt")):
            files.append(name)
    rng = random.Random(FB.SHUFFLE_SEED)
    rng.shuffle(files)
    return files[:N]
FB.calib_names = _calib_names

def jload(p):
    with open(p, encoding="utf8") as f: return json.load(f)

R = {}
t_start = time.time()

# ---------- 1. 全量重标定验证 ----------
_stale = os.path.join(FB.REV5, "budget_calib_N197.json")
if os.path.exists(_stale):
    os.remove(_stale)   # 本脚本自产缓存, 失败运行残留, 强制重算
d197 = FB.calibrate(197)
ref = jload(os.path.join(PKG, "07_权重值", "reference_calib.json"))
R["full_recalib"] = {"recomputed": {k: d197[k] for k in ("k", "st", "bmf", "kl", "win", "lmd", "use_local_bg", "tau", "calib_F2")},
                     "reference": ref,
                     "match": all(abs(d197[k]-ref[k]) < 1e-9 for k in ("k", "st", "bmf", "kl", "win", "lmd", "tau")) and abs(d197["calib_F2"]-ref["calib_F2"]) < 5e-4}
print("FULL RECALIB:", json.dumps(R["full_recalib"], ensure_ascii=False), flush=True)

# ---------- 2. 候选缓存与 F2 矩阵 ----------
names = FB.calib_names(197)
cache = FB.load_calib_cache(197)
proto = DiagParams()
over = dict(k_sigma=d197["k"], use_local_rescue=True, k_local=d197["kl"],
            local_win=d197["win"], local_min_delta=d197["lmd"], use_local_bg=d197["use_local_bg"])
t0 = time.time()
per_img = FB.detect_cache_for(cache, **over)
print("cache built %.1fs, images=%d" % (time.time()-t0, len(per_img)), flush=True)

ST = FB.ST_GRID; TAU = FB.TAU_GRID
bmf = d197["bmf"]
# F2 matrix [st, tau] on full set
def f2_of(sel_names, st, tau):
    TP = FP = FN = 0
    for nm in sel_names:
        e = per_img[nm]
        acc = FB.score_candidates(e["cands"], e["H"], e["W"], proto, st, bmf, tau)
        used = FB.greedy_match(acc, e["gt"])
        tp = sum(used)
        TP += tp; FN += len(e["gt"]) - tp; FP += len(acc) - tp
    p = TP / max(TP+FP, 1); r = TP / max(TP+FN, 1)
    return FB.fbeta(p, r), p, r, TP, FP, FN

full_mat = {}
for st in ST:
    for tau in TAU:
        f2, p, r, TP, FP, FN = f2_of(names, st, tau)
        full_mat["%.2f_%.2f" % (st, tau)] = {"F2": round(f2, 4), "P": round(p, 4), "R": round(r, 4), "TP": TP, "FP": FP, "FN": FN}
R["gate_plateau_full"] = full_mat
sel = full_mat["0.42_0.30"]
R["plateau_check"] = {"g042_tau030": sel,
                      "range_st036_044_note": "ST_GRID={0.38,0.40,0.42,0.45}; plateau claim g in [0.36,0.44] maps to grid 0.38-0.42(0.45 outside)",
                      "F2_at_st_grid_tau030": {str(st): full_mat["%.2f_0.30" % st]["F2"] for st in ST}}

# ---------- 3. bootstrap 校准集 ----------
B = 200
rng = random.Random(42)
boot_rows = []
t0 = time.time()
for b in range(B):
    sample = [names[rng.randrange(len(names))] for _ in range(len(names))]
    best = None
    for st in ST:
        for tau in TAU:
            f2, p, r, TP, FP, FN = f2_of(sample, st, tau)
            if best is None or f2 > best["F2"]:
                best = {"st": st, "tau": tau, "F2": f2}
    # plateau-centre rule among st with F2 within 0.01 of max at chosen tau
    tau_b = best["tau"]
    f2max = best["F2"]
    plateau = [st for st in ST if full_mat and f2_of(sample, st, tau_b)[0] >= f2max - 0.01]
    centre = plateau[len(plateau)//2] if plateau else best["st"]
    boot_rows.append({"st_argmax": best["st"], "st_plateau_centre": centre, "tau": tau_b, "F2": round(best["F2"], 4)})
    if (b+1) % 50 == 0:
        print("boot %d/%d (%.0fs)" % (b+1, B, time.time()-t0), flush=True)
cnt_arg = Counter(r["st_argmax"] for r in boot_rows)
cnt_cen = Counter(r["st_plateau_centre"] for r in boot_rows)
f2s = [r["F2"] for r in boot_rows]
R["calib_bootstrap"] = {"B": B,
                        "st_argmax_dist": {str(k): v/B for k, v in sorted(cnt_arg.items())},
                        "st_plateau_centre_dist": {str(k): v/B for k, v in sorted(cnt_cen.items())},
                        "tau_dist": {str(k): v/B for k, v in sorted(Counter(r["tau"] for r in boot_rows).items())},
                        "F2_mean": round(float(np.mean(f2s)), 4), "F2_sd": round(float(np.std(f2s, ddof=1)), 4),
                        "F2_CI95": [round(float(np.percentile(f2s, 2.5)), 4), round(float(np.percentile(f2s, 97.5)), 4)]}
with open(os.path.join(OUT, "phaseB_boot_rows.json"), "w", encoding="utf8") as f:
    json.dump(boot_rows, f)
print("BOOTSTRAP:", json.dumps(R["calib_bootstrap"], ensure_ascii=False), flush=True)

# ---------- 4. 扰动副本重标定 ----------
pert_rows = []
for i in range(1, 7):
    FB.BASELINE_LABEL_DIR = os.path.join(CAL, "perturb_s%d" % i)
    cache_p = FB.load_calib_cache(197)
    per_p = FB.detect_cache_for(cache_p, **over)
    best = None
    for st in ST:
        for tau in TAU:
            TP = FP = FN = 0
            for nm in names:
                e = per_p[nm]
                acc = FB.score_candidates(e["cands"], e["H"], e["W"], proto, st, bmf, tau)
                used = FB.greedy_match(acc, e["gt"])
                tp = sum(used); TP += tp; FN += len(e["gt"]) - tp; FP += len(acc) - tp
            p = TP/max(TP+FP, 1); r = TP/max(TP+FN, 1); f2 = FB.fbeta(p, r)
            if best is None or f2 > best["F2"]:
                best = {"st": st, "tau": tau, "F2": round(f2, 4)}
    pert_rows.append(best)
    print("perturb s%d:" % i, best, flush=True)
FB.BASELINE_LABEL_DIR = os.path.join(CAL, "labels_A")
R["perturb_recalib_fixed_mask_params"] = pert_rows

R["elapsed_sec"] = round(time.time()-t_start, 1)
with open(os.path.join(OUT, "phaseB_stats.json"), "w", encoding="utf8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
print("PHASE B DONE %.0fs" % R["elapsed_sec"], flush=True)
