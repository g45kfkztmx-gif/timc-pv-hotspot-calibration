# -*- coding: utf-8 -*-
"""本机执行 Major 3 基线（GTX1650, ultralytics 8.1.0, 与论文同超参）。
臂:
  ts_inhouse : 复刻包内自实现 teacher-student (conf>=0.5 硬伪标签), 验证管线 (期望≈0.933)
  ct_yolo    : 公开配方替代臂 = YOLO族 teacher-student + 强弱一致伪标签筛选
               (teacher 弱增强推理产标签; 仅保留在强增强视图下仍 conf>=0.5 且 IoU>=0.5 的框)
训练协议与 args.yaml 完全一致: 60ep batch32 imgsz320 workers2 cache=ram amp=False seed0 deterministic。
评估: Ultralytics val(test) + 自有 evaluator 前缀指标。
"""
import os, sys, json, shutil, random, time

BUILD = r"C:\Users\YJP\AppData\Local\Qianwen\User Data\qwen-agent-workspaces\runs\3612b721-61de-458e-89c4-b16c3f3594e1\2026-11-17T09-51-51-635Z\build"
PKG = os.path.join(BUILD, "repro", "TIMC_repro_package")
BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
CODE = os.path.join(PKG, "06_环境与代码", "code")
OUT = os.path.join(BUILD, "baseline_runs")
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
_orig = cv2.imread
cv2.imread = lambda p, f=cv2.IMREAD_COLOR: cv2.imdecode(np.fromfile(p, dtype=np.uint8), f)

import exp__rev5_f_budget as FB
FB.REV5 = os.path.join(OUT, "rev5_local")
os.makedirs(FB.REV5, exist_ok=True)
FB.BASELINE_HOT_DIR = os.path.join(PKG, "04_校准集_图像与人工框", "images")
FB.BASELINE_LABEL_DIR = os.path.join(PKG, "04_校准集_图像与人工框", "labels_A")
def _calib_names(N):
    files = [n for n in FB.ap_list_images(FB.BASELINE_HOT_DIR)
             if os.path.exists(os.path.join(FB.BASELINE_LABEL_DIR, os.path.splitext(n)[0] + ".txt"))]
    rng = random.Random(FB.SHUFFLE_SEED); rng.shuffle(files)
    return files[:N]
FB.calib_names = _calib_names

from ultralytics import YOLO
WEIGHTS = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\yolov8n.pt"
spec = importlib.util.spec_from_file_location("tim_eval", os.path.join(CODE, "exp__tim_eval.py"))
TE = importlib.util.module_from_spec(spec); spec.loader.exec_module(TE)

TRAIN_KW = dict(epochs=60, imgsz=320, batch=32, workers=2, device=0, amp=False,
                cache="ram", seed=0, deterministic=True, verbose=False, plots=False,
                optimizer="auto", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                warmup_epochs=3.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, erasing=0.4)

def log(*a):
    print(*a, flush=True)

def train_eval(tag, yaml_path):
    m = YOLO(WEIGHTS)
    t0 = time.time()
    m.train(data=yaml_path, project=OUT, name=tag, exist_ok=True, **TRAIN_KW)
    mt = m.val(data=yaml_path, split="test", imgsz=320, batch=32, workers=2, device=0, verbose=False)
    res = {"tag": tag, "train_seconds": round(time.time()-t0, 1),
           "ultralytics_map50": round(float(mt.box.map50), 4),
           "ultralytics_map50_95": round(float(mt.box.map), 4),
           "P": round(float(mt.box.mp), 4), "R": round(float(mt.box.mr), 4)}
    # 自有 evaluator: 导出 test 预测
    pred_dir = os.path.join(OUT, tag + "_pred")
    m.predict(os.path.join(BASE, "test", "hot"), conf=0.001, iou=0.7, imgsz=320, device=0,
              save_txt=True, save_conf=True, project=OUT, name=tag + "_pred", exist_ok=True, verbose=False)
    # normal 图也需预测(空GT)以计 FP: 合并目录
    hot_pred = os.path.join(OUT, tag + "_pred", "labels")
    m.predict(os.path.join(BASE, "test", "normal"), conf=0.001, iou=0.7, imgsz=320, device=0,
              save_txt=True, save_conf=True, project=OUT, name=tag + "_pred", exist_ok=True, verbose=False)
    preds = TE.read_pred_txt(hot_pred)
    gt = TE.load_gt(os.path.join(BASE, "test", "hot_labels"))
    res["own_map50"] = round(TE.ap_at(preds, gt, 0.5), 4)
    res.update(TE.prf_at_fp_budget(preds, gt, 350))
    with open(os.path.join(OUT, tag + ".json"), "w", encoding="utf8") as f:
        json.dump(res, f, indent=2)
    log("[RESULT]", tag, json.dumps(res))
    return res

def copy_test(root):
    for sub, src in [("images", os.path.join(BASE, "test", "hot")),
                     ("images", os.path.join(BASE, "test", "normal")),
                     ("labels", os.path.join(BASE, "test", "hot_labels"))]:
        d = os.path.join(root, sub, "test"); os.makedirs(d, exist_ok=True)
        for n in os.listdir(src):
            shutil.copy2(os.path.join(src, n), os.path.join(d, n))
    d = os.path.join(root, "labels", "test")
    for n in os.listdir(os.path.join(BASE, "test", "normal")):
        open(os.path.join(d, os.path.splitext(n)[0] + ".txt"), "w").close()

def build_teacher_yaml():
    root = os.path.join(OUT, "data_teacher")
    if os.path.exists(os.path.join(root, "dataset.yaml")):
        return os.path.join(root, "dataset.yaml")
    names = list(_calib_names(197)); rng = random.Random(42); rng.shuffle(names)
    n_val = max(1, int(round(len(names)*0.2))); val_n, tr_n = names[:n_val], names[n_val:]
    normals = sorted(os.listdir(os.path.join(PKG, "04_校准集_图像与人工框", "normal")))
    for split, subset, nrm in [("train", tr_n, normals[:260]), ("val", val_n, normals[260:320])]:
        di, dl = os.path.join(root, "images", split), os.path.join(root, "labels", split)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for f in subset:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(FB.BASELINE_HOT_DIR, f), os.path.join(di, f))
            shutil.copy2(os.path.join(FB.BASELINE_LABEL_DIR, stem + ".txt"), os.path.join(dl, stem + ".txt"))
        for f in nrm:
            shutil.copy2(os.path.join(PKG, "04_校准集_图像与人工框", "normal", f), os.path.join(di, f))
            open(os.path.join(dl, os.path.splitext(f)[0] + ".txt"), "w").close()
    copy_test(root)
    return FB.write_yaml(root)

def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1); inter = iw*ih
    sa = (a[2]-a[0])*(a[3]-a[1]); sb = (b[2]-b[0])*(b[3]-b[1])
    return inter/max(1e-9, sa+sb-inter)

def strong_view(bgr):
    """强增强视图: HSV 抖动 + 水平翻转 + 亮度gamma, 用于一致性筛选。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    rng = np.random.default_rng(0)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1]*rng.uniform(0.6, 1.4), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2]*rng.uniform(0.6, 1.4), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.5:
        out = out[:, ::-1].copy()
    return out

def build_student_yaml(tag, teacher_w, consistent):
    root = os.path.join(OUT, "data_" + tag)
    yp = os.path.join(root, "dataset.yaml")
    if os.path.exists(yp):
        return yp
    names = list(_calib_names(197)); rng = random.Random(42); rng.shuffle(names)
    n_val = max(1, int(round(len(names)*0.2))); val_n, tr_n = names[:n_val], names[n_val:]
    normals = sorted(os.listdir(os.path.join(PKG, "04_校准集_图像与人工框", "normal")))
    for split, subset, nrm in [("train", tr_n, normals[:260]), ("val", val_n, normals[260:320])]:
        di, dl = os.path.join(root, "images", split), os.path.join(root, "labels", split)
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for f in subset:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(FB.BASELINE_HOT_DIR, f), os.path.join(di, f))
            shutil.copy2(os.path.join(FB.BASELINE_LABEL_DIR, stem + ".txt"), os.path.join(dl, stem + ".txt"))
        for f in nrm:
            shutil.copy2(os.path.join(PKG, "04_校准集_图像与人工框", "normal", f), os.path.join(di, f))
            open(os.path.join(dl, os.path.splitext(f)[0] + ".txt"), "w").close()
    teacher = YOLO(teacher_w)
    stat = {"kept": 0, "drop": 0}
    for split, d in [("train", os.path.join(BASE, "train", "hot")),
                     ("val", os.path.join(BASE, "val", "hot"))]:
        for name in sorted(os.listdir(d)):
            src = os.path.join(d, name)
            bgr = cv2.imread(src)
            H, W = bgr.shape[:2]
            r0 = teacher.predict(bgr, conf=0.5, iou=0.5, imgsz=320, verbose=False)[0]
            boxes = [] if r0.boxes is None else [b.xyxy[0].tolist() for b in r0.boxes]
            if consistent and boxes:
                sv = strong_view(bgr)
                r1 = teacher.predict(sv, conf=0.5, iou=0.5, imgsz=320, verbose=False)[0]
                vb = [] if r1.boxes is None else [b.xyxy[0].tolist() for b in r1.boxes]
                # 若强视图做了水平翻转, 把验证框翻回原坐标 (RNG 顺序与 strong_view 一致)
                rr = np.random.default_rng(0)
                rr.uniform(-10, 10); rr.uniform(0.6, 1.4); rr.uniform(0.6, 1.4)
                did_flip = rr.random() < 0.5
                if did_flip:
                    vb = [[W-x2, y1, W-x1, y2] for x1, y1, x2, y2 in vb]
                kept = []
                for b in boxes:
                    if any(iou_xyxy(b, v) >= 0.5 for v in vb):
                        kept.append(b); stat["kept"] += 1
                    else:
                        stat["drop"] += 1
                boxes = kept
            di, dl = os.path.join(root, "images", split), os.path.join(root, "labels", split)
            shutil.copy2(src, os.path.join(di, name))
            lines = ["0 %.6f %.6f %.6f %.6f" % ((x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H)
                     for x1, y1, x2, y2 in boxes]
            with open(os.path.join(dl, os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
    log("[%s] pseudo kept/drop:" % tag, stat)
    copy_test(root)
    return FB.write_yaml(root)

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if which in ("all", "ts_inhouse"):
        ty = build_teacher_yaml()
        tr = train_eval("teacher197", ty)
        tw = os.path.join(OUT, "teacher197", "weights", "best.pt")
        sy = build_student_yaml("ts_inhouse", tw, consistent=False)
        results["ts_inhouse"] = train_eval("ts_inhouse_student", sy)
    if which in ("all", "ct_yolo"):
        tw = os.path.join(OUT, "teacher197", "weights", "best.pt")
        sy = build_student_yaml("ct_yolo", tw, consistent=True)
        TRAIN_KW.update(workers=0, cache=False, batch=16)  # 4GB卡+页面文件受限, 降载
        results["ct_yolo"] = train_eval("ct_yolo_student", sy)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf8") as f:
        json.dump(results, f, indent=2)
    log("DONE", json.dumps(results))

if __name__ == "__main__":
    main()
