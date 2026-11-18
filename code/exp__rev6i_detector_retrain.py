# -*- coding: utf-8 -*-
"""
rev6i_detector_retrain.py
=========================
Table XI（检测器对比）在**新操作点 (g=0.42)** 的伪标签集上重训。

用户决策: **只重训 5 个 YOLO 检测器，删除 RT-DETR-l**
（Transformer 对比移入 Discussion 作为 future work，避免"4 新 + 1 旧"的混合基座）。

  tag               模型
  ------------------------------------
  det_yolov5n       yolov5n.pt
  det_yolov5s       yolov5s.pt
  det_yolov8s       yolov8s.pt
  det_yolov10n      yolov10n.pt
  det_yolov11n      yolo11n.pt

（YOLOv8n 行 = 已训好的 rev6_g42_r1_seed0，不再重复训练。）

数据集: yolo_fin42（主配置 = 无精炼 A3/R1 的伪标签集，effective gate 0.42）
协议  : 60 epochs / imgsz 320 / batch 32 / workers 2 / amp=False / cache=ram / seed 0
        OOM 自动降 batch 16 重试。

等待: 先等 Phase 3 (rev6_phase3_5seed.py) 收尾，避免与 5-seed 训练抢 GPU。

输出: rev6i_det_<tag>.json  +  rev6i_detector_summary.json
"""
import json
import os
import subprocess
import sys
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
ROOT = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, _HERE)

DATA = os.path.join(BASE, "yolo_fin42", "dataset.yaml")
SUMMARY = os.path.join(_HERE, "rev6i_detector_summary.json")

JOBS = [
    ("det_yolov5n", os.path.join(ROOT, "yolov5n.pt")),
    ("det_yolov5s", os.path.join(ROOT, "yolov5s.pt")),
    ("det_yolov8s", os.path.join(ROOT, "yolov8s.pt")),
    ("det_yolov10n", os.path.join(ROOT, "yolov10n.pt")),
    ("det_yolov11n", os.path.join(ROOT, "yolo11n.pt")),
]

PHASE3_SUMMARY = os.path.join(_HERE, "rev6_phase3_summary.json")
R1_SEED4 = os.path.join(_HERE, "rev6_g42_r1_seed4.json")
WAIT_MAX_S = 4 * 3600
POLL_S = 60


def running(substr):
    """用 CIM 枚举 python 进程 (wmic 在本环境被拦截)。返回 None = 无法判定。"""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name == 'python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            stderr=subprocess.DEVNULL, timeout=60)
        txt = out.decode("utf8", "ignore")
        if not txt.strip():
            return False
        return substr in txt
    except Exception:
        return None


def wait_for_gpu():
    """等 Phase 3 **真正**收尾。

    坑（2026-09-10 踩到）：rev6_phase3_5seed.py 是**每跑完一条臂就重写** summary，
    所以 summary 文件在 A5 臂结束时（16:22）就存在了，此时 A3 的 seed1-4 还在排队。
    若以 summary 存在为判据，就会和 Phase 3 抢 GPU（实测 epoch 17s -> 30s）。
    因此判据必须锚定 **A3 最后一个 seed 的 json**。
    """
    done = lambda: os.path.exists(R1_SEED4)
    if done():
        print("[wait] Phase 3 最后一个 A3 seed 产物已存在 -> 直接开跑", flush=True)
        return
    t0 = time.time()
    while True:
        if done():
            print("[wait] Phase 3 收尾（产物就绪）-> 开始重训", flush=True)
            return
        alive = running("rev6_phase3_5seed")
        if alive is False:
            print("[wait] Phase 3 进程确已退出（且无产物）-> 继续", flush=True)
            return
        if time.time() - t0 > WAIT_MAX_S:
            print("[wait] 等待超时 %.1f h -> 仍继续" % (WAIT_MAX_S / 3600.0), flush=True)
            return
        time.sleep(POLL_S)


def train_and_eval(tag, weights, batch=32):
    from ultralytics import YOLO

    print("\n" + "=" * 62, flush=True)
    print("TRAINING: %s (batch=%d)" % (tag, batch), flush=True)
    print("=" * 62, flush=True)

    model = YOLO(weights)
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = round(float(get_flops(model.model, 320)), 1)
    except Exception:
        pass

    t0 = time.time()
    model.train(data=DATA, epochs=60, imgsz=320, batch=batch, workers=2,
                device=0, amp=False, cache="ram",
                project=os.path.join(BASE, "runs"), name="pseudo_" + tag,
                exist_ok=True, verbose=False, seed=0)
    train_sec = time.time() - t0

    print(">>> evaluating on TEST split ...", flush=True)
    m = model.val(data=DATA, split="test", imgsz=320, batch=32, workers=4,
                  device=0, verbose=False)
    r = {"tag": tag, "weights": os.path.basename(weights),
         "params_M": round(n_params, 2), "gflops": gflops, "batch": batch,
         "train_seconds": round(train_sec, 1),
         "test_map50": round(float(m.box.map50), 4),
         "test_map50_95": round(float(m.box.map), 4),
         "test_precision": round(float(m.box.mp), 4),
         "test_recall": round(float(m.box.mr), 4),
         "train_min": round(train_sec / 60.0, 1)}
    print("[%s] mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f (%.0f s)"
          % (tag, r["test_map50"], r["test_map50_95"], r["test_precision"],
             r["test_recall"], train_sec), flush=True)
    return r


def main():
    wait_for_gpu()

    import torch
    results = []
    for tag, weights in JOBS:
        rj = os.path.join(_HERE, "rev6i_det_%s.json" % tag)
        if os.path.exists(rj):
            print("[skip] %s exists" % tag, flush=True)
            results.append(json.load(open(rj, encoding="utf8")))
            continue
        try:
            r = train_and_eval(tag, weights, batch=32)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print("[OOM] %s -> batch 16" % tag, flush=True)
            torch.cuda.empty_cache()
            r = train_and_eval(tag, weights, batch=16)
        with open(rj, "w", encoding="utf8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        results.append(r)

    # 把 YOLOv8n 的结果一并收进汇总（来自 Phase 1；主配置为无精炼的 R1）
    y8 = os.path.join(_HERE, "rev6_g42_r1_seed0.json")
    if os.path.exists(y8):
        d = json.load(open(y8, encoding="utf8"))
        results.append({"tag": "det_yolov8n", "weights": "rev6_g42_r1_seed0/best.pt",
                        "params_M": 3.16, "gflops": None,
                        "batch": 32,
                        "train_seconds": d.get("train_seconds"),
                        "test_map50": d["test_map50"], "test_map50_95": d["test_map50_95"],
                        "test_precision": d["test_precision"], "test_recall": d["test_recall"],
                        "note": "reused from Phase 1 (identical protocol and label set)"})

    out = {"dataset": DATA, "operating_point": {"effective_gate": 0.42},
           "note": "RT-DETR-l intentionally excluded: retrained rows must share one pseudo-label basis",
           "results": results,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(SUMMARY, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved ->", SUMMARY, flush=True)
    for r in results:
        print("  %-14s mAP50=%.4f  mAP50-95=%.4f  P=%.4f R=%.4f"
              % (r["tag"], r["test_map50"], r["test_map50_95"],
                 r["test_precision"], r["test_recall"]), flush=True)


if __name__ == "__main__":
    main()
