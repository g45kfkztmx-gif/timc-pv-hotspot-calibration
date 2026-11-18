# -*- coding: utf-8 -*-
"""
rev6k_latency.py
================
延迟实测（**固定计时协议**，等所有训练结束、GPU 空闲后才跑）。

为什么必须等：
  在训练占用 GPU 时测出的数字（历史上测到 30.7 ms）不可比也不公平。

计时协议（写进论文的那一套）：
  * 输入 320 x 320, batch size = 1, FP32
  * model.eval(), torch.inference_mode()（无梯度）
  * 预热 100 次，正式计时 1000 次，报告 mean ± std（另附 p50 / p95）
  * 两个口径分别报告：
      (a) raw forward：只算网络前向（不含预处理、不含 NMS）
      (b) full predict：Ultralytics predict，含 letterbox 预处理 + NMS + 后处理
          （conf 0.25 / NMS IoU 0.7，与论文的部署口径一致）
  * 记录硬件与软件版本（GPU / CUDA / PyTorch / Ultralytics）

输出: rev6k_latency.json，并打印可直接粘进论文的句子。
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
WEIGHTS = os.path.join(BASE, "rev5_runs", "pseudo_rev6_g42_r2_seed0", "weights", "best.pt")
IMAGE = os.path.join(BASE, "yolo_rev6_g42_r2", "images", "test")
DET_SUMMARY = os.path.join(_HERE, "rev6i_detector_summary.json")
OUT = os.path.join(_HERE, "rev6k_latency.json")

WARMUP = 100
TIMED = 1000
IMGSZ = 320
CONF, NMS_IOU = 0.25, 0.70
WAIT_MAX_S = 4 * 3600


def gpu_state():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=30)
        m, u = [int(v.strip()) for v in out.decode().strip().split("\n")[0].split(",")]
        return m, u
    except Exception:
        return None


SKIP_DET_WAIT = os.environ.get("SKIP_DET_WAIT") == "1"


def wait_for_idle():
    t0 = time.time()
    if SKIP_DET_WAIT:
        print("[wait] SKIP_DET_WAIT=1 -> waiting for an idle GPU only", flush=True)
    else:
        while not os.path.exists(DET_SUMMARY):
            if time.time() - t0 > WAIT_MAX_S:
                print("[wait] timeout waiting for detector retrain summary", flush=True)
                break
            time.sleep(60)
        print("[wait] detector retrain finished -> waiting for an idle GPU", flush=True)
    quiet = 0
    while True:
        st = gpu_state()
        if st is None:
            print("[wait] nvidia-smi unavailable -> proceed", flush=True)
            return
        mem, util = st
        if mem < 600 and util < 10:
            quiet += 1
            print("[wait] gpu mem=%dMiB util=%d%% (quiet %d/3)" % (mem, util, quiet), flush=True)
            if quiet >= 3:
                print("[wait] GPU idle -> starting timing", flush=True)
                return
        else:
            quiet = 0
            print("[wait] gpu busy (mem=%dMiB util=%d%%) -> keep waiting" % (mem, util), flush=True)
        time.sleep(20)


def stats(ts):
    x = sorted(ts)
    n = len(x)
    mean = sum(x) / n
    sd = (sum((v - mean) ** 2 for v in x) / (n - 1)) ** 0.5
    return {"mean_ms": round(mean, 3), "std_ms": round(sd, 3),
            "p50_ms": round(x[n // 2], 3),
            "p95_ms": round(x[int(0.95 * n)], 3),
            "min_ms": round(x[0], 3), "max_ms": round(x[-1], 3),
            "fps_from_mean": round(1000.0 / mean, 1), "n": n}


def main():
    import torch
    import ultralytics
    import numpy as np

    wait_for_idle()

    dev = torch.device("cuda:0")
    print("[torch] %s | cuda %s | gpu %s | ultralytics %s"
          % (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0),
             ultralytics.__version__), flush=True)

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    net = model.model.to(dev).eval().float()
    names = {0: "hotspot"}

    files = sorted(os.listdir(IMAGE))
    # 手工复现 Ultralytics 的 letterbox（pad=114），避免依赖其内部 API 版本
    import cv2
    img0 = cv2.imread(os.path.join(IMAGE, files[0]))
    h, w = img0.shape[:2]
    r = min(IMGSZ / float(w), IMGSZ / float(h))
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img0, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = torch.from_numpy(canvas[:, :, ::-1].copy().transpose(2, 0, 1)[None]).to(dev).float() / 255.0
    print("[input] tensor", tuple(x.shape), "file", files[0], flush=True)

    # ---------- (a) raw forward ----------
    with torch.inference_mode():
        for _ in range(WARMUP):
            net(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(TIMED):
            t = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000.0)
    raw = stats(ts)

    # ---------- (b) full predict (preprocess + NMS + post) ----------
    src = os.path.join(IMAGE, files[0])
    for _ in range(max(20, WARMUP // 5)):
        model.predict(source=src, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, device=0,
                      half=False, verbose=False, save=False, max_det=300)
    torch.cuda.synchronize()
    ts = []
    n_det = None
    for _ in range(TIMED):
        t = time.perf_counter()
        r = model.predict(source=src, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, device=0,
                          half=False, verbose=False, save=False, max_det=300)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000.0)
        if n_det is None:
            try:
                n_det = int(len(r[0].boxes))
            except Exception:
                n_det = None
    full = stats(ts)

    rep = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": WEIGHTS,
        "hardware": {"gpu": torch.cuda.get_device_name(0),
                     "cuda": torch.version.cuda, "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__},
        "protocol": {"imgsz": IMGSZ, "batch": 1, "precision": "FP32",
                     "mode": "eval + torch.inference_mode()",
                     "warmup_iters": WARMUP, "timed_iters": TIMED,
                     "predict_conf": CONF, "predict_nms_iou": NMS_IOU,
                     "includes_nms": {"raw_forward": False, "full_predict": True}},
        "raw_forward_no_nms": raw,
        "full_predict_with_nms": full,
        "sample_detections": n_det,
        "note": ("Full-predict numbers include letterbox pre-processing, NMS and "
                 "post-processing; raw-forward numbers are the network pass only."),
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("DETECTOR-INFERENCE LATENCY (320x320, batch 1, FP32, idle GPU)")
    print("=" * 72)
    print("hardware : %s | CUDA %s | PyTorch %s | ultralytics %s"
          % (rep["hardware"]["gpu"], rep["hardware"]["cuda"],
             rep["hardware"]["torch"], rep["hardware"]["ultralytics"]))
    print("raw forward (no NMS) : %.2f ± %.2f ms  (p50 %.2f, p95 %.2f)  -> %.1f img/s"
          % (raw["mean_ms"], raw["std_ms"], raw["p50_ms"], raw["p95_ms"], raw["fps_from_mean"]))
    print("full predict (NMS)   : %.2f ± %.2f ms  (p50 %.2f, p95 %.2f)  -> %.1f img/s"
          % (full["mean_ms"], full["std_ms"], full["p50_ms"], full["p95_ms"], full["fps_from_mean"]))
    print("\nPAPER SENTENCE:")
    print("The detector alone requires %.1f ± %.1f ms per 320 × 320 image including "
          "non-maximum suppression (%.1f ms for the raw forward pass without it), i.e. "
          "%.0f images/s at this input resolution (NVIDIA %s, CUDA %s, PyTorch %s)."
          % (full["mean_ms"], full["std_ms"], raw["mean_ms"], full["fps_from_mean"],
             rep["hardware"]["gpu"], rep["hardware"]["cuda"], rep["hardware"]["torch"]))
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
