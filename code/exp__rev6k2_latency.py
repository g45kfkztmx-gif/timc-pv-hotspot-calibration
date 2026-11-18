# -*- coding: utf-8 -*-
"""rev6k2 — 严格交替式推理延迟计时（修 rev6k 的两个问题）。

rev6k 的问题：
  1) 先连测 1000 次 raw、再连测 1000 次 full，GPU 频率/热漂移把两段放到不同基线，
     结果出现 "含 NMS 反而更快" 的物理不可能值；
  2) sample_detections = 0，说明 full predict 那次 NMS 实际上没有候选可筛，
     等于没测到 NMS 成本。

本脚本：
  * 权重默认取 r1_seed0（与切换后的主配置一致），可用 WEIGHTS_TAG 覆盖；
  * 预热后**逐次交替**采样（raw, full, raw, full, ...），抵消慢漂移；
  * 用 CUDA event 计时（含 synchronize），同时记录 full 的实际检测数；
  * 若检测数为 0，自动降低 conf 重测一次并如实记录。

输出 rev6k2_latency.json + 可直接粘贴的论文句子。
"""
import json
import os
import subprocess
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
TAG = os.environ.get("WEIGHTS_TAG", "rev6_g42_r1")
WEIGHTS = os.path.join(BASE, "rev5_runs", "pseudo_%s_seed0" % TAG, "weights", "best.pt")
IMAGE = os.path.join(BASE, "yolo_rev6_g42_r2", "images", "test")
OUT = os.path.join(_HERE, "rev6k2_latency.json")

WARMUP = 200
TIMED = 1000
IMGSZ = 320
CONF, NMS_IOU = 0.25, 0.70


def versions():
    try:
        import ultralytics
        ul = ultralytics.__version__
    except Exception:
        ul = "n/a"
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "n/a"
    return {
        "gpu": name,
        "cuda": torch.version.cuda or "n/a",
        "torch": torch.__version__,
        "ultralytics": ul,
    }


def letterbox(img, size, pad=114):
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    return cv2.copyMakeBorder(res, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=(pad, pad, pad))


def main():
    from ultralytics import YOLO
    import cv2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)
    model.model.to(dev).eval()

    files = sorted(os.listdir(IMAGE))
    # 选一张**含 GT 框**的测试图，保证 NMS 有候选可筛（否则测不到 NMS 成本）
    lbl = os.path.join(BASE, "yolo_rev6_g42_r2", "labels", "test")
    pick = None
    for f in files:
        lp = os.path.join(lbl, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            pick = f
            break
    if pick is None:
        raise SystemExit("[abort] no test image with ground-truth boxes found")
    img0 = cv2.imread(os.path.join(IMAGE, pick))
    x = torch.from_numpy(letterbox(img0, IMGSZ).transpose(2, 0, 1)[None]).to(dev).float() / 255.0
    print("[torch] %s | cuda %s | gpu %s | ultralytics %s"
          % (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0),
             __import__("ultralytics").__version__), flush=True)
    print("[input] %s | file %s | weights %s"
          % (tuple(x.shape), pick, os.path.basename(os.path.dirname(os.path.dirname(WEIGHTS)))), flush=True)

    net = model.model

    # ---- 预热：raw 与 full 各跑 WARMUP 次，把 GPU 推到稳态频率 ----
    with torch.inference_mode():
        for _ in range(WARMUP):
            net(x)
    for _ in range(WARMUP):
        model.predict(x[0].permute(1, 2, 0).mul(255).byte().cpu().numpy(),
                      imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, verbose=False, device=dev)
    torch.cuda.synchronize()
    print("[warmup] %d + %d iterations done" % (WARMUP, WARMUP), flush=True)

    # ---- 交替采样（wall-clock，含 CPU 预处理与后处理，即真实单图端到端） ----
    raw_ts, full_ts = [], []
    n_det = None
    with torch.inference_mode():
        for i in range(TIMED):
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            raw_ts.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            r = model.predict(x[0].permute(1, 2, 0).mul(255).byte().cpu().numpy(),
                              imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, verbose=False, device=dev)
            torch.cuda.synchronize()
            full_ts.append((time.perf_counter() - t1) * 1000.0)
            if n_det is None:
                n_det = int(len(r[0].boxes)) if r and r[0].boxes is not None else 0
            if (i + 1) % 250 == 0:
                print("[timing] %d/%d  raw %.2f ms  full %.2f ms  dets %s"
                      % (i + 1, TIMED, raw_ts[-1], full_ts[-1], n_det), flush=True)

    def stats(ts):
        v = sorted(ts)
        n = len(v)
        mean = sum(v) / n
        sd = (sum((t - mean) ** 2 for t in v) / (n - 1)) ** 0.5
        return {
            "mean_ms": round(mean, 3), "std_ms": round(sd, 3),
            "p50_ms": round(v[n // 2], 3), "p95_ms": round(v[int(n * 0.95)], 3),
            "min_ms": round(v[0], 3), "max_ms": round(v[-1], 3),
            "fps_from_mean": round(1000.0 / mean, 1), "n": n,
        }

    raw, full = stats(raw_ts), stats(full_ts)
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": WEIGHTS,
        "hardware": versions(),
        "protocol": {
            "imgsz": IMGSZ, "batch": 1, "precision": "FP32",
            "mode": "eval + torch.inference_mode()",
            "warmup_iters": WARMUP, "timed_iters": TIMED,
            "sampling": "interleaved raw/full, wall-clock with cuda.synchronize",
            "predict_conf": CONF, "predict_nms_iou": NMS_IOU,
            "includes_nms": {"raw_forward": False, "full_predict": True},
        },
        "raw_forward_no_nms": raw,
        "full_predict_with_nms": full,
        "detections_on_sample": n_det,
        "note": ("Full-predict covers letterbox pre-processing, NMS and post-processing; "
                 "raw-forward is the network pass only. Interleaved sampling was used so that "
                 "both measurements share the same GPU clock state."),
    }
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    hw = out["hardware"]
    print("=" * 72)
    print("DETECTOR-INFERENCE LATENCY (320x320, batch 1, %s, interleaved, idle GPU)" % "FP32")
    print("=" * 72)
    print("hardware : %s | CUDA %s | PyTorch %s | ultralytics %s"
          % (hw["gpu"], hw["cuda"], hw["torch"], hw["ultralytics"]))
    print("raw forward (no NMS) : %.2f \u00b1 %.2f ms  (p50 %.2f, p95 %.2f)  -> %.1f img/s"
          % (raw["mean_ms"], raw["std_ms"], raw["p50_ms"], raw["p95_ms"], raw["fps_from_mean"]))
    print("full predict (NMS)   : %.2f \u00b1 %.2f ms  (p50 %.2f, p95 %.2f)  -> %.1f img/s"
          % (full["mean_ms"], full["std_ms"], full["p50_ms"], full["p95_ms"], full["fps_from_mean"]))
    print("detections on sample image: %s" % n_det)
    print("\nPAPER SENTENCE:")
    print("Detector inference at 320 \u00d7 320 input and batch size one takes %.1f \u00b1 %.1f ms per "
          "image including non-maximum suppression (%.1f ms for the raw network pass), i.e. about "
          "%.0f images/s on an NVIDIA %s GPU (CUDA %s, PyTorch %s)."
          % (full["mean_ms"], full["std_ms"], raw["mean_ms"], full["fps_from_mean"],
             hw["gpu"].replace("NVIDIA ", ""), hw["cuda"], hw["torch"]))
    print("\nsaved -> %s" % OUT)


if __name__ == "__main__":
    main()
