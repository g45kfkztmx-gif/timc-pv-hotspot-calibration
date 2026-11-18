# -*- coding: utf-8 -*-
"""
rev5_common.py
v5 修订实验的公共工具: 训练+test评估 (协议与原实验完全一致),
yaml 读写, 结果断点续跑。
协议: epochs=60, imgsz=320, batch=32, workers=2, cache=ram,
      amp=False(GTX1650防NaN), seed 可指定; test 350 GT 评估一次。
"""
import os
import json
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
WEIGHTS = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\yolov8n.pt"
REV5 = os.path.join(BASE, "rev5_experiments")


def out_json(tag, seed):
    return os.path.join(REV5, "%s_seed%d.json" % (tag, seed))


def result_exists(tag, seed):
    p = out_json(tag, seed)
    return os.path.exists(p) and os.path.getsize(p) > 10


def train_and_eval(tag, yaml_path, seed=0, project=None):
    """训练 + test(GT) 评估一次, 返回 dict。"""

    from ultralytics import YOLO

    project = project or os.path.join(BASE, "runs")

    print("\n" + "=" * 64)
    print("TRAIN: %s | seed=%d | %s" % (tag, seed, yaml_path))
    print("=" * 64)

    model = YOLO(WEIGHTS)
    t0 = time.time()

    model.train(
        data=yaml_path,
        epochs=60,
        imgsz=320,
        batch=32,
        workers=2,
        device=0,
        amp=False,
        cache="ram",
        project=project,
        name="pseudo_%s_seed%d" % (tag, seed),
        exist_ok=True,
        verbose=False,
        seed=seed,
    )
    train_sec = time.time() - t0

    print("\n>>> Evaluating on TEST split ...")
    metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=320,
        batch=32,
        workers=2,
        device=0,
        verbose=False,
    )

    result = {
        "tag": tag,
        "seed": seed,
        "train_seconds": round(train_sec, 1),
        "test_map50": round(float(metrics.box.map50), 4),
        "test_map50_95": round(float(metrics.box.map), 4),
        "test_precision": round(float(metrics.box.mp), 4),
        "test_recall": round(float(metrics.box.mr), 4),
    }

    print("[%s|seed%d] mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f (%.0fs)" % (
        tag, seed, result["test_map50"], result["test_map50_95"],
        result["test_precision"], result["test_recall"], train_sec))

    with open(out_json(tag, seed), "w", encoding="utf8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def write_yaml(root, names=("hotspot",)):
    p = os.path.join(root, "dataset.yaml")
    with open(p, "w", encoding="utf8") as f:
        f.write("path: %s\n" % root.replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n  0: %s\n" % names[0])
    return p


def copy_test_from_weak(root):
    """test 冻结集: 照抄 yolo_weak 的人工GT test。"""
    src = os.path.join(BASE, "yolo_weak")
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(root, sub, "test"), exist_ok=True)
        for name in os.listdir(os.path.join(src, sub, "test")):
            s = os.path.join(src, sub, "test", name)
            d = os.path.join(root, sub, "test", name)
            if not os.path.exists(d):
                shutil_copy(s, d)


def shutil_copy(s, d):
    import shutil
    shutil.copy2(s, d)


def list_images(d):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(exts):
                out.append(f)
    return out
