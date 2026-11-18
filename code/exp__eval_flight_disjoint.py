# -*- coding: utf-8 -*-
"""
eval_flight_disjoint.py —— S8 分组鲁棒性复核 (不重训, 只换评估子集)
====================================================================
背景: 审计 (audit_split_leakage.py) 发现 image-level split 存在**架次级泄漏**:
350 张测试图中 273 张 (78%) 所在 DJI 飞行架次也贡献了训练/校准图, 相邻帧
内容高度相似 -> 报告的 test mAP 可能乐观偏置。

本脚本用**冻结权重**在一个"架次与 train/val/calib 完全不相交"的测试子集
(默认 77 张 / 35 个实例) 上重新评估, 与全测试集结果对照。

用法:
  python eval_flight_disjoint.py <weights...>
输出:
  rev5_experiments/flight_disjoint_eval.json
"""
import json
import os
import shutil
import sys

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(BASE, "yolo_test_flightdisjoint")


def build_subset():
    """把架次不相交的测试图做成一个可被 ultralytics val 读取的数据集。"""
    meta = json.load(open(os.path.join(HERE, "test_flight_disjoint.json"),
                          encoding="utf8"))
    yaml_path = os.path.join(SUB, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("[gen] subset exists, skip")
        return yaml_path, meta

    for split in ("val", "test"):
        os.makedirs(os.path.join(SUB, "images", split), exist_ok=True)
        os.makedirs(os.path.join(SUB, "labels", split), exist_ok=True)

    src_img = os.path.join(BASE, "test")
    src_lbl = os.path.join(BASE, "yolo_weak", "labels", "test")
    n_hot = n_norm = 0
    for f in meta["disjoint_files"]:
        stem = os.path.splitext(f)[0]
        hits = [os.path.join(src_img, s, f) for s in ("hot", "normal")]
        src = next((h for h in hits if os.path.exists(h)), None)
        if src is None:
            continue
        # val 留空(不参与选模型), 全部放 test
        shutil.copy2(src, os.path.join(SUB, "images", "test", f))
        lab = os.path.join(src_lbl, stem + ".txt")
        dst = os.path.join(SUB, "labels", "test", stem + ".txt")
        if os.path.exists(lab):
            shutil.copy2(lab, dst)
        else:
            open(dst, "w").close()
        if os.path.getsize(dst) > 0:
            n_hot += 1
        else:
            n_norm += 1

    with open(yaml_path, "w", encoding="utf8") as f:
        f.write("path: %s\n" % SUB.replace("\\", "/"))
        f.write("train: images/test\n")
        f.write("val: images/test\n")
        f.write("test: images/test\n")
        f.write("names:\n  0: hotspot\n")
    print("[gen] subset: hot=%d normal=%d (train split points here too)"
          % (n_hot, n_norm))
    return yaml_path, meta


def main():
    weights = sys.argv[1:]
    if not weights:
        print("usage: eval_flight_disjoint.py <weights> [weights...]")
        return
    from ultralytics import YOLO

    yaml_path, meta = build_subset()
    out_path = os.path.join(HERE, "flight_disjoint_eval.json")
    res = {"subset": {k: v for k, v in meta.items() if k != "disjoint_files"},
           "models": []}

    for w in weights:
        if not os.path.exists(w):
            print("[skip] %s not found" % w)
            continue
        m = YOLO(w)
        mt = m.val(data=yaml_path, split="test", imgsz=320, batch=16,
                   workers=2, device=0, verbose=False)
        row = {"weights": w,
               "mAP50": round(float(mt.box.map50), 4),
               "mAP50_95": round(float(mt.box.map), 4),
               "P": round(float(mt.box.mp), 4),
               "R": round(float(mt.box.mr), 4)}
        print("[flight-disjoint] %-52s mAP50=%.4f P=%.4f R=%.4f"
              % (os.path.basename(os.path.dirname(os.path.dirname(w))),
                 row["mAP50"], row["P"], row["R"]))
        res["models"].append(row)

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
