# -*- coding: utf-8 -*-
"""
rev7_thermal_characteristics.py
===============================
Infrared Physics & Technology (IPT) repositioning: quantitative characterisation
of the *target class* (PV thermal hotspots) and of the detector's sensitivity to
its physical appearance axes.

Reads the per-box dump of the frozen reported detector
(rev6h3_boxes_<TAG>.json, produced by rev6h3_dump_boxes.py from
rev5_runs/pseudo_rev6_g42_r1_seed0/weights/best.pt at the reported operating
point g = 0.42) and reports, for the 279 annotated test instances:

  * appearance distributions  : footprint area, relative thermal contrast,
    peak prominence (z), elongation (aspect), fill, edge density
  * detection rate by bin     : contrast / area / aspect
  * binned counts             : so the paper table can be rebuilt exactly

No model is re-run and no new threshold is chosen: this is a post-hoc,
single-run description of the frozen reported configuration, exactly as used
for Table VII of the manuscript (matched 239 / FP 58 / mislocalised 27 /
missed 40 at IoU 0.50, confidence 0.25).

Output: rev7_thermal_characteristics.json
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("WEIGHTS_TAG", "rev6_g42_r1")

# bins: (low, high) with high = None meaning +inf
BINS = {
    "contrast": [(0, 20), (20, 40), (40, 60), (60, None)],
    "area": [(0, 30), (30, 60), (60, 100), (100, None)],
    "aspect": [(0, 1.3), (1.3, 1.6), (1.6, 2.2), (2.2, None)],
}
DIMS = ["area", "contrast", "peak_z", "aspect", "fill", "edge_density"]


def stats(v):
    return {
        "n": int(v.size),
        "median": round(float(np.median(v)), 4),
        "q25": round(float(np.percentile(v, 25)), 4),
        "q75": round(float(np.percentile(v, 75)), 4),
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
    }


def bin_label(lo, hi):
    return "%g-" % lo if hi is None else "%g-%g" % (lo, hi)


def main():
    with open(os.path.join(_HERE, "rev6h3_boxes_%s.json" % TAG), encoding="utf8") as f:
        meta = json.load(f)
    recs = meta["records"]

    gt = [r for r in recs if r["kind"] in ("gt_matched", "gt_missed")]
    pred = [r for r in recs if r["kind"].startswith("pred_")]

    def arr(rows, k):
        v = np.array([r[k] for r in rows], dtype=float)
        return v[np.isfinite(v)]

    appearance = {d: stats(arr(gt, d)) for d in DIMS}

    # fractions that a practitioner would quote as "how hard is this corpus"
    frac = {
        "area_lt_30_px2": round(float(np.mean(arr(gt, "area") < 30)), 4),
        "contrast_lt_20_gray": round(float(np.mean(arr(gt, "contrast") < 20)), 4),
        "aspect_gt_1p6": round(float(np.mean(arr(gt, "aspect") > 1.6)), 4),
        "crop_area_median_pct": round(
            100.0 * float(np.median(arr(gt, "area"))) / (60.0 * 110.0), 4
        ),
    }

    # detection rate per bin (matched / instances), same criterion as Table VII
    detect = {}
    for dim, bins in BINS.items():
        v = arr(gt, dim)
        rows = []
        for lo, hi in bins:
            sel = [r for r in gt if (r[dim] >= lo) and (hi is None or r[dim] < hi)]
            m = sum(1 for r in sel if r["kind"] == "gt_matched")
            rows.append({
                "bin": bin_label(lo, hi),
                "instances": len(sel),
                "detected": m,
                "rate": round(m / len(sel), 4) if sel else None,
            })
        detect[dim] = rows

    out = {
        "source": "rev6h3_boxes_%s.json" % TAG,
        "weights": meta.get("weights"),
        "criterion": {
            "iou_match": 0.50,
            "confidence": meta.get("conf_op", 0.25),
            "note": "single frozen run; identical criterion to the manuscript's residual-error table",
        },
        "counts": {
            "gt_instances": len(gt),
            "gt_matched": sum(1 for r in gt if r["kind"] == "gt_matched"),
            "gt_missed": sum(1 for r in gt if r["kind"] == "gt_missed"),
            "detections": len(pred),
        },
        "appearance": appearance,
        "appearance_fractions": frac,
        "detection_rate_by_bin": detect,
    }

    p = os.path.join(_HERE, "rev7_thermal_characteristics.json")
    with open(p, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("appearance (median [q25-q75]):")
    for d in DIMS:
        a = appearance[d]
        print("  %-13s %.2f [%.2f-%.2f]  n=%d" % (d, a["median"], a["q25"], a["q75"], a["n"]))
    print("fractions:", json.dumps(frac))
    for dim in BINS:
        print("detection rate by %s:" % dim)
        for row in detect[dim]:
            print("   %-10s n=%3d det=%3d rate=%s" %
                  (row["bin"], row["instances"], row["detected"], row["rate"]))
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
