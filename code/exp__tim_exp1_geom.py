# -*- coding: utf-8 -*-
"""
tim_exp1_geom.py —— TIM 审稿意见 #1/#2 的测量学事实采集
======================================================================
输出 (tim_geom_summary.json):
  1. 语料各 split 的采集时间戳分布 (文件名 DJI_YYYYMMDDHHMMSS_*)
  2. 代表性东中国站点的太阳几何 + clear-sky GHI 估计 (0.65-0.75 透过率带)
     -> 估计采集时刻的 GHI 上界, 与 IEC TS 62446-3 的 600 W/m^2 对照
  3. 各采集日的 GT 框像素尺寸中位数 -> 推断语料内 GSD 是否存在平台间差异
     (H20T IFOV=1.107 mrad vs M30T IFOV=1.664 mrad, 相差 1.5x)
纯 CPU, 无 cv2 依赖。
"""
import os
import re
import json
import math
import statistics as st

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
HERE = os.path.join(BASE, "rev5_experiments")
OUT = os.path.join(HERE, "tim_geom_summary.json")

DAYS = ["20230222", "20230223", "20230227", "20230228",
        "20230302", "20230308", "20231114", "20231121"]

# 代表性东中国站点 (语料 8 电站均在中国, 源发布不给 GPS; 取 3 个代表点做敏感性)
SITES = {"east_cn_32N115E": (32.0, 115.0),
         "south_28N110E": (28.0, 110.0),
         "north_40N120E": (40.0, 120.0)}
TIMEZONE = 8            # DJI 时间戳默认本地时间 (UTC+8)
GSC = 1361.0            # 太阳常数 W/m^2


def ts_of(name):
    """DJI_20230222130057_0210_T_000011.png -> (2023-02-22, 13:00:57)"""
    m = re.match(r"DJI_(\d{8})(\d{6})_", name)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    return d, t


def day_of_year(d):
    y, mo, dd = int(d[:4]), int(d[4:6]), int(d[6:8])
    n = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0:
        n[1] = 29
    return sum(n[:mo - 1]) + dd, y


def solar_geom(d, tstr, lat_deg, lon_deg):
    """NOAA 太阳位置算法 -> cos(zenith), 大气层外水平辐照度 G_o (W/m^2)"""
    doy, y = day_of_year(d)
    hh = int(tstr[0:2]) + int(tstr[2:4]) / 60.0 + int(tstr[4:6]) / 3600.0
    gamma = 2 * math.pi / 365.0 * (doy - 1 + (hh - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))       # minutes
    decl = (0.006918 - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.001480 * math.sin(3 * gamma))                  # rad
    time_offset = eqtime + 4 * lon_deg - 60 * TIMEZONE
    tst = hh * 60 + time_offset
    ha = math.radians(tst / 4.0 - 180.0)
    lat = math.radians(lat_deg)
    cosz = (math.sin(lat) * math.sin(decl)
            + math.cos(lat) * math.cos(decl) * math.cos(ha))
    cosz = max(0.0, cosz)
    gon = GSC * (1 + 0.033 * math.cos(360.0 * doy / 365.0 * math.pi / 180.0))
    go = gon * cosz
    return cosz, go


def list_imgs(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))


def main():
    res = {}

    # ---------- 1. 各 split 时间戳 ----------
    groups = {
        "calibration": os.path.join(BASE, "baseline", "hot"),
        "train_pool": None, "val_pool": None, "test": None,
    }
    stamps = {}
    for gname, gdir in [("calibration", groups["calibration"]),
                        ("train_pool_hot", os.path.join(BASE, "train", "hot")),
                        ("train_pool_normal", os.path.join(BASE, "train", "normal")),
                        ("val_pool_hot", os.path.join(BASE, "val", "hot")),
                        ("val_pool_normal", os.path.join(BASE, "val", "normal")),
                        ("test", os.path.join(BASE, "yolo_weak", "images", "test"))]:
        ts = []
        for f in list_imgs(gdir):
            t = ts_of(f)
            if t:
                ts.append(t)
        stamps[gname] = {"n": len(ts), "days": sorted({d for d, _ in ts}),
                         "hours": sorted({h[0:2] for _, h in ts})}
    res["timestamps"] = stamps

    # 全语料 (论文用到的全部图像) 的 GHI 估计
    all_ts = []
    for gname, gdir in [("calibration", groups["calibration"]),
                        ("train_pool_hot", os.path.join(BASE, "train", "hot")),
                        ("train_pool_normal", os.path.join(BASE, "train", "normal")),
                        ("val_pool_hot", os.path.join(BASE, "val", "hot")),
                        ("val_pool_normal", os.path.join(BASE, "val", "normal")),
                        ("test", os.path.join(BASE, "yolo_weak", "images", "test"))]:
        for f in list_imgs(gdir):
            t = ts_of(f)
            if t:
                all_ts.append(t)
    res["n_images_with_timestamp"] = len(all_ts)

    ghi_lo_all, ghi_hi_all = [], []
    hour_hist = {}
    for d, t in all_ts:
        hour_hist[t[0:2]] = hour_hist.get(t[0:2], 0) + 1
        cosz, go = solar_geom(d, t, *SITES["east_cn_32N115E"])
        ghi_lo_all.append(0.65 * go)
        ghi_hi_all.append(0.75 * go)
    res["hour_hist"] = dict(sorted(hour_hist.items()))
    res["clear_sky_ghi"] = {
        "model": "extraterrestrial horizontal (NOAA solar position) x clear-sky "
                 "transmittance 0.65-0.75 (typical clear winter/late-autumn sky, "
                 "eastern China); representative site 32N/115E",
        "ghi_low_wm2": round(st.mean(ghi_lo_all), 1),
        "ghi_high_wm2": round(st.mean(ghi_hi_all), 1),
        "frac_ge600_low": round(sum(1 for g in ghi_lo_all if g >= 600) / len(ghi_lo_all), 4),
        "frac_ge600_high": round(sum(1 for g in ghi_hi_all if g >= 600) / len(ghi_hi_all), 4),
        "frac_ge400_low": round(sum(1 for g in ghi_lo_all if g >= 400) / len(ghi_lo_all), 4),
    }
    # 站点敏感性 (同时刻三站点 GHI 估计差异)
    sens = {}
    for d, t in all_ts[:0]:
        pass
    samp = all_ts[::max(1, len(all_ts) // 200)]
    diffs = []
    for d, t in samp:
        vals = []
        for sname, (la, lo) in SITES.items():
            _c, go = solar_geom(d, t, la, lo)
            vals.append(0.70 * go)
        diffs.append(max(vals) - min(vals))
    res["clear_sky_ghi"]["site_sensitivity_maxdiff_wm2"] = round(st.mean(diffs), 1)

    # ---------- 2. 各采集日 GT 框像素尺寸 ----------
    lbl_dir = os.path.join(BASE, "yolo_weak", "labels", "test")
    per_day = {}
    for f in list_imgs(os.path.join(BASE, "yolo_weak", "images", "test")):
        d = ts_of(f)
        if not d:
            continue
        day = d[0]
        p = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf8"):
            parts = line.split()
            if len(parts) < 5:
                continue
            w = float(parts[3]) * 60.0
            h = float(parts[4]) * 110.0
            per_day.setdefault(day, []).append((w, h))
    day_stat = {}
    for day, boxes in sorted(per_day.items()):
        ws = [b[0] for b in boxes]
        hs = [b[1] for b in boxes]
        day_stat[day] = {
            "n_inst": len(boxes),
            "median_w_px": round(st.median(ws), 2),
            "median_h_px": round(st.median(hs), 2),
            "median_area_px": round(st.median([a * b for a, b in boxes]), 1),
        }
    res["per_day_gt_box_px"] = day_stat

    # IFOV / GSD 参考表
    ifov_h20t = math.radians(40.6 / 640.0)
    ifov_m30t = math.radians(61.0 / 640.0)
    res["ifov_mrad"] = {"H20T": round(ifov_h20t * 1000, 3),
                        "M30T": round(ifov_m30t * 1000, 3),
                        "ratio_M30T_over_H20T": round(ifov_m30t / ifov_h20t, 3)}
    res["gsd_mm_per_px"] = {
        "H20T": {str(h): round(ifov_h20t * h * 1000, 1) for h in (15, 20, 25, 30)},
        "M30T": {str(h): round(ifov_m30t * h * 1000, 1) for h in (15, 20, 25, 30)},
    }
    res["crop_footprint_m_at_20m"] = {
        "H20T": [round(60 * ifov_h20t * 20, 2), round(110 * ifov_h20t * 20, 2)],
        "M30T": [round(60 * ifov_m30t * 20, 2), round(110 * ifov_m30t * 20, 2)],
    }

    with open(OUT, "w", encoding="utf8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
