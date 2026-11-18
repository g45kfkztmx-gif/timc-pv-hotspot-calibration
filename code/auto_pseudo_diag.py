# -*- coding: utf-8 -*-

"""
auto_pseudo_diag.py
诊断增强版热斑伪标签系统 —— 把每个错误样本变成可定位的问题

在原 auto_pseudo.py 基础上新增环节:

1. 全候选诊断
   保留所有连通域候选(包括被拒绝的), 记录拒绝原因:
   empty_roi / small_area / bad_aspect / low_score

2. 边缘距离惩罚 (情况五)
   抑制光伏板边框、焊带、图像边缘的误检,
   惩罚系数并入综合评分。

3. 多面板可视化
   原图+框 / Base Mask / Low Mask / High Mask / 连通域着色 / 拒绝候选
   红=accepted预测 绿=人工GT 黄=small_area 青=bad_aspect 橙=low_score

4. GT 逐框错误分类 (对应五种情况)
   matched        检测正确
   oversize       情况一: 掩膜覆盖太大, 候选框远大于GT
   undersize      情况二: 掩膜过小, 候选框在GT内部
   filtered_area  情况三: 被面积过滤
   filtered_score 情况四: 被质量分过滤(给出最弱指标)
   filtered_aspect 被长宽比过滤
   fragmented     掩膜有覆盖但连通域碎裂
   mask_too_small 掩膜覆盖不足(需要降K)
   no_signal      完全无信号

5. FP 逐框分析
   border_fp      情况五: 靠近图像边缘的误检
   interior_fp    内部误检

6. JSON 诊断报告 + 汇总统计 + 参数调整建议
"""


import os
import cv2
import json
import time

import numpy as np

from dataclasses import dataclass, asdict

from auto_pseudo import (
    DetectParams,
    sigmoid,
    list_images,
    load_labels,
    extract_components,
    expand_box,
    IoU,
    calc_peak,
    calc_fill,
    calc_edge,
    TEST_HOT_DIR,
    TEST_LABEL_DIR,
    VIS_DIR,
    PARAM_FILE,
)


# ==========================================================
# 配置
# ==========================================================

NUM_IMAGES = 20
SCALE = 5                       # 可视化放大倍数
IOU_MATCH = 0.5                 # TP判定阈值
IOU_PARTIAL = 0.25              # 部分重叠判定阈值

DIAG_DIR = os.path.join(VIS_DIR, "diagnosis")
os.makedirs(DIAG_DIR, exist_ok=True)


@dataclass
class DiagParams(DetectParams):

    # 情况一: 可关闭闭运算, 减少粘连
    use_close: bool = True

    # 情况五: 边缘惩罚强度
    # 最终分 = 基础分 * (border_min_factor + (1-border_min_factor)*边缘距离)
    # border_min_factor=0.6 表示贴边候选分数打6折, 0=不惩罚
    border_min_factor: float = 0.6

    # 情况四 contrast 偏低: 改用局部环形背景计算对比度
    use_local_bg: bool = False

    # 情况一结构性修复: 用高阈值掩膜把粘连的连通域拆成多个"核心"框
    # 解决相邻热斑合并导致预测框远大于GT的问题
    fit_to_core: bool = False
    core_expand_ratio: float = 0.2
    core_min_area: int = 3

    # 核心掩膜阈值倍数: k_core = k_sigma + delta_k * core_delta_mult
    # 越大核心越紧, 用于压缩仍然oversize的核心框
    core_delta_mult: float = 1.0

    # 高置信豁免: 基础分>=border_exempt_score的候选(真实角落热斑
    # 通常contrast/peak极高), 边缘惩罚因子不低于border_exempt_factor
    border_exempt_score: float = 0.90
    border_exempt_factor: float = 0.80

    # 微小ROI(像素数<该值)时剔除edge指标并重新归一化权重
    # (Canny在极小区域上不稳定, 会把真实小热斑的分数拉低)
    small_roi_px: int = 30

    # 正扩展时每侧最少扩展像素数(小框int截断会导致扩展为0)
    expand_min_px: int = 0

    # ---------- 局部自适应阈值救援 (召回率关键模块) ----------
    # 诊断发现: no_signal 漏检GT的全局z低(中位2.37)但局部z高(中位2.49),
    # 因为全图std被其他亮区抬高。用滑动窗口局部均值/方差做二次阈值,
    # 与全局掩膜取并集, 救回这类弱对比热斑。
    use_local_rescue: bool = False
    local_win: int = 25           # 局部窗口尺寸(奇数化)
    k_local: float = 2.0          # 局部z阈值 (k = k_local + dk偏移)
    local_min_delta: float = 3.0  # 最小绝对灰度差, 抑制平坦区噪声


# ==========================================================
# 边缘距离惩罚 (情况五)
# ==========================================================

def calculate_image_border_penalty(box, W, H):

    x, y, w, h = box

    center_x = x + w / 2
    center_y = y + h / 2

    border_distance = min(
        center_x,
        center_y,
        W - center_x,
        H - center_y
    )

    normalized_distance = border_distance / max(
        min(W, H) * 0.15,
        1
    )

    return float(np.clip(normalized_distance, 0.0, 1.0))


# ==========================================================
# 可控闭运算的掩膜生成
# ==========================================================

def _apply_morph(mask, params):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params.morph_size, params.morph_size)
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if params.use_close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def generate_mask_diag(gray, k, params):

    blur = cv2.GaussianBlur(
        gray,
        (params.blur_size, params.blur_size),
        0
    )

    mean = float(np.mean(blur))
    std = float(np.std(blur))
    threshold = mean + k * std

    mask = (blur > threshold).astype(np.uint8) * 255

    mask = _apply_morph(mask, params)

    return mask, blur, threshold


def local_rescue_mask(blur, params, dk=0.0):
    """
    局部自适应阈值掩膜。

    滑动窗口(local_win)计算局部均值/标准差,
    条件: (blur - 局部均值) > k*lstd 且 > local_min_delta
    用于救回全局z低、局部z高的弱对比热斑(no_signal漏检)。

    dk: 与全局掩膜一致的阈值偏移 (low=-delta_k, high=+delta_k)
    """

    k = params.k_local + dk

    win = params.local_win

    if win % 2 == 0:
        win += 1

    win = max(5, win)

    f = blur.astype(np.float32)

    lmean = cv2.blur(f, (win, win))
    lsq = cv2.blur(f * f, (win, win))

    lvar = np.maximum(lsq - lmean * lmean, 0.0)
    lstd = np.sqrt(lvar)

    dz = f - lmean

    mask = ((dz > k * lstd) & (dz > params.local_min_delta))

    return (mask.astype(np.uint8) * 255)


def generate_mask_full(gray, k, params):
    """
    全局阈值掩膜 + 局部救援掩膜 的并集。
    未开启 use_local_rescue 时等价于 generate_mask_diag。
    """

    mask, blur, thr = generate_mask_diag(gray, k, params)

    if not params.use_local_rescue:
        return mask, blur, thr

    dk = k - params.k_sigma

    lmask = local_rescue_mask(blur, params, dk)
    lmask = _apply_morph(lmask, params)

    merged = cv2.bitwise_or(mask, lmask)

    return merged, blur, thr


# ==========================================================
# 质量评分 (支持局部背景)
# ==========================================================

def calc_contrast_diag(gray, box, params):

    x, y, w, h = box
    H, W = gray.shape

    roi = gray[y:y + h, x:x + w]

    if roi.size == 0:
        return 0.0

    if params.use_local_bg:
        # 局部环形背景: 框外扩 0.5*max(w,h) 的区域, 扣除框内
        pad = int(0.5 * max(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        local = gray[y1:y2, x1:x2]
        m = np.ones(local.shape, dtype=bool)
        m[y - y1:y - y1 + h, x - x1:x - x1 + w] = False
        bg = local[m]
    else:
        m = np.ones(gray.shape, dtype=bool)
        m[y:y + h, x:x + w] = False
        bg = gray[m]

    if len(bg) == 0:
        return 0.0

    diff = np.mean(roi) - np.mean(bg)

    return float(sigmoid(diff / 20))


def calc_stability_diag(gray, box, params):

    x, y, w, h = box

    scores = []

    for dk in [-params.delta_k, 0, params.delta_k]:

        mask, _, _ = generate_mask_full(gray, params.k_sigma + dk, params)

        roi = mask[y:y + h, x:x + w]

        if roi.size:
            scores.append(np.mean(roi > 0))

    if len(scores) < 2:
        return 0.0

    return float(1 - np.std(scores))


def quality_score_diag(gray, box, area, params):

    c = calc_contrast_diag(gray, box, params)
    p = calc_peak(gray, box)
    s = calc_stability_diag(gray, box, params)
    e = calc_edge(gray, box)
    f = calc_fill(area, box)

    roi_px = box[2] * box[3]

    if roi_px < params.small_roi_px:

        # 微小ROI: Canny不稳定, 剔除edge并重新归一化权重
        wsum = (
            params.w_contrast
            + params.w_peak
            + params.w_stability
            + params.w_fill
        )

        total = (
            params.w_contrast * c
            + params.w_peak * p
            + params.w_stability * s
            + params.w_fill * f
        ) / max(wsum, 1e-8)

        e = -1.0    # 标记未参与评分

    else:

        total = (
            params.w_contrast * c
            + params.w_peak * p
            + params.w_stability * s
            + params.w_edge * e
            + params.w_fill * f
        )

    metrics = {
        "contrast": c,
        "peak": p,
        "stability": s,
        "edge": e,
        "fill": f
    }

    return float(total), metrics


# ==========================================================
# 带最小像素保证的框扩展
# ==========================================================

def expand_box_diag(box, W, H, ratio, min_px):

    x, y, w, h = box

    dx = int(w * ratio)
    dy = int(h * ratio)

    if ratio > 0:
        dx = max(dx, min_px)
        dy = max(dy, min_px)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(W, x + w + dx)
    y2 = min(H, y + h + dy)

    return (x1, y1, x2 - x1, y2 - y1)


# ==========================================================
# 核心拆分 (情况一结构性修复)
# ==========================================================

def split_component_by_core(comp_id, bbox, base_labels, high_mask, params):
    """
    在基础连通域的外接框范围内, 用高阈值掩膜寻找核心团,
    把粘连在一起的多个热斑拆成多个候选框。
    """

    x, y, w, h = bbox

    roi_high = high_mask[y:y + h, x:x + w].copy()
    roi_lbl = base_labels[y:y + h, x:x + w]

    # 只保留属于当前连通域的像素
    roi_high[roi_lbl != comp_id] = 0

    n, lbls, stats, _ = cv2.connectedComponentsWithStats(roi_high, 8)

    cores = []

    for i in range(1, n):

        cx, cy, cw, ch, carea = stats[i]

        if carea < params.core_min_area:
            continue

        cores.append(
            {
                "box": (int(x + cx), int(y + cy), int(cw), int(ch)),
                "area": int(carea)
            }
        )

    return cores


# ==========================================================
# 全候选诊断检测
# ==========================================================

def make_candidate(gray, box, area, params, W, H, origin="component"):

    x, y, w, h = box

    aspect = max(w, h) / max(1, min(w, h))

    cand = {
        "raw_box": (int(x), int(y), int(w), int(h)),
        "box": (int(x), int(y), int(w), int(h)),
        "area": int(area),
        "aspect": round(float(aspect), 2),
        "origin": origin,
        "status": "accepted",
        "reject_reason": None,
        "base_score": 0.0,
        "score": 0.0,
        "border_penalty": 1.0,
        "metrics": {},
        "weakest": None,
        "matched_gt": None,
    }

    min_area = H * W * params.min_area_ratio

    # 核心候选来自高阈值掩膜, 可靠性更高, 面积门槛减半
    if origin == "core":
        min_area *= 0.5

    roi = gray[y:y + h, x:x + w]

    if roi.size == 0:
        cand["status"] = "rejected"
        cand["reject_reason"] = "empty_roi"
        return cand

    # ---------- 硬规则过滤 ----------

    if area < min_area:
        cand["status"] = "rejected"
        cand["reject_reason"] = "small_area"
    elif aspect > params.max_aspect:
        cand["status"] = "rejected"
        cand["reject_reason"] = "bad_aspect"

    # ---------- 质量评分 (无论是否被硬规则过滤都算, 便于诊断) ----------

    base_score, metrics = quality_score_diag(gray, (x, y, w, h), area, params)

    bp = calculate_image_border_penalty((x, y, w, h), W, H)

    factor = (
        params.border_min_factor
        + (1 - params.border_min_factor) * bp
    )

    # 高置信豁免: 真实角落热斑contrast/peak极高, 不应被边缘惩罚杀死
    if base_score >= params.border_exempt_score:
        factor = max(factor, params.border_exempt_factor)

    score = base_score * factor

    cand["base_score"] = round(base_score, 4)
    cand["score"] = round(score, 4)
    cand["border_penalty"] = round(bp, 3)
    cand["metrics"] = {
        k: round(float(v), 3) for k, v in metrics.items()
    }

    valid = {k: v for k, v in metrics.items() if v >= 0}
    cand["weakest"] = min(valid, key=valid.get) if valid else None

    # ---------- 质量分过滤 ----------

    if cand["status"] == "accepted" and score < params.score_threshold:
        cand["status"] = "rejected"
        cand["reject_reason"] = "low_score"

    # ---------- 框扩展 ----------

    if cand["status"] == "accepted":

        ratio = (
            params.core_expand_ratio
            if origin == "core"
            else params.expand_ratio
        )

        cand["box"] = expand_box_diag(
            (x, y, w, h), W, H, ratio, params.expand_min_px
        )

    return cand


def detect_diag(gray, params):

    H, W = gray.shape

    base_mask, blur, thr = generate_mask_full(gray, params.k_sigma, params)
    low_mask, _, _ = generate_mask_full(
        gray, params.k_sigma - params.delta_k, params
    )
    high_mask, _, _ = generate_mask_full(
        gray, params.k_sigma + params.delta_k, params
    )

    comps, labels = extract_components(base_mask)

    # 核心掩膜: 阈值更严, 用于框拟合与粘连拆分
    core_mask, _, _ = generate_mask_full(
        gray,
        params.k_sigma + params.delta_k * params.core_delta_mult,
        params
    )

    cands = []

    for comp in comps:

        box = comp["box"]

        # ---------- 核心拆分 ----------

        if params.fit_to_core:

            cores = split_component_by_core(
                comp["id"], box, labels, core_mask, params
            )

            if len(cores) >= 1:

                # 保留原始合并框作为诊断记录
                merged = make_candidate(
                    gray, box, comp["area"], params, W, H,
                    origin="merged"
                )
                merged["status"] = "rejected"
                merged["reject_reason"] = "merged_split"
                cands.append(merged)

                for core in cores:

                    cands.append(
                        make_candidate(
                            gray, core["box"], core["area"],
                            params, W, H, origin="core"
                        )
                    )

                continue

        cands.append(
            make_candidate(
                gray, box, comp["area"], params, W, H,
                origin="component"
            )
        )

    rescue_px = 0

    if params.use_local_rescue:
        rmask = local_rescue_mask(blur, params, 0.0)
        rescue_px = int(np.count_nonzero(rmask))

    info = {
        "base_mask": base_mask,
        "low_mask": low_mask,
        "high_mask": high_mask,
        "labels": labels,
        "threshold": round(thr, 2),
        "mean": round(float(np.mean(blur)), 2),
        "std": round(float(np.std(blur)), 2),
        "rescue_px": rescue_px,
    }

    return cands, info


# ==========================================================
# 贪心匹配 (按分数从高到低)
# ==========================================================

def match_diag(cands, gts, thr=IOU_MATCH):

    acc = sorted(
        [c for c in cands if c["status"] == "accepted"],
        key=lambda c: -c["score"]
    )

    matched = [False] * len(gts)

    for c in acc:

        best = 0
        idx = -1

        for i, g in enumerate(gts):

            if matched[i]:
                continue

            v = IoU(c["box"], g)

            if v > best:
                best = v
                idx = i

        if best >= thr and idx >= 0:
            matched[idx] = True
            c["matched_gt"] = idx

    return matched


# ==========================================================
# GT 逐框错误分类 (五种情况)
# ==========================================================

def classify_gt(gt, cands, info):

    x, y, w, h = gt
    gt_area = max(w * h, 1)

    acc = [c for c in cands if c["status"] == "accepted"]

    # ---------- accepted 中与GT IoU 最高的 ----------

    best_acc = None
    best_acc_iou = 0.0

    for c in acc:
        v = IoU(c["box"], gt)
        if v > best_acc_iou:
            best_acc = c
            best_acc_iou = v

    if best_acc_iou >= IOU_MATCH:
        detail = "IoU=%.2f" % best_acc_iou
        if best_acc_iou < 0.65:
            detail += ", IoU偏低可微调expand_ratio"
        return "matched", best_acc_iou, detail

    # ---------- accepted 但 IoU 不足: 框尺寸问题 ----------

    if best_acc_iou >= IOU_PARTIAL:

        bx, by, bw, bh = best_acc["box"]
        ratio = (bw * bh) / gt_area

        if ratio > 1.6:
            return (
                "oversize",
                best_acc_iou,
                "情况一: 预测框=%.1fxGT, k_sigma+=0.1或关闭use_close或expand_ratio=-0.05"
                % ratio
            )

        if ratio < 0.6:
            return (
                "undersize",
                best_acc_iou,
                "情况二: 预测框=%.0f%%GT, k_sigma-=0.1或expand_ratio+=0.05"
                % (ratio * 100)
            )

        return "shift", best_acc_iou, "框尺寸接近但位置偏移"

    # ---------- 被过滤的候选压在GT上: 情况三 / 情况四 ----------

    rej_over = [
        c for c in cands
        if c["status"] != "accepted" and IoU(c["raw_box"], gt) >= 0.2
    ]

    if rej_over:

        c0 = max(rej_over, key=lambda c: IoU(c["raw_box"], gt))
        reason = c0["reject_reason"]

        if reason == "small_area":
            return (
                "filtered_area",
                IoU(c0["raw_box"], gt),
                "情况三: area=%d被面积过滤, min_area_ratio降到0.0010~0.0015"
                % c0["area"]
            )

        if reason == "bad_aspect":
            return (
                "filtered_aspect",
                IoU(c0["raw_box"], gt),
                "长宽比=%.1f被过滤, 真实细长热斑可放宽max_aspect" % c0["aspect"]
            )

        if reason == "low_score":
            return (
                "filtered_score",
                IoU(c0["raw_box"], gt),
                "情况四: score=%.2f<%.2f, 最弱指标=%s(%s)"
                % (
                    c0["score"],
                    0.55,
                    c0["weakest"],
                    json.dumps(c0["metrics"])
                )
            )

        return "filtered_other", 0.0, str(reason)

    # ---------- 没有任何候选压住GT: 看掩膜覆盖 ----------

    base_cov = float(np.mean(info["base_mask"][y:y + h, x:x + w] > 0))
    low_cov = float(np.mean(info["low_mask"][y:y + h, x:x + w] > 0))

    if base_cov > 0.3:
        return (
            "fragmented",
            0.0,
            "Base掩膜覆盖%.0f%%但连通域碎裂, 考虑开use_close或k_sigma-=0.1"
            % (base_cov * 100)
        )

    if low_cov > 0.3:
        return (
            "mask_too_small",
            0.0,
            "情况二: Base覆盖%.0f%%/Low覆盖%.0f%%, k_sigma-=0.1"
            % (base_cov * 100, low_cov * 100)
        )

    return (
        "no_signal",
        0.0,
        "Base/Low掩膜均未覆盖(%.0f%%/%.0f%%), k_sigma-=0.2或样本对比度过低"
        % (base_cov * 100, low_cov * 100)
    )


# ==========================================================
# FP 逐框分析
# ==========================================================

def classify_fp(c):

    if c["border_penalty"] < 0.5:
        return (
            "border_fp",
            "情况五: 贴边误检 bp=%.2f edge=%.2f, 加强边缘惩罚或降低w_edge"
            % (c["border_penalty"], c["metrics"].get("edge", 0))
        )

    return (
        "interior_fp",
        "内部误检 score=%.2f contrast=%.2f peak=%.2f"
        % (
            c["score"],
            c["metrics"].get("contrast", 0),
            c["metrics"].get("peak", 0)
        )
    )


# ==========================================================
# 可视化
# ==========================================================

REJ_COLORS = {
    "small_area": (0, 255, 255),    # 黄
    "bad_aspect": (255, 255, 0),    # 青
    "low_score": (0, 140, 255),     # 橙
    "empty_roi": (128, 128, 128),   # 灰
    "merged_split": (255, 0, 200),  # 紫: 已被核心拆分替代的合并框
}

REJ_TAGS = {
    "small_area": "S",
    "bad_aspect": "A",
    "low_score": "L",
    "empty_roi": "E",
    "merged_split": "M",
}


def upscale(img, S=SCALE):
    return cv2.resize(
        img,
        (img.shape[1] * S, img.shape[0] * S),
        interpolation=cv2.INTER_NEAREST
    )


def draw_box_scaled(img, box, color, thick=2):
    x, y, w, h = box
    cv2.rectangle(
        img,
        (x * SCALE, y * SCALE),
        ((x + w) * SCALE, (y + h) * SCALE),
        color,
        thick
    )


def build_visual(name, bgr, cands, gts, info, gt_report, fp_report, params):

    H, W = bgr.shape[:2]
    pw, ph = W * SCALE, H * SCALE

    # ---------- 面板1: 原图 + 全部框 ----------

    p1 = upscale(bgr)

    for c in cands:

        if c["status"] == "accepted":
            draw_box_scaled(p1, c["box"], (0, 0, 255), 2)
            cv2.putText(
                p1, "%.2f" % c["score"],
                (c["box"][0] * SCALE, max(12, c["box"][1] * SCALE - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
            )
        else:
            col = REJ_COLORS.get(c["reject_reason"], (128, 128, 128))
            draw_box_scaled(p1, c["raw_box"], col, 1)

    for g in gts:
        draw_box_scaled(p1, g, (0, 255, 0), 2)

    cv2.putText(
        p1, "RED:pred GREEN:GT yel:S cya:A org:L",
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
    )

    # ---------- 面板2/3/4: 三级掩膜 ----------

    p2 = cv2.cvtColor(upscale(info["base_mask"]), cv2.COLOR_GRAY2BGR)
    p3 = cv2.cvtColor(upscale(info["low_mask"]), cv2.COLOR_GRAY2BGR)
    p4 = cv2.cvtColor(upscale(info["high_mask"]), cv2.COLOR_GRAY2BGR)

    cv2.putText(
        p2, "BASE k=%.1f thr=%.0f" % (params.k_sigma, info["threshold"]),
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1
    )
    cv2.putText(
        p3, "LOW k=%.1f" % (params.k_sigma - params.delta_k),
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1
    )
    cv2.putText(
        p4, "HIGH k=%.1f" % (params.k_sigma + params.delta_k),
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1
    )

    for p in (p2, p3, p4):
        for g in gts:
            draw_box_scaled(p, g, (0, 255, 0), 1)

    # ---------- 面板5: 连通域着色 ----------

    lbl = info["labels"]
    n = int(lbl.max())

    if n > 0:
        rs = np.random.RandomState(7)
        colors = (rs.rand(n + 1, 3) * 200 + 40).astype(np.uint8)
        colors[0] = 0
        comp_img = colors[lbl]
    else:
        comp_img = np.zeros((H, W, 3), dtype=np.uint8)

    p5 = upscale(comp_img)

    cv2.putText(
        p5, "COMPONENTS n=%d" % n,
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
    )

    for c in cands:
        col = (0, 0, 255) if c["status"] == "accepted" else (0, 255, 255)
        draw_box_scaled(p5, c["raw_box"], col, 1)

    # ---------- 面板6: 被拒绝候选 + 原因标签 ----------

    p6 = upscale(bgr)

    for c in cands:

        if c["status"] == "accepted":
            continue

        col = REJ_COLORS.get(c["reject_reason"], (128, 128, 128))
        tag = REJ_TAGS.get(c["reject_reason"], "?")

        draw_box_scaled(p6, c["raw_box"], col, 1)

        cv2.putText(
            p6, tag,
            (c["raw_box"][0] * SCALE + 2, c["raw_box"][1] * SCALE + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1
        )

    for g in gts:
        draw_box_scaled(p6, g, (0, 255, 0), 1)

    cv2.putText(
        p6, "REJECTED S:area A:aspect L:score",
        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
    )

    # ---------- 拼接画布 ----------

    panels = [p1, p2, p3, p4, p5, p6]
    gap = 8
    cols = 3

    report_lines = build_report_lines(gt_report, fp_report, cands)

    header_h = 28
    report_h = 24 + 17 * len(report_lines)

    canvas_w = cols * pw + (cols + 1) * gap
    canvas_h = header_h + 2 * (ph + gap) + report_h

    canvas = np.full((canvas_h, canvas_w, 3), (30, 30, 30), np.uint8)

    cv2.putText(
        canvas,
        "%s  mean=%.1f std=%.1f thr=%.1f  close=%s border_factor=%.2f"
        % (
            name, info["mean"], info["std"], info["threshold"],
            params.use_close, params.border_min_factor
        ),
        (gap, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1
    )

    for i, p in enumerate(panels):

        r = i // cols
        c = i % cols

        x = gap + c * (pw + gap)
        y = header_h + r * (ph + gap)

        canvas[y:y + ph, x:x + pw] = p

    # ---------- 文字报告区 ----------

    ty = header_h + 2 * (ph + gap) + 16

    for line in report_lines:

        cv2.putText(
            canvas, line,
            (gap, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1
        )

        ty += 17

    return canvas


def build_report_lines(gt_report, fp_report, cands):

    lines = []

    n_acc = sum(1 for c in cands if c["status"] == "accepted")
    n_rej = len(cands) - n_acc

    lines.append(
        "candidates=%d accepted=%d rejected=%d | GT=%d FP=%d"
        % (len(cands), n_acc, n_rej, len(gt_report), len(fp_report))
    )

    for i, g in enumerate(gt_report):

        lines.append(
            "GT%d [%d,%d,%d,%d] -> %s : %s"
            % (
                i, g["gt"][0], g["gt"][1], g["gt"][2], g["gt"][3],
                g["tag"], g["detail"][:90]
            )
        )

    for j, f in enumerate(fp_report):

        lines.append(
            "FP%d %s : %s" % (j, f["tag"], f["detail"][:100])
        )

    return lines[:34]


# ==========================================================
# 单图诊断
# ==========================================================

def diagnose_image(bgr_path, params):

    bgr = cv2.imread(bgr_path)

    if bgr is None:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape

    cands, info = detect_diag(gray, params)

    label_path = os.path.join(
        TEST_LABEL_DIR,
        os.path.splitext(os.path.basename(bgr_path))[0] + ".txt"
    )

    gts = load_labels(label_path, W, H)

    matched = match_diag(cands, gts)

    # ---------- GT 分类 ----------

    gt_report = []

    for i, g in enumerate(gts):

        if matched[i]:
            gt_report.append({
                "gt": list(g),
                "tag": "matched",
                "iou": 1.0,
                "detail": "TP"
            })
            continue

        tag, iou, detail = classify_gt(g, cands, info)

        gt_report.append({
            "gt": list(g),
            "tag": tag,
            "iou": round(iou, 3),
            "detail": detail
        })

    # ---------- FP 分类 ----------

    fp_report = []

    for c in cands:

        if c["status"] != "accepted":
            continue

        if c["matched_gt"] is not None:
            continue

        tag, detail = classify_fp(c)

        fp_report.append({
            "box": list(c["box"]),
            "score": c["score"],
            "tag": tag,
            "detail": detail,
            "metrics": c["metrics"],
            "border_penalty": c["border_penalty"]
        })

    return {
        "bgr": bgr,
        "cands": cands,
        "gts": gts,
        "info": info,
        "gt_report": gt_report,
        "fp_report": fp_report,
    }


# ==========================================================
# 汇总建议
# ==========================================================

def summarize(all_gt, all_fp):

    from collections import Counter

    gt_cnt = Counter(g["tag"] for g in all_gt)
    fp_cnt = Counter(f["tag"] for f in all_fp)

    weakest_cnt = Counter()

    for g in all_gt:
        if g["tag"] == "filtered_score":
            pass

    suggestions = []

    if gt_cnt.get("oversize", 0):
        suggestions.append(
            "情况一 x%d: k_sigma+=0.1; 或 use_close=False; 或 expand_ratio=-0.05"
            % gt_cnt["oversize"]
        )

    n_under = gt_cnt.get("undersize", 0) + gt_cnt.get("mask_too_small", 0)
    if n_under:
        suggestions.append(
            "情况二 x%d: k_sigma-=0.1; 或 expand_ratio+=0.05~0.10" % n_under
        )

    if gt_cnt.get("filtered_area", 0):
        suggestions.append(
            "情况三 x%d: min_area_ratio=0.0010~0.0015, 同步 score_threshold+=0.05"
            % gt_cnt["filtered_area"]
        )

    if gt_cnt.get("filtered_score", 0):
        suggestions.append(
            "情况四 x%d: 看最弱指标 contrast->局部背景 peak->自适应 "
            "stability->delta_k=0.15 edge->降权重/轮廓框 fill->降权重"
            % gt_cnt["filtered_score"]
        )

    if gt_cnt.get("no_signal", 0):
        suggestions.append(
            "无信号 x%d: k_sigma-=0.2, 或该样本热斑本身对比度过低"
            % gt_cnt["no_signal"]
        )

    if gt_cnt.get("fragmented", 0):
        suggestions.append(
            "连通域碎裂 x%d: 开 use_close 或 k_sigma-=0.1"
            % gt_cnt["fragmented"]
        )

    if fp_cnt.get("border_fp", 0):
        suggestions.append(
            "情况五 x%d: border_min_factor 降到 0.4~0.5, 或降低 w_edge"
            % fp_cnt["border_fp"]
        )

    if fp_cnt.get("interior_fp", 0):
        suggestions.append(
            "内部误检 x%d: score_threshold+=0.05, 或提高 w_contrast"
            % fp_cnt["interior_fp"]
        )

    return gt_cnt, fp_cnt, suggestions


# ==========================================================
# main
# ==========================================================

def main(params=None):

    start = time.time()

    # ---------- 参数 ----------

    if params is None:

        if os.path.exists(PARAM_FILE):

            with open(PARAM_FILE, "r", encoding="utf8") as f:
                data = json.load(f)

            params = DiagParams(**data)
            print("loaded best_params.json")

        else:

            params = DiagParams()
            print("no best_params.json, using defaults")

    print(json.dumps(asdict(params), indent=2))

    # ---------- 逐图诊断 ----------

    files = list_images(TEST_HOT_DIR)[:NUM_IMAGES]

    print("\ndiagnosing %d images ..." % len(files))

    all_gt = []
    all_fp = []

    TP = FP = FN = 0

    for idx, name in enumerate(files):

        path = os.path.join(TEST_HOT_DIR, name)

        res = diagnose_image(path, params)

        if res is None:
            continue

        tp = sum(1 for g in res["gt_report"] if g["tag"] == "matched")
        fn = len(res["gt_report"]) - tp
        fp = len(res["fp_report"])

        TP += tp
        FN += fn
        FP += fp

        all_gt.extend(res["gt_report"])
        all_fp.extend(res["fp_report"])

        # ---------- 可视化 ----------

        canvas = build_visual(
            name, res["bgr"], res["cands"], res["gts"], res["info"],
            res["gt_report"], res["fp_report"], params
        )

        cv2.imwrite(os.path.join(DIAG_DIR, name), canvas)

        # ---------- JSON ----------

        json_path = os.path.join(
            DIAG_DIR, os.path.splitext(name)[0] + ".json"
        )

        with open(json_path, "w", encoding="utf8") as f:

            json.dump(
                {
                    "file": name,
                    "params": asdict(params),
                    "mask_info": {
                        "threshold": res["info"]["threshold"],
                        "mean": res["info"]["mean"],
                        "std": res["info"]["std"],
                    },
                    "candidates": res["cands"],
                    "gt_report": res["gt_report"],
                    "fp_report": res["fp_report"],
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            "[%2d/%d] %-40s TP=%d FN=%d FP=%d"
            % (idx + 1, len(files), name, tp, fn, fp)
        )

    # ---------- 汇总 ----------

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    gt_cnt, fp_cnt, suggestions = summarize(all_gt, all_fp)

    summary = {
        "num_images": len(files),
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gt_error_distribution": dict(gt_cnt),
        "fp_distribution": dict(fp_cnt),
        "suggestions": suggestions,
        "params": asdict(params),
    }

    with open(
        os.path.join(DIAG_DIR, "summary.json"), "w", encoding="utf8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("TP=%d FP=%d FN=%d  P=%.4f R=%.4f F1=%.4f" % (TP, FP, FN, precision, recall, f1))
    print("\nGT error distribution:")
    for k, v in sorted(gt_cnt.items(), key=lambda kv: -kv[1]):
        print("  %-16s %d" % (k, v))
    print("\nFP distribution:")
    for k, v in sorted(fp_cnt.items(), key=lambda kv: -kv[1]):
        print("  %-16s %d" % (k, v))
    print("\nSuggestions:")
    for s in suggestions:
        print("  " + s)
    print("\noutput dir: %s" % DIAG_DIR)
    print("elapsed: %.1fs" % (time.time() - start))


if __name__ == "__main__":
    main()
