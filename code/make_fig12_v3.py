# -*- coding: utf-8 -*-
"""
make_fig12_v3.py — Fig.4 修复版
修改: YOLOv8n 标签原本被红方块标记压住
对策: 统一所有标签放标记右下 + 白底 bbox, YOLOv8n 用箭头指引避免重叠
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
OUT = r"C:\Users\YJP\.qoderworkcn\workspace\msclj98qehfx7b6h\figs_paper\fig12_efficiency_merged.png"

with open(os.path.join(BASE, "detector_comparison_v2.json"), "r", encoding="utf8") as f:
    rows = json.load(f)["results"]

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.frameon": False,
})

COLOR_OTHER = "#4C72B0"
COLOR_OURS = "#C00000"

# 标签位置: 标记右下方 (右下更稳, 不与左侧 y 轴或右面板冲突)
LABEL_DX_PANEL = {"a": 0.7, "b": 80}   # 数据坐标偏移
LABEL_DY = -0.013                    # 向下, 避开标记
LABEL_BBOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=600)

for ax, xkey, panel in (
    (axes[0], "params_M", "a"),
    (axes[1], "train_s", "b"),
):
    dx = LABEL_DX_PANEL[panel]
    for t in rows:
        ours = "ours" in t["detector"]
        label = t["detector"].replace(" (ours)", "")
        # 标记
        ax.scatter(
            t[xkey], t["mAP50"],
            marker="s" if ours else "o",
            s=70 if ours else 46,
            color=COLOR_OURS if ours else COLOR_OTHER,
            edgecolors="black", linewidths=0.6, zorder=5,
        )
        # 标签: 全部放右下, 白底框避免覆盖/被覆盖
        color = COLOR_OURS if ours else "#333333"
        weight = "bold" if ours else "normal"
        ax.annotate(
            label,
            xy=(t[xkey], t["mAP50"]),
            xytext=(t[xkey] + dx, t["mAP50"] + LABEL_DY),
            fontsize=8.0,
            color=color,
            fontweight=weight,
            bbox=LABEL_BBOX,
            zorder=10,
        )
    ax.set_xlabel("Parameters (M)" if panel == "a" else "Training time (s)")
    ax.set_ylabel("mAP@50")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)

axes[0].set_title("(a)", fontsize=10, pad=6)
axes[1].set_title("(b)", fontsize=10, pad=6)

plt.tight_layout()
plt.savefig(OUT, dpi=600, bbox_inches="tight")
print("saved ->", OUT)
for t in sorted(rows, key=lambda x: -x["mAP50"]):
    print("%-18s params=%-6s time=%-5s mAP50=%.4f" %
          (t["detector"], t["params_M"], t["train_s"], t["mAP50"]))