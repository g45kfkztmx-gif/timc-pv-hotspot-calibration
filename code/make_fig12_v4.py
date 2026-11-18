# -*- coding: utf-8 -*-
"""
make_fig12_v4.py — Fig.4 修复版 v4
上一次的 v3 改动后, v10n/v5s 标签仍然在同一 y 线上挤在左侧,
按到 mAP 排序的奇偶索引交错标签上下位置 (above/below marker), 避免遮挡.
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

# 按 mAP@50 降序(同图视觉直觉), 排序后用索引决定上下交错
rows_sorted = sorted(rows, key=lambda r: -r["mAP50"])

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

# 标签 x 偏移 (数据坐标): panel a 用相对小值, panel b 大值
DX = {"a": 0.6, "b": 90}
# 标签 y 偏移: 交替使用 + / - 让相邻检测器标签上下错开
DY_ABOVE = 0.014
DY_BELOW = -0.014

LABEL_BBOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=600)

for ax, xkey, panel in (
    (axes[0], "params_M", "a"),
    (axes[1], "train_s", "b"),
):
    for i, t in enumerate(rows_sorted):
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
        # 标签: 奇偶索引上下交错, Ours 标签强制放在标记下方
        if ours:
            dy = DY_BELOW
        else:
            dy = DY_ABOVE if (i % 2 == 0) else DY_BELOW
        ax.annotate(
            label,
            xy=(t[xkey], t["mAP50"]),
            xytext=(t[xkey] + DX[panel], t["mAP50"] + dy),
            fontsize=8.0,
            color=COLOR_OURS if ours else "#222222",
            fontweight="bold" if ours else "normal",
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
for t in rows_sorted:
    print("%-18s %-6s  %-5s  mAP50=%.4f" %
          (t["detector"], t["params_M"], t["train_s"], t["mAP50"]))