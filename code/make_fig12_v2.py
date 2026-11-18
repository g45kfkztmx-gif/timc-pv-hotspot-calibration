# -*- coding: utf-8 -*-
"""
make_fig12_v2.py
Fig.4 (fig12_efficiency_merged.png) 重绘: 7 个检测器
(a) Parameters vs mAP@50   (b) Training time vs mAP@50
Times New Roman, 600 dpi, 通栏双面板, 本文 YOLOv8n 红色高亮。
输入: detector_comparison_v2.json (由 summarize_v2.py 生成)
输出: figs_paper/fig12_efficiency_merged.png (直接覆盖论文用图)
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

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=600)

for ax, xkey, xlabel in (
    (axes[0], "params_M", "Parameters (M)"),
    (axes[1], "train_s", "Training time (s)"),
):
    for t in rows:
        ours = "ours" in t["detector"]
        ax.scatter(
            t[xkey], t["mAP50"],
            marker="s" if ours else "o",
            s=70 if ours else 46,
            color=COLOR_OURS if ours else COLOR_OTHER,
            edgecolors="black", linewidths=0.6, zorder=5,
        )
        dx = 0.12 if xkey == "params_M" else 25
        ax.annotate(
            t["detector"].replace(" (ours)", ""),
            xy=(t[xkey], t["mAP50"]),
            xytext=(t[xkey] + dx, t["mAP50"] - 0.004),
            fontsize=7.5,
            color=COLOR_OURS if ours else "#333333",
        )
    ax.set_xlabel(xlabel)
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
