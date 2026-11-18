# -*- coding: utf-8 -*-
"""重画 Figure 4：数值以对账后 JSON 为准，标签去重叠，300 DPI。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\YJP\AppData\Local\Qianwen\User Data\qwen-agent-workspaces\runs\3612b721-61de-458e-89c4-b16c3f3594e1\2026-11-17T09-00-25-475Z\outputs"
N = [20, 50, 100, 150, 197]
sup_m = [0.8233, 0.8401, 0.9055, 0.9290, 0.9076]
ts_m  = [0.7904, 0.9033, 0.9415, 0.9328, 0.9325]
our_m = [0.8065, 0.7311, 0.8045, 0.8020, 0.8020]
ref_m = [0.7994, 0.7886, 0.7590, 0.8003, 0.8003]
sup_r = [0.7861, 0.8029, 0.8495, 0.9176, 0.9129]
ts_r  = [0.7276, 0.8280, 0.9104, 0.8925, 0.9091]
our_r = [0.7814, 0.7491, 0.7778, 0.7921, 0.7921]
ref_r = [0.7885, 0.7921, 0.7660, 0.7954, 0.7954]

series = [("Direct supervision", sup_m, sup_r, "#d62728", "s", "-"),
          ("Teacher-student", ts_m, ts_r, "#1f77b4", "^", "-"),
          ("Calibration-guided (reported)", our_m, our_r, "#2ca02c", "o", "-"),
          ("+ refinement (optional)", ref_m, ref_r, "#9467bd", "o", "--")]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, col, title, ylab in [(axes[0], 1, "(a) Test mAP@50", "mAP@50"),
                             (axes[1], 2, "(b) Test recall", "recall")]:
    placed = {x: [] for x in N}
    for name, m, r, c, mk, ls in series:
        y = m if col == 1 else r
        ax.plot(N, y, color=c, marker=mk, ls=ls, lw=1.8, ms=6, label=name)
        for x, v in zip(N, y):
            # 防碰撞：同 x 列先向上堆叠，超出上限则向下堆叠；最小间距 0.016
            stack = placed[x]
            cand = v + 0.013
            while any(abs(cand - p) < 0.016 for p in stack):
                cand += 0.016
            if cand > 0.985:
                cand = v - 0.013
                while any(abs(cand - p) < 0.016 for p in stack):
                    cand -= 0.016
            stack.append(cand)
            ax.annotate("%.3f" % v, (x, v), xytext=(x, cand),
                        ha="center", va="center", fontsize=8.5, color=c,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.5,
                                        shrinkA=0, shrinkB=3) if abs(cand - v) > 0.02 else None)
    ax.set_xticks(N)
    ax.set_xlabel("Number of manually annotated images N")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.68, 1.0)
axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
fig.suptitle("Annotation budget and calibration stability (values reconciled to Table 3(b))", y=1.00)
fig.tight_layout()
p = os.path.join(OUT, "figure4_redrawn.png")
fig.savefig(p, dpi=300, bbox_inches="tight")
print("saved", p)
