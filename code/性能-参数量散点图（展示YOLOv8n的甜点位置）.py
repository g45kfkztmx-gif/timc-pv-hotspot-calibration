import matplotlib.pyplot as plt

# ========== SCI全局样式设置（直接复制） ==========
plt.rcParams.update({
    "font.family": "Arial",        # SCI常用无衬线字体
    "font.size": 10,
    "axes.linewidth":1.0,
    "xtick.major.width":1.0,
    "ytick.major.width":1.0,
    "legend.frameon":False,        # 图例不要方框
})

# ---------------------- 填入你的实验数据（示例，替换成你真实mAP/参数量）
models = {
    "YOLOv5n": {"params":2.6, "map50":0.72, "marker":"o"},
    "YOLOv8n": {"params":3.2, "map50":0.85, "marker":"s"}, # sweet spot方块
    "YOLOv5s": {"params":9.1, "map50":0.76, "marker":"o"},
    "YOLOv8s": {"params":11.2, "map50":0.782, "marker":"o"},
}

# 帕累托前沿点，你原图虚线两个端点
pareto_x = [2.6, 3.2]
pareto_y = [0.72, 0.85]

fig, ax = plt.subplots(figsize=(6,5), dpi=300)

# 绘制各个模型散点
for name, info in models.items():
    ax.scatter(info["params"], info["map50"], marker=info["marker"], s=220, label=name, edgecolors="black", zorder=5)

# 画帕累托前沿虚线
ax.plot(pareto_x, pareto_y, linestyle="--", color="#555555", linewidth=1.5, label="Pareto frontier")

# sweet spot标注（替换掉手绘箭头，学术简洁版）
ax.annotate("Sweet spot", xy=(3.2,0.85), xytext=(4.0,0.86),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"), fontsize=9)

# 坐标轴
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("mAP50")
ax.set_xlim(1,12)
ax.set_ylim(0.68,0.90)

# 【关键SCI修改】隐藏上、右两条边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例放到图外面，解决原图点和文字挤在一起的问题
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)

plt.tight_layout()
plt.savefig("pareto_map_params.pdf", format="pdf", bbox_inches="tight") #矢量图输出
# plt.show()  # 取消注释可弹窗预览