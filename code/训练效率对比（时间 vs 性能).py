import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========== SCI全局样式设置 ==========
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.frameon": False,
})

fig, ax = plt.subplots(figsize=(6, 5))

detectors_time = [
    ('YOLOv5n', 887, 0.7210, 2.65, 'tab:blue'),
    ('YOLOv8n', 779, 0.8494, 3.2, 'tab:orange'),
    ('YOLOv5s', 1399, 0.7581, 9.12, 'tab:green'),
    ('YOLOv8s', 1648, 0.7815, 11.14, 'tab:red'),
]

for name, time, mAP50, params, color in detectors_time:
    size = 200 + params * 30  # 气泡大小反映参数量
    marker = 's' if 'v8n' in name else 'o'
    edgewidth = 3 if 'v8n' in name else 1.5
    ax.scatter(time, mAP50, s=size, color=color, marker=marker,
               edgecolor='black', linewidth=edgewidth, alpha=0.7, label=name)
    
    ax.text(time + 50, mAP50 + 0.008, name, fontsize=9, ha='left',
            fontweight='bold' if 'v8n' in name else 'normal')

ax.axhline(y=0.8494, color='#555555', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axvline(x=779, color='#555555', linestyle='--', linewidth=1.5, alpha=0.5)

ax.set_xlabel('Training Time (seconds, 60 epochs)')
ax.set_ylabel('mAP50')
ax.set_xlim(700, 1750)
ax.set_ylim(0.70, 0.87)

# 隐藏上、右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例放到图外右侧，避免拥挤
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), title='Bubble size = Params', title_fontsize=8)

plt.tight_layout()
plt.savefig('detector_training_efficiency.png', dpi=300, bbox_inches='tight')
plt.savefig('detector_training_efficiency.pdf', format='pdf', bbox_inches='tight')
print('Done: detector_training_efficiency.png / .pdf')