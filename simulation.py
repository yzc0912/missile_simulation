import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import re

# 读取 CSV 数据，处理列数不一致问题
filename = 'measurement_data_vis.csv'
with open(filename, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

# 转换为 DataFrame，并处理列名与数值
df = pd.DataFrame(data)
df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')

# 提取所有目标的前缀列表
target_prefixes = set()
for col in df.columns:
    if col is not None and isinstance(col, str):  # 确保列名不为空且为字符串
        match = re.match(r'(Target\d+)_x', col)
        if match:
            target_prefixes.add(match.group(1))
target_prefixes = sorted(list(target_prefixes), key=lambda x: int(re.findall(r'\d+', x)[0]))

# 获取所有时间步
time_steps = sorted(df['TimeStep'].dropna().unique())

# 设置颜色映射
colors = plt.cm.get_cmap('tab10', len(target_prefixes))

# 创建图形和轴
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Current Position and Variance Ellipses')
ax.grid(True)

# 设置固定的坐标轴范围（根据实际数据范围调整）
ax.set_xlim(15, 35)  # 设置 X 轴范围
ax.set_ylim(15, 35)  # 设置 Y 轴范围

# 初始化图形元素
scatter_plots = {}
ellipses = {}
for i, tp in enumerate(target_prefixes):
    scatter_plots[tp], = ax.plot([], [], 'o', color=colors(i), label=tp, markersize=5)
    ellipses[tp] = Ellipse((0, 0), width=0, height=0, angle=0,
                           edgecolor=colors(i), facecolor='none', lw=2)
    ax.add_patch(ellipses[tp])


# 更新函数
def update(frame):
    timestep = time_steps[frame]
    current_data = df[df['TimeStep'] == timestep]
    # 清除之前的置信度文本
    for txt in ax.texts:
        txt.remove()

    for i, tp in enumerate(target_prefixes):
        if f"{tp}_x" in current_data.columns:
            row = current_data.iloc[0]
            x = row.get(f"{tp}_x", np.nan)
            y = row.get(f"{tp}_y", np.nan)
            major = row.get(f"{tp}_MajorAxis", np.nan)
            minor = row.get(f"{tp}_MinorAxis", np.nan)
            angle = row.get(f"{tp}_AngleRad", np.nan)
            confidence = row.get(f"{tp}_Confidence", np.nan)

            # 更新当前点位置
            if not np.isnan(x) and not np.isnan(y):
                scatter_plots[tp].set_data([x], [y])  # 只显示当前位置
            else:
                scatter_plots[tp].set_data([], [])

            # 更新误差椭圆参数
            if not np.isnan(major) and not np.isnan(minor) and not np.isnan(angle):
                ellipses[tp].set_visible(True)
                ellipses[tp].center = (x, y)
                ellipses[tp].width = 2 * major
                ellipses[tp].height = 2 * minor
                ellipses[tp].angle = angle * 180 / np.pi
            else:
                ellipses[tp].set_visible(False)

            # 显示置信度
            if not np.isnan(confidence):
                ax.text(x, y, f'{confidence:.2f}', color=colors(i), fontsize=9)

    ax.set_title(f'Timestep {timestep}')
    return list(scatter_plots.values()) + list(ellipses.values())

# 创建动画
anim = FuncAnimation(fig, update, frames=len(time_steps), interval=50, blit=False, repeat=True)

plt.show()
