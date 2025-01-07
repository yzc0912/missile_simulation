import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from carrier import Carrier  # 使用上面修改后的 Carrier 类

class MissileCarrierSimulation3D:
    def __init__(self, root):
        self.root = root
        self.root.title("Missile and Carrier Simulation 3D")

        # ========== 控制面板框架 ==========
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # ========== 载具数量 ==========
        ttk.Label(control_frame, text="Number of Carriers:").pack()
        self.carrier_count_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.carrier_count_var).pack()

        # ========== 导弹数量 ==========
        ttk.Label(control_frame, text="Number of Missiles:").pack()
        self.missile_count_var = tk.IntVar(value=1)
        ttk.Entry(control_frame, textvariable=self.missile_count_var).pack()

        # ========== 载具速度 ==========
        ttk.Label(control_frame, text="Carrier Speed:").pack()
        self.carrier_speed_var = tk.DoubleVar(value=0.0015)
        ttk.Entry(control_frame, textvariable=self.carrier_speed_var).pack()

        # ========== 导弹速度 ==========
        ttk.Label(control_frame, text="Missile Speed:").pack()
        self.missile_speed_var = tk.DoubleVar(value=0.03)
        ttk.Entry(control_frame, textvariable=self.missile_speed_var).pack()

        # ========== 最大步数 ==========
        ttk.Label(control_frame, text="Max Steps:").pack()
        self.max_steps_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.max_steps_var).pack()

        # ========== 箔条干扰出现次数 ==========
        ttk.Label(control_frame, text="Chaff Appear Times:").pack()
        self.chaff_appear_times_var = tk.IntVar(value=3)
        ttk.Entry(control_frame, textvariable=self.chaff_appear_times_var).pack()

        # ========== 角反射器出现次数 ==========
        ttk.Label(control_frame, text="Corner Reflector Appear Times:").pack()
        self.corner_reflector_appear_times_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.corner_reflector_appear_times_var).pack()

        # ========== 按钮 ==========
        ttk.Button(control_frame, text="Start Simulation", command=self.start_simulation).pack(pady=5)
        ttk.Button(control_frame, text="Reset Simulation", command=self.reset_simulation).pack(pady=5)

        # ========== 显示当前时间步 ==========
        self.time_step = 0
        self.time_label = ttk.Label(control_frame, text=f"Time Step: {self.time_step}")
        self.time_label.pack(pady=5)

        # ========== 初始化仿真状态 ==========
        self.is_running = False
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = None
        self.animation = None
        self.missiles = []
        self.carrier = None
        self.max_steps = None  # 最大步数

        # ========== 干扰相关变量 ==========
        # 箔条干扰（Chaff）
        self.chaff_appear_times = 0          # 总共允许出现多少次
        self.chaff_appear_count = 0          # 已出现次数
        self.is_chaff_active = False         # 当前是否有箔条在场
        self.chaff_timer = 0                # 当前箔条还剩多少步消失
        self.chaff_scatter = None           # 箔条散点对象

        # 角反射器（Corner Reflector）
        self.corner_reflector_appear_times = 0
        self.corner_reflector_appear_count = 0
        self.is_corner_reflector_active = False
        self.corner_reflector_timer = 0
        self.corner_reflector_scatter = None

        # corner_reflector_type = 'fixed' 或 'moving'
        self.corner_reflector_type = None   

        # 用于存储“固定角反射器”或“移动角反射器”的坐标数据：
        # - 如果是 fixed => 存的是绝对坐标，形状 (N, 3)
        # - 如果是 moving => 存的是 [ship_idx, offset_x, offset_y, 0.0]，形状 (N, 4)
        self.corner_reflector_positions = None  

        # ========== 创建Matplotlib图形的画布 ==========
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.create_canvas()

    def create_canvas(self):
        """创建并放置 Matplotlib 画布"""
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def start_simulation(self):
        """开始仿真"""
        if self.is_running:
            return

        self.is_running = True
        self.reset_simulation_data()
        self.time_step = 0
        self.update_time_label()

        # 读取用户设定的最大步数
        self.max_steps = self.max_steps_var.get()

        # 读取箔条、角反射器可出现的最大次数
        self.chaff_appear_times = self.chaff_appear_times_var.get()
        self.corner_reflector_appear_times = self.corner_reflector_appear_times_var.get()

        # 设置动画
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=100,
            blit=False,
            cache_frame_data=False
        )
        self.canvas.draw()

    def reset_simulation(self):
        """重置仿真"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.reset_simulation_data()
        self.canvas.draw()

    def reset_simulation_data(self):
        """重置仿真中使用的所有数据"""
        self.ax.clear()
        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(0, 60)
        self.ax.set_zlim(0, 60)
        self.ax.set_title("Missile and Carrier Simulation 3D")

        # 初始化载具
        carrier_count = self.carrier_count_var.get()
        carrier_speed = self.carrier_speed_var.get()
        self.carrier = Carrier(carrier_count, carrier_speed)

        # ========== 重置干扰相关变量 ==========

        # (1) 箔条
        self.chaff_appear_count = 0
        self.is_chaff_active = False
        self.chaff_timer = 0
        if self.chaff_scatter:
            self.chaff_scatter.remove()
        self.chaff_scatter = None

        # (2) 角反射器
        self.corner_reflector_appear_count = 0
        self.is_corner_reflector_active = False
        self.corner_reflector_timer = 0
        if self.corner_reflector_scatter:
            self.corner_reflector_scatter.remove()
        self.corner_reflector_scatter = None
        self.corner_reflector_type = None
        self.corner_reflector_positions = None

        # ========== 初始化导弹 ==========
        missile_count = self.missile_count_var.get()
        self.missiles = np.array([[0, 0, 15] for _ in range(missile_count)], dtype=np.float64)

        # ========== 初始化散点 ==========
        self.carrier_scatter = self.ax.scatter(
            *self.carrier.get_positions().T,
            c="blue",
            label="Carriers"
        )
        self.missile_scatter = self.ax.scatter(
            self.missiles[:, 0],
            self.missiles[:, 1],
            self.missiles[:, 2],
            c="red",
            label="Missiles"
        )

        self.ax.legend()

    def update(self, frame):
        """每次动画刷新时调用，用来更新仿真状态"""
        missile_speed = self.missile_speed_var.get()

        # 载具移动
        self.carrier.move()

        # 随机生成或移除箔条干扰 (Chaff)
        self.update_chaff()

        # 随机生成或移除角反射器
        self.update_corner_reflector()

        # 导弹朝载具方向移动
        carrier_positions = self.carrier.get_positions()
        for i in range(len(self.missiles)):
            target_idx = i % len(carrier_positions)  # 每个导弹对应一个目标（循环分配）
            direction = carrier_positions[target_idx] - self.missiles[i]
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.missiles[i] += missile_speed * direction / norm

        # 更新载具和导弹散点数据
        self.carrier_scatter._offsets3d = (*carrier_positions.T,)
        self.missile_scatter._offsets3d = (
            self.missiles[:, 0],
            self.missiles[:, 1],
            self.missiles[:, 2]
        )

        # 如果角反射器是 "moving"，计算其跟随船的绝对位置
        self.update_moving_reflector_positions()

        # 时间步自增
        self.time_step += 1
        self.update_time_label()

        # 判断是否达到最大步数
        if self.max_steps is not None and self.time_step >= self.max_steps:
            self.is_running = False
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None

    # ========== 箔条干扰逻辑 ==========

    def update_chaff(self):
        """
        随机生成或移除箔条干扰 (Chaff)。
        - 每次出现持续 50 个时间步
        - 总共可出现 chaff_appear_times 次
        """
        if self.is_chaff_active:
            self.chaff_timer -= 1
            if self.chaff_timer <= 0:
                if self.chaff_scatter:
                    self.chaff_scatter.remove()
                self.chaff_scatter = None
                self.is_chaff_active = False
        else:
            if self.chaff_appear_count < self.chaff_appear_times:
                # 1% 的概率触发
                if np.random.rand() < 0.01:
                    self.spawn_chaff()

    def spawn_chaff(self):
        """生成一次箔条干扰，持续 50 步"""
        self.is_chaff_active = True
        self.chaff_timer = 50
        self.chaff_appear_count += 1

        # 使用 carrier.generate_chaff() 生成绝对坐标
        chaff = self.carrier.generate_chaff()
        self.chaff_scatter = self.ax.scatter(
            chaff[:, 0],
            chaff[:, 1],
            chaff[:, 2],
            c="yellow",
            marker="o",
            label="Chaff"
        )
        self.ax.legend()

    # ========== 角反射器逻辑 ==========

    def update_corner_reflector(self):
        """
        随机生成或移除角反射器：
        - 每次出现持续 100 个时间步
        - 总共可出现 corner_reflector_appear_times 次
        - 出现时随机决定 "fixed" 或 "moving"
        """
        if self.is_corner_reflector_active:
            self.corner_reflector_timer -= 1
            if self.corner_reflector_timer <= 0:
                if self.corner_reflector_scatter:
                    self.corner_reflector_scatter.remove()
                self.corner_reflector_scatter = None
                self.is_corner_reflector_active = False
                self.corner_reflector_type = None
                self.corner_reflector_positions = None
        else:
            if self.corner_reflector_appear_count < self.corner_reflector_appear_times:
                # 1% 的概率在任意步触发
                if np.random.rand() < 0.01:
                    self.spawn_corner_reflector()

    def spawn_corner_reflector(self):
        """生成一次角反射器（fixed 或 moving），持续 100 步"""
        self.is_corner_reflector_active = True
        self.corner_reflector_timer = 100
        self.corner_reflector_appear_count += 1

        # 随机选择类型
        self.corner_reflector_type = np.random.choice(["fixed", "moving"])

        if self.corner_reflector_type == "fixed":
            # 固定角反射器 => 直接返回绝对坐标
            corner_reflectors = self.carrier.generate_fixed_corner_reflectors()
            color = "green"
            marker = "x"
            # 记录绝对坐标
            self.corner_reflector_positions = corner_reflectors
        else:
            # 移动角反射器 => 返回 [ship_idx, offset_x, offset_y, 0.0]
            corner_reflectors = self.carrier.generate_moving_corner_reflectors()
            color = "purple"
            marker = "^"
            # 记录相对信息
            self.corner_reflector_positions = corner_reflectors

        # 初始散点（固定直接用其坐标，移动的先把“相对信息”转为绝对坐标，再绘制）
        if self.corner_reflector_type == "fixed":
            xs = self.corner_reflector_positions[:, 0]
            ys = self.corner_reflector_positions[:, 1]
            zs = self.corner_reflector_positions[:, 2]
        else:
            # moving => 将 ship_idx 和 offset_x,y 转成绝对坐标
            carrier_positions = self.carrier.get_positions()
            abs_positions = []
            for row in self.corner_reflector_positions:
                ship_idx, offx, offy, _ = row
                ship_idx = int(ship_idx)
                sx, sy, sz = carrier_positions[ship_idx]
                abs_positions.append([sx + offx, sy + offy, sz])
            abs_positions = np.array(abs_positions)
            xs = abs_positions[:, 0]
            ys = abs_positions[:, 1]
            zs = abs_positions[:, 2]

        self.corner_reflector_scatter = self.ax.scatter(
            xs, ys, zs,
            c=color,
            marker=marker,
            label=f"{self.corner_reflector_type.capitalize()} Corner Reflectors"
        )
        self.ax.legend()

    def update_moving_reflector_positions(self):
        """
        如果角反射器是 "moving"，则每帧更新它的绝对坐标（跟随对应船移动）。
        """
        if (
            self.is_corner_reflector_active
            and self.corner_reflector_type == "moving"
            and self.corner_reflector_positions is not None
            and self.corner_reflector_scatter is not None
        ):
            carrier_positions = self.carrier.get_positions()
            abs_positions = []
            for row in self.corner_reflector_positions:
                ship_idx, offx, offy, _ = row
                ship_idx = int(ship_idx)
                sx, sy, sz = carrier_positions[ship_idx]
                abs_positions.append([sx + offx, sy + offy, sz])

            abs_positions = np.array(abs_positions)
            xs = abs_positions[:, 0]
            ys = abs_positions[:, 1]
            zs = abs_positions[:, 2]
            self.corner_reflector_scatter._offsets3d = (xs, ys, zs)

    # ========== 时间步显示 ==========

    def update_time_label(self):
        """更新界面上的时间步显示"""
        self.time_label.config(text=f"Time Step: {self.time_step}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MissileCarrierSimulation3D(root)
    root.mainloop()
