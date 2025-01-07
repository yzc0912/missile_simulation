import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 从 carrier.py 和 missile.py 引入相关类
from carrier import Carrier
from missile import Missile

class MissileCarrierSimulation3D:
    def __init__(self, root):
        self.root = root
        self.root.title("Missile and Carrier Simulation 3D")

        # ========== 控制面板 ==========
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="Number of Carriers:").pack()
        self.carrier_count_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.carrier_count_var).pack()

        ttk.Label(control_frame, text="Number of Missiles:").pack()
        self.missile_count_var = tk.IntVar(value=1)
        ttk.Entry(control_frame, textvariable=self.missile_count_var).pack()

        ttk.Label(control_frame, text="Carrier Speed:").pack()
        self.carrier_speed_var = tk.DoubleVar(value=0.0015)
        ttk.Entry(control_frame, textvariable=self.carrier_speed_var).pack()

        ttk.Label(control_frame, text="Missile Speed:").pack()
        self.missile_speed_var = tk.DoubleVar(value=0.03)
        ttk.Entry(control_frame, textvariable=self.missile_speed_var).pack()

        ttk.Label(control_frame, text="Max Steps:").pack()
        self.max_steps_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.max_steps_var).pack()

        ttk.Label(control_frame, text="Chaff Appear Times:").pack()
        self.chaff_appear_times_var = tk.IntVar(value=3)
        ttk.Entry(control_frame, textvariable=self.chaff_appear_times_var).pack()

        ttk.Label(control_frame, text="Corner Reflector Appear Times:").pack()
        self.corner_reflector_appear_times_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.corner_reflector_appear_times_var).pack()

        ttk.Button(control_frame, text="Start Simulation", command=self.start_simulation).pack(pady=5)
        ttk.Button(control_frame, text="Reset Simulation", command=self.reset_simulation).pack(pady=5)

        self.time_step = 0
        self.time_label = ttk.Label(control_frame, text=f"Time Step: {self.time_step}")
        self.time_label.pack(pady=5)

        # ========== Matplotlib 3D画布 ==========
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = None
        self.animation = None
        self.create_canvas()

        # ========== 仿真对象与数据存储 ==========
        self.is_running = False
        self.carrier = None
        self.missile = None  # Missile对象
        self.missiles = []   # 导弹位置数组
        self.max_steps = None

        # 载具 + 导弹 散点
        self.carrier_scatter = None
        self.missile_scatter = None

        # 干扰相关
        self.chaff_appear_times = 0
        self.chaff_appear_count = 0
        self.is_chaff_active = False
        self.chaff_timer = 0
        self.chaff_scatter = None
        # 记录当前帧箔条干扰的绝对坐标，用于给导弹测量
        self.chaff_positions = np.array([])

        self.corner_reflector_appear_times = 0
        self.corner_reflector_appear_count = 0
        self.is_corner_reflector_active = False
        self.corner_reflector_timer = 0
        self.corner_reflector_scatter = None
        self.corner_reflector_type = None
        self.corner_reflector_positions = None  # fixed => (N,3); moving => (N,4)
        # 用于保存本帧的角反射器绝对坐标，以便导弹测量
        self.current_corner_abs_positions = np.array([])

    def create_canvas(self):
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def start_simulation(self):
        if self.is_running:
            return
        self.is_running = True

        self.reset_simulation_data()
        self.time_step = 0
        self.update_time_label()

        self.max_steps = self.max_steps_var.get()
        self.chaff_appear_times = self.chaff_appear_times_var.get()
        self.corner_reflector_appear_times = self.corner_reflector_appear_times_var.get()

        self.animation = FuncAnimation(self.fig, self.update, interval=100, blit=False, cache_frame_data=False)
        self.canvas.draw()

    def reset_simulation(self):
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None

        # 在这里也可以选择将测量数据导出到CSV
        if self.missile:
            self.missile.export_to_csv("measurement_data.csv")

        self.reset_simulation_data()
        self.canvas.draw()

    def reset_simulation_data(self):
        self.ax.clear()
        self.ax.set_xlim(0, 60)
        self.ax.set_ylim(0, 60)
        self.ax.set_zlim(0, 60)
        self.ax.set_title("Missile and Carrier Simulation 3D")

        # (1) 初始化载具
        carrier_count = self.carrier_count_var.get()
        carrier_speed = self.carrier_speed_var.get()
        self.carrier = Carrier(carrier_count, carrier_speed)

        # (2) 初始化导弹
        missile_count = self.missile_count_var.get()
        # 示例：所有导弹初始都在 (0,0,15) 附近
        self.missiles = np.array([[0, 0, 15] for _ in range(missile_count)], dtype=np.float64)

        # 创建 Missile 对象，后面用来采集测量数据
        self.missile = Missile(self.missiles)

        # (3) 重置散点对象
        if self.carrier_scatter:
            self.carrier_scatter.remove()
        if self.missile_scatter:
            self.missile_scatter.remove()

        self.carrier_scatter = self.ax.scatter(*self.carrier.get_positions().T, c="blue", label="Carriers")
        self.missile_scatter = self.ax.scatter(self.missiles[:,0], self.missiles[:,1], self.missiles[:,2],
                                               c="red", label="Missiles")

        # (4) 重置干扰相关
        self.chaff_appear_count = 0
        self.is_chaff_active = False
        self.chaff_timer = 0
        if self.chaff_scatter:
            self.chaff_scatter.remove()
        self.chaff_scatter = None
        self.chaff_positions = np.array([])

        self.corner_reflector_appear_count = 0
        self.is_corner_reflector_active = False
        self.corner_reflector_timer = 0
        if self.corner_reflector_scatter:
            self.corner_reflector_scatter.remove()
        self.corner_reflector_scatter = None
        self.corner_reflector_type = None
        self.corner_reflector_positions = None
        self.current_corner_abs_positions = np.array([])

        self.ax.legend()

    def update(self, frame):
        # 1) 载具移动
        self.carrier.move()

        # 2) 随机生成或移除干扰
        self.update_chaff()
        self.update_corner_reflector()

        # 3) 更新导弹位置（朝载具前进示例）
        missile_speed = self.missile_speed_var.get()
        carrier_positions = self.carrier.get_positions()
        for i in range(len(self.missiles)):
            target_idx = i % len(carrier_positions)
            direction = carrier_positions[target_idx] - self.missiles[i]
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.missiles[i] += missile_speed * direction / norm

        # 4) 更新载具和导弹散点
        self.carrier_scatter._offsets3d = (*carrier_positions.T,)
        self.missile_scatter._offsets3d = (self.missiles[:,0], self.missiles[:,1], self.missiles[:,2])

        # 5) 如果是 moving corner，就计算本帧的绝对坐标，更新散点
        self.update_moving_reflector_positions()

        # 6) ★★ 让导弹执行一次“测量” ★★
        #    - 船的当前位置：carrier_positions
        #    - 箔条干扰：self.chaff_positions（如果无箔条则为空数组）
        #    - 角反射器：self.current_corner_abs_positions（如果无角反射器则为空数组）
        self.missile.generate_sensor_measurements(
            carriers_positions=carrier_positions,
            chaff_positions=self.chaff_positions,
            corner_positions=self.current_corner_abs_positions,
            time_step=self.time_step
        )

        # 7) 时间步+1，检查是否到达最大步数
        self.time_step += 1
        self.update_time_label()

        if self.max_steps is not None and self.time_step >= self.max_steps:
            self.is_running = False
            if self.animation:
                self.animation.event_source.stop()
                self.animation = None
            # 仿真结束 => 导出测量数据
            self.missile.export_to_csv("measurement_data.csv")

    # ========== 箔条干扰（Chaff） ==========

    def update_chaff(self):
        """
        随机生成或移除箔条干扰：
        - 每次出现持续 50 个时间步
        - 总共可出现 chaff_appear_times 次
        """
        if self.is_chaff_active:
            self.chaff_timer -= 1
            if self.chaff_timer <= 0:
                # 移除散点
                if self.chaff_scatter:
                    self.chaff_scatter.remove()
                self.chaff_scatter = None
                self.is_chaff_active = False
                self.chaff_positions = np.array([])
        else:
            # 如果当前没有箔条，且出现次数还不超限，则有一定概率生成一次
            if self.chaff_appear_count < self.chaff_appear_times:
                if np.random.rand() < 0.01:
                    self.spawn_chaff()

    def spawn_chaff(self):
        self.is_chaff_active = True
        self.chaff_timer = 50
        self.chaff_appear_count += 1

        # 由 carrier.generate_chaff() 生成箔条的绝对坐标
        chaff = self.carrier.generate_chaff()
        self.chaff_positions = chaff  # 存储，给导弹测量用
        self.chaff_scatter = self.ax.scatter(chaff[:,0], chaff[:,1], chaff[:,2],
                                             c="yellow", marker="o", label="Chaff")
        self.ax.legend()

    # ========== 角反射器（Corner Reflector） ==========

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
                self.current_corner_abs_positions = np.array([])
        else:
            if self.corner_reflector_appear_count < self.corner_reflector_appear_times:
                if np.random.rand() < 0.01:
                    self.spawn_corner_reflector()

    def spawn_corner_reflector(self):
        self.is_corner_reflector_active = True
        self.corner_reflector_timer = 100
        self.corner_reflector_appear_count += 1

        self.corner_reflector_type = np.random.choice(["fixed", "moving"])
        if self.corner_reflector_type == "fixed":
            corner_positions = self.carrier.generate_fixed_corner_reflectors()
            color = "green"
            marker = "x"
            # 固定角反射器直接是绝对坐标
            self.corner_reflector_positions = corner_positions
        else:
            corner_positions = self.carrier.generate_moving_corner_reflectors()
            color = "purple"
            marker = "^"
            # 移动角反射器保存相对数据
            self.corner_reflector_positions = corner_positions

        # 初次绘制散点
        if self.corner_reflector_type == "fixed":
            self.current_corner_abs_positions = self.corner_reflector_positions
        else:
            # 先根据船位置 + offset 计算一次绝对坐标
            self.current_corner_abs_positions = self._compute_moving_corner_abs_positions()

        xs = self.current_corner_abs_positions[:,0]
        ys = self.current_corner_abs_positions[:,1]
        zs = self.current_corner_abs_positions[:,2]

        self.corner_reflector_scatter = self.ax.scatter(xs, ys, zs,
                                                        c=color, marker=marker,
                                                        label=f"{self.corner_reflector_type.capitalize()} Corner")
        self.ax.legend()

    def update_moving_reflector_positions(self):
        """如果是 moving，就实时更新它的绝对坐标 + 散点。"""
        if self.is_corner_reflector_active and self.corner_reflector_type == "moving":
            self.current_corner_abs_positions = self._compute_moving_corner_abs_positions()
            xs = self.current_corner_abs_positions[:,0]
            ys = self.current_corner_abs_positions[:,1]
            zs = self.current_corner_abs_positions[:,2]
            if self.corner_reflector_scatter:
                self.corner_reflector_scatter._offsets3d = (xs, ys, zs)

    def _compute_moving_corner_abs_positions(self):
        """
        根据 self.corner_reflector_positions (N,4) => [ship_idx, offset_x, offset_y, 0]，
        结合当前船位置，计算出绝对坐标。
        """
        carrier_positions = self.carrier.get_positions()
        abs_positions = []
        for row in self.corner_reflector_positions:
            ship_idx, offx, offy, _ = row
            ship_idx = int(ship_idx)
            sx, sy, sz = carrier_positions[ship_idx]
            abs_positions.append([sx + offx, sy + offy, sz])
        return np.array(abs_positions)

    # ========== 时间步显示 ==========

    def update_time_label(self):
        self.time_label.config(text=f"Time Step: {self.time_step}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MissileCarrierSimulation3D(root)
    root.mainloop()
