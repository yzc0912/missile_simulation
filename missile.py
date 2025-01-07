import numpy as np
import csv
import random
import math

class Missile:
    def __init__(self, missile_positions, sensor_categories=None):
        """
        :param missile_positions: (N, 3) 数组，表示所有导弹在三维空间的初始位置
        :param sensor_categories: 传感器类别列表，如 [0.1, 0.2, 0.3, 0.4, 0.6] 表示不同传感器的角度测量误差(度)
        """
        self.missiles = missile_positions
        self.num_missiles = self.missiles.shape[0]

        # 如果未指定传感器类别，就给个默认
        if sensor_categories is None:
            self.sensor_categories = [0.1, 0.2, 0.3, 0.4, 0.6]
        else:
            self.sensor_categories = sensor_categories

        # ☆ 要求 4：同一枚导弹的测量误差固定不变
        #   这里为每枚导弹随机选取一个“角度测量误差”
        self.missile_error_degs = []
        for _ in range(self.num_missiles):
            chosen_error = random.choice(self.sensor_categories)
            self.missile_error_degs.append(chosen_error)

        # 用于输出到 CSV 的测量数据；每元素是一行（对应一个 time_step、一个 missile）
        self.measurement_data = []

    def generate_sensor_measurements(
        self,
        carriers_positions,
        chaff_positions,
        corner_positions,
        time_step
    ):
        """
        对“船舶 + 干扰”视为统一的潜在目标，每个导弹在当前时间步进行测量。
        
        本需求中：
        1) 不区分目标类型，故不输出目标类型；
        2) 每个时间步 -> 对每个导弹输出一行，行内依次列出各目标的测量数据；
        3) 置信度用 0~1 的高斯随机数；
        4) 相同导弹的误差固定不变。
        """
        # 汇总所有潜在目标（不区分类型）
        all_targets = []
        all_targets.extend(carriers_positions)   # M 条船
        all_targets.extend(chaff_positions)      # 若干箔条
        all_targets.extend(corner_positions)     # 若干角反射器

        # 对每枚导弹生成一行数据
        for missile_id, missile_pos in enumerate(self.missiles):
            # 取得该导弹的固定测量误差（度）
            sensor_error_deg = self.missile_error_degs[missile_id]
            sensor_error_rad = math.radians(sensor_error_deg)

            # sub_result 用来收集该导弹对每个目标的测量结果
            sub_result = []

            for target_pos in all_targets:
                dx = target_pos[0] - missile_pos[0]
                dy = target_pos[1] - missile_pos[1]
                dz = target_pos[2] - missile_pos[2]

                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                # 真实方位角
                azimuth = math.atan2(dy, dx)

                # 加入测量误差
                angle_error = random.uniform(-sensor_error_rad, sensor_error_rad)
                measured_azimuth = azimuth + angle_error

                # 反投影到二维平面
                x_meas = missile_pos[0] + distance * math.cos(measured_azimuth)
                y_meas = missile_pos[1] + distance * math.sin(measured_azimuth)

                # ☆ 要求 3：置信度使用 0~1 高斯随机
                confidence = random.gauss(0.5, 0.2)  # 均值 0.5，方差 0.2
                confidence = min(max(confidence, 0.0), 1.0)  # 截断到 [0, 1]

                # measurement_scatter = 这里简单直接用 sensor_error_deg
                # 也可以根据距离/其他模型定义
                measurement_scatter = sensor_error_deg

                # 把单个目标测量结果 (x_meas, y_meas, scatter, confidence) 连到 sub_result
                sub_result.extend([x_meas, y_meas, measurement_scatter, confidence])

            # 组装一行：[time_step, missile_id, x_meas1, y_meas1, scatter1, conf1, x_meas2, y_meas2, scatter2, conf2, ...]
            row = [time_step, missile_id] + sub_result
            self.measurement_data.append(row)

    def export_to_csv(self, filename="measurement_data.csv"):
        """
        将 measurement_data 导出成 CSV。
        每个 time_step + missile 的结果占一行，对应若干目标数据的列。
        
        注意：若每帧目标数不一样，则列数会随之变化，CSV 列数也会不一致。
        若需固定列数，应在此处做补空白 / 截断处理。
        """
        if not self.measurement_data:
            return

        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # 由于目标数可变，列数也可变，这里示例性地不写表头
            # 如果需要表头，可以在此根据最大目标数构造列名
            for row in self.measurement_data:
                writer.writerow(row)

        print(f"Measurement data exported to {filename}.")
