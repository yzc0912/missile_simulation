import numpy as np
import csv
import random
import math

class Missile:
    MAX_TARGETS = 20   # 每个传感器测量的最大目标数
    NUM_SENSORS = 5    # 每个导弹拥有的传感器数量

    def __init__(self, missile_positions, sensor_categories=None):
        """
        :param missile_positions: (N, 3) 数组，表示所有导弹在三维空间的初始位置
        :param sensor_categories: 传感器类别列表, 例如 [0.1, 0.2, 0.3, 0.4, 0.6]
                                 表示不同的角度测量误差(度)可供选用
        """
        self.missiles = missile_positions
        self.num_missiles = self.missiles.shape[0]

        # 如果未指定传感器类别，就给一个默认列表
        if sensor_categories is None:
            self.sensor_categories = [0.1, 0.2, 0.3, 0.4, 0.6]
        else:
            self.sensor_categories = sensor_categories

        # 对于每枚导弹，随机选取 "NUM_SENSORS" 个互不相同的误差，分别对应 5 个传感器
        # 注意：需保证 sensor_categories 的长度 >= NUM_SENSORS，否则 sample 会报错
        self.missile_sensor_errors = []  # 形状: [ [err_sen1, err_sen2, ..., err_sen5], [...], ... ]

        for _ in range(self.num_missiles):
            chosen_errors = random.sample(self.sensor_categories, self.NUM_SENSORS)
            self.missile_sensor_errors.append(chosen_errors)

        # 用于输出到 CSV 的测量数据（每元素是一行：time_step, missile_id, sensor_id, ...）
        self.measurement_data = []

    def generate_sensor_measurements(
        self,
        carriers_positions,
        chaff_positions,
        corner_positions,
        time_step
    ):
        """
        在当前 time_step，让每枚导弹的 5 个传感器分别对所有目标(船+干扰)进行测量。
        
        1) 不区分目标类型于输出，但内部依旧识别(ship/chaff/corner) 给不同置信度。
        2) 每个时间步，对每个导弹输出 5 行(对应 5 个传感器)。
        3) 每行的目标测量内容与之前类似：最多记录 MAX_TARGETS 个目标，每目标 8 个字段。
        4) 置信度在 [0,1] 内的高斯随机；同一传感器的测量误差固定不变。
        
        输出格式 (每行)：
        [
          time_step,
          missile_id,
          sensor_id,        # 新增，用于区分同导弹上的 5 个传感器
          x_meas_目标1, y_meas_目标1, z_meas_目标1, major_axis_目标1, minor_axis_目标1, angle_rad_目标1, scatter_目标1, confidence_目标1,
          x_meas_目标2, y_meas_目标2, z_meas_目标2, ...
          ...
        ]
        """

        # 将所有可能目标(船 + 箔条 + 角反射器)汇总
        detection_prob = getattr(self, "detection_prob", 0.9)

        all_targets = []
        for pos in carriers_positions:
            all_targets.append((pos, "ship"))
        for pos in chaff_positions:
            all_targets.append((pos, "chaff"))
        for pos in corner_positions:
            all_targets.append((pos, "corner"))

        # 对每枚导弹进行测量
        for missile_id, missile_pos in enumerate(self.missiles):

            # 对该导弹的每个传感器都做测量
            for sensor_id in range(self.NUM_SENSORS):
                sensor_error_deg = self.missile_sensor_errors[missile_id][sensor_id]

                sub_result = []
                target_count = 0

                for (target_pos, target_type) in all_targets:
                    if target_count >= self.MAX_TARGETS:
                        break  # 达到最大目标数量

                    # (A) 是否探测到目标
                    if random.random() > detection_prob:
                        # 探测失败 => 填充 None
                        sub_result.extend([None, None, None, None, None, None, None, None])
                    else:
                        # 1) 计算导弹->目标的 3D 距离 & 真实方位角 / 仰角
                        dx = target_pos[0] - missile_pos[0]
                        dy = target_pos[1] - missile_pos[1]
                        dz = target_pos[2] - missile_pos[2]

                        r_true = math.sqrt(dx*dx + dy*dy + dz*dz)
                        az_true = math.atan2(dy, dx)
                        el_true = 0.0

                        xy_dist = math.sqrt(dx*dx + dy*dy)
                        if xy_dist > 1e-8:
                            el_true = math.atan2(dz, xy_dist)

                        # 2) 生成角度误差
                        r_rand = random.uniform(0, sensor_error_deg)
                        phi = random.uniform(0, 2*math.pi)

                        error_az_deg = r_rand * math.cos(phi)
                        error_el_deg = r_rand * math.sin(phi)

                        # 转弧度
                        error_az_rad = math.radians(error_az_deg)
                        error_el_rad = math.radians(error_el_deg)

                        az_meas = az_true + error_az_rad
                        el_meas = el_true + error_el_rad

                        # 3) 反投影到 (x_meas, y_meas, z_meas)
                        x_meas = missile_pos[0] + r_true * math.cos(el_meas) * math.cos(az_meas)
                        y_meas = missile_pos[1] + r_true * math.cos(el_meas) * math.sin(az_meas)
                        z_meas = missile_pos[2] + r_true * math.sin(el_meas)

                        # 4) 误差传播(雅可比 + 协方差)
                        dx_daz = -r_true * math.cos(el_true) * math.sin(az_true)
                        dx_del = -r_true * math.sin(el_true) * math.cos(az_true)
                        dy_daz =  r_true * math.cos(el_true) * math.cos(az_true)
                        dy_del = -r_true * math.sin(el_true) * math.sin(az_true)

                        J = np.array([
                            [dx_daz, dx_del],
                            [dy_daz, dy_del]
                        ])

                        sigma_az_rad = math.radians(sensor_error_deg)
                        sigma_el_rad = math.radians(sensor_error_deg)
                        Cov_ae = np.diag([sigma_az_rad**2, sigma_el_rad**2])

                        Cov_xy = J @ Cov_ae @ J.T
                        eigvals, eigvecs = np.linalg.eig(Cov_xy)

                        idx = np.argsort(eigvals)[::-1]
                        eigvals = eigvals[idx]

                        major_axis = 0.0
                        minor_axis = 0.0
                        if eigvals[0] > 0:
                            major_axis = 2.0 * math.sqrt(eigvals[0])
                        if eigvals[1] > 0:
                            minor_axis = 2.0 * math.sqrt(eigvals[1])

                        v_major = eigvecs[:, 0]
                        angle_rad = math.atan2(v_major[1], v_major[0])

                        measurement_scatter = sensor_error_deg

                        # 5) 置信度（不同目标类型 => 不同分布）
                        if target_type == "ship":
                            confidence = random.gauss(0.8, 0.1)
                        elif target_type == "chaff":
                            confidence = random.gauss(0.3, 0.2)
                        else:  # corner
                            confidence = random.gauss(0.3, 0.2)

                        confidence = max(0.0, min(1.0, confidence))

                        sub_result.extend([
                            x_meas, y_meas, z_meas,
                            major_axis, minor_axis, angle_rad,
                            measurement_scatter, confidence
                        ])

                    target_count += 1

                # 若目标数不足 MAX_TARGETS，补 None
                if target_count < self.MAX_TARGETS:
                    padding = [None] * 8 * (self.MAX_TARGETS - target_count)
                    sub_result += padding

                # 组装行 => [time_step, missile_id, sensor_id, ...sub_result...]
                row = [time_step, missile_id, sensor_id] + sub_result
                self.measurement_data.append(row)

    def export_to_csv(self, filename="measurement_data.csv"):
        """
        导出 CSV，每个 time_step + missile + sensor 占一行
        对应固定 MAX_TARGETS 个目标（每目标 8 字段）。
        """
        if not self.measurement_data:
            return

        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # 生成表头
            headers = ["TimeStep", "MissileID", "SensorID"]
            for i in range(1, Missile.MAX_TARGETS + 1):
                headers.extend([
                    f"Target{i}_x", f"Target{i}_y", f"Target{i}_z",
                    f"Target{i}_MajorAxis", f"Target{i}_MinorAxis",
                    f"Target{i}_AngleRad", f"Target{i}_Scatter",
                    f"Target{i}_Confidence"
                ])

            writer.writerow(headers)

            for row in self.measurement_data:
                writer.writerow(row)

        print(f"Measurement data exported to {filename}.")
