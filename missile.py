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

        # 如果未指定传感器类别，就给一个默认列表，确保有足够的唯一值
        if sensor_categories is None:
            self.sensor_categories = [0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            # 过滤传感器类别，确保在0.2到1.0度之间
            self.sensor_categories = [err for err in sensor_categories if 0.2 <= err <= 1.0]
            if len(set(self.sensor_categories)) < self.NUM_SENSORS:
                raise ValueError(f"sensor_categories必须包含至少{self.NUM_SENSORS}个在0.2到1.0度之间的唯一值")

        # 对于每枚导弹，随机选取 "NUM_SENSORS" 个互不相同的误差，分别对应 5 个传感器
        self.missile_sensor_errors = []  # 形状: [ [err_sen1, err_sen2, ..., err_sen5], [...], ... ]
        for _ in range(self.num_missiles):
            chosen_errors = random.sample(self.sensor_categories, self.NUM_SENSORS)
            self.missile_sensor_errors.append(chosen_errors)

        # 用于输出到 CSV 的测量数据（每元素是一行：time_step, missile_id, sensor_id, ...）
        self.measurement_data = []

        # 用于记录船舶真实位置数据
        self.ship_locations_data = []
        self.max_ships = 0  # 动态追踪最大船舶数量

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

        # 更新最大船舶数量
        num_ships = carriers_positions.shape[0]
        if num_ships > self.max_ships:
            self.max_ships = num_ships

        # 记录当前时间步的船舶真实位置
        ship_row = [time_step]
        for pos in carriers_positions:
            ship_row.extend([pos[0], pos[1], pos[2]])
        # 如果当前时间步船舶数量少于之前的最大数量，填充 None
        if num_ships < self.max_ships:
            padding_ships = self.max_ships - num_ships
            ship_row.extend([None, None, None] * padding_ships)
        self.ship_locations_data.append(ship_row)

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

    def export_to_csv(self, measurement_filename="measurement_data.csv", ship_loc_filename="ship_loc.csv"):
        """
        导出 CSV 文件：
        1. measurement_data.csv - 包含每个 time_step, missile_id, sensor_id 的测量数据。
        2. ship_loc.csv - 包含每个 time_step 所有船舶的真实位置。
        
        measurement_data.csv 格式:
        [TimeStep, MissileID, SensorID, Target1_x, Target1_y, Target1_z, Target1_MajorAxis, Target1_MinorAxis, Target1_AngleRad, Target1_Scatter, Target1_Confidence, ..., Target20_x, Target20_y, Target20_z, Target20_MajorAxis, Target20_MinorAxis, Target20_AngleRad, Target20_Scatter, Target20_Confidence]
        
        ship_loc.csv 格式:
        [TimeStep, ship1_x, ship1_y, ship1_z, ship2_x, ship2_y, ship2_z, ..., shipN_x, shipN_y, shipN_z]
        """
        # 导出 measurement_data.csv
        if self.measurement_data:
            with open(measurement_filename, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)

                # 生成表头
                headers = ["TimeStep", "MissileID", "SensorID"]
                for i in range(1, self.MAX_TARGETS + 1):
                    headers.extend([
                        f"Target{i}_x", f"Target{i}_y", f"Target{i}_z",
                        f"Target{i}_MajorAxis", f"Target{i}_MinorAxis",
                        f"Target{i}_AngleRad", f"Target{i}_Scatter",
                        f"Target{i}_Confidence"
                    ])

                writer.writerow(headers)  # 写入表头

                for row in self.measurement_data:
                    writer.writerow(row)

            print(f"Measurement data exported to {measurement_filename}.")

        # 导出 ship_loc.csv
        if self.ship_locations_data:
            with open(ship_loc_filename, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)

                # 确定最大船舶数量
                max_ships = self.max_ships

                # 生成表头
                headers = ["TimeStep"]
                for i in range(1, max_ships + 1):
                    headers.extend([f"ship{i}_x", f"ship{i}_y", f"ship{i}_z"])

                writer.writerow(headers)  # 写入表头

                for ship_row in self.ship_locations_data:
                    writer.writerow(ship_row)

            print(f"Ship locations exported to {ship_loc_filename}.")