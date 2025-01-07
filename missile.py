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
        
        要求：
        1) 不区分目标类型，故不输出目标类型；
        2) 每个时间步 -> 对每个导弹输出一行，行内依次列出各目标的测量数据；
        3) 置信度用 0~1 的高斯随机数；
        4) 相同导弹的误差固定不变（即：self.missile_error_degs[missile_id] 固定）。

        输出格式：
        [ time_step, missile_id,
        x_meas_目标1, y_meas_目标1, z_meas_目标1, scatter_目标1, confidence_目标1,
        x_meas_目标2, y_meas_目标2, z_meas_目标2, scatter_目标2, confidence_目标2,
        ...
        ]
        """

        # 将所有可能的目标(船 + 箔条 + 角反射器)汇总
        all_targets = []
        all_targets.extend(carriers_positions)
        all_targets.extend(chaff_positions)
        all_targets.extend(corner_positions)

        # 对每枚导弹生成一行数据
        for missile_id, missile_pos in enumerate(self.missiles):
            # 取得该导弹的固定测量误差（度）
            sensor_error_deg = self.missile_error_degs[missile_id]

            # 收集该导弹对每个目标的测量结果（x_meas, y_meas, z_meas, scatter, confidence）
            sub_result = []

            for target_pos in all_targets:
                # ------------------ 1) 计算导弹->目标的 3D 距离 & 真实角度 ------------------
                dx = target_pos[0] - missile_pos[0]
                dy = target_pos[1] - missile_pos[1]
                dz = target_pos[2] - missile_pos[2]

                r_true = math.sqrt(dx*dx + dy*dy + dz*dz)  # 3D 距离
                az_true = math.atan2(dy, dx)               # 方位角：-pi ~ pi
                el_true = 0.0
                # 如果距离足够大再计算仰角，避免除 0
                xy_dist = math.sqrt(dx*dx + dy*dy)
                if xy_dist > 1e-8:
                    el_true = math.atan2(dz, xy_dist)      # 仰角：-pi/2 ~ pi/2

                # -------------- 2) 生成合并后不超过 sensor_error_deg 的角度误差 --------------
                # 在 [0, sensor_error_deg] 内选一个半径 r_rand
                r_rand = random.uniform(0, sensor_error_deg)
                # 随机方向 phi ∈ [0, 2π)
                phi = random.uniform(0, 2*math.pi)

                error_az_deg = r_rand * math.cos(phi)
                error_el_deg = r_rand * math.sin(phi)

                # 转为弧度
                error_az_rad = math.radians(error_az_deg)
                error_el_rad = math.radians(error_el_deg)

                # 带误差的方位角 / 仰角
                az_meas = az_true + error_az_rad
                el_meas = el_true + error_el_rad

                # ------------------ 3) 反投影到 (x_meas, y_meas, z_meas) ------------------
                # 假设 r_true 本身比较精准，没有单独加距离误差
                x_meas = missile_pos[0] + r_true * math.cos(el_meas) * math.cos(az_meas)
                y_meas = missile_pos[1] + r_true * math.cos(el_meas) * math.sin(az_meas)
                z_meas = missile_pos[2] + r_true * math.sin(el_meas)

                # ------------------ 4) 置信度(0~1 的高斯随机) & measurement_scatter ------------------
                # 例：以均值 0.8、标准差 0.2 生成一个值，并截断在 [0, 1]
                confidence = random.gauss(0.8, 0.2)
                confidence = min(max(confidence, 0.0), 1.0)

                 # 4.1) 在 (az_true, el_true) 处计算雅可比矩阵 J
            dx_daz = -r_true * math.cos(el_true) * math.sin(az_true)
            dx_del = -r_true * math.sin(el_true) * math.cos(az_true)
            dy_daz =  r_true * math.cos(el_true) * math.cos(az_true)
            dy_del = -r_true * math.sin(el_true) * math.sin(az_true)

            J = np.array([
                [dx_daz, dx_del],
                [dy_daz, dy_del]
            ])

            # 4.2) 将 sensor_error_deg 视为方位角、仰角的“最大标准差”
            sigma_az_rad = math.radians(sensor_error_deg)
            sigma_el_rad = math.radians(sensor_error_deg)
            Cov_ae = np.diag([sigma_az_rad**2, sigma_el_rad**2])

            # 4.3) 投影到 (x,y): Cov_xy = J * Cov_ae * J^T
            Cov_xy = J @ Cov_ae @ J.T
            eigvals, eigvecs = np.linalg.eig(Cov_xy)
            # 排序，使 eigvals[0] >= eigvals[1]
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]

            # 4.4) 1σ 椭圆 => 长轴 = 2*sqrt(λ1), 短轴 = 2*sqrt(λ2)，这里需要注意是2*sqrt(λ)！！！！
            major_axis = 0.0
            minor_axis = 0.0
            if eigvals[0] > 0:
                major_axis = 2.0 * math.sqrt(eigvals[0])
            if eigvals[1] > 0:
                minor_axis = 2.0 * math.sqrt(eigvals[1])

            # 4.6) 椭圆方向 => 与最大特征值对应的特征向量
            v_major = eigvecs[:, 0]  # 形如 [vx, vy]
            # atan2(y, x) 得到相对于X轴的角度 (-pi, pi)
            angle_rad = math.atan2(v_major[1], v_major[0])
            angle_deg = math.degrees(angle_rad)

            # 测量散度，简单直接用 sensor_error_deg
            measurement_scatter = sensor_error_deg

            # ========== 6) 把单个目标的测量结果延伸到 sub_result ==========
            # 这里在一行中输出: x_meas, y_meas, z_meas, major_axis, minor_axis, scatter, confidence
            sub_result.extend([
                x_meas, 
                y_meas, 
                z_meas, 
                major_axis, 
                minor_axis, 
                angle_rad,
                measurement_scatter, 
                confidence
            ])

        # ========== 7) 组装一行并写入 self.measurement_data ==========
        # 格式: [time_step, missile_id,
        #  x_meas1, y_meas1, z_meas1, maj1, min1, scatter1, conf1,
        #  x_meas2, y_meas2, z_meas2, maj2, min2, scatter2, conf2,
        #  ...
        # ]
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
