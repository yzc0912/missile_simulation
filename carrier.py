import numpy as np
import random

class Carrier:
    def __init__(self, carrier_count, carrier_speed):
        """
        :param carrier_count: 船的数量
        :param carrier_speed: 船的移动速度（单位：在 x、y 平面上的移动速度）
        """
        self.carrier_count = carrier_count
        self.carrier_speed = carrier_speed

        # 初始化每艘船在 x, y 平面的初始位置 (z=0)
        self.positions = self._initialize_positions()

        # 每艘船在 x, y 平面的移动方向向量 (dx, dy)，归一化后再乘以速度
        self.directions = self._initialize_directions()

    def _initialize_positions(self):
        """
        船初始位置的设定，这里假设随机分布在 [5, 35] 范围内。
        """
        positions = []
        for _ in range(self.carrier_count):
            x = random.uniform(5, 35)
            y = random.uniform(5, 35)
            z = 0.0
            positions.append([x, y, z])
        return np.array(positions, dtype=np.float64)

    def _initialize_directions(self):
        """
        生成随机的移动方向（只在 x,y 平面），然后乘以速度。
        """
        directions = []
        for _ in range(self.carrier_count):
            angle = random.uniform(0, 2 * np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)
            # 归一化后乘以 speed
            norm = np.sqrt(dx**2 + dy**2)
            dx = (dx / norm) * self.carrier_speed
            dy = (dy / norm) * self.carrier_speed
            directions.append([dx, dy, 0.0])  # z方向为0
        return np.array(directions, dtype=np.float64)

    def move(self):
        """
        每一步让船根据自己的方向移动。如果位置超出一定范围，就让它反弹。
        """
        self.positions += self.directions

        # 如果超出 [0, 40] 区域，就让它反弹
        for i in range(self.carrier_count):
            for axis in range(2):  # 只检测 x,y
                if self.positions[i, axis] < 0 or self.positions[i, axis] > 40:
                    self.directions[i, axis] = -self.directions[i, axis]

    def get_positions(self):
        """
        :return: 返回所有船的当前位置（numpy数组，形状 (carrier_count, 3)）
        """
        return self.positions

    def generate_chaff(self):
        """
        生成箔条干扰点（相对于每艘船做一定随机偏移）。
        这里演示：对每艘船生成 1~3 个箔条点。
        返回的为绝对坐标 (N, 3)。
        """
        chaff_positions = []
        for pos in self.positions:
            for _ in range(random.randint(1, 3)):
                offset_x = random.uniform(-1, 1)
                offset_y = random.uniform(-1, 1)
                chaff_positions.append([
                    pos[0] + offset_x,
                    pos[1] + offset_y,
                    0.0
                ])
        return np.array(chaff_positions, dtype=np.float64)

    def generate_fixed_corner_reflectors(self):
        """
        生成固定型角反射器（在出现那一刻，根据船的当前位置 + 随机偏移得到绝对坐标），
        之后不随船移动。返回 (N, 3)。
        """
        fixed_positions = []
        for pos in self.positions:
            for _ in range(random.randint(1, 3)):
                offset_x = random.uniform(-1, 1)
                offset_y = random.uniform(-1, 1)
                fixed_positions.append([
                    pos[0] + offset_x,
                    pos[1] + offset_y,
                    0.0
                ])
        return np.array(fixed_positions, dtype=np.float64)

    def generate_moving_corner_reflectors(self):
        """
        生成移动型角反射器：
        不直接返回绝对坐标，而是返回 [ship_idx, offset_x, offset_y, 0.0]，
        表示这是第 ship_idx 条船、相对偏移 (offset_x, offset_y, 0)。
        后续主程序中，会通过船的当前位置 + offset 计算绝对位置，从而实现“跟随船移动”。
        """
        moving_data = []
        # 枚举每条船的索引和当前位置
        for i, pos in enumerate(self.positions):
            for _ in range(random.randint(1, 3)):
                offset_x = random.uniform(-1, 1)
                offset_y = random.uniform(-1, 1)
                # 记录: 该反射器属于第 i 条船 + 相对偏移
                moving_data.append([i, offset_x, offset_y, 0.0])
        return np.array(moving_data, dtype=np.float64)
