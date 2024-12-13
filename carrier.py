import numpy as np

class Carrier:
    def __init__(self, count, speed):
        self.count = count
        self.speed = speed
        self.positions = np.random.uniform(5, 35, size=(count, 3))
        self.positions[:, 2] = 0  # Carriers start at z=0

    def move(self):
        """Randomly move the carriers within bounds."""
        self.positions += np.random.uniform(-self.speed, self.speed, self.positions.shape)
        self.positions = np.clip(self.positions, [5, 5, 0], [35, 35, 0])

    def get_positions(self):
        """Return the current positions of the carriers."""
        return self.positions