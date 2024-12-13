import tkinter as tk
from tkinter import ttk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class MissileCarrierSimulation3D:
    def __init__(self, root):
        self.root = root
        self.root.title("Missile and Carrier Simulation 3D")

        # Set up control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Add control elements
        ttk.Label(control_frame, text="Number of Carriers:").pack()
        self.carrier_count_var = tk.IntVar(value=5)
        ttk.Entry(control_frame, textvariable=self.carrier_count_var).pack()

        ttk.Label(control_frame, text="Number of Missiles:").pack()
        self.missile_count_var = tk.IntVar(value=5)
        ttk.Entry(control_frame, textvariable=self.missile_count_var).pack()

        ttk.Button(control_frame, text="Start Simulation", command=self.start_simulation).pack(pady=5)
        ttk.Button(control_frame, text="Pause Simulation", command=self.pause_simulation).pack(pady=5)
        ttk.Button(control_frame, text="Resume Simulation", command=self.resume_simulation).pack(pady=5)
        ttk.Button(control_frame, text="Reset Simulation", command=self.reset_simulation).pack(pady=5)

        # Initialize simulation state
        self.is_running = False
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = None
        self.animation = None
        self.missiles = []
        self.carriers = []
        self.missile_targets = []

        # Create Matplotlib canvas
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.create_canvas()

    def create_canvas(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def start_simulation(self):
        if self.is_running:
            return

        self.is_running = True
        self.reset_simulation_data()

        # Set up animation
        self.animation = FuncAnimation(
            self.fig, self.update, interval=100, blit=False, cache_frame_data=False
        )
        self.canvas.draw()

    def pause_simulation(self):
        if self.animation:
            self.animation.event_source.stop()

    def resume_simulation(self):
        if self.animation:
            self.animation.event_source.start()

    def reset_simulation(self):
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        self.reset_simulation_data()
        self.canvas.draw()

    def reset_simulation_data(self):
        self.ax.clear()
        self.ax.set_xlim(0, 40)
        self.ax.set_ylim(0, 40)
        self.ax.set_zlim(0, 20)
        self.ax.set_title("Missile and Carrier Simulation 3D")

        # Initialize carriers
        carrier_count = self.carrier_count_var.get()
        self.carriers = np.random.uniform(5, 35, size=(carrier_count, 3))
        self.carriers[:, 2] = 0  # Carriers are at z=0
        self.missile_targets = self.carriers.copy()

        # Initialize missiles
        missile_count = self.missile_count_var.get()
        self.missiles = np.array([[0, 0, 15] for _ in range(missile_count)], dtype=np.float64)  # Starting positions at (0, 0, 15)

        self.carrier_scatter = self.ax.scatter(
            self.carriers[:, 0], self.carriers[:, 1], self.carriers[:, 2], c="blue", label="Carriers"
        )
        self.missile_scatter = self.ax.scatter(
            self.missiles[:, 0], self.missiles[:, 1], self.missiles[:, 2], c="red", label="Missiles"
        )
        self.ax.legend()

    def update(self, frame):
        if not self.is_running:
            return []

        # Update missile positions
        for i in range(len(self.missiles)):
            if i < len(self.missile_targets):  # Ensure each missile has a target
                direction = self.missile_targets[i % len(self.missile_targets)] - self.missiles[i]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    self.missiles[i] += 0.5 * direction / norm  # Move missiles towards targets

        # Check if all missiles have reached z=0
        if np.all(self.missiles[:, 2] <= 0):
            self.is_running = False
            if self.animation:
                self.animation.event_source.stop()

        # Update scatter plot data
        self.carrier_scatter._offsets3d = (self.carriers[:, 0], self.carriers[:, 1], self.carriers[:, 2])
        self.missile_scatter._offsets3d = (self.missiles[:, 0], self.missiles[:, 1], self.missiles[:, 2])
        return []

if __name__ == "__main__":
    root = tk.Tk()
    app = MissileCarrierSimulation3D(root)
    root.mainloop()
