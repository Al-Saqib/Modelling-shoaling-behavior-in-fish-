import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Parameters
NUM_FISH = 100
PERCEPTION_RADIUS = 30
ATTRACTION_DIST = 10
REPULSION_DIST = 5
MAX_SPEED = 2
WIDTH, HEIGHT, DEPTH = 100, 100, 100
STEPS = 100

class Fish:
    def __init__(self, x, y, z, vx, vy, vz):
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.speed = np.linalg.norm(self.velocity)

    def update_position(self):
        self.position += self.velocity
        # Boundary conditions for 3D space
        self.position = self.position % np.array([WIDTH, HEIGHT, DEPTH])

    def apply_behaviors(self, fishes):
        nearest_neighbor = self.find_nearest_neighbor(fishes)
        if nearest_neighbor is not None:
            self.apply_attraction(nearest_neighbor)
            self.apply_repulsion(nearest_neighbor)

    def find_nearest_neighbor(self, fishes):
        distances = [np.linalg.norm(fish.position - self.position) for fish in fishes if fish != self]
        if distances:
            nearest_neighbor = fishes[np.argmin(distances)]
            if np.min(distances) < PERCEPTION_RADIUS:
                return nearest_neighbor
        return None

    def apply_attraction(self, neighbor):
        if np.linalg.norm(neighbor.position - self.position) > ATTRACTION_DIST:
            self.velocity += (neighbor.position - self.position) / ATTRACTION_DIST

    def apply_repulsion(self, neighbor):
        if np.linalg.norm(neighbor.position - self.position) < REPULSION_DIST:
            self.velocity -= (neighbor.position - self.position) / REPULSION_DIST

        # Limiting the speed
        if np.linalg.norm(self.velocity) > MAX_SPEED:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * MAX_SPEED

# Initialize schools of fish in 3D space
fishes = [Fish(np.random.rand() * WIDTH, np.random.rand() * HEIGHT, np.random.rand() * DEPTH, 
               np.random.randn(), np.random.randn(), np.random.randn()) for _ in range(NUM_FISH)]

# Create initial 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([fish.position[0] for fish in fishes], 
                  [fish.position[1] for fish in fishes], 
                  [fish.position[2] for fish in fishes])
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_zlim(0, DEPTH)
ax.set_title("3D Schooling Fish Simulation")

def update(frame):
    for fish in fishes:
        fish.apply_behaviors(fishes)
        fish.update_position()

    # Update scatter plot data for 3D
    scat._offsets3d = ([fish.position[0] for fish in fishes], 
                       [fish.position[1] for fish in fishes], 
                       [fish.position[2] for fish in fishes])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=STEPS, blit=False, interval=20, repeat=False)
plt.show()

