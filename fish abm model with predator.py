import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
NUM_FISH = 100
PREDATOR_PERCEPTION_RADIUS = 60
FISH_PERCEPTION_RADIUS = 30
ATTRACTION_DIST = 10
REPULSION_DIST = 5
MAX_SPEED = 2
PREDATOR_MAX_SPEED = 5
WIDTH, HEIGHT = 100, 100
STEPS = 100

def calculate_cohesion(fishes):
    positions = np.array([fish.position for fish in fishes])
    center_of_mass = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center_of_mass, axis=1)
    return np.mean(distances)


class Predator:
    def __init__(self):
        self.position = np.array([np.random.rand() * WIDTH, np.random.rand() * HEIGHT])
        self.velocity = np.random.randn(2) * 0.5

    def update_position(self):
        self.position += self.velocity
        self.position = self.position % np.array([WIDTH, HEIGHT])

    def hunt(self, fishes):
        distances = [np.linalg.norm(fish.position - self.position) for fish in fishes]
        if distances:
            nearest_fish = fishes[np.argmin(distances)]
            distance_to_nearest = np.min(distances)
            if distance_to_nearest < PREDATOR_PERCEPTION_RADIUS:
                acceleration_vector = (nearest_fish.position - self.position) / distance_to_nearest
                self.velocity += acceleration_vector * 0.8
                # Limiting the predator's speed
                if np.linalg.norm(self.velocity) > PREDATOR_MAX_SPEED:
                    self.velocity = self.velocity / np.linalg.norm(self.velocity) * PREDATOR_MAX_SPEED
            
            if distance_to_nearest < 10:  # Predator catches the fish
                return nearest_fish
        return None


class Fish:
    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.speed = np.linalg.norm(self.velocity)

    def update_position(self):
        self.position += self.velocity
        # Boundary conditions
        self.position = self.position % np.array([WIDTH, HEIGHT])

    def apply_behaviors(self, fishes):
        nearest_neighbor = self.find_nearest_neighbor(fishes)
        if nearest_neighbor is not None:
            self.apply_attraction(nearest_neighbor)
            self.apply_repulsion(nearest_neighbor)
        
        self.react_to_predator(predator)
        
    def react_to_predator(self, predator):
        if np.linalg.norm(predator.position - self.position) < FISH_PERCEPTION_RADIUS:
            # Increase speed away from the predator
            escape_direction = self.position - predator.position
            self.velocity += escape_direction / np.linalg.norm(escape_direction) * (MAX_SPEED / 3)
            # Limit speed
            if np.linalg.norm(self.velocity) > MAX_SPEED * 1.25:
                self.velocity = self.velocity / np.linalg.norm(self.velocity) * MAX_SPEED * 1.25

    def find_nearest_neighbor(self, fishes):
        distances = [np.linalg.norm(fish.position - self.position) for fish in fishes if fish != self]
        if distances:
            nearest_neighbor = fishes[np.argmin(distances)]
            if np.min(distances) < FISH_PERCEPTION_RADIUS:
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

# Initialize fishes
fishes = [Fish(np.random.rand() * WIDTH, np.random.rand() * HEIGHT, np.random.randn(), np.random.randn()) for _ in range(NUM_FISH)]
predator = Predator()

cohesion_Values = []
# Update the environment
# Create initial scatter plot
fig, ax = plt.subplots()
scat_fishes = ax.scatter([fish.position[0] for fish in fishes], [fish.position[1] for fish in fishes], color='blue')
scat_predator = ax.scatter(predator.position[0], predator.position[1], color='red')
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_title("Schooling Fish with Predator Simulation")

def update(frame):
    
    global fishes
    caught_fish = predator.hunt(fishes)
    if caught_fish:
        fishes.remove(caught_fish)
    
    for fish in fishes:
        fish.apply_behaviors(fishes)
        fish.update_position()
    predator.update_position()
    
    cohesion_Values.append(calculate_cohesion(fishes))
    
    # Update scatter plot data
    scat_fishes.set_offsets([fish.position for fish in fishes])
    scat_predator.set_offsets([predator.position])
    return scat_fishes, scat_predator

ani = animation.FuncAnimation(fig, update, frames=STEPS, blit=True, interval=20, repeat=False)
plt.show()

# Calculate and print the average cohesion metric
average_cohesion = sum(cohesion_Values) / len(cohesion_Values)
print(f"Average Cohesion Metric: {average_cohesion}")

remaining_fish = len(fishes)
print(f"Number of Remaining Fish: {remaining_fish}")