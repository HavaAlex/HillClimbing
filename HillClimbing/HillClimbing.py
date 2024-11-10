import random
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print("UJJMENEEEEEEEEEEET")
# The matrix given
matrix = [
    [10, 11, 12, 13, 14, 13, 12, 11, 12, 11, 12, 11, 10, 11, 12, 11, 10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 11],
    [11, 12, 13, 14, 15, 16, 15, 14, 13, 14, 15, 14, 13, 12, 13, 14, 13, 14, 13, 14, 13, 12, 13, 12, 13, 12, 11, 10, 11, 12],
    [12, 13, 14, 15, 16, 17, 16, 15, 16, 15, 14, 15, 16, 15, 16, 17, 16, 15, 14, 13, 12, 13, 14, 15, 14, 13, 12, 11, 12, 13],
    [13, 14, 15, 16, 17, 18, 17, 16, 17, 16, 17, 18, 17, 18, 17, 16, 15, 14, 13, 12, 13, 14, 13, 12, 13, 14, 15, 14, 15, 16],
    [14, 15, 16, 17, 18, 19, 18, 19, 18, 19, 18, 17, 16, 15, 16, 17, 16, 17, 16, 15, 14, 13, 12, 13, 12, 13, 12, 13, 14, 15],
    [13, 14, 15, 16, 15, 16, 15, 14, 13, 12, 13, 12, 13, 14, 15, 16, 15, 16, 17, 18, 17, 16, 17, 18, 19, 18, 17, 18, 19, 20],
    [14, 13, 12, 13, 14, 15, 16, 17, 16, 17, 16, 15, 14, 13, 14, 15, 14, 15, 16, 15, 16, 17, 18, 17, 16, 15, 14, 13, 12, 13],
    [13, 14, 15, 16, 17, 18, 17, 16, 15, 16, 17, 18, 19, 18, 19, 20, 19, 18, 17, 16, 15, 14, 13, 12, 13, 12, 13, 14, 15, 16],
    [12, 13, 14, 15, 14, 13, 12, 11, 10, 11, 12, 13, 12, 13, 12, 11, 12, 13, 14, 13, 14, 15, 16, 15, 16, 17, 16, 15, 14, 13],
    [11, 12, 13, 12, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 12, 13, 14, 13, 14, 13, 12, 13, 14, 13, 14, 15, 16, 17, 16],
    [12, 13, 14, 13, 14, 15, 14, 13, 12, 13, 12, 13, 14, 15, 16, 17, 18, 17, 18, 19, 18, 17, 16, 15, 14, 13, 12, 13, 12, 13],
    [13, 14, 15, 16, 15, 14, 13, 14, 15, 14, 15, 16, 17, 18, 19, 18, 17, 16, 15, 14, 15, 16, 15, 16, 15, 14, 15, 16, 17, 16],
    [12, 13, 14, 15, 16, 15, 16, 17, 16, 15, 16, 15, 14, 15, 14, 13, 14, 13, 14, 13, 12, 11, 10, 11, 10, 11, 12, 13, 12, 13],
    [11, 12, 13, 12, 11, 12, 13, 12, 13, 14, 13, 14, 13, 12, 13, 12, 13, 12, 13, 12, 11, 12, 13, 14, 13, 12, 13, 12, 13, 12],
    [10, 11, 12, 13, 14, 15, 14, 13, 12, 13, 12, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 15, 14, 13, 12, 13, 12, 13, 14, 15],
    [11, 12, 13, 14, 15, 16, 15, 14, 13, 14, 13, 12, 13, 14, 13, 12, 11, 10, 11, 10, 11, 12, 11, 12, 13, 14, 13, 14, 13, 12],
    [12, 13, 14, 15, 16, 15, 14, 13, 14, 13, 14, 13, 12, 11, 12, 13, 12, 11, 12, 11, 10, 11, 12, 13, 12, 11, 12, 13, 12, 13],
    [13, 14, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 15, 16, 17, 16, 17, 18, 17, 18, 17, 16, 17, 16, 15, 16, 15, 14, 13, 12],
    [12, 13, 12, 11, 10, 11, 12, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 11, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13],
    [11, 10, 11, 12, 13, 12, 11, 12, 11, 12, 13, 12, 11, 12, 11, 10, 11, 12, 11, 12, 13, 14, 13, 14, 15, 14, 13, 12, 13, 12],
    [10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 12, 13, 12, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 11, 12, 13, 12, 11, 10, 11],
    [11, 12, 13, 14, 15, 14, 13, 12, 13, 12, 11, 10, 11, 12, 13, 14, 13, 14, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 15, 14],
    [12, 13, 12, 11, 12, 13, 12, 13, 12, 13, 13, 12, 11, 10, 11, 12, 13, 12, 13, 14, 13, 12, 13, 12, 12, 13, 14, 15, 16, 17],
    [12, 13, 14, 15, 16, 17, 16, 15, 14, 13, 14, 15, 14, 13, 12, 11, 10, 11, 12, 13, 14, 13, 14, 15, 14, 13, 12, 13, 12, 11],
    [13, 14, 13, 12, 13, 14, 15, 14, 13, 14, 15, 14, 13, 12, 11, 10, 10, 11, 12, 13, 14, 13, 12, 13, 14, 15, 14, 13, 14, 13],
    [14, 13, 12, 13, 14, 15, 16, 15, 16, 15, 14, 15, 14, 13, 12, 13, 14, 13, 14, 15, 14, 13, 12, 13, 14, 13, 14, 15, 16, 17]
]

# Find the largest value in the matrix (global max_value)
max_value = max(max(row) for row in matrix)

# Directions for neighbors (up, down, left, right, and diagonals)
directions = [
    (-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
    (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals: Up-Left, Up-Right, Down-Left, Down-Right
]

# Function to check if a position is within the matrix bounds
def in_bounds(x, y):
    return 0 <= x < len(matrix) and 0 <= y < len(matrix[0])

# Function to initialize the 3D plot
def init_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(len(matrix))
    y = np.arange(len(matrix[0]))
    x, y = np.meshgrid(x, y, indexing='ij')
    z = np.array(matrix)

    # Plot the surface and return figure and axis
    surface = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
    scatter = ax.scatter([], [], [], color='red', s=100)  # Initial empty scatter for the red dot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    plt.ion()  # Interactive mode on
    plt.show()
    return fig, ax, scatter

# Function to update the position of the red dot on the 3D plot
def update_plot(ax, scatter, current_x, current_y):
    scatter._offsets3d = ([current_x], [current_y], [matrix[current_x][current_y]])
    plt.draw()
    plt.pause(0.1)

# Modified random walk function with random movement in all eight directions
def random_walk_to_target(current_x, current_y, ax, scatter, steps=5, round_counter=1):
    print(f"Random walk initiated from ({current_x}, {current_y}) for {steps} steps")

    for _ in range(steps):
        round_counter += 1  # Increment round counter
        print(f"Round {round_counter}")
        
        update_plot(ax, scatter, current_x, current_y)  # Visualize current position
        
        # Choose a random direction to move, including diagonals
        dx, dy = random.choice(directions)
        new_x, new_y = current_x + dx, current_y + dy

        # Check if the new position is within bounds
        if in_bounds(new_x, new_y):
            current_x, current_y = new_x, new_y
            print(f"Moving to ({current_x}, {current_y})")

    print(f"Random walk completed, resuming hill climbing.")
    hill_climb(matrix, max_value, current_x, current_y, ax, scatter, round_counter, is_first_round=False)

# Hill-climbing algorithm
def hill_climb(matrix, max_value, current_x, current_y, ax, scatter, round_counter=1, is_first_round=True):
    if is_first_round:
        print(f"Starting at ({current_x}, {current_y}) with value {matrix[current_x][current_y]}")
    
    current_value = matrix[current_x][current_y]
    
    while True:
        round_counter += 1  # Increment round counter
        print(f"Round {round_counter}")
        
        update_plot(ax, scatter, current_x, current_y)  # Visualize current position
        
        neighbors = []
        for dx, dy in directions:
            nx, ny = current_x + dx, current_y + dy
            if in_bounds(nx, ny):
                neighbors.append((nx, ny))
        
        best_neighbor = (current_x, current_y)
        best_value = current_value
        
        for nx, ny in neighbors:
            if matrix[nx][ny] > best_value:
                best_neighbor = (nx, ny)
                best_value = matrix[nx][ny]

        print(f"At position ({current_x}, {current_y}) with value {current_value}.")

        if best_neighbor == (current_x, current_y):
            print(f"Local maximum reached at ({current_x}, {current_y}) with value {current_value}")
            break
        
        current_x, current_y = best_neighbor
        current_value = best_value
    
    if current_value == max_value:
        print(f"Global maximum found at ({current_x}, {current_y}) with value {current_value}")
    else:
        print(f"Local maximum found at ({current_x}, {current_y}) with value {current_value}. Initiating random walk...")
        
        random_walk_to_target(current_x, current_y, ax, scatter, steps=5, round_counter=round_counter)

# Starting position
current_x = random.randint(0, len(matrix) - 1)
current_y = random.randint(0, len(matrix[0]) - 1)

# Initialize the plot
fig, ax, scatter = init_plot()

# Run the hill climbing algorithm
hill_climb(matrix, max_value, current_x, current_y, ax, scatter)