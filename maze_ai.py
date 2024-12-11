import numpy as np
import random
import pygame
import time
from collections import deque


# Maze generation
def create_maze(size):
    maze = np.ones((size, size), dtype=int)  # 1 for walls, 0 for paths
    start_x, start_y = 1, 1  # Start position
    maze[start_x, start_y] = 0  # Mark start as open

    # Function to generate the maze using DFS
    def dfs(x, y):
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]  # Directions to explore
        random.shuffle(directions)  # Randomize direction order to ensure randomness
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                maze[nx, ny] = 0
                maze[x + dx // 2, y + dy // 2] = 0  # Remove wall between cells
                dfs(nx, ny)

    # Start maze generation from the top left corner
    dfs(start_x, start_y)
    return maze


# BFS to find the farthest point from the start
def find_farthest_point(maze, start):
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([(start[0], start[1], 0)])  # (x, y, distance)
    visited[start[0], start[1]] = True

    farthest_point = start
    max_distance = -1

    # Directions for 4 possible movements (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y, dist = queue.popleft()

        # Update farthest point if this is further
        if dist > max_distance:
            max_distance = dist
            farthest_point = (x, y)

        # Explore neighboring cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0 and not visited[nx, ny]:
                visited[nx, ny] = True
                queue.append((nx, ny, dist + 1))

    return farthest_point


# Environment setup
class MazeEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.SIZE = maze.shape[0]
        self.state = (1, 1)

        # Find the farthest point from the start using BFS
        self.end_state = find_farthest_point(self.maze, (1, 1))

    def reset(self):
        self.state = (1, 1)  # Start at the top-left corner
        return self.state

    def step(self, action):
        x, y = self.state
        new_state = {
            'up': (x - 1, y),
            'down': (x + 1, y),
            'left': (x, y - 1),
            'right': (x, y + 1)
        }[action]

        # Check if the new state is within bounds and not a wall
        if 0 <= new_state[0] < self.SIZE and 0 <= new_state[1] < self.SIZE and self.maze[new_state] == 0:
            self.state = new_state
        else:
            return self.state, -10, False  # Penalty for hitting a wall

        # If the agent reaches the goal
        if self.state == self.end_state:
            return self.state, 100, True  # Reward for reaching the exit

        return self.state, -1, False  # Small penalty for each step


# Q-learning agent and training
ACTIONS = ['up', 'down', 'left', 'right']
SIZE = 10
Q_table = np.zeros((SIZE, SIZE, len(ACTIONS)))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 10

# Initialize Pygame
pygame.init()
CELL_SIZE = 40
WINDOW_SIZE = SIZE * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Maze Training Visualization")
clock = pygame.time.Clock()

# Performance tracking variables
start_time = 0
episode_times = []
highest_time = -float('inf')
lowest_time = float('inf')
total_time = 0
step_counts = 0
error_percentage = []
total_steps = 0


# Draw the maze and agent
def draw_maze(env, agent_pos, episode, move_count):
    screen.fill((0, 0, 0))  # Fill the screen with black

    # Draw each cell of the maze
    for x in range(SIZE):
        for y in range(SIZE):
            color = (255, 255, 255) if env.maze[x, y] == 0 else (0, 0, 0)  # White for paths, Black for walls
            pygame.draw.rect(screen, color, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw agent (blue square)
    pygame.draw.rect(screen, (0, 0, 255), (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw goal (green square) at the fixed goal position on the track
    pygame.draw.rect(screen, (0, 255, 0),
                     (env.end_state[1] * CELL_SIZE, env.end_state[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Show episode number
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Episode: {episode} | Moves: {move_count}', True, (255, 255, 255))
    screen.blit(img, (10, 10))

    pygame.display.flip()
    clock.tick(10)  # Control the speed


def train_agent():
    maze = create_maze(SIZE)
    env = MazeEnvironment(maze)
    global epsilon

    for episode in range(episodes):
        state = env.reset()
        done = False
        step_count = 0  # Count steps per episode
        while not done:
            x, y = state
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            draw_maze(env, state, episode + 1, step_count)  # Draw the maze with move count

            ###############################
            # adds print out for ai after each episode
            episode_time = time.time() - start_time
            episode_times.append(episode_time)
            total_time += episode_time
            print(episode_time)
            highest_time = max(highest_time, episode_time)
            lowest_time = min(lowest_time, episode_time)
            step_counts.append(step_count)
            total_steps += step_count





            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                action_idx = np.argmax(Q_table[x, y])

            action = ACTIONS[action_idx]
            new_state, reward, done = env.step(action)
            nx, ny = new_state

            # Q-value update
            best_future_q = np.max(Q_table[nx, ny])
            Q_table[x, y, action_idx] += learning_rate * (
                        reward + discount_factor * best_future_q - Q_table[x, y, action_idx])
            state = new_state
            step_count += 1

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    avg_time = total_time / len(episode_times) if episode_times else 0
    avg_error_percentage = np.mean(error_percentage) if error_percentage else 0

    print("Training complete")
    ###############################
    # adds print out for ai after each episode
    print(f"Longest time: {highest_time:.2f}s")
    print(f"Shortest time: {lowest_time:.2f}s")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Average error percentage: {avg_error_percentage:.2f}%")

    return Q_table, env


# Testing the trained agent
def test_agent(Q_table, env):
    state = env.reset()
    done = False
    steps = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        draw_maze(env, state, "Test Run", steps)  # Draw the maze with move count

        x, y = state
        action_idx = np.argmax(Q_table[x, y])
        action = ACTIONS[action_idx]
        state, _, done = env.step(action)
        steps += 1
        time.sleep(0.1)

    print(f"Total steps to solve the maze: {steps}")


# Main script to run training and testing
if __name__ == "__main__":
    # Train the agent
    Q_table, env = train_agent()

    # Test the trained agent
    print("\nTesting the trained agent:\n")
    test_agent(Q_table, env)

    pygame.quit()
