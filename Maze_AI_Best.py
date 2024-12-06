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

    def dfs(x, y):
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                maze[nx, ny] = 0
                maze[x + dx // 2, y + dy // 2] = 0
                dfs(nx, ny)

    dfs(start_x, start_y)
    return maze


# BFS to calculate the shortest path (optimal path) from start to goal
def bfs_shortest_path(maze, start, goal):
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([(start[0], start[1], 0)])  # (x, y, distance)
    visited[start[0], start[1]] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y, dist = queue.popleft()

        if (x, y) == goal:
            return dist  # Return the number of steps to reach the goal

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0 and not visited[nx, ny]:
                visited[nx, ny] = True
                queue.append((nx, ny, dist + 1))

    return float('inf')  # If goal is unreachable, return infinity


# Environment setup
class MazeEnvironment:
    def __init__(self, maze):
        self.maze = maze
        self.SIZE = maze.shape[0]
        self.state = (1, 1)
        self.end_state = self.find_farthest_point((1, 1))

    def find_farthest_point(self, start):
        rows, cols = self.maze.shape
        visited = np.zeros_like(self.maze, dtype=bool)
        queue = deque([(start[0], start[1], 0)])
        visited[start[0], start[1]] = True

        farthest_point = start
        max_distance = -1
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y, dist = queue.popleft()

            if dist > max_distance:
                max_distance = dist
                farthest_point = (x, y)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and self.maze[nx, ny] == 0 and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append((nx, ny, dist + 1))

        return farthest_point

    def reset(self):
        self.state = (1, 1)
        return self.state

    def step(self, action):
        x, y = self.state
        new_state = {
            'up': (x - 1, y),
            'down': (x + 1, y),
            'left': (x, y - 1),
            'right': (x, y + 1)
        }[action]

        if 0 <= new_state[0] < self.SIZE and 0 <= new_state[1] < self.SIZE and self.maze[new_state] == 0:
            self.state = new_state
            return self.state, 10 if self.state == self.end_state else -1, self.state == self.end_state
        return self.state, -10, False


# Q-learning agent
ACTIONS = ['up', 'down', 'left', 'right']
SIZE = 30
Q_table = np.zeros((SIZE, SIZE, len(ACTIONS)))

learning_rate = 0.5
discount_factor = 0.9
epsilon = 0.05  # Lower epsilon to heavily favor exploitation
epsilon_decay = 0.999  # Slow decay to maintain low epsilon
min_epsilon = 0.01  # Epsilon will not decay below 0.01
episodes = 200

# Initialize Pygame
pygame.init()
CELL_SIZE = 20
WINDOW_SIZE = SIZE * CELL_SIZE + 200
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Maze Training Visualization")
clock = pygame.time.Clock()

# Performance tracking variables
episode_times = []
highest_time = -float('inf')
lowest_time = float('inf')
total_time = 0
step_counts = []
error_percentages = []
total_steps = 0


def draw_maze(env, agent_pos, episode, move_count):
    screen.fill((0, 0, 0))

    for x in range(SIZE):
        for y in range(SIZE):
            color = (255, 255, 255) if env.maze[x, y] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, (0, 0, 255), (agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, (0, 255, 0),
                     (env.end_state[1] * CELL_SIZE, env.end_state[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Episode: {episode} | Moves: {move_count}', True, (255, 255, 255))
    screen.blit(img, (SIZE * CELL_SIZE + 10, 10))

    img = font.render(f'Longest Time: {highest_time:.2f}s', True, (255, 255, 255))
    screen.blit(img, (SIZE * CELL_SIZE + 10, 40))

    img = font.render(f'Shortest Time: {lowest_time:.2f}s', True, (255, 255, 255))
    screen.blit(img, (SIZE * CELL_SIZE + 10, 70))

    avg_time_display = f'{total_time / len(episode_times):.2f}s' if episode_times else 'N/A'
    img = font.render(f'Avg. Time: {avg_time_display}', True, (255, 255, 255))
    screen.blit(img, (SIZE * CELL_SIZE + 10, 100))

    avg_error_display = f'{error_percentages[-1]:.2f}%' if error_percentages else 'N/A'
    img = font.render(f'Error%: {avg_error_display}', True, (255, 255, 255))
    screen.blit(img, (SIZE * CELL_SIZE + 10, 130))

    pygame.display.flip()
    clock.tick(200)


def train_agent():
    maze = create_maze(SIZE)
    env = MazeEnvironment(maze)
    global epsilon, highest_time, lowest_time, episode_times, step_counts, error_percentages, total_time, total_steps

    for episode in range(episodes):
        start_time = time.time()
        state = env.reset()
        done = False
        move_count = 0

        optimal_steps = bfs_shortest_path(env.maze, state, env.end_state)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            x, y = state
            action_idx = random.randint(0, len(ACTIONS) - 1) if random.uniform(0, 1) < epsilon else np.argmax(
                Q_table[x, y])
            action = ACTIONS[action_idx]
            new_state, reward, done = env.step(action)
            nx, ny = new_state

            Q_table[x, y, action_idx] += learning_rate * (
                    reward + discount_factor * np.max(Q_table[nx, ny]) - Q_table[x, y, action_idx]
            )
            state = new_state
            move_count += 1

            draw_maze(env, state, episode + 1, move_count)

        episode_time = time.time() - start_time
        episode_times.append(episode_time)
        total_time += episode_time
        highest_time = max(highest_time, episode_time)
        lowest_time = min(lowest_time, episode_time)
        step_counts.append(move_count)
        total_steps += move_count

        if optimal_steps != float('inf') and optimal_steps > 0:
            error_percentage = ((move_count - optimal_steps) / optimal_steps) * 100
        else:
            error_percentage = 0
        error_percentages.append(error_percentage)

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    avg_time = total_time / len(episode_times) if episode_times else 0
    avg_error_percentage = np.mean(error_percentages) if error_percentages else 0

    print("Training complete")
    print(f"Longest time: {highest_time:.2f}s")
    print(f"Shortest time: {lowest_time:.2f}s")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Average error percentage: {avg_error_percentage:.2f}%")

    return Q_table, env


# Start training
train_agent()
