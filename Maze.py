import numpy as np
import random

class Maze:
    __dimensions = None

    def __init__(self, dim):
        self.set_dim(dim)

    def generate_maze(self):
        maze_grid = np.ones((self.get_dim() * 2 + 1, self.get_dim() *2 + 1))

        cell_x, cell_y = (0, 0)
        maze_grid[2 * cell_x + 1, 2 * cell_y + 1] = 0

        cell_stack = [(cell_x, cell_y)]

        while len(cell_stack) > 0:

            cell_x, cell_y = cell_stack[-1]

            possible_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(possible_directions)

            for direction_x, direction_y in possible_directions:
                neighbor_x, neighbor_y = cell_x + direction_x, cell_y + direction_y

                if (0 <= neighbor_x < self.get_dim() and 0 <= neighbor_y < self.get_dim()
                    and maze_grid[2 * neighbor_x + 1, 2 * neighbor_y + 1] == 1):

                    maze_grid[2 * neighbor_x + 1, 2 * neighbor_y + 1] = 0
                    maze_grid[2 * cell_x + 1 + direction_x, 2 * cell_y + 1 + direction_y] = 0

                    cell_stack.append((neighbor_x, neighbor_y))
                    break

            else:
                cell_stack.pop()
        maze_grid[1,0] = 0
        maze_grid[-2, -1] = 0

        return maze_grid


    def get_dim(self):
        return self.__dimensions

    def set_dim(self, dim):
        self.__dimensions = dim

