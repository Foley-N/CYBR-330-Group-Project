import numpy as np
import random

class Maze:
    __dimensions = None # private variable to store dimensions of the maze

    def __init__(self, dim):
        self.set_dim(dim) # setting the dimensions of the maze when creating Maze object

    def generate_maze(self):
        maze_grid = np.ones((self.get_dim() * 2 + 1, self.get_dim() *2 + 1))
        # creating a grid filled with 1s (walls) of size (2*dim +1) x (2*dim +1)

        cell_x, cell_y = (0, 0) # starting at the top left corner, i.e. (0, 0)
        maze_grid[2 * cell_x + 1, 2 * cell_y + 1] = 0

        cell_stack = [(cell_x, cell_y)] # stack to keep track of the cells that need to be explored

        while len(cell_stack) > 0:
         
            cell_x, cell_y = cell_stack[-1]
            # while there are still cells in the stack to explore, get current cell from top of stack
            possible_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            # possible directions to move: right, down, left, up
            random.shuffle(possible_directions)

            for direction_x, direction_y in possible_directions:
                neighbor_x, neighbor_y = cell_x + direction_x, cell_y + direction_y
                # checks each direction

                if (0 <= neighbor_x < self.get_dim() and 0 <= neighbor_y < self.get_dim()
                    and maze_grid[2 * neighbor_x + 1, 2 * neighbor_y + 1] == 1):
                    # check if the neighbor cell is within bounds and not yet visited (still a wall)

                    maze_grid[2 * neighbor_x + 1, 2 * neighbor_y + 1] = 0
                    # mark the neighbor cell as open (0)
                    maze_grid[2 * cell_x + 1 + direction_x, 2 * cell_y + 1 + direction_y] = 0
                    # mark the wall between the current cell and the neighbor cell as open (0)

                    cell_stack.append((neighbor_x, neighbor_y))
                    # add the neighbor cell to the stack to explore it later
                    break # exit the for loop to continue with the new cell

            else:
                cell_stack.pop()
                # if no unvisited neighbors are found, pop the current cell from the stack
        maze_grid[1,0] = 0 # maze entry point
        maze_grid[-2, -1] = 0 # maze exit point

        return maze_grid # returns the completed maze


    def get_dim(self):
        return self.__dimensions # returns the dimensions of the maze 

    def set_dim(self, dim):
        self.__dimensions = dim # set the dimensions of the maze 

