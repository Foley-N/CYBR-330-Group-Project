class Path:
    def __init__(self, maze):
        self.maze = maze # initialize with the maze to solve 
        self.start = (0, 1)  # Entry point
        self.end = (maze.shape[0] - 2, maze.shape[1] - 1)  # Exit point
        self.solution_path = [] # list that stores the solution path

    def solve(self):
        # Wrapper to solve the maze
        visited = set() # set to keep track of visited cells
        path = [] # list to store the current path
        self._dfs(self.start, visited, path) # starts the depth-first search (dfs) from entry point
        return self.solution_path # returns the solution path

    def _dfs(self, position, visited, path):
        x, y = position # get current position

        # Add current position to visited and path
        visited.add(position)
        path.append(position)

        # If we've reached the end, save the path
        if position == self.end:
            self.solution_path = path.copy() # saves the current path as solution
            return True # exit the function

        # Explore neighbors
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy # calculates the new position
            new_position = (new_x, new_y)

            if (0 <= new_x < self.maze.shape[0] and 0 <= new_y < self.maze.shape[1]
                    and self.maze[new_x, new_y] == 0 and new_position not in visited):
            # check if the new position is within bounds, not a wall, and not visited 
                if self._dfs(new_position, visited, path): # recursively perform dfs on new position
                    return True # if solution is found, exit the function

        # Backtrack if no solution is found
        path.pop() # removes the last positon from the path
        return False # returns false to indicate no solution found from this position
