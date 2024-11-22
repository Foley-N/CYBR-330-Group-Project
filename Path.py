class Path:
    def __init__(self, maze):
        self.maze = maze
        self.start = (0, 1)  # Entry point
        self.end = (maze.shape[0] - 2, maze.shape[1] - 1)  # Exit point
        self.solution_path = []

    def solve(self):
        # Wrapper to solve the maze
        visited = set()
        path = []
        self._dfs(self.start, visited, path)
        return self.solution_path

    def _dfs(self, position, visited, path):
        x, y = position

        # Add current position to visited and path
        visited.add(position)
        path.append(position)

        # If we've reached the end, save the path
        if position == self.end:
            self.solution_path = path.copy()
            return True

        # Explore neighbors
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            new_position = (new_x, new_y)

            if (0 <= new_x < self.maze.shape[0] and 0 <= new_y < self.maze.shape[1]
                    and self.maze[new_x, new_y] == 0 and new_position not in visited):
                if self._dfs(new_position, visited, path):
                    return True

        # Backtrack
        path.pop()
        return False