from Maze import Maze
from Path import Path
import matplotlib.pyplot as plt # importing matplotlib for plotting the maze

def main():
    m1 = Maze(20) # create a Maze object with dimension 20
    maze = m1.generate_maze() # generate the maze

    path_solver = Path(maze) # create a Path object with the generated maze
    solution = path_solver.solve() # solve the maze to find the path
    display_maze(maze, solution) # display the maze and the solution path

def display_maze(maze, path=None):
    fig, ax = plt.subplots(figsize=(10, 10)) # set up the plot

    fig.patch.set_edgecolor('white') # set the edge color of the figure
    fig.patch.set_linewidth(0) # set the line width of the figure

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest') # display the maze using binary color map

    if path is not None: # if solution not provided, plot it
        x_coords = [pos[1] for pos in path] # gets the x-coordinates of the path
        y_coords = [pos[0] for pos in path] # gets the y-coordinates of the path
        ax.plot(x_coords, y_coords, color='red', linewidth=2) # plot the path in red

    ax.set_xticks([]) # remove x-axis ticks 
    ax.set_yticks([]) # remove y-axis ticks
    # draw arrows for entry (green) and exit (blue) points
    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3) # entry point
    ax.arrow(maze.shape[1] - 1, maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3,
             head_length=0.3) # exit points

    plt.show() # show the plot 

if __name__ == '__main__':
    main() 
