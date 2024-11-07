from Maze import Maze
import matplotlib.pyplot as plt

def main():
    m1 = Maze(100)


    display_maze(m1.generate_maze())

def display_maze(maze, path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')

    if path is not None:
        x_coords = [pos[1] for pos in path]
        y_coords = [pos[0] for pos in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.arrow(0, 1, 0.4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1] - 1, maze.shape[0] - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3,
             head_length=0.3)

    plt.show()

if __name__ == '__main__':
    main()