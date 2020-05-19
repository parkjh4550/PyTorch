import numpy as np
import matplotlib.pyplot as plt


def draw_maze():
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()

    # draw red walls
    # plot([x_range, y_range])
    plt.plot([1,1], [0,1], color='red', linewidth=2)
    plt.plot([1,2], [2,2], color='red', linewidth=2)
    plt.plot([2,2], [2,1], color='red', linewidth=2)
    plt.plot([2,3], [1,1], color='red', linewidth=2)

    # show state
    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')

    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')

    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')

    # set the range and remove the gradtions.
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, labelleft=False)

    # show the current state S0 as a green circle.
    line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)

    return fig, plt, line

if __name__ == '__main__':
    _, plt, _ = draw_maze()
    plt.show()
