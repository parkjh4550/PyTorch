from matplotlib import animation, rc
from IPython.display import HTML, display
from Maze_environment import draw_maze
from random_agent import goal_maze, theta_0, simple_convert_into_pi_from_theta

if __name__ == '__main__':
    fig, plt, line = draw_maze()

    pi_0=simple_convert_into_pi_from_theta(theta_0)
    state_history = goal_maze(pi_0)

    print(len(state_history))
    def init():
        # initialize a background image
        line.set_data([], [])

        return (line,)


    def animate(i):
        # generate images frame by frame
        state = state_history[i]  # current location
        x = (state % 3) + 0.5  # x coordinate
        y = 2.5 - int(state / 3)  # y coordinate
        print(i)
        line.set_data(x, y)
        return (line,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), repeat=True)
    plt.show()
    html = HTML(anim.to_jshtml())
    rc('animation', html='html5')
    print(anim)