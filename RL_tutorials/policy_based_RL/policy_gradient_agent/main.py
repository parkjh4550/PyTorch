from matplotlib import animation, rc
from IPython.display import HTML, display
from Maze_environment import draw_maze
from policy_gradient_agent import goal_maze_ret_s_a, theta_0, softmax_convert_into_pi_from_theta, update_theta

import numpy as np

if __name__ == '__main__':

    ### 1. Learning
    stop_epsilon = 10**-4  # if the policy difference is lower than this, stop learning

    theta = theta_0
    pi =  softmax_convert_into_pi_from_theta(theta)

    is_continue = True
    count = 1
    while is_continue:
        s_a_history = goal_maze_ret_s_a(pi)
        new_theta = update_theta(theta, pi, s_a_history)
        new_pi = softmax_convert_into_pi_from_theta(new_theta)

        diff = np.sum(np.abs(new_pi - pi))
        print('policy change : ', diff) # policy difference
        print('num of steps taken to the goal : '+str(len(s_a_history)))

        if diff < stop_epsilon:
            is_continue = False
        else:
            theta = new_theta
            pi = new_pi

    ### 2. Show the agent actions
    fig, plt, line = draw_maze()

    print(len(s_a_history))
    def init():
        # initialize a background image
        line.set_data([], [])

        return (line,)

    def animate(i):
        # generate images frame by frame
        state, action = s_a_history[i]  # current location
        x = (state % 3) + 0.5  # x coordinate
        y = 2.5 - int(state / 3)  # y coordinate
        print(i)
        line.set_data(x, y)
        return (line,)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(s_a_history), repeat=True)
    plt.show()
    html = HTML(anim.to_jshtml())
    rc('animation', html='html5')
    print(anim)
