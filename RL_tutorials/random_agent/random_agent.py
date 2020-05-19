import numpy as np

# initialize policy
# row : state 0~7
# column : action direction(up, right, down, left)
theta_0 = np.array([[np.nan, 1, 1, np.nan], # s0
                    [np.nan, 1, np.nan, 1], # s1
                    [np.nan, np.nan, 1, 1], # s2
                    [1, 1, 1, np.nan],  # s3
                    [np.nan, np.nan, 1, 1], # s4
                    [1, np.nan, np.nan, np.nan],    # s5
                    [1, np.nan, np.nan, np.nan],    # s6
                    [1, 1, np.nan, np.nan]  #s7
                    ])  # s8 is a goal region, so we don't have a policy for it.

def simple_convert_into_pi_from_theta(theta):
    """
    policy param "theta" -> action policy "pi"
    :param theta: list
    :return: list
    """
    [m, n] = theta.shape # get the matrix shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i,:]) # calc the ratio

    pi = np.nan_to_num(pi) # nan -> 0

    return pi

def get_next_s(pi, s):
    """
    randomly select a direction and return the next state
    :param pi: list
    :param s: int
    :return: int
    """
    direction = ['up', 'right', 'down', 'left']

    # randomly select a direction
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction =='up':
        s_next = s - 3
    elif next_direction == 'right':
        s_next = s + 1
    elif next_direction == 'down':
        s_next = s + 3
    elif next_direction == 'left':
        s_next = s - 1

    return s_next

def goal_maze(pi):
    """
    start to find the goal
    :param pi: list
    :return: list
    """
    s = 0  # start point
    state_history = [0] # historty of the agent

    while(True):
        next_s = get_next_s(pi, s)
        state_history.append(next_s)

        if next_s == 8: # goal state
            break
        else:
            s = next_s
    return state_history

if __name__ == '__main__':
    pi_0 = simple_convert_into_pi_from_theta(theta_0)
    print('inital pi \n', pi_0)

    # agent history
    state_history = goal_maze(pi_0)
    print('history \n', state_history)