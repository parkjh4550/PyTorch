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

def softmax_convert_into_pi_from_theta(theta):
    """
    policy param "theta" -> action policy "pi"
    :param theta: list
    :return: list
    """
    beta = 1.0
    [m, n] = theta.shape # get the matrix shape
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta)   # theta -> exp(theta)

    for i in range(0, m):
        # softmax
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i,:])

    pi = np.nan_to_num(pi) # nan -> 0

    return pi

def get_action_and_next_s(pi, s):
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
        action = 0
        s_next = s - 3
    elif next_direction == 'right':
        action = 1
        s_next = s + 1
    elif next_direction == 'down':
        action = 2
        s_next = s + 3
    elif next_direction == 'left':
        action = 3
        s_next = s - 1

    return [action, s_next]

def goal_maze_ret_s_a(pi):
    """
    start to find the goal
    :param pi: list
    :return: list
    """
    s = 0  # start point
    s_a_history = [[0, np.nan]] # [state, action] history of the agent

    while(True):
        [action, next_s] = get_action_and_next_s(pi, s)

        s_a_history[-1][1] = action

        s_a_history.append([next_s, np.nan])

        if next_s == 8: # goal state
            break
        else:
            s = next_s
    return s_a_history

def update_theta(theta, pi, s_a_history):
    eta = 0.1 # learning rate
    T = len(s_a_history) - 1    # num of steps taken to the goal

    [m, n] = theta.shape    # shape of theta
    delta_theta = theta.copy()

    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i,j])):

                # get history which state is i-th element
                SA_i = [SA for SA in s_a_history if SA[0] == i]

                SA_ij = [SA for SA in s_a_history if SA == [i,j]]

                N_i = len(SA_i)
                N_ij = len(SA_ij)

                delta_theta[i,j] = (N_ij - pi[i,j] * N_i)/T

    new_theta = theta + eta * delta_theta
    return new_theta

if __name__ == '__main__':
    pi_0 = softmax_convert_into_pi_from_theta(theta_0)
    print('inital pi \n', pi_0)

    # agent history
    s_a_history = goal_maze_ret_s_a(pi_0)
    print('history \n', s_a_history)