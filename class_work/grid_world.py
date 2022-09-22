import numpy as np

def update(action, current_state):  # get next State
    left_wall = max(1, (current_state//4)*4) 
    right_wall =  min(14, (current_state//4)*4 + 3) 

    if action == 'l':  # move left
        if  current_state - 1 >= left_wall:
            return  current_state - 1 , -1
        elif    current_state - 1 == 0:
            return 0, 0
        else:
            return  current_state, -1
    if action == 'r':  # move right
        if  current_state + 1 <= right_wall:
            return  current_state + 1, -1
        elif    current_state + 1 == 15:
            return 0, 0
        else:
            return  current_state, -1
    if action == 'u':  # move up
        if  current_state - 4 >= 1:
            return  current_state - 4, -1
        elif    current_state - 4 == 0:
            return 0, 0
        else:
            return  current_state, -1
    if action == 'd':  # move down
        if  current_state + 4 <= 14:
            return  current_state + 4, -1
        elif    current_state + 4 == 15:
            return 0, 0
        else:
            return  current_state, -1

if __name__ == '__main__':

    states = range(1,15)
    gamma = 1
    theta = 0.00001
    actions = ['l', 'r', 'u', 'd']
    v = np.zeros(len(states)+2)
    V = np.zeros(len(states)+2)
    
    while True:
        delta = 0
        v = np.copy(V)
        V = np.zeros(len(states)+2)
        for state in states:
            for action in actions:
                new_state, reward = update(action, state)
                V[state] += 0.25 * (reward + gamma*v[new_state])
        delta = max(delta, np.max(abs(v - V)))
        if delta < theta: 
            break
    print(V.reshape(4,4))