#Reset and repeat the above, but with a deterministic greedy behavior policy: at each state st use the best
#action at 2 arg maxa Q(st; a) indicated by the current Q table. If there is a tie, prefer move.

import numpy as np
import random
# Initialize q-table values to 0

#np.zeros((state_size, action_size))
Q = np.zeros((2, 2))
curr_state = 0
alpha = 0.5
gamma = 0.9

for i in range(200):
    
    next_state = 0
    
    if Q[curr_state][0] < Q[curr_state][1]:
        action = 1
    elif Q[curr_state][0] > Q[curr_state][1]:
        action = 0
        #next_state = 0
    elif Q[curr_state][0] == Q[curr_state][1]:
        action = 1
        
        
    if action == 1:
        reward = 0
        if curr_state == 0:
            next_state = 1
        else:
            next_state = 0
        
    else:
        reward = 1
        next_state = curr_state
        
    maxval = max(Q[next_state][0], Q[next_state][1])
    Q[curr_state][action] = (1-alpha)*Q[curr_state][action] + alpha*(reward + gamma*(maxval))
    curr_state = next_state
    
print(Q[0][0])
print(Q[0][1])
print(Q[1][0])
print(Q[1][1])