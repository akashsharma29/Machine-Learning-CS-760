#2. Reset and repeat the above, but with an E greedy behavior policy: at each state st, with probability 1 􀀀 
#choose what the current Q table says is the best action: argmaxa Q(st; a); Break ties arbitrarily. Otherwise
#(with probability ) uniformly chooses between move and stay. Use  = 0:5.

import numpy as np
import random
# Initialize q-table values to 0

#np.zeros((state_size, action_size))
Q = np.zeros((2, 2))
curr_state = 0
alpha = 0.5
gamma = 0.9

for i in range(200):
    chooseNext = random.choice([0,1])
    next_action = 0
    
    if chooseNext == 0:
         next_action = 0
         if Q[curr_state][0] < Q[curr_state][1]:
             next_action = 1
         elif Q[curr_state][0] > Q[curr_state][1]:
            next_action = 0
         elif Q[curr_state][0] == Q[curr_state][1]:
            if curr_state == 0:
                next_action = 1
            else:
                next_action = 0
    
         if curr_state == next_action:
            reward = 1
         else:
            reward = 0
        
    else:
        next_action = random.choice([0,1])
        if curr_state == next_action:
            reward = 1
        else:
            reward = 0
        
        
    maxval = max(Q[next_action][0], Q[next_action][1])
    Q[curr_state][next_action] = (1-alpha)*Q[curr_state][next_action] + alpha*(reward + gamma*(maxval))
    curr_state = next_action
   
print(Q[0][0])
print(Q[0][1])
print(Q[1][0])
print(Q[1][1])