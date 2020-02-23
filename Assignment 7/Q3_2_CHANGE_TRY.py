#2. Reset and repeat the above, but with an E greedy behavior policy: at each state st, with probability 1 ô€€€ 
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
    next_state = 0
    action = 0
    
    #Max Value : Greedy
    if chooseNext == 0:
        
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
        
    #Random
    else:
        next_state = random.choice([0,1])
        action = 0
    
        if curr_state == next_state:
            reward = 1
            action = 0
        else:
            reward = 0
            action = 1

        
        
    maxval = max(Q[next_state][0], Q[next_state][1])
    Q[curr_state][action] = (1-alpha)*Q[curr_state][action] + alpha*(reward + gamma*(maxval))
    curr_state = next_state
   
print("A, Stay")
print(Q[0][0])

print("A, Move")
print(Q[0][1])

print("B, Stay")
print(Q[1][0])

print("B, Move")
print(Q[1][1])