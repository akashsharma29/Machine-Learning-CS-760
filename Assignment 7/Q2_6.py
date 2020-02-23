import math
import numpy as np
import matplotlib.pyplot as plt

def diffcalculate(h):
    delta = 0.05
    n = 200
    val = 2*(h*math.log(n) + h*math.log((2*math.e)/h) + math.log(2/delta))/n;
    val = 2*math.sqrt(val)
    print(val)
    
#diffcalculate(1)

def findAfunction(list1, listoutput):
    Aval = 5;
    for i in range(len(listoutput)):
        if listoutput[i] == 1:
            if list1[i] < Aval:
                Aval = list1[i] 
            
    
    return Aval    

FinalAList = []
for i in range(10000):
    list1 = []
    listoutput = []
    for j in range(200):
        val = np.random.uniform(-1,1)
        output = 1
        if val < 0:
            output = 0
        
        list1.append(val)
        listoutput.append(output)
    
    A = findAfunction(list1, listoutput)
    FinalAList.append(A/2)
    
#print(FinalAList)
plt.hist(FinalAList, bins  = 50, label = "ok")
plt.xlabel("R(fa) - R^S(fa)")
plt.ylabel("Frequency")
print(np.percentile(FinalAList, 95))