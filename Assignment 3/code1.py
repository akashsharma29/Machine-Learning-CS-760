import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

# making function for calculating euclidean distance
def E_Distance(x1, x2, length):
    #print(x1, x2)
    distance = 0.0
    for x in range(length):
        #print(np.square(x1[x] - x2[x]))    
        distance += np.square(x1[x] - x2[x])
    return np.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
    
	for x in range(len(trainingSet)):
		dist = E_Distance(testInstance, trainingSet[x], 2)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

file = open('C:/Users/akash/OneDrive/Documents/D2z.txt', 'r')
dataset = []
for line in file:
    dataset.append([ float(x) for x in line.split()])
    
#print(dataset[0][0])
    
file = open('C:/Users/akash/OneDrive/Documents/test_data.txt', 'r')
testset = []
for line in file:
    testset.append([ float(x) for x in line.split()])
    
#print(testset[1][1])
#print(E_Distance((1,2), (3,4), 2))
#print(testset[0])

finalList = []

for x in testset:
    listmine = getNeighbors(dataset, x, 1)
    finalList.append((x, listmine[0][2]))
#print(listmine[0][2])
#print(finalList)

print(finalList[0][1])
x1 = []
x2 = []
x3 = []
x4 = []
y = []
#x1 = data[0]
#x2 = data[1]
#y = data[2]
ypr = []

for t in finalList:
    x3.append(t[0][0])
    x4.append(t[0][1])
    ypr.append(t[1])

for x in dataset:
    x1.append(x[0])
    x2.append(x[1])
    y.append(x[2])

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('red')
        #elif l==1:
        #    cols.append('blue')
        else:
            cols.append('yellow')
    return cols


def pltcolorfinal(lst):
    cols=[]
    for l in lst:
        if l==0:
            cols.append('green')
        #elif l==1:
        #    cols.append('blue')
        else:
            cols.append('blue')
    return cols

# Create the colors list using the function above

cols1 = pltcolorfinal(ypr)
plt.scatter(x=x3,y=x4,c=cols1)

cols = pltcolor(y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('D2z.txt Plot')
plt.scatter(x=x1,y=x2,c=cols)
#plt.show()


plt.show()







