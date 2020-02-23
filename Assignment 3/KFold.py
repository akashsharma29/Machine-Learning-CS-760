from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from statistics import mean 
from statistics import stdev
from numpy import array
from sklearn.model_selection import KFold

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

def func(list1, list2):
    print(list1)

file = open('C:/Users/akash/OneDrive/Documents/D2a.txt', 'r')
dataset = []
dataset1 = []

for line in file:
    dataset.append([ float(x) for x in line.split()])
    dataset1.append([ float(x) for x in line.split()])

data = array(dataset)
#print(data)

kfold = KFold(5, True, 1)
#train =  kfold.split(dataset)
#print(train)

FinalParts = []

for train, test in kfold.split(data):
	FinalParts.append((data[train], data[test]))
    
#print(len(FinalParts[0][0]))
meanlist = []

#tmpval = mean[FinalParts[0][0][0]]
print(len(FinalParts[0][1]))
 
meanCol = mean([row[0] for row in FinalParts[0][0]])
print(meanCol)

stdDevCol = stdev([row[0] for row in FinalParts[0][0]])
print(stdDevCol)

#for i in [row[0] for row in FinalParts[0][0]]:
#    print(i)    

"""for x in FinalParts:
    tmpMean = mean(x[0])
    meanlist.append()
    
print(meanlist)
   """ 
"""
Ylist = [i[6] for i in dataset]

for j in dataset:
    del j[6]

knn_cv = KNeighborsClassifier(n_neighbors=1)

cv_scores = cross_val_score(knn_cv, dataset, Ylist, cv=5)
print(np.mean(cv_scores))
Xlist0 = [i[0] for i in dataset]
mean0 = mean(Xlist0)
std0 = stdev(Xlist0)

Xlist1 = [i[1] for i in dataset]
mean1 = mean(Xlist1)
std1 = stdev(Xlist1)"""