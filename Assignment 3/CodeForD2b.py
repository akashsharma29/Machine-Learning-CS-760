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
import csv

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
    
#print(FinalParts[0][0])
meanlist = []

#print(len(FinalParts[0][1]))
 
meanCol = mean([row[0] for row in FinalParts[0][0]])
#print(meanCol)

stdDevCol = stdev([row[0] for row in FinalParts[0][0]])
#print(stdDevCol)

#for i in [row[0] for row in FinalParts[0][0]]:
#    print(i)    



x1 = dataset[0:200]
print(x1)

with open("outputD2A.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x1)



"""for x in FinalParts:
    tmpMean = mean(x[0])
    meanlist.append()
    
print(meanlist)
""" 
"""
Ylist = [i[2] for i in dataset]
#print(Ylist)

for j in dataset:
    del j[2]

knn_cv = KNeighborsClassifier(n_neighbors=1)

cv_scores = cross_val_score(knn_cv, dataset, Ylist, cv=5)
#print(1-cv_scores)
#print(1-np.mean(cv_scores))
"""

