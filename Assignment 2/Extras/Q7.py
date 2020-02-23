# -*- coding: utf-8 -*-
import numpy as np
import scipy as scipy
from scipy.stats import entropy
from math import log, e
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
import csv
#class Node:
class Node:
    def __init__(self, threshold, id1, left, right, isTerminal: bool, feature, classify):
        self.threshold = threshold
        self.id = id1
        self.left = left
        self.right = right
        self.isTerminal = isTerminal
        self.feature = feature
        self.classify = classify

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] >= value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def calculateGain(listTotal, list1, list2):
    
     labelsTotal = [i[2] for i in listTotal]
     labels1 = [i[2] for i in list1]
     labels2 = [i[2] for i in list2]
    
  #   if len(labels1) == 0 and len(labels2)== 0: return 0  
  #   if len(labels2) == 0: return 0;
    
     a = calculateEntropy(labelsTotal)
     b = calculateEntropy(labels1)
     c = calculateEntropy(labels2)
    
     n1 = len(labels1)
     n2 = len(labels2)
     
     gain = a - (((1.0*n1)/(n1+n2))*b + ((1.0*n2)/(n1+n2))*c)
     return gain

def calculateEntropy(X):
    no1 = 0
    no0 = 0
    for i in X: 
        if i == 1:  no1 = no1 + 1    
        else: no0 = no0 + 1
    
    if no1 + no0 == 0: return 1
    
    p1 = no1/(no1 + no0)
    p2 = no0/(no1 + no0)
    
    if p1 == 0: return 0
    if p2 == 0: return 0
    
    #print(p2)
    prob = [p1, p2]
    ent = 0
    
    for i in prob:
        ent = ent - i * log(i, 2)
    return ent

def getdot1(root, idot):
    if root is None:
        return idot
    if root.isTerminal is False:
        idot.node(str(root.id), label=str(root.feature)+" >= "+str(round(root.threshold, 6)), shape='box')
    else:
        idot.node(str(root.id), label=str(root.classify), shape='box')
    if root.left is not None:
        idot = getdot1(root.left, idot)
        idot.edge(str(root.id), str(root.left.id), label="T")
    if root.right is not None:
        idot = getdot1(root.right, idot)
        idot.edge(str(root.id), str(root.right.id), label="F")
    return idot

def visualize(root):
    dot = Digraph(comment='Decision Tree', format='png')
    dot = getdot1(root, dot)
    dot.render('test', view=True)

def majority(list):
    no0 = 0
    no1 = 0
    labelsTotal = [i[2] for i in list]
    for i in labelsTotal: 
        if i == 1:  no1 = no1 + 1    
        else: no0 = no0 + 1        
    
    if no1 < no0: return 0
    else: return 1


label = 0
def createTree(dataset):
        
        if len(dataset) == 0: return None
    
        dataset = sorted(dataset)
        #root = Node(0, label,  None,None, True, '')
        x1list = ([row[0] for row in dataset])
        maxGain = -1
        ansIndex = 0
        x1x2 = 0
        val = 0
    
        for i in range(len(x1list)):
            left, right = test_split(0, x1list[i], dataset)
            #print(len(left))
            gain = calculateGain(dataset, left, right)
            if gain > maxGain: 
                val = x1list[i]
                maxGain = gain
                x1x2 = 0
                ansIndex = i
        
        dataset = sorted(dataset, key = lambda x: x[1])
        x2list = ([row[1] for row in dataset])
        
        for i in range(len(x2list)):
            left, right = test_split(1, x2list[i], dataset)
            gain = calculateGain(dataset, left, right)
            if gain > maxGain: 
                val = x2list[i]
                maxGain = gain
                x1x2 = 1
                ansIndex = i
                
        #print(x1x2 ,  maxGain, val)
        
        if x1x2 == 0:
            left, right = test_split(0, val, sorted(dataset))
        else :
            left, right = test_split(1, val, sorted(dataset, key = lambda x: x[1]))
        
        global label
        label = label+1
        
        if maxGain == 0:
            ans = majority(dataset)
            root = Node(val, label, None,None, True, '', ans)
            #print(len(dataset))
        else:
            if x1x2 == 0:
                root = Node(val, label, createTree(left),createTree(right), False, 'x1', -1)
            else:
                root = Node(val, label, createTree(left),createTree(right), False, 'x2', -1)
        return root
    
#threshold, id1, left, right, isTerminal, feature, classify
def predict_Value(x,root):
        
    #print(x[0] , x[1])    
    while root.isTerminal != 1:
        
        #if root == None: return
        if root.isTerminal == True:
            #print('hi')
            print(root.threshold)
            return root.classify
    
        if root.feature == 'x1':
            #print('bi')
            if x[0] >= root.threshold: 
                #print('bi1')
                root = root.left
            else: 
                root = root.right
                #print('bi2')
                
        elif root.feature == 'x2':
            #print('ci')
            if x[1] >= root.threshold: 
                root = root.left
                #print('ci1')
            else: 
                root = root.right
                #print('ci2')
    return root.classify
    
    
def plot_decision_boundry(dataset,root:Node):
        plotstep = 0.001
        
        x1 = ([row[0] for row in dataset])
        x2 = ([row[1] for row in dataset])
        
        x_min,x_max = min(x1) - 5*plotstep, max(x1) + 5*plotstep
        y_min,y_max = min(x2) - 5*plotstep, max(x2) + 5*plotstep
        xx,yy = np.meshgrid(np.arange(x_min,x_max,plotstep),np.arange(y_min,y_max,plotstep))
        y_pred = []
        
        #print(xx.shape)
        
        for i in range(len(xx)):
            y_pred_temp =[]
            for j in range(len(xx[0])):
                #print(predict_Value([xx[i][j],yy[i][j]],root))
                y_pred_temp.append(predict_Value([xx[i][j],yy[i][j]],root))
            y_pred.append(y_pred_temp)
        
        y_pred = np.asarray(y_pred,'float64')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.contourf(xx,yy,y_pred,cmap=plt.cm.Spectral)
        plt.show()

def measureError(root, test):
    n = len(test)
    wrong = 0
    for i in test:
        #print(i[0] , i[1])
        a = predict_Value([i[0], i[1]],root)
        if a != i[2]:
            wrong = wrong + 1
    
    wrongpercentage = (100.0*wrong)/n
    return wrongpercentage
    
        
file = open('C:/Users/akash/OneDrive/Documents/Dbig.txt', 'r')
dataset = []
for line in file:
    dataset.append([ float(x) for x in line.split()])

np.random.shuffle(dataset)

candidate = dataset[: 8192]
test = dataset[ 8192 : ]

x = list()

x1 = candidate[: 32]
x2 = candidate[: 128]
x3 = candidate[: 512]
x4 = candidate[: 2048]
x5 = candidate[: 8192]
#print(x1[0], x2[0],x3[0], x4[0], x5[0])

x.append(x1)
x.append(x2)
x.append(x3)
x.append(x4)
x.append(x5)
#print(len(x))

nvalues= []
testerrornvalues = []

for i in x:
    root = createTree(i)
    error = measureError(root, test)
    nvalues.append(len(i))
    testerrornvalues.append(error)
    plot_decision_boundry(i,root)


print(nvalues)
print(testerrornvalues)

with open("testset.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(test)

with open("output1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x1)
    
with open("output2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x2)
    
with open("output3.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x3)

with open("output4.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x4)

with open("output5.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(x5)

    