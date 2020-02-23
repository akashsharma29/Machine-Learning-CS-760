import numpy as np
import scipy as scipy
from scipy.stats import entropy
from math import log, e
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
#class Node:
class Node:
    def __init__(self, threshold, id1, left, right, isTerminal, feature, classify):
        self.threshold = threshold
        self.id = id1
        self.left = left
        self.right = right
        self.isTerminal = isTerminal
        self.feature = feature
        self.classify = classify 

#DONE DONE DONE DONE
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

"""def getdot1(root, idot):
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
    dot.render('test', view=True)"""

def majority(list):
    no0 = 0
    no1 = 0
    labelsTotal = [i[2] for i in list]
    for i in labelsTotal: 
        if i == 1:  no1 = no1 + 1    
        else: no0 = no0 + 1        
    
    if no1 >= no0: return 1
    else: return 0   

def createTree(dataset, label):
        
        print("In create Decision Tree")
        if len(dataset) == 0: return None
    
        dataset = sorted(dataset)
        #print(dataset)
        #root = Node(0, label,  None,None, True, '')
        x1list = ([row[0] for row in dataset])
        print(x1list)
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
        
        label = label+1
        
        if maxGain == 0:
            ans = majority(dataset)
            root = Node(val, label, None,None, True, '', ans)
            #print(len(dataset))
        else:
            if x1x2 == 0:
                root = Node(val, label, createTree(left, label+1),createTree(right, label+2), False, 'x1', 0)
            else:
                root = Node(val, label, createTree(left, label+1),createTree(right, label+2), False, 'x2', 0)
        return root
    
        
file = open('C:/Users/akash/OneDrive/Documents/D3leaves.txt', 'r')
dataset = []
for line in file:
    dataset.append([ float(x) for x in line.split()])

dataset = sorted(dataset)
root = createTree(dataset, 0)
visualize(root)