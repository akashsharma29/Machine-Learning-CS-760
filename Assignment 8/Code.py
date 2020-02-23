# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 08:04:13 2019

@author: akash
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

file = open('C:/Users/akash/OneDrive/Documents/three.txt', 'r')
dataset3 = []
for line in file:
    dataset3.append([ float(x) for x in line.split()])
    
#print(dataset3[0][12])
#print(len(data))
    
MatrixTmp = np.array(dataset3[0]).reshape(16,16)
#print(MatrixTmp)

MatrixTmp = MatrixTmp.transpose()        
#print(MatrixTmp)

    
#plt.imsave('filename.png', np.array(dataset3[0]).reshape(16,16), cmap=cm.gray)
plt.imshow(np.array(MatrixTmp), cmap = cm.gray)


file = open('C:/Users/akash/OneDrive/Documents/eight.txt', 'r')
dataset8 = []
for line in file:
    dataset8.append([ float(x) for x in line.split()])

#print(dataset3[0][12])
#print(len(data))
    
MatrixTmp = np.array(dataset8[0]).reshape(16,16)
MatrixTmp = MatrixTmp.transpose()        

    
#plt.imsave('filename.png', np.array(dataset3[0]).reshape(16,16), cmap=cm.gray)
plt.imshow(np.array(MatrixTmp), cmap = cm.gray)