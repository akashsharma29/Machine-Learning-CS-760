

import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import normalize
from scipy.special import expit
import math
from mpl_toolkits import mplot3d

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def Q1():
    # x = (1,1)
    arrX = [[1,1],[1,-1],[-1,-1]]
    for x in arrX:
        b = [1]
        X = np.asarray(b+x)
        W = np.asarray([1,1,1])
    
        o = list()
        for i in range(10):
            o.append(np.matmul(X,np.transpose(W)))
    
        #ReLU output
        o_ReLU = np.maximum(np.asarray(o),0)
        #print(o_ReLU)
    
        o_ReLU = np.append(o_ReLU,np.asarray([1]))
        w_ReLU = np.ones([1,len(o_ReLU)])
        y = sigmoid(np.matmul(o_ReLU, np.transpose(w_ReLU)))
        print("Y is ")
        print(y)


def main():
    Q1()
    

if __name__ == '__main__':
    main()