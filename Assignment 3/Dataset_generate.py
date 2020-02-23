import numpy as np
import scipy as scipy
from scipy.stats import entropy
from math import log, e
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt

file = open('C:/Users/akash/OneDrive/Documents/test_data.txt', 'r+')
dataset = []

values = [-0.2, -1.9, -1.8,-1.7, -1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]


for i in values:
    s = ''
    for j in values:
        s = str(i)+' ' +str(j)+'\n' 
        file.write(s)
        