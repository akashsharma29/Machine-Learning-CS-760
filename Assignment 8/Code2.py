import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

file = open('C:/Users/akash/OneDrive/Documents/three.txt', 'r')
dataset = []
for line in file:
    dataset.append([ float(x) for x in line.split()])

file = open('C:/Users/akash/OneDrive/Documents/eight.txt', 'r')
for line in file:
    dataset.append([ float(x) for x in line.split()])
    
#print(len(dataset[0]))
#print(len(dataset))

output = []

for i in range(len(dataset[0])):
    sum = 0
    for j in range(len(dataset)):
        sum = sum + dataset[j][i]
    sum = (1.0*sum)/400
    output.append(sum)
    
#print(len(output))   
#MatrixTmp = np.array(output).reshape(16,16)
#MatrixTmp = MatrixTmp.transpose()        
#plt.imshow(np.array(MatrixTmp), cmap = cm.gray)
    
for i in range(len(dataset)):
    for j in range(len(dataset[0])):
        dataset[i][j] = dataset[i][j] - output[j]

#print(len(dataset[0]))
        
MatrixTmp = np.array(dataset).reshape(400,256)
TransMat = MatrixTmp.transpose()

CoVarMatrix = np.matmul(TransMat, MatrixTmp)
#print(len(CoVarMatrix[0]))

for i in range(len(CoVarMatrix)):
    for j in range(len(CoVarMatrix[0])):
        CoVarMatrix[i][j] = (1.0*CoVarMatrix[i][j])/399

print(CoVarMatrix[0:5, 0:5])
#print(CoVarMatrix[0])
#print(CoVarMatrix[np.ix_([0,6],[0,6])])

egnval, egnvctr = np.linalg.eig(CoVarMatrix)
egnvctr = np.transpose(egnvctr)
print(egnval)

tmp = np.transpose(np.array(egnvctr[0]).reshape(16,16))
#plt.colorbar(cmap = 'gray')
#plt.imshow(np.array(tmp))


tmp1 = np.transpose(np.array(egnvctr[1]).reshape(16,16))
plt.imshow(np.array(tmp1), cmap = cm.gray)

V = []
V.append(egnvctr[0])
V.append(egnvctr[1])
V = np.transpose(V)
print(V)

Projected = np.matmul(dataset,V)
print(Projected[0])
print(Projected[200])
#c = np.array(Projected)

for i in range(len(c)):
    if(i<200):
        plt.scatter(c[i][0], c[i][1], label = "three.txt", color = "blue",marker =".",s = 2)
    else:
        plt.scatter(c[i][0], c[i][1], label = "three.txt", color = "red",marker =".",s = 2)

#plt.show()

#cols = pltcolor()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('D2z.txt Plot')
#mappable = plt.cm.ScalarMappable(cmap='gray')
#mappable.set_array([0,255])
#plt.colorbar(mappable)
#plt.show()
#plt.scatter(x=x1,y=x2,c=cols)
#plt.show()
#plt.show()