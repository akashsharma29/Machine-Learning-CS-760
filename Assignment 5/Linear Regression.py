from matplotlib import pyplot as plt
import statistics 

f1 = open("MendotaFinalTxt").readlines()
f2 = open("MononaFinalTxt").readlines()
year1 = []
days1 = []

for i in f1:
  row = i.strip().split()
  year1.append(int(row[0]))
  days1.append(int(row[1]))

year1_train = []
days1_train = []
year1_test = []
days1_test = []
for i in range(len(year1)):
  if year1[i] <= 1970:
    year1_train.append(year1[i])
    days1_train.append(days1[i])
  else:
    year1_test.append(year1[i])
    days1_test.append(days1[i])

import math
m1 = statistics.mean(days1_train)
s1 = statistics.stdev(days1_train)
m1 = 0.0
s1 = 0.0
for day in days1_train:
  m1 += day
m1 = float(m1/len(days1_train)-1)
for day in days1_train:
  s1 += math.pow(day-m1,2)
s1 = float(math.sqrt(s1/(len(days1_train)-1)))
print(" Mean for Mendota training set = "+str(m1))
print(" Standard Deviation for Mendota training set = "+str(s1))

norm_days1 = []
for i in days1:
  nday = float((i - m1)/s1)
  norm_days1.append(nday)


year2 = []
days2 = []
for i in f2:
  row = i.strip().split()
  #year1.append(int(row[0]))
  #days1.append(int(row[1]))
  year2.append(int(row[0]))
  days2.append(int(row[1]))

year2_train = []
days2_train = []
year2_test = []
days2_test = []
for i in range(len(year2)):
  if year2[i] <= 1970:
    year2_train.append(year2[i])
    days2_train.append(days2[i])
  else:
    year2_test.append(year2[i])
    days2_test.append(days2[i])


m2 = statistics.mean(days2_train)
s2 = statistics.stdev(days2_train)
m2 = 0.0
s2 = 0.0
for day in days2_train:
  m2 += day
m2 = float(m2/len(days2_train)-1)
for day in days2_train:
  s2 += math.pow(day-m2,2)
s2 = float(math.sqrt(s2/(len(days2_train)-1)))
print("Sample Mean for Monona training set = "+str(m2))
print("Sample Standard Deviation for Monona training set = "+str(s2))

# print(year1_train == year2_train)
# print(year2_train)
term1 = []

import numpy as np
sm = np.array([0,0,0])
x_array = []
for i in range(len(year1_train)):
  x_array.append([1,year1_train[i],days2_train[i]])
x = np.array(x_array)
term1 = np.matmul(x.T,x)
#print(term1)
y_arr = np.array(days1_train)
term2 = np.matmul(x.T,y_arr)
print(term2)

res = np.matmul(np.linalg.inv(term1),term2)
print(res)



# beta = res.tolist()
# y_cap = []
# for i in range(len(days2_test)):
#   pred = beta[0]+beta[1]*year1_test[i]+beta[2]*days2_test[i]
#   y_cap.append(pred)

# mean_sum = 0.0
# for i in range(len(y_cap)):
#   mean_sum += math.pow((y_cap[i]-days1_test[i]),2)
# explained_variance = mean_sum
# mean_sum /= len(y_cap)
# avg_actual_value = 0.0
# for i in days1_test:
#   avg_actual_value += i
# avg_actual_value /= len(days1_test)

# sum_avg_actual_pred = 0.0
# for i in y_cap:
#   sum_avg_actual_pred += math.pow(i - avg_actual_value,2)
# r_squared = 1-(explained_variance/sum_avg_actual_pred)

# print(mean_sum)
# print(r_squared)