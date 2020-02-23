import matplotlib.pyplot as plt
import pandas as pd

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

plt.xlabel('year')
plt.ylabel('ice days')
plt.title('year vs ice days, Orange: Monona, Blue: Mendota')
plt.plot(year1,days1)
plt.plot(year2,days2)
#plt.scatter(x1,x2, 'o-', linestyle='dashed')
plt.show()

"""ymon_men = []
for i in range(len(days1)):
  ymon_men.append(days2[i]-days1[i])

#plt.figure(figsize=(5, 4))
plt.title('Year vs yMonona-yMendota')
plt.xlabel('Year')
plt.ylabel('ice days(yMonona-yMendota)')
#plt.plot(year1,norm_days1,'r-',label='Mendota')
plt.plot(year2,ymon_men)
plt.legend()
plt.show()"""