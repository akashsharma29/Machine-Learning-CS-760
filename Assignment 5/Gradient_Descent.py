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
    

alpha = 0.01 #Step size
iterations = 10 #No. of iterations
m = year1_train.size #No. of data points
np.random.seed(123) #Set the seed
theta = np.random.rand(2) #Pick some random values to start with


def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			sum_error += error**2
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef

"""def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs"""