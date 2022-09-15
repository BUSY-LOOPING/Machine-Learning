import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D


#load the data
#first 2 columns are x1, x2 and 3rd columns is y

X = []
Y = []

df = pd.read_csv('E:\jupyter\csvs\data_2d.csv', header=None)

Y = df.iloc[:, 2].values

for x in df.iloc[:, 0 : 2].values :
    X.append([x[0], x[1], 1])
    
X = np.array(X)

#plotting the data
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], Y)

#calculating the weights 
w = np.linalg.solve((X.T).dot(X), (X.T).dot(Y))
yhat = np.dot(X, w)

#computing the r-square
d1 =  Y - yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print('The accuracy is {:.2f}%'.format(r2 * 100))

ax.plot(sorted(X[:, 0]), sorted(X[:, 1]) , sorted(yhat)) 
plt.show()