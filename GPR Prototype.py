#!/usr/bin/env python3
import numpy as np 
np.random.seed(6)               # Ensures same random var. values used each execution of the code

def Kernel(X, Y, L = 1):
    """
    Generates a kernel matrix from 
    X = array of shape [N, D] where N is the number of points, D is the dimensionality of each point
    Y = array of shape [M, D] where M is the number of poitns in the array, same dimensionality for each point
    """
    def k_func(x, y):
        """
        kernel function 
        """
        return np.exp(-(1/(2*(L**2)))*(np.linalg.norm(x - y)))
    XC = X.shape[0]
    YC = Y.shape[0]
    kernel = np.zeros((XC, YC))
    for i in range(XC):
        for j in range(YC):
            kernel[i,j] = k_func(X[i], Y[j])
    return kernel

# Get Y_train (the large dataset)
load_set = np.loadtxt("A_Dataset.csv", dtype=float, delimiter=',')
dataset = load_set.reshape(load_set.shape[0], load_set.shape[1]//4, 4)
Y_train = np.reshape(dataset, ((dataset.shape[0] * dataset.shape[1], dataset.shape[2])))

# Get X_train (larger set of sample inputs)
low_v = 300.0
high_v = 390.0
low_psi = 5.0
high_psi = 5.16
v = np.arange(low_v, (high_v + 1), 10.0)
psi = np.arange(low_psi,  (high_psi), 0.0175, dtype='single')
X_train = np.zeros((100,2))
point = 0
for i in range(v.shape[0]):
    for j in range(psi.shape[0]):
        X_train[point] = np.array([v[i], psi[j]])
        point += 1

# Get X_test and Y_test (smaller set of sample inputs and smaller respective dataset respectively)
numberOfInputsPerPt = 2
numberOfStates = 4   # x, y, theta, delta
testSetPts = 5
trainingInputsCol_1 = np.zeros(testSetPts)[:, np.newaxis]
trainingInputsCol_2 = np.zeros(testSetPts)[:, np.newaxis]
X_test = np.zeros((testSetPts**2, numberOfInputsPerPt))  #16x2
Y_test = np.zeros((testSetPts**2, numberOfStates))  #16x4

# Select random test 
indexSet1 = np.random.choice((v.shape[0]), size = testSetPts, replace=False)
indexSet2 = np.random.choice((psi.shape[0]), size = testSetPts, replace=False)

counter = 0
for i in range(testSetPts):
    for j in range(testSetPts):
        X_test[counter] = np.array([v[indexSet1[i]], psi[indexSet2[j]]])
        index = 10*indexSet1[i] + indexSet2[j]
        Y_test[counter] += Y_train[index]
        counter += 1

# Get the required kernels
Kss = Kernel(Y_train, Y_train)
Kst = Kernel(Y_train, Y_test)
Kts = Kernel(Y_test, Y_train) # This equals Kst.T
Ktt = Kernel(Y_test, Y_test)


u = np.ones((Y_train.shape[1])) * 0. # I made it 0 for simplicity... but other values work
nc = 0.01
fs = Kts @ np.linalg.inv(Kss + nc * np.eye(X_train.shape[0])) @ (Y_train - u) + u
print("Predictions:")
print(fs)
print()
print("Ground Truth:")
print(Y_test)

print()
print("Average Error:")
error = np.average(np.linalg.norm(fs - Y_test, axis=1))
print(error)