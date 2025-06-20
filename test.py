import numpy as np

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
print(f'X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
