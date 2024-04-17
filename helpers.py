import numpy as np

def linear_activation(X):
    return X

def linear_derivative():
    return 1 

def predict(X, W):
    return linear_activation(np.dot(X, W))
