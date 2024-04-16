import numpy as np

def sigmoid_activation(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    return X*(1-X)

def predict(X, W):
    return sigmoid_activation(np.dot(X, W))
