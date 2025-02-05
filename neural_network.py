import numpy as np

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu_activation(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return np.where(X > 0, 1, 0)

def linear_activation(X):
    return X  # Linear activation for regression output

class NeuralNetwork:

    def __init__(self, layers, epochs, activation="relu", lr=0.01):
        self.layers = layers
        self.lr = lr
        self.activation = activation
        self.epochs = epochs
        self.W = []

        # Weights initialization
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # Output layer with no bias
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

        # Activation functions and derivatives
        self.activation_name = activation
        self.activations = {
            "sigmoid": sigmoid_activation,
            "relu": relu_activation
        }
        self.derivatives = {
            "sigmoid": sigmoid_derivative,
            "relu": relu_derivative
        }

    def fit(self, X, y, display=10):
        # bias trick
        X = np.c_[X, np.ones((X.shape[0], 1))]

        for epoch in np.arange(self.epochs):
            for x, target in zip(X, y):
                self.backpropagation(x, target)

            if epoch == 0 or (epoch + 1) % display == 0:
                loss = self.loss(X, y)
                print(f"[INFO] epoch={epoch + 1}, loss={loss:.7f}")

    def backpropagation(self, x, y):
        # FORWARD PASS
        Ac = [np.atleast_2d(x)]

        for i in np.arange(0, len(self.W) - 1):
            activation = np.dot(Ac[i], self.W[i])
            Ac.append(self.activations[self.activation_name](activation))

        # Output layer
        activation = np.dot(Ac[-1], self.W[-1])
        output = linear_activation(activation)
        Ac.append(output)

        # BACKPROPAGATION
        error = Ac[-1] - y
        deltas = [error]

        for layer in np.arange(len(Ac) - 2, 0, -1):
            delta = deltas[-1].dot(self.W[layer].T) * self.derivatives[self.activation_name](Ac[layer])
            deltas.append(delta)

        deltas = deltas[::-1]

        # WEIGHTS UPDATE 
        for layer in np.arange(len(self.W)):
            gradients = Ac[layer].T.dot(deltas[layer])
            self.W[layer] -= self.lr * gradients

    def predict(self, x, bias=True):
        
        if bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))

        for layer in range(len(self.W) - 1):
            x = self.activations[self.activation_name](np.dot(x, self.W[layer]))

        return linear_activation(np.dot(x, self.W[-1]))

    def loss(self, X, y):
        predictions = self.predict(X, bias=False)
        return np.mean((y - predictions) ** 2)  # MSE loss
