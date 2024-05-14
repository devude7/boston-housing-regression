import numpy as np


# ex. layers = [2, 2, 1] = len = 3
class NeuralNetwork:
    def __init__(self, layers, alpha=0.01):
        self.layers = layers
        self.alpha = alpha
        self.W = []

        # initialize weights
        for layer in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[layer] + 1, layers[layer+1] + 1) / np.sqrt(layers[layer])
            self.W.append(w)

        w = np.random.randn(layers[-2] + 1, layers[-1]) / np.sqrt(layers[-2])
        self.W.append(w)
    def __repr__(self):
        return 'Neural Network: {}'.format(str(layer) for layer in self.layers)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display=100):
        # bias trick
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        for epoch in np.arange(epochs):
            for x, target in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % display == 0:
                loss = self.calc_loss(X, y)
                print(f"[INFO] epoch={epoch + 1}, loss={loss:.7f}")


    def fit_partial(self, x, y):
        # FORWARD PASS
        activations = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.W)):
            preds = np.dot(activations[layer], self.W[layer])
            out = self.sigmoid(preds)

            activations.append(out)

        # BACKPROPAGATION
        error = activations[-1] - y
        deltas = [error * self.sigmoid_deriv(activations[-1])]

        for layer in np.arange(len(self.W) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.W[layer].T)
            delta = delta * self.sigmoid_deriv(activations[layer])

            deltas.append(delta)

        deltas = deltas[::-1]

        # UPDATE WEIGHTS
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * np.dot(activations[layer].T, deltas[layer])


    def predict(self, x, add_bias=True):
        # bias trick
        pred = np.atleast_2d(x)
        if add_bias:
            pred = np.hstack((pred, np.ones((pred.shape[0], 1))))

        for layer in np.arange(0, len(self.W)):
            pred = self.sigmoid(np.dot(pred, self.W[layer]))

        return pred

    def calc_loss(self, x, targets):
        targets = np.atleast_2d(targets)
        preds = self.predict(x, add_bias=False)
        loss = 0.5 * np.sum((preds - targets) ** 2)

        return loss
    

