import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import predict, sigmoid_derivative, sigmoid_activation

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# hyperparameters
epochs = 20
lr = 0.01

X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=32)

W = np.random.randn(X.shape[1])

losses = []
for epoch in np.arange(0, epochs):

    #activation
    preds = sigmoid_activation(X_train.dot(W))
    errors = preds - y_train
    
    loss = np.sum(errors ** 2)
    losses.append(loss)

    # gradient descent
    d = errors * sigmoid_derivative(preds)
    gradient = X_train.T.dot(d)
    
    # update weights
    W += -lr * gradient
    
    if epoch == 0 or (epoch + 1) % 1 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))
