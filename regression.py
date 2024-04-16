import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import predict, sigmoid_derivative

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# hyperparameters
epochs = 2
lr = 0.01

X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=32)

W = np.random.randn(X.shape[1])

losses = []
for epoch in range(epochs):
    prediction = predict(X_train, W)
    errors = y_train - prediction
    losses.append(errors.sum())
    W += -lr * X_train.T.dot(errors)

    print(losses)
