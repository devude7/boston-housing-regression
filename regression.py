import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers import predict, linear_derivative, linear_activation

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# hyperparameters
epochs = 25
lr = 0.001

scaler = StandardScaler()
X = scaler.fit_transform(data.drop('MEDV', axis=1))
y = scaler.fit_transform(data['MEDV'].values.reshape(-1, 1))
y = y.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=32)

W = np.random.randn(X.shape[1])

losses = []
for epoch in np.arange(0, epochs):

    # activation
    preds = linear_activation(np.dot(X_train, W))
    errors = preds - y_train
    
    loss = np.sum(errors ** 2)
    losses.append(loss)

    # gradient descent
    d = errors * linear_derivative()
    gradient = X_train.T.dot(d)
    
    # update weights
    W += -lr * gradient
    
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))
