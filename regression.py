import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from backpropagation import NeuralNetwork

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# hyperparameters
epochs = 25
lr = 0.001

data = data.drop('BIAS_COL', axis=1)
X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=32)

nn = NeuralNetwork([X.shape[1], 16, 8, 1], lr)
nn.fit(X_train, y_train, epochs=1000)


