import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

data = data.drop('BIAS_COL', axis = 1)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)


