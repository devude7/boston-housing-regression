import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# Hyperparameters
epochs = 200
lr = 0.01

data = data.drop('BIAS_COL', axis=1)
X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=32)

# Train the network
nn = NeuralNetwork([X.shape[1], 64, 32, 16, 1], epochs, "sigmoid", lr)
nn.fit(X_train, y_train)

print("[INFO] training network...")

# Evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(X_test)

mse = np.mean((y_test - predictions) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
