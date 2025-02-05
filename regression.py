import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# Hyperparameters
epochs = 100
lr = 0.01

data = data.drop('BIAS_COL', axis=1)
X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train 
nn = NeuralNetwork([X.shape[1], 64, 32, 16, 1], epochs, "sigmoid", lr)
nn.fit(X_train, y_train)

# Evaluate 
predictions = nn.predict(X_test)
mse = np.mean((y_test - predictions) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

#  Regresion graph
plt.scatter(y_test, predictions)
plt.title("Regression - Boston Housing")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

# Plot real and predicted values
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real values')
plt.scatter(range(len(predictions)), predictions, color='green', label='Predicted values', marker='x')
plt.title("Real vs. Predicted")
plt.legend()
plt.show()

# Change to 1-dim arrays for plotting
y_test = np.array(y_test).flatten()
predictions = np.array(predictions).flatten()

# Plot lines to link predicted to real values
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Real values')
plt.scatter(range(len(predictions)), predictions, color='green', label='Predicted values', marker='x')

for i, (yt, yp) in enumerate(zip(y_test, predictions)):
    plt.plot([i, i], [yt, yp], 'gray', linestyle='dotted', alpha=0.6)

plt.title("Real vs. Predicted")
plt.legend()
plt.show()