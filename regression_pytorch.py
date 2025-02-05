import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("hou_all.csv")
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

# Hyperparameters
epochs = 1000
lr = 0.005
activation = "sigmoid"

data = data.drop('BIAS_COL', axis=1)
X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]

y = np.array(y)
y = np.reshape(y, (-1, 1))

# Normalization 
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform nparrays to torch.tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

Model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.Sigmoid() if activation == "sigmoid" else nn.ReLU(),
    nn.Linear(64, 32),
    nn.Sigmoid() if activation == "sigmoid" else nn.ReLU(),
    nn.Linear(32, 16),
    nn.Sigmoid() if activation == "sigmoid" else nn.ReLU(),
    nn.Linear(16, 1),
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(Model.parameters(), lr)

for epoch in np.arange(0, epochs):
    optimizer.zero_grad() # Reset gradients
    outputs = Model(X_train) # Forward pass
    loss = criterion(outputs, y_train) # Calculate loss with Mean Squared Error
    loss.backward() # Backpropagation - compute gradients 
    optimizer.step() # Weight update based on gradients and optimizer

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Eval
with torch.no_grad():
    y_pred = Model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

