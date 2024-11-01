# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive')

# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import numpy as np

# Read Files
demographics = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/telecom_demographics.csv")
usage = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/telecom_usage.csv")
# EDA
print(demographics.head())
print(usage.head())
print(demographics['customer_id'].nunique())
print(usage['customer_id'].nunique())
print(demographics.shape)
print(usage.shape)

# Join 2 dataset: inner join as shape and unique count for join key is the same.
churn_df = pd.merge(demographics, usage, how = 'inner', on = 'customer_id')
churn_df['registration_event'] = pd.to_datetime(churn_df['registration_event'])
churn_df['registration_year'] = churn_df['registration_event'].dt.year
churn_df['registration_month'] = churn_df['registration_event'].dt.month
print(churn_df.head())
print(churn_df.isna().sum())
churn_df = pd.get_dummies(churn_df, columns=['telecom_partner', 'gender', 'city'], drop_first=True)
churn_df = churn_df.drop(columns=["state", 'customer_id', 'pincode','registration_event'])

# Train_test_split
X = churn_df.drop(columns='churn')
y = churn_df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state=42)

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
non_num_cols = X_train.select_dtypes(include=['uint8']).columns

# Standardize
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
X_train.head()

# Convert boolean columns to integers
X_train = X_train.astype({col: 'int' for col in X_train.select_dtypes(include=['bool']).columns})
X_test = X_test.astype({col: 'int' for col in X_test.select_dtypes(include=['bool']).columns})

# Convert df to tensor
X_train_tensor = torch.tensor(X_train.values).float()
X_test_tensor = torch.tensor(X_test.values).float()
y_train_tensor = torch.tensor(y_train.values).float()
y_test_tensor = torch.tensor(y_test.values).float()

# Batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)
test_loader = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

# Define Neural Network Structure
class ChurnPredictNN(nn.Module):
    def __init__(self, input_dimension):
        super(ChurnPredictNN, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 256)
        self.leaky_relu1 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(256,128)
        self.leaky_lrelu2 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(128,64)
        self.leaky_lrelu3 = nn.LeakyReLU(0.01)
        self.fc4 = nn.Linear(64,32)
        self.leaky_lrelu4 = nn.LeakyReLU(0.01)
        self.fc5 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid() # activation for output
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_lrelu2(x)
        x = self.fc3(x)
        x = self.leaky_lrelu3(x)
        x = self.fc4(x)
        x = self.leaky_lrelu4(x)
        x = self.fc5(x)
        return self.sigmoid(x)

# Initialize Neural Network
input_dimension = X_train_tensor.shape[1]
model = ChurnPredictNN(input_dimension)

# Loss Function & Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0004, momentum = 0.7)

# Training Loop
num_epoch = 5000
loss_values = [] # for loss visualizing
for epoch in range(num_epoch):
    for batch_X, batch_y in train_loader:
        model.zero_grad() # clear previous gradients
        # Forward Pass
        output = model(batch_X)
        loss = criterion(output, batch_y.view(-1,1))

        # Backward Pass & Optimize
        loss.backward()
        optimizer.step()

    # Print loss every n epochs
    if (epoch+1) % 25 == 0:
        loss_values.append(loss.item())
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}') #round loss to 4 decimals

# Visualize loss in training loop
plt.plot(loss_values)
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

"""##Hyperparameter Tuning:

- num_epoch: 5000
- batch_size: 80
- learning_rate: 0.004
- momentom: 0.7
"""

# Save the model
torch.save(model.state_dict(), 'model_weights.pth')
# Set model to evaludation mode
model.eval()

# Set up bianry accuracy metric
acc = Accuracy(task='binary')

# Evaludate on Testing set
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        # Forward pass to get predictions
        outputs = model(batch_X)
        # Convert the output to binary class
        predicted = (outputs >= 0.5).float()
        acc(predicted, batch_y.view(-1,1))

# Calculate accuracy
accuracy = acc.compute()
print(f"Test Accuracy: {accuracy:.4f}")
