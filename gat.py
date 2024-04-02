import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load tabular data
data = pd.read_csv("SampledData/day/train_10.csv")

# Preprocess data (handle missing values, encode categorical variables, etc.)
# For simplicity, we assume the data is already preprocessed and ready for modeling

# Split data into features (X) and labels (y)
X = data.drop(columns=["starttime","endtime","user","day","week","insider"]).values  # Features (excluding the label column)
y = data["insider"].values  # Labels

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Define a simple Graph Attention Network (GAT) model
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        self.gat_layer = nn.MultiheadAttention(input_dim, num_heads)
        self.fc = nn.Linear(input_dim * num_heads, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # Apply GAT layer
        attn_output, _ = self.gat_layer(X, X, X)
        attn_output = torch.flatten(attn_output, start_dim=1)

        # Apply fully connected layer
        fc_output = torch.relu(self.fc(attn_output))

        # Apply output layer
        output = self.output_layer(fc_output)

        return output

# Set hyperparameters
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 64
output_dim = 4  # Multi-Class classification task
num_heads = 1

# Initialize GAT model
model = GAT(input_dim, hidden_dim, output_dim, num_heads)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Plotting training and validation loss
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
    precision = precision_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    recall = recall_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())
    print("Confusion Matrix:")
    print(cm)
