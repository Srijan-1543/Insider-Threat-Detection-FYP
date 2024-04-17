import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load tabular data
data = pd.read_csv("your_data.csv")

# Preprocess data (handle missing values, encode categorical variables, etc.)
# For simplicity, we assume the data is already preprocessed and ready for modeling

# Split data into features (X) and labels (y)
X = data.drop(columns=["insider"]).values  # Features (excluding the label column)
y = data["insider"].values  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)  # Assuming labels are integers

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)  # Assuming labels are integers

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
num_heads = 2

# Initialize GAT model
model = GAT(input_dim, hidden_dim, output_dim, num_heads)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data into PyTorch DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Accuracy on test set: {accuracy:.2f}")
