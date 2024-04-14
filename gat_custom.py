import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def create_edge_index(X, y):
    num_nodes = X.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    data_col = X.columns.tolist() 
    X=X.values
    y=y.values
    columns = ['f_unit', 'dept', 'team']

    for col in columns:
        unique_values = np.unique(X[:, data_col.index(col)])
        for value in unique_values:       
            indices = np.where(X[:, data_col.index(col)] == value)[0] # Filter array to get indices of rows with the same value in the column
            adj_matrix[indices[:, None], indices] = 1
            
    return adj_matrix

# Define a simple Graph Attention Network (GAT) layer
class GATLayer(nn.Module):
    def __init__(self, input_dim):
        super(GATLayer, self).__init__()
        self.W_query = nn.Linear(input_dim, input_dim)
        self.W_key = nn.Linear(input_dim, input_dim)
        self.W_value = nn.Linear(input_dim, input_dim)

    def forward(self, X, adj):
        # Linear transformation to obtain query, key, and value matrices
        query = self.W_query(X)
        key = self.W_key(X)
        value = self.W_value(X)
        
        # Calculate dot product attention
        dot_products = torch.matmul(query, key.transpose(0, 1)) / torch.sqrt(torch.tensor(X.size(-1), dtype=torch.float32))
        dot_products[adj == 0] = 0
        # Softmax to get attention weights
        attention_weights = F.softmax(dot_products, dim=1)

        # Weighted sum to compute attention output
        attention_output = torch.matmul(attention_weights, value)

        return attention_output

# Define the GAT model
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout):
        super(GAT, self).__init__()
        self.gat_layer = GATLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj):
        gat_output = self.gat_layer(X, adj) # Apply GAT layer
        gat_output = torch.flatten(gat_output, start_dim=1) # Flatten the output
        gat_output = self.dropout(gat_output) # Apply dropout
        fc_output = F.leaky_relu(self.fc1(gat_output), negative_slope=0.2) # Apply fully connected layer with Leaky ReLU activation
        output = F.softmax(self.fc2(fc_output), dim=1) 
        return output

# Load tabular data
data = pd.read_csv("sampledData_multi\day_train_10.csv")
test = pd.read_csv("sampledData_multi\day_test.csv")

# Split data into features (X) and labels (y)
X_train = data.drop(columns=["insider"])
y_train = data["insider"]

X_test = test.drop(columns=["insider"])
y_test = test["insider"]

# Split data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
adj_train = create_edge_index(X_train, y_train)
adj_test = create_edge_index(X_test, y_test)
adj_val =create_edge_index(X_val, y_val)

adj_train_tensor = torch.tensor(adj_train, dtype=torch.float)
adj_test_tensor = torch.tensor(adj_test, dtype=torch.float)
adj_val_tensor = torch.tensor(adj_val, dtype=torch.float)

X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)

X_val_tensor = torch.FloatTensor(X_val.values)
y_val_tensor = torch.LongTensor(y_val.values)

X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Define hyperparameters
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 4
num_heads = 1
dropout = 0.5

# Initialize GAT model
model = GAT(input_dim, hidden_dim, output_dim, num_heads, dropout)

# Define loss function and optimizer
weight_decay = 1e-5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')  # Initialize with a large value
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor, adj_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    _, predicted_train = torch.max(outputs, 1)
    train_accuracy = accuracy_score(y_train_tensor.numpy(), predicted_train.numpy())
    train_accuracies.append(train_accuracy)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor, adj_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
        _, predicted_val = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val_tensor.numpy(), predicted_val.numpy())
        val_accuracies.append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved the best model with validation loss:", val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")

# Save the trained model
torch.save(model.state_dict(), "gat_model.pth")

# Plotting training and validation loss
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('gat_training_validation_loss_plot.png')
plt.show()

# Plotting training and validation accuracy
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('gat_training_validation_accuracy_plot.png')
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor, adj_test_tensor)
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