import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

# Load train data from CSV -- embeddings
train_df = pd.read_csv(r"F:\FYP\sampled_week dataset\sampled_week_embedding\embedding_train_10.csv")
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Load test data from CSV
test_df = pd.read_csv(r"F:\FYP\sampled_week dataset\sampled_week_embedding\embedding_test.csv")
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Split data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Load UMAP embeddings from pickle file

# Function to create edge index based on label similarity
def create_edge_index(labels):
    edge_index = []
    for i, j in combinations(range(len(labels)), 2):    # using this instead of 2 for loops
        if labels[i] == labels[j]:
            edge_index.append([i, j])
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    return edge_index

# Usage:
edge_index_train = create_edge_index(y_train)
edge_index_val = create_edge_index(y_val)
edge_index_test = create_edge_index(y_test)

# # Visualize graph structure before training
# plt.figure(figsize=(10, 10))
# G = nx.Graph()
# G.add_edges_from(edge_index_train.t().numpy())
# nx.draw(G, with_labels=False, node_size=10)
# plt.title('Graph Structure (Before Training)')
# plt.savefig('graph_structure_before_training.png')
# plt.show()

# # Visualize the embeddings before training
# plt.figure(figsize=(8, 8))
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=10)
# plt.title('UMAP Embeddings Before Training')
# plt.colorbar(label='Label')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.savefig('umap_embeddings_before_training.png')
# plt.show()


# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)

input_dim = X_train.shape[1]  # Dimension of UMAP embedding
output_dim = 4 

# Set hyperparameters -- using grid/ random search
hidden_dim = 16  
learning_rate = 0.01
weight_decay_l2 = 5e-4     # L2 regularization
# weight_decay_l1 = 1e-5     # L1 regularization

model = GCN(input_dim, hidden_dim, output_dim)

# Training loop
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_l2)

num_epochs = 100
best_val_loss = float('inf')

# Lists to store training and validation losses
train_losses = []
val_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train, edge_index_train)  # embedding, adj
    loss = criterion(out, y_train)  

    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    loss += weight_decay_l2 * l2_reg
    # l1_reg = sum(p.abs().sum() for p in model.parameters())
    # loss += weight_decay_l1 * l1_reg
    loss.backward()
    optimizer.step()
    

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val, edge_index_val)
        val_loss = criterion(val_outputs, y_val)

        val_l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        val_loss += weight_decay_l2 * val_l2_reg
        # val_l1_reg = sum(p.abs().sum() for p in model.parameters())
        # val_loss += weight_decay_l1 * val_l1_reg
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Visualization: Plot training and validation losses
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('gcn_train_val_loss_10.png')
plt.show()

# # After the training loop, get the embeddings after training
# model.eval()
# with torch.no_grad():
#     # Obtain embeddings for the entire dataset (train + val + test)
#     all_outputs = model(torch.cat([X_train, X_val, X_test], dim=0), edge_index_train)
#     all_embeddings = all_outputs.cpu().numpy()

# # Visualize graph structure after training
# plt.figure(figsize=(10, 10))
# G = nx.Graph()
# G.add_edges_from(edge_index_train.t().numpy())
# nx.draw(G, with_labels=False, node_size=10)
# plt.title('Graph Structure (After Training)')
# plt.savefig('graph_structure_after_training.png')
# plt.show()

# # Visualize the embeddings
# plt.figure(figsize=(8, 8))
# plt.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=y_train.tolist()+y_val.tolist()+y_test.tolist(), cmap='viridis', s=10)
# plt.title('UMAP Embeddings After Training')
# plt.colorbar(label='Label')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.savefig('umap_embeddings_after_training.png')
# plt.show()


# Test
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test, edge_index_val)  
    test_loss = criterion(test_outputs, y_test)
    test_losses.append(test_loss.item())
    print(f"Test Loss: {test_loss.item()}")

    # Model predictions
    _, predicted = torch.max(test_outputs, 1)   # class with the highest logit value

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    precision = precision_score(y_test.numpy(), predicted.numpy(), average='weighted')
    recall = recall_score(y_test.numpy(), predicted.numpy(), average='weighted')
    f1 = f1_score(y_test.numpy(), predicted.numpy(), average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test.numpy(), predicted.numpy())
    print("Confusion Matrix:")
    print(cm)

    TP = cm[1, 1]
    print('True Positive:', TP)

    FP = cm[0, 1]
    print('False Positive:', FP)

    FN = cm[1, 0]
    print('False Negative:', FN)

    TN = cm[0, 0]
    print('True Negative:', TN)

    epsilon = 1e-7  # Small value to avoid division by zero

    Accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    print('Accuracy:', Accuracy)

    Precision = TP / (TP + FP + epsilon) if (TP + FP) != 0 else 0
    print('Precision:', Precision)

    Recall = TP / (TP + FN + epsilon) if (TP + FN) != 0 else 0
    print('Recall:', Recall)

    F1_score = 2 * Precision * Recall / (Precision + Recall + epsilon) if (Precision + Recall) != 0 else 0
    print('F1 Score:', F1_score)

    True_negative_rate = TN / (TN + FN + epsilon) if (TN + FN) != 0 else 0
    print('True Negative Rate:', True_negative_rate)

    False_negative_rate = FN / (FN + TP + epsilon) if (FN + TP) != 0 else 0
    print('False Negative Rate:', False_negative_rate)

    True_positive_rate = TP / (TP + FN + epsilon) if (TP + FN) != 0 else 0
    print('True Positive Rate:', True_positive_rate)

    False_positive_rate = FP / (FP + TN + epsilon) if (FP + TN) != 0 else 0
    print('False Positive Rate:', False_positive_rate)

