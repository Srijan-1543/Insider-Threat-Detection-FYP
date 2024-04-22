import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from itertools import combinations
from torch_geometric.nn import GATConv
import sys
import random
from tqdm import tqdm

# output_file = "gat_output.txt"
# def save_output_to_file(output_file):
#     sys.stdout = open(output_file, "w")
# save_output_to_file(output_file)

def create_edge_index(x, user_dict, adj):
    num_rows = x.shape[0]
    print(f"Rows: {num_rows}")
    adj_matrix = np.zeros((num_rows, num_rows)) 
    user_connections = {}
    
    for user, idx in user_dict.items():
        user_connections[user] = set(np.where(adj[idx] == 1)[0])
        
    for i in tqdm(range(num_rows), desc="Processing Row"):
        user_i = x.iloc[i]['user']         
        adj_matrix[i][i] = 1
        for conn_idx in user_connections[user_i]:
            rows_to_connect = x.index[x['user'] == list(user_dict.keys())[conn_idx]].tolist()
            for j in rows_to_connect:
                adj_matrix[i][j] = 1
                            
    edge_index = torch.LongTensor(np.nonzero(adj_matrix)) 
    return edge_index

# Load tabular data
data = pd.read_csv(r"Test Purpose\Bcitd\day10f_train.csv")
test = pd.read_csv(r"Test Purpose\Bcitd\day10f_test.csv")
users = pd.read_csv(r"C:\Users\srija\Desktop\FYP\Data\userListFromExtraction.csv")
adj = pd.read_csv(r"Knowledge Graph\KnowledgeGraph_AdjacencyMatrix.csv")

user_dict = {i: id for (i, id) in enumerate(users.uid)}
# data = data.groupby(['user', 'insider']).mean().reset_index()

# Split data into features (X) and labels (y)
X_train = data.drop(columns=["insider"])  # Features (excluding the label column)
y_train = data["insider"]  # Labels

X_test = test.drop(columns=["insider"])  # Features (excluding the label column)
y_test = test["insider"]  # Labels

# Split data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)

# Convert data to PyTorch tensors
edge_index_train = create_edge_index(X_train, user_dict, adj)
edge_index_test = create_edge_index(X_test, user_dict, adj)
edge_index_val =create_edge_index(X_val, user_dict, adj)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)

X_val = torch.FloatTensor(X_val.values)
y_val = torch.LongTensor(y_val.values)

X_test = torch.FloatTensor(X_test.values)
y_test = torch.LongTensor(y_test.values)

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads,dropout):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=output_dim, heads=num_heads, concat=False)
        self.gat = GATConv(in_channels=input_dim, out_channels=output_dim, heads=num_heads, concat=False)
        self.fc = nn.Linear(hidden_dim*num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return F.softmax(x, dim=1)  
   
# Set hyperparameters
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 64
output_dim = 2  # Multi-Class classification task
num_heads = 3
dropout = 0.2

# Initialize GAT model
model = GAT(input_dim, hidden_dim, output_dim, num_heads, dropout)

# Define loss function and optimizer
weight_decay = 1e-5
weight_decay_l2 = 5e-4  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

# Training loop
num_epochs = 100 
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')  # Initialize with a large value
best_model_path = "gat_best_model.pth"

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # permuted_indices = torch.randperm(X_train_tensor.size(0))
    # X_train_tensor = X_train_tensor[permuted_indices]
    # edge_index_train = edge_index_train[permuted_indices]
    outputs = model(X_train, edge_index_train)
    loss = criterion(outputs, y_train)
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    loss += weight_decay_l2 * l2_reg
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    _, predicted_train = torch.max(outputs, 1)
    train_accuracy = accuracy_score(y_train.numpy(), predicted_train.numpy())
    train_accuracies.append(train_accuracy)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val, edge_index_val)
        val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
        _, predicted_val = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val.numpy(), predicted_val.numpy())
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
    outputs = model(X_test, edge_index_test)
    _, predicted = torch.max(outputs, 1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    precision = precision_score(y_test.numpy(), predicted.numpy(), average='weighted')
    recall = recall_score(y_test.numpy(), predicted.numpy(), average='weighted')
    f1 = f1_score(y_test.numpy(), predicted.numpy(), average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    target_names = ["0","1"]
    print(classification_report(predicted.numpy(), y_test.numpy(), target_names=target_names))

    classes = ["0","1"]
    cm = confusion_matrix(y_test.numpy(), predicted.numpy())
    print('Confusion Matrix:')
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title("Confusion Matrix - GAT")
    disp = disp.plot(ax=ax)
    plt.savefig('Confusion Matrix - GAT.png')
    plt.show()

    False_positive_rate = confusion_matrix(y_test.numpy(), predicted.numpy())[0, 1] / (confusion_matrix(y_test.numpy(), predicted.numpy())[0, 1] + confusion_matrix(y_test.numpy(), predicted.numpy())[0, 0])
    print('False Positive Rate:', False_positive_rate)

    False_negative_rate = confusion_matrix(y_test.numpy(), predicted.numpy())[1, 0] / (confusion_matrix(y_test.numpy(), predicted.numpy())[1, 0] + confusion_matrix(y_test.numpy(), predicted.numpy())[1, 1])
    print('False Negative Rate:', False_negative_rate)

    True_positive_rate = confusion_matrix(y_test.numpy(), predicted.numpy())[1, 1] / (confusion_matrix(y_test.numpy(), predicted.numpy())[1, 1] + confusion_matrix(y_test.numpy(), predicted.numpy())[1, 0])
    print('True Positive Rate:', True_positive_rate)

    True_negative_rate = confusion_matrix(y_test.numpy(), predicted.numpy())[0, 0] /  (confusion_matrix(y_test.numpy(), predicted.numpy())[0, 0] + confusion_matrix(y_test.numpy(), predicted.numpy())[0, 1])
    print('True Negative Rate:', True_negative_rate)

    Balanced_accuracy = balanced_accuracy_score(y_test.numpy(), predicted.numpy())
    print('Balanced Accuracy:', Balanced_accuracy)

    # TP = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())[1, 1]
    # print('True Positive:', TP)

    # FP = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())[0, 1]
    # print('False Positive:', FP)

    # FN = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())[1, 0]
    # print('False Negative:', FN)

    # TN = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())[0, 0]
    # print('True Negative:', TN)
    
    # Calculate true positive rate (Sensitivity) for each class
    TP = [cm[i, i] / sum(cm[i, :]) for i in range(len(cm))]
    print('True Positive Rates (Sensitivity):', TP)

    # Calculate false positive rate for each class
    FP = [(sum(cm[:, i]) - cm[i, i]) / (sum(cm[:, i])) for i in range(len(cm))]
    print('False Positive Rates:', FP)

    # Calculate true negative rate (Specificity) for each class
    TN = [(sum(sum(cm)) - sum(cm[i, :]) - sum(cm[:, i]) + cm[i, i]) / (sum(sum(cm)) - sum(cm[:, i])) for i in range(len(cm))]
    print('True Negative Rates (Specificity):', TN)

    # Calculate false negative rate for each class
    FN = [(sum(cm[i, :]) - cm[i, i]) / sum(cm[i, :]) for i in range(len(cm))]
    print('False Negative Rates:', FN)

    # Calculate Overall Accuracy
    overall_accuracy = sum(np.diag(cm)) / sum(sum(cm))
    print('Overall Accuracy:', overall_accuracy)

    # Calculate Overall Precision
    overall_precision = sum([cm[i, i] / sum(cm[:, i]) if sum(cm[:, i]) != 0 else 0 for i in range(len(cm))]) / len(cm)
    print('Overall Precision:', overall_precision)

    # Calculate Overall Recall
    overall_recall = sum([cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) != 0 else 0 for i in range(len(cm))]) / len(cm)
    print('Overall Recall:', overall_recall)

    # Calculate Overall F1 Score
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) != 0 else 0
    print('Overall F1 Score:', overall_f1)
    
    # True_negative_rate = TN / (TN + FN)
    # print('True Negative Rate:', True_negative_rate)

    # False_negative_rate = FN / (FN + TP)
    # print('False Negative Rate:', False_negative_rate)

    # True_positive_rate = TP / (TP + FN)
    # print('True Positive Rate:', True_positive_rate)

    # False_positive_rate = FP / (FP + TN)
    # print('False Positive Rate:', False_positive_rate)
    
sys.stdout.close()
sys.stdout = sys.__stdout__
