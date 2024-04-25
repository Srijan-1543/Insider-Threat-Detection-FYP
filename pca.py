import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
path = r'BinaryData\day.csv'  
dataset = pd.read_csv(path)

X = dataset.drop(["insider"], axis=1)
y = dataset["insider"]

# Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA on the data
pca = PCA()
pca.fit(X_scaled)

# Calculate the cumulative explained variance
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

# Determine the number of components to explain at least 90% of the variance
n_components = np.argmax(explained_variance_ratio >= 0.9) + 1
print(f"Number of components to explain at least 90% of the variance: {n_components}")

# Perform PCA with the optimal number of components
optimal_pca = PCA(n_components=n_components)
X_pca = optimal_pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

# Concatenate the labels to the transformed data
final_df = pd.concat([pca_df, y.reset_index(drop=True)], axis=1)

# Save the output as a CSV file
output_path = r'Dimensionality Reduction\pca\pca_out.csv'
final_df.to_csv(output_path, index=False)

print(f"Transformed data with labels saved to {output_path}")

# Optional: Visualize the data after PCA transformation (for 2D or 3D)
if n_components <= 3:
    plt.figure(figsize=(8, 6))
    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
    plt.title("PCA Transformed Data")
    plt.show()
