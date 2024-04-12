import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib

df = pd.read_csv('C:/Users/srija/Desktop/FYP/Data/week.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_space = {
    'n_neighbors': (5, 50),
    'min_dist': (0.01, 0.5),
    # 'n_components': (2, 10)
}

# Define UMAP model
umap = UMAP()
search = BayesSearchCV(umap, param_space, n_iter=20, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose= 1)
search.fit(X_train, y_train)

best_umap = search.best_estimator_
best_params = search.best_params_
print("Best parameters:", best_params)
joblib.dump(best_umap, 'best_umap_model.pkl')

y_pred = best_umap.transform(X_test)
# Calculate silhouette score
silhouette = silhouette_score(y_pred, y_test)
print("Silhouette score on test set:", silhouette)

np.savetxt('umap_embedding.csv', y_pred, delimiter=',')

# Visualize the UMAP embedding with the best parameters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], c=y_test)
ax.set_title('3D UMAP Embedding with Best Parameters')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')
plt.savefig('umap_embedding_3d_best_params.png')
plt.show()