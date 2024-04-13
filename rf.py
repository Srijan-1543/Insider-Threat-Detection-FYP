from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = 'Data/day.csv'  
dataset = pd.read_csv(path)

X = dataset.drop(["insider"], axis=1)
y = dataset["insider"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 8, 12]}
rf_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)  # 5-fold cross-validation
rf_grid.fit(X_scaled, y)

best_rf = rf_grid.best_estimator_
best_params = rf_grid.best_params_
print("Best parameters:", best_params)
# Get feature importances with corresponding column names
feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)

# Sort features by importances in descending order
sorted_features = feature_importances.sort_values(ascending=False)

# Print sorted feature names and importances
print("Sorted Features by Importance:")
for idx, name in sorted_features.items():
    print(f"  - {name}: {feature_importances[idx]}")

# Create a DataFrame for saving results
results_df = pd.DataFrame({'Feature Name': sorted_features.index, 'Importance': sorted_features.values})
results_df.to_csv('rf_feature_importances.csv', index=False)

selected_features = sorted_features[sorted_features > 0]
selected_features_csv_path = "rf_selected_features.csv"
X_selected = X[selected_features.index.tolist()]
X_selected.to_csv(selected_features_csv_path, index=False)
print(f"Dataset with selected features saved to {selected_features_csv_path}")

plt.figure(figsize=(10, 6))
plt.bar(sorted_features.index, sorted_features.values)
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)  # Rotate feature names for readability
plt.tight_layout()
plot_image_path = 'rf_feature_importance_plot.png'
plt.savefig(plot_image_path)
print(f"Feature importance plot saved to {plot_image_path}")
plt.show()

