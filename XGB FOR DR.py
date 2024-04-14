# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# path = r'XGB DR\train_10.csv'  
# dataset = pd.read_csv(path)

# X = dataset.drop(["insider"], axis=1)
# y = dataset["insider"]

# # Convert labels to binary values (0 and 1)
# y_binary = np.where(y == "Yes", 1, 0)

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Convert data to DMatrix format for XGBoost
# dtrain = xgb.DMatrix(X_scaled, label=y_binary)

# # Define parameters for XGBoost
# params = {
#     'objective': 'binary:logistic',  # Binary classification
#     'eval_metric': 'auc'  # Evaluation metric
# }

# # Train XGBoost model
# num_rounds = 100  # Number of boosting rounds
# xgb_classifier = xgb.train(params, dtrain, num_rounds)

# # Get feature importances
# importance_dict = xgb_classifier.get_fscore()

# # Convert feature importances to DataFrame
# sorted_features = pd.DataFrame(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
#                                columns=['Feature', 'Importance'])

# # Print sorted feature names and importances
# print("Sorted Features by Importance:")
# print(sorted_features)

# # Save results to a CSV file
# results_csv_path = 'xgb_feature_importances.csv'
# sorted_features.to_csv(results_csv_path, index=False)
# print(f"Feature importances saved to {results_csv_path}")

# # Select features with importance greater than 0
# selected_features = sorted_features[sorted_features['Importance'] > 0]

# # Save the dataset with the selected features as a CSV file
# selected_features_csv_path = "xgb_selected_features.csv"
# data = X[selected_features['Feature']]
# data["insider"] = y_binary
# data.to_csv(selected_features_csv_path, index=False)
# print(f"Dataset with selected features saved to {selected_features_csv_path}")

# # Plot the feature importance scores
# plt.figure(figsize=(10, 6))
# plt.bar(selected_features['Feature'], selected_features['Importance'])
# plt.xlabel('Feature Name')
# plt.ylabel('Feature Importance')
# plt.title('Feature Importance Scores')
# plt.xticks(rotation=45)  # Rotate feature names for readability
# plt.tight_layout()
# plot_image_path = 'xgb_feature_importance_plot.png'
# plt.savefig(plot_image_path)
# print(f"Feature importance plot saved to {plot_image_path}")
# plt.show()

# print(data.shape)
# print(data.head())

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the dataset
path = r'C:\Users\srija\Desktop\ML COMP -BALA\DR-BALA\XGB DR\train_10.csv'  
dataset = pd.read_csv(path)

# Separate features and target variable
X = dataset.drop(["insider"], axis=1)
y = dataset["insider"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Incremental PCA for dimensionality reduction
ipca = IncrementalPCA(n_components=10, batch_size=1000)
X_ipca = ipca.fit_transform(X_scaled)

# Define hyperparameters grid for XGBoost
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 500]
}

# Perform GridSearchCV to find the best XGBoost model
xgb_grid = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5)  # 5-fold cross-validation
xgb_grid.fit(X_ipca, y)

# Get the best XGBoost model and its parameters
best_xgb = xgb_grid.best_estimator_
best_params = xgb_grid.best_params_
print("Best parameters:", best_params)

# Get feature importances from the best XGBoost model
importance_dict = best_xgb.feature_importances_

# Convert feature importances to dictionary
importance_dict = dict(enumerate(importance_dict))

# Convert feature importances to DataFrame with original column names
sorted_features = pd.DataFrame(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
                               columns=['Feature Index', 'Importance'])
sorted_features['Feature'] = X.columns[sorted_features['Feature Index']]

# Save feature importances to a CSV file
results_csv_path = 'xgb_feature_importances.csv'
sorted_features.to_csv(results_csv_path, index=False)
print(f"Feature importances saved to {results_csv_path}")

# Select features with importance greater than 0
selected_features = sorted_features[sorted_features['Importance'] > 0]

# Save the dataset with the selected features as a CSV file
selected_features_csv_path = "xgb_selected_features.csv"
data = X[selected_features['Feature']]
data["insider"] = y
data.to_csv(selected_features_csv_path, index=False)
print(f"Dataset with selected features saved to {selected_features_csv_path}")

# Plot the feature importance scores
plt.figure(figsize=(10, 6))
plt.bar(selected_features['Feature'], selected_features['Importance'])
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)  # Rotate feature names for readability
plt.tight_layout()
plt.show()

