import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

path = 'Data/day.csv'  
dataset = pd.read_csv(path)

X = dataset.drop(["insider"], axis=1)
y = dataset["insider"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_scaled, y)

# Get feature importances with corresponding column names
feature_importances = pd.Series(gb_classifier.feature_importances_, index=X.columns)

# Sort features by importances in descending order
sorted_features = feature_importances.sort_values(ascending=False)

# Print sorted feature names and importances
print("Sorted Features by Importance:")
for feature, importance in sorted_features.items():
    print(f"  - {feature}: {importance}")

# Create a DataFrame for saving results
results_df = pd.DataFrame({'Feature Name': sorted_features.index, 'Importance': sorted_features.values})

# Save results to a CSV file
results_csv_path = 'gb_feature_importances.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Feature importances saved to {results_csv_path}")

selected_features = sorted_features[sorted_features > 0]

# Save the dataset with the selected features as a CSV file
selected_features_csv_path = "gb_selected_features.csv"
data = X[selected_features.index]
data["insider"] = y
data.to_csv(selected_features_csv_path, index=False)
print(f"Dataset with selected features saved to {selected_features_csv_path}")

# Plot the feature importance scores
plt.figure(figsize=(10, 6))
plt.bar(selected_features.index, selected_features.values)
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)  # Rotate feature names for readability
plt.tight_layout()
plot_image_path = 'gb_feature_importance_plot.png'
plt.savefig(plot_image_path)
print(f"Feature importance plot saved to {plot_image_path}")
plt.show()

print(data.shape)
print(data.head())