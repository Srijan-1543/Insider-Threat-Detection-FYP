from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path = 'Data/day.csv'  # Assuming 'Data' folder exists
dataset = pd.read_csv(path)

# Separate features and labels
X = dataset.drop(["starttime", "endtime", "user", "day", "week", "insider"], axis=1)
y = dataset["insider"]

# Initialize and fit Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Get feature importances with corresponding column names
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)

# Sort features by importances in descending order
sorted_features = feature_importances.sort_values(ascending=False)

# Print sorted feature names and importances
print("Sorted Features by Importance:")
for idx, name in sorted_features.items():
    print(f"  - {name}: {feature_importances[idx]}")

# Create a DataFrame for saving results
results_df = pd.DataFrame({'Feature Name': sorted_features.index, 'Importance': sorted_features.values})

# Save results to a CSV file
results_df.to_csv('feature_importances.csv', index=False)

plt.figure(figsize=(10, 6))
plt.bar(sorted_features.index, sorted_features.values)
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)  # Rotate feature names for readability
plt.tight_layout()
plt.show()

