import pandas as pd
import matplotlib.pyplot as plt

# Read the feature importances from the CSV file
importance_df = pd.read_csv("rf_feature_importances.csv")

# Sort features by importances in descending order
sorted_features = importance_df["Feature Name"].sort_values(ascending=False)

# Print sorted feature names and importances
print("Sorted Features by Importance:")
for name, importance in importance_df.values:
    print(f"- {name}: {importance}")

# Select features with scores greater than 0
selected_features = sorted_features[importance_df["Importance"] > 0]

# Read the original dataset "day.csv" to include the "insider" column
dataset = pd.read_csv("Data/day.csv")

# Extract X (features) and y (target)
X = dataset[selected_features.tolist()]  # Only select features, don't add "insider" yet
y = dataset["insider"]

# Combine X and y into a single DataFrame
df_out = pd.concat([X, y], axis=1)

# Save the combined DataFrame to "rf_out.csv"
df_out.to_csv("rf_out.csv", index=False)

# Create the feature importance plot
plt.figure(figsize=(10, 6))
plt.bar(sorted_features, importance_df["Importance"])
plt.xlabel('Feature Name')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45)  # Rotate feature names for readability
plt.tight_layout()
plt.savefig('rf_feature_importance_plot.png')
print(f"Feature importance plot saved to rf_feature_importance_plot.png")
