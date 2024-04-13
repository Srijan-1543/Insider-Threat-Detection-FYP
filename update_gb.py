import pandas as pd

# Load the dataset with selected features
selected_features_df = pd.read_csv("DR/gb_out_142.csv")

# Load the original dataset with insider labels
original_data = pd.read_csv("binary_data/day.csv")

# Append the 'insider' column from the original dataset
selected_features_df['insider'] = original_data['insider']

# Save the updated dataset with the 'insider' column
updated_data_csv_path = "gb_out_142_with_insider.csv"
selected_features_df.to_csv(updated_data_csv_path, index=False)
print("Dataset with selected features and insider label saved to", updated_data_csv_path)
