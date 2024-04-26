# df = pd.read_csv(r"F:\FYP\Data\multi\Preprocessed_Data\week.csv")
# final_df.to_csv('binary_week_preprocessed.csv', index=False)

import pandas as pd

# Read the input CSV file
df = pd.read_csv(r"F:\FYP\Data\multi\Preprocessed_Data\week.csv")

# Convert 1, 2, 3 to 1 in the 'insider' column
df.loc[df['insider'].isin([1, 2, 3]), 'insider'] = 1

# Aggregate the data by 'user' and 'insider', counting occurrences
user_counts = df.groupby(['user', 'insider']).size().reset_index(name='count')

# Identify users classified as both 0 and 1 at different times
conflict_users = user_counts[user_counts.groupby('user')['insider'].transform('nunique') > 1]['user'].unique()

# Create a new DataFrame to store the final result
final_df = pd.DataFrame(columns=df.columns)

# Process non-conflicting users
non_conflict_records = df[~df['user'].isin(conflict_users)]
final_df = pd.concat([final_df, non_conflict_records], ignore_index=True)

# Process conflicting users
for user in conflict_users:
    user_records = df[(df['user'] == user) & (df['insider'] == 1)]
    final_df = pd.concat([final_df, user_records], ignore_index=True)

# Print the final counts
print(final_df['insider'].value_counts())

# Print the shape (number of rows) for insider and non-insider classes
print(f"Shape of insider class (1): {final_df[final_df['insider'] == 1].shape[0]}")
print(f"Shape of non-insider class (0): {final_df[final_df['insider'] == 0].shape[0]}")

# Write the final DataFrame to a CSV file
final_df.to_csv('binary_week_preprocessed.csv', index=False)