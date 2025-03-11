import sportslabkit as slk
# Load the CSV
df = slk.load_df('/home/lars/Downloads/archive(3)/top_view/annotations/D_20220220_1_0000_0030.csv')

print(df.head())
# Show the multi-index column structure (just for verification)


# Filter columns where the first level is 'BALL'
ball_cols = [col for col in df.columns if col[0] == 'BALL']

# Extract just the BALL data
ball_df = df[ball_cols].copy()

# Create a new frame column (incremental starting from 1)
ball_df.insert(0, 'frame', range(1, len(ball_df) + 1))

# Flatten the column names: we only want the attribute name (3rd level of the MultiIndex)
new_columns = ['frame'] + [col[2] for col in ball_cols]

# Assign the new flat column names
ball_df.columns = new_columns

# Save to CSV
ball_df.to_csv('ball_with_flat_headers.csv', index=False)

print("BALL data with flattened headers saved as 'ball_with_flat_headers.csv'")