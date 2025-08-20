import pandas as pd
import numpy as np

csv_input = 'encodings/output_encodings_verysparse25e2.csv'
csv_output = 'encodings/normalized_encodings_verysparse25e2.csv'

def normalize(arr):
    arr[arr < 0] = 0
    max_val, min_val = arr.max(), arr.min()
    if max_val != min_val:
        normalized = (arr - min_val) / (max_val - min_val)
    else:
        normalized = arr - min_val
    return np.round(normalized, decimals=5)

# Read input data
#res_labels = pd.read_csv(csv_input, usecols=[0])
df = pd.read_csv(csv_input).dropna()
print(df.shape)
res_labels = df.iloc[:, 0]
print(res_labels.shape)
print(res_labels)
df = df.iloc[:, 1:]
print(df.shape)

# Normalize all columns in one pass
normalized_array = np.apply_along_axis(normalize, 0, df.to_numpy())

# Build normalized DataFrame in one step
normalized_df = pd.DataFrame(normalized_array, columns=df.columns)
print(normalized_df.shape)
print(res_labels.shape)
# Combine and save
combined = pd.concat([res_labels, normalized_df], axis=1)
print(f"Final shape: {combined.shape}")
print(combined.columns)
combined.to_csv(csv_output, index=False)
print("csv written", flush=True)
