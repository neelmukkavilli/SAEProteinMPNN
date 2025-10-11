import pandas as pd
import numpy as np
import os

csv = str(os.getenv("CSV"))
if csv == 'None':
    csv = 'dense'

csv = "firstedgemodel"    
csv_input = 'encodings/output_encodings_' + csv + '.csv'
csv_output = 'encodings/normalized_encodings_' + csv + '.csv'

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
df = df.iloc[:, 1:]

# Normalize all columns in one pass
normalized_array = np.apply_along_axis(normalize, 0, df.to_numpy())

# Build normalized DataFrame in one step
normalized_df = pd.DataFrame(normalized_array, columns=df.columns)
# Combine and save
combined = pd.concat([res_labels, normalized_df], axis=1)
print(f"Final shape: {combined.shape}")
print(combined.columns)
combined.to_csv(csv_output, index=False)
print("csv written", flush=True)
