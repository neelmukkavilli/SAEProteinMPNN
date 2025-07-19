import numpy as np
import csv
import pandas as pd

csv_input = 'output_encodings_lsamples500_2.csv'
csv_output = 'normalized_encodings_lsamples500_2.csv'

def clean(csv_input):
    df = pd.read_csv(csv_input, skiprows=1)
    col_name = df.columns[0]
    mask = ~df[col_name].astype(str).str.startswith('1a3b')

    df = df[mask]
    df.to_csv(csv_input, index = False)

def normalize(arr):
    arr[arr < 0] = 0
    max_val = arr.max()
    min_val = arr.min()
    if max_val != min_val:
        normalized_col = (arr - min_val)/(max_val - min_val)
    else:
        normalized_col = (arr - min_val)

    return np.round(normalized_col, decimals=5)

res_labels = pd.read_csv(csv_input, usecols=[0])
df = pd.read_csv(csv_input, usecols=range(1, 1025))
print(df.shape)
normalized_df = pd.DataFrame()

for col_name in df.columns:
    col = pd.to_numeric(df[col_name], errors='coerce')
    normalized_df[col_name] = normalize(col)
    print(f"Column '{col_name}' normalized")

#df = df.apply(lambda col: normalize(pd.to_numeric(col, errors='coerce')), axis=0)

combined = pd.concat([res_labels, normalized_df], axis=1)
print(combined.shape)
combined.to_csv(csv_output, index=False)

'''
#normalized_encodings = pd.DataFrame([])
res_labels = pd.read_csv(csv_input, usecols=[0])
res_labels.to_csv(csv_output, header=True, index=False)

for col in range(1, 10):
    df = pd.read_csv(csv_input, usecols=[col], index_col=False)
    arr = pd.to_numeric(df.iloc[:,0], errors='coerce')
    new_arr = normalize(arr)

    new_arr.to_frame().T.to_csv(csv_output, mode='a', header=(col == 1))
    print(f"Col: {col} completed")
'''
#normalized_encodings.columns = list(range(1, 1025))
#res_labels = pd.read_csv(csv_input, usecols=[0])
#res_labels = df.iloc[:, 0]
#res_labels.columns = ['identifier']
#print(res_labels.columns)
#normalized_encodings = pd.concat([res_labels, normalized_encodings], axis=1)
#normalized_encodings.to_csv(csv_output, header=True, index=False)
