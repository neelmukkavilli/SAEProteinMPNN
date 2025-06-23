import numpy as np
import csv
import pandas as pd

csv_input = 'output_encodings.csv'
csv_output = 'normalized_encodings.csv'

def normalize(arr):
    max_val = arr.max()
    min_val = arr.min()
    if max_val != min_val:
        normalized_col = (arr - min_val)/(max_val - min_val)
    else:
        normalized_col = (arr - min_val)

    return np.round(normalized_col, decimals=5)

df = pd.read_csv(csv_input)
normalized_encodings = pd.DataFrame([])

for col in range(1, df.shape[1]):
    arr = df.iloc[:,col].astype(float)
    new_arr = normalize(arr)
    normalized_encodings = pd.concat([normalized_encodings, new_arr], axis=1)
    
normalized_encodings.columns = list(range(1, 1025))
res_labels = df.iloc[:, 0]
#res_labels.columns = ['identifier']
#print(res_labels.columns)
normalized_encodings = pd.concat([res_labels, normalized_encodings], axis=1)
normalized_encodings.to_csv(csv_output, header=True, index=False)
