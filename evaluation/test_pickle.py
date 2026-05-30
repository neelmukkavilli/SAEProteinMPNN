
import pickle
import pandas as pd
import numpy as np

# Node Encodings
csv_outputs = ['created_data/encodings/output_dense_0.pkl']
#for i in range(3):
#    csv_outputs.append(f'created_data/encodings/output_' + 'validlosstest' + '_' + str(i) + '.pkl')

for file in csv_outputs:
    dfs = []
    with open(file, "rb") as f:
        while True:
            try:
                dfs.append(pickle.load(f))
            except EOFError:
                break

    full_df = pd.concat(dfs, ignore_index=True)
    print(full_df)

# Features
'''
file = "data_labeling/test_edge_features.pkl"
with open(file, "rb") as pkl:
    df = pickle.load(pkl)
print(df)
'''
#import pickle

# Edge Encodings
'''
encoded_all = None
labels_all = None

with open(file, "rb") as f:
    while True:
        try:
            data = pickle.load(f)

            if encoded_all is None:
                encoded_all = data["encoded"]
                labels_all = data["labels"]
            else:
                encoded_all = np.concatenate((encoded_all, data["encoded"]), axis=0)
                labels_all = np.concatenate((labels_all, data["labels"]), axis=0)

        except EOFError:
            break

def build_df(encoded_all, labels_all, neighbor):
    encoded_sel = encoded_all[:, neighbor, :]
    labels_sel = labels_all[:, neighbor]
    df = pd.DataFrame(encoded_sel, index=labels_sel)
    df.index.name = "identifier"
    df = df.sort_values(by='identifier')
    return df

print(build_df(encoded_all, labels_all, 1))
'''