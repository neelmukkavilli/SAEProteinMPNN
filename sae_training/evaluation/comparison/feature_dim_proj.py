import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr

csv = 'newlsamples500_2'

encodings_csv = '../encodings/normalized_encodings_' + csv + '.csv'
features_csv = '../node_features.csv'

encodings = pd.read_csv(encodings_csv, header=0)
features = pd.read_csv(features_csv, header=0)

# Filter out incomplete or missing rows
encodings = encodings.dropna(axis='index')
features = features.dropna(axis='index')
matching_ids = set(encodings['identifier']) & set(features['identifier'])
encodings = encodings[encodings['identifier'].isin(matching_ids)]
encodings = encodings.sort_values('identifier').reset_index(drop=True)
features = features[features['identifier'].isin(matching_ids)]
features = features.sort_values('identifier').reset_index(drop=True)

# Split up labels from encodings
id_labels = encodings.iloc[:, 0]
encodings = encodings.iloc[:, 1:]
features = features.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier (0.1 cluster mask level)
#print("Encoding data shape before mask", encodings.shape)
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < 0.01
encodings = encodings.iloc[mask, :].reset_index(drop=True)
id_labels = id_labels.iloc[mask].reset_index(drop=True)
features = features.iloc[mask, :].reset_index(drop=True)


encodings = normalize(encodings, 'l2', axis=1)
encodings = pd.DataFrame(encodings)

'''
encodings_csv = '../encodings/normalized_encodings_' + csv + '.csv'
features_csv = '../node_features.csv'

encodings = pd.read_csv(encodings_csv, header=0)
features = pd.read_csv(features_csv, header=0)
encodings = encodings.dropna(axis='index')
matching_ids = set(encodings['identifier']) & set(features['identifier'])

encodings = encodings[encodings['identifier'].isin(matching_ids)]
encodings = encodings.sort_values('identifier').reset_index(drop=True)
features = features[features['identifier'].isin(matching_ids)]
features = features.sort_values('identifier').reset_index(drop=True)
print(encodings.shape)
print(encodings.iloc[:5, :5])

res_labels = encodings.iloc[:, 0]
encodings = encodings.iloc[:, 1:]
encodings = pd.DataFrame(normalize(encodings, 'l2', axis=1))

np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < 0.1
encodings = encodings.iloc[mask, :]
res_labels = res_labels.iloc[mask]
print(encodings.shape)
'''

feature = 'bfactor'
characteristic = 10
res_labels = features.index[features[feature] < characteristic]
low_ASA_encodings = encodings.loc[res_labels]
low_mean = low_ASA_encodings.mean(axis=0)

characteristic = 50
res_labels = features.index[features[feature] > characteristic]
high_ASA_encodings = encodings.loc[res_labels]
high_mean = high_ASA_encodings.mean(axis=0)
bfactor_basis_dim = high_mean - low_mean

#bfactor_dim = encodings @ direction / (direction @ direction)
#bfactor_vals = features[feature]

#bfactor_dim = encodings @ bfactor_dim / (bfactor_dim @ bfactor_dim)
#bfactor_vals * (direction @ direction) = encodings @ direction
#bfactor_vals * (direction @ direction)


#plt.hist(bfactor_dim, bins=20)
#plt.savefig('bfactor_hist.png')
#plt.show()

#print(bfactor_vals.max())
#print(bfactor_vals.min())

#ASA_vals = features['ASA']

#plt.figure()
#plt.scatter(ASA_vals, bfactor_vals)
#plt.xlabel('bfactor dimension projection')
#plt.ylabel('bfactor value')
#plt.savefig('bfactor_proj.png')
#plt.show()

#print(pearsonr(bfactor_dim_scale, bfactor_vals))

feature = 'sec_struct'
characteristic = 'E'
res_labels = features.index[features[feature] == characteristic]
yes_SS_encodings = encodings.loc[res_labels]
high_mean = yes_SS_encodings.mean(axis=0)

res_labels = features.index[features[feature] != characteristic]
not_SS_encodings = encodings.loc[res_labels]
low_mean = not_SS_encodings.mean(axis=0)
SS_basis_dim = (high_mean - low_mean)

feature = 'residue'
characteristic = list('HAVLMI')
res_labels = features.index[features[feature].isin(characteristic)]
yes_AA_encodings = encodings.loc[res_labels]
high_mean = yes_AA_encodings.mean(axis=0)

res_labels = features.index[~features[feature].isin(characteristic)]#.isin(characteristic)]
not_AA_encodings = encodings.loc[res_labels]
low_mean = not_AA_encodings.mean(axis=0)
AA_basis_dim = (high_mean - low_mean)


direction_x = AA_basis_dim
x_vals = encodings @ direction_x / (direction_x @ direction_x)

direction_y = SS_basis_dim
y_vals = encodings @ direction_y / (direction_y @ direction_y)

direction_z = bfactor_basis_dim
z_vals = encodings @ direction_z / (direction_z @ direction_z)

plt.scatter(z_vals, x_vals)
plt.xlabel('bfactor')
plt.ylabel('hydrophobicity')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_vals, y_vals, z_vals)
ax.set_xlabel('hydrophobicity')
ax.set_ylabel('In a SS')
ax.set_zlabel('bfactor')
plt.show()