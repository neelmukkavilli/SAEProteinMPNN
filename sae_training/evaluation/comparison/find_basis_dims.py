import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

csv = 'newlsamples500_2'

encodings_csv = '../encodings/normalized_encodings_' + csv + '.csv'
features_csv = '../node_features.csv'

encodings = pd.read_csv(encodings_csv, index_col=0)
features = pd.read_csv(features_csv, index_col=0)

print(encodings.shape)
encodings = encodings.dropna(axis='index')
matching_ids = set(encodings['identifier']) & set(features['identifier'])

encodings = encodings[encodings['identifier'].isin(matching_ids)]
encodings = encodings.sort_values('identifier').reset_index(drop=True)
features = features[features['identifier'].isin(matching_ids)]
features = features.sort_values('identifier').reset_index(drop=True)

print(encodings.shape)
print(encodings.columns)
print(features.shape)
print(features.columns)

feature = 'ASA'
characteristic = 0.01
res_labels = features.index[features[feature] < characteristic]
ASA_encodings = encodings.loc[res_labels]
stdv = ASA_encodings.std(axis=0)
mean = ASA_encodings.mean(axis=0)
high_ASA_basis_dim = pd.Series(0.0, index=mean.index)
high_ASA_basis_dim[stdv < 0.05] = mean[stdv < 0.05]

feature = 'ASA'
characteristic = 0.90
res_labels = features.index[features[feature] > characteristic]
low_ASA_encodings = encodings.loc[res_labels]
stdv = low_ASA_encodings.std(axis=0)
mean = low_ASA_encodings.mean(axis=0)
low_ASA_basis_dim = pd.Series(0.0, index=mean.index)
low_ASA_basis_dim[stdv < 0.05] = mean[stdv < 0.05]

ASA_basis_dim = high_ASA_basis_dim - low_ASA_basis_dim


feature = 'sec_struct'
characteristic = 'E'
res_labels = features.index[features[feature] == characteristic]
yes_SS_encodings = encodings.loc[res_labels]

stdv = yes_SS_encodings.std(axis=0)
mean = yes_SS_encodings.mean(axis=0)
yes_SS_basis_dim = pd.Series(0.0, index=mean.index)
yes_SS_basis_dim[stdv < 0.05] = mean[(stdv < 0.05)]

feature = 'sec_struct'
characteristic = 'E'
res_labels = features.index[features[feature] != characteristic]
not_SS_encodings = encodings.loc[res_labels]

stdv = not_SS_encodings.std(axis=0)
mean = not_SS_encodings.mean(axis=0)
not_SS_basis_dim = pd.Series(0.0, index=mean.index)
not_SS_basis_dim[stdv < 0.05] = mean[(stdv < 0.05)]

SS_basis_dim = yes_SS_basis_dim - not_SS_basis_dim

point = SS_basis_dim + ASA_basis_dim
point.to_csv('point.csv')
