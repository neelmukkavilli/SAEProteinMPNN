import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, BisectingKMeans
from sklearn.cluster import HDBSCAN, OPTICS, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import os
import argparse
import pickle

expansion = 2
model = f'log17_exp{expansion}'
SAE_type = 'node'
layer = 0
cluster_mask_level = 0.01

encodings_path = f'../created_data/encodings/{SAE_type}_{model}/output_{model}_{layer}.pkl'
dfs = []
with open(encodings_path, "rb") as f:
    while True:
        try:
            dfs.append(pickle.load(f))
        except EOFError:
            break
    encodings = pd.concat(dfs, ignore_index=True)

# Filter out incomplete or missing rows
print("Encoding data shape before filtering", encodings.shape)
encodings = encodings.dropna(axis='index')
encodings = encodings.sort_values('identifier').reset_index(drop=True)

# Split up labels from encodings
encodings = encodings.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier (0.1 cluster mask level)
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < cluster_mask_level
encodings = encodings.iloc[mask, :]
print("Encoding data shape after filtering", encodings.shape)

X = StandardScaler().fit_transform(encodings)
kmeans = KMeans(n_clusters=25).fit(X)
labels = kmeans.lables_
print("kmeans done")

'''
idp_encodings_path = f'../created_data/encodings/idp_{SAE_type}_{model}/output_{model}_{layer}.pkl'
idp_dfs = []
with open(idp_encodings_path, "rb") as f:
    while True:
        try:
            idp_dfs.append(pickle.load(f))
        except EOFError:
            break
    idp_encodings = pd.concat(idp_dfs, ignore_index=True)
# Filter out incomplete or missing rows
print("Encoding data shape before filtering", idp_encodings.shape)
idp_encodings = idp_encodings.dropna(axis='index')
idp_encodings = idp_encodings.sort_values('identifier').reset_index(drop=True)

# Split up labels from encodings
idp_encodings = idp_encodings.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier (0.1 cluster mask level)
np.random.seed(0)
mask = np.random.rand(idp_encodings.shape[0]) < 0.5
idp_encodings = idp_encodings.iloc[mask, :]
print("Encoding data shape after filtering", idp_encodings.shape)
X2 = StandardScaler().fit_transform(idp_encodings)
labels = [0]*encodings.shape[0] + [1]*idp_encodings.shape[0]
X = np.vstack((X, X2))
'''

tsne = TSNE(n_components=2, random_state=0)
tsne_graph = tsne.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(tsne_graph[:, 0], tsne_graph[:, 1], c=labels)
ax.set_title(f"SAE with {expansion*128} Neurons: Layer {layer+1}")
ax.set_xlabel('tSNE Dimension 1')
ax.set_ylabel('tSNE Dimension 2')
plt.savefig(f'idp tSNE reduction with kmeans for {model} layer{layer}')
plt.show()

if None:#args.show_img:
    # Get distances for each point to its cluster centroid and return k closest points
    distances_to_centroids = kmeans.transform(encodings)
    bottom_k = np.argsort(distances_to_centroids, axis=0)[:args.bottom_k, :]

    # Number of clusters to output, you may not always want to look at all 100 clusters, returning fewer clusters means querying fewer Uniprot IDs
    clusters = args.return_num_clusters

    pdb_info = pd.DataFrame(())
    for c in range(clusters):
        pdb_info['identifier_' + str(c)] = [str(identifier) for identifier in res_labels.iloc[bottom_k[:, c]]]

#with open(output_pdb_list, 'w') as file:
#    for c in range(clusters):
#        for i in range(args.bottom_k):
#                file.write(pdb_info['identifier_' + str(c)][i][:4] + '\n')

#pdb_info.to_csv(output_pdb_info)

#pd.DataFrame(kmeans.cluster_centers_).to_csv(csv+'_centroids.csv')
