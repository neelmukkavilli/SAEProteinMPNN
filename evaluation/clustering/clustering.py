import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, BisectingKMeans
from sklearn.cluster import HDBSCAN, OPTICS, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import argparse
import pickle

expansion = 2
model = 'log17_exp2'  #ex: dense, log17_exp2
SAE_type = 'node'
layer = 1 # 0-indexed
cluster_mask_level = 0.01


def load_data(SAE_type, model, layer, mask_lvl, load_dssp=True):
    # Load feature data and encodings
    dssp_path = '/home/neelm/SAEProteinMPNN/evaluation/created_data/features/node_features.csv'
    encodings_path = '/home/neelm/SAEProteinMPNN/evaluation/created_data/encodings/' + SAE_type + '_' + model + '/output_' + model + '_' + str(layer) + '.pkl'

    # Read and format pkl files
    dfs = []
    with open(encodings_path, "rb") as f:
        while True:
            try:
                dfs.append(pickle.load(f))
            except EOFError:
                break
        encodings = pd.concat(dfs, ignore_index=True)
    dfs = []
    np.random.seed(0)
    mask = np.random.rand(encodings.shape[0]) < mask_lvl
    encodings = encodings.iloc[mask, :]

    if load_dssp:
        dssp = pd.read_csv(dssp_path)
        print("read data")
        # Filter data to make sure there are entries for both sets of data and remove duplicates
        encodings = encodings.drop_duplicates(subset='identifier')
        dssp = dssp.drop_duplicates(subset='identifier')
        encodings = encodings.dropna() # Drop rows with NA values
        dssp = dssp.dropna()
        encodings['identifier'] = encodings['identifier'].astype(str).str.strip().str.lower()
        dssp['identifier'] = dssp['identifier'].astype(str).str.strip().str.lower()
        matching_ids = set(encodings['identifier']) & set(dssp['identifier'])
        encodings = encodings[encodings['identifier'].isin(matching_ids)]
        encodings = encodings.sort_values('identifier').reset_index(drop=True)
        dssp = dssp[dssp['identifier'].isin(matching_ids)]
        dssp = dssp.sort_values('identifier').reset_index(drop=True)
        print("filtered data")
    else:
        dssp = None

    # Isolate sample labels and set up mask (default 10% of samples -> ~43,000 samples)
    res_labels = encodings.iloc[:, 0]
    encodings = encodings.iloc[:, 1:]
    print(f'{res_labels.shape[0]} samples')
    n_dims = encodings.shape[1]
    print(f'{n_dims} dimensions')

    return encodings, res_labels, n_dims, dssp

encodings, res_labels, n_dims, dssp = load_data(SAE_type, model, layer, cluster_mask_level)

print("Encoding data shape after filtering", encodings.shape)

X = StandardScaler().fit_transform(encodings)

# Compare encodings to IDP encodings
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
print("IDP Encoding data shape before filtering", idp_encodings.shape)
idp_encodings = idp_encodings.dropna(axis='index')
idp_encodings = idp_encodings.sort_values('identifier').reset_index(drop=True)

# Split up labels from encodings
idp_encodings = idp_encodings.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier (0.1 cluster mask level)
np.random.seed(0)
mask = np.random.rand(idp_encodings.shape[0]) < 0.5
idp_encodings = idp_encodings.iloc[mask, :]
print("IDP Encoding data shape after filtering", idp_encodings.shape)
X2 = StandardScaler().fit_transform(idp_encodings)
labels = [0]*encodings.shape[0] + [1]*idp_encodings.shape[0]
X = np.vstack((X, X2))

tsne = TSNE(n_components=2, random_state=0)
tsne_graph = tsne.fit_transform(X)
tsne_idp_graph = tsne.fit_transform(X2)


fig = plt.figure()
ax = fig.add_subplot()
#for label in []#np.unique(dssp['sec_struct']):

#mask = (dssp['sec_struct'] == 'H') + (dssp['sec_struct'] == 'G') + (dssp['sec_struct'] == 'I')
#ax.scatter(tsne_graph[:, 0][mask], tsne_graph[:, 1][mask], label='Helix')

#mask = (dssp['sec_struct'] == 'E')
#ax.scatter(tsne_graph[:, 0][mask], tsne_graph[:, 1][mask], label='Beta Strand')


#mask = (dssp['sec_struct'] == 'B') + (dssp['sec_struct'] == 'T') + (dssp['sec_struct'] == 'S') + (dssp['sec_struct'] == '-')
#ax.scatter(tsne_graph[:, 0][mask], tsne_graph[:, 1][mask], label='Turn or Bend')

#ax.scatter(tsne_graph[:, 0], tsne_graph[:, 1], label=dssp['sec_struct'])

ax.scatter(tsne_graph[:4304, 0], tsne_graph[:4304, 1], label='Ordinary Residues')
ax.scatter(tsne_graph[4304:, 0], tsne_graph[4304:, 1], label='IDP Residues')

ax.set_title(f"SAE with {expansion*128} Neurons: Layer {layer+1}")
ax.set_xlabel('tSNE Dimension 1')
ax.set_ylabel('tSNE Dimension 2')
ax.legend()#title='Secondary Structure', loc='best')
plt.savefig(f'IDP tSNE reduction for {model} layer{layer}')#tSNE reduction with sec_struct for {model} layer{layer}')
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
