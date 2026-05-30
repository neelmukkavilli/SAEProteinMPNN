import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, BisectingKMeans
from sklearn.cluster import HDBSCAN, OPTICS, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import numpy as np
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv", type=str, default='llb')
argparser.add_argument("--num_clusters", type=int, default=100)
argparser.add_argument("--bottom_k", type=int, default=25)
argparser.add_argument("--return_num_clusters", type=int, default=100, help="Number of clusters whose info will be printed in a csv")
argparser.add_argument("--cluster_mask_level", type=float, default=0.01, help="Fraction of datapoints used for clustering calculation")
argparser.add_argument("--show_img", action='store_true', help="If the 3D PCA Plot should be shown")
argparser.add_argument("--img_mask_level", type=float, default=0.1, help="Fraction of datapoints used for plotting, multiplied by cluster_mask_level")
args = argparser.parse_args()

csv = args.csv

encodings_csv = 'created_data/encodings/ki_encodings/output_' + csv + '.csv'
features_csv = 'created_data/features/node_features.csv'
encodings = pd.read_csv(encodings_csv, header=0)
features = pd.read_csv(features_csv, header=0)

# Filter out incomplete or missing rows
print(encodings.shape)
encodings = encodings.dropna(axis='index')
matching_ids = set(encodings['identifier']) & set(features['identifier'])
print(encodings.shape)
encodings = encodings[encodings['identifier'].isin(matching_ids)]
encodings = encodings.sort_values('identifier').reset_index(drop=True)
print(encodings.shape)
output_pdb_list = 'data_labeling/uniprot/pdb_list.txt'
output_pdb_info = 'data_labeling/' + csv + '_pdb_info.csv'

# Split up labels from encodings
res_labels = encodings.iloc[:, 0]
encodings = encodings.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier (0.1 cluster mask level)
#print("Encoding data shape before mask", encodings.shape)
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < args.cluster_mask_level
encodings = encodings.iloc[mask, :]
res_labels = res_labels.iloc[mask]

kmeans = KMeans(n_clusters=20).fit(encodings)
print("kmeans done")

pca = PCA(n_components=3)
pca_graph = pca.fit_transform(encodings)

if args.show_img:
    # Filter points again ~43,000 -> ~4,300 to make plotting easier (0.1 cluster mask level and 0.1 img mask level), kmeans data is not filtered
    print("PCA Shape before mask", pca_graph.shape)
    np.random.seed(0)
    mask = np.random.rand(pca_graph.shape[0]) < args.img_mask_level
    pca_graph = pca_graph[mask, :]
    print("PCA Shape after mask", pca_graph.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pca_graph[:, 0], pca_graph[:, 1], pca_graph[:, 2], c=kmeans.labels_)#[mask])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(f'PCA Reduction with KMeans for {csv}')
    plt.show()
    

# Get distances for each point to its cluster centroid and return k closest points
distances_to_centroids = kmeans.transform(encodings)
bottom_k = np.argsort(distances_to_centroids, axis=0)[:args.bottom_k, :]

# Number of clusters to output, you may not always want to look at all 100 clusters, returning fewer clusters means querying fewer Uniprot IDs
clusters = args.return_num_clusters

pdb_info = pd.DataFrame(())
for c in range(clusters):
    pdb_info['identifier_' + str(c)] = [str(identifier) for identifier in res_labels.iloc[bottom_k[:, c]]]

with open(output_pdb_list, 'w') as file:
    for c in range(clusters):
        for i in range(args.bottom_k):
                file.write(pdb_info['identifier_' + str(c)][i][:4] + '\n')

pdb_info.to_csv(output_pdb_info)

pd.DataFrame(kmeans.cluster_centers_).to_csv(csv+'_centroids.csv')
