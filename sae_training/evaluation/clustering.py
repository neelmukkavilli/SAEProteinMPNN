import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, BisectingKMeans
from sklearn.cluster import HDBSCAN, OPTICS, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

encodings_csv = 'encodings/normalized_encodings_verysparse25e2.csv'

encodings = pd.read_csv(encodings_csv, header=0)

output_pdb_list = 'data_labeling/uniprot/pdb_list.txt'
output_pdb_info = 'data_labeling/pdb_info.csv'

# Split up labels from encodings
res_labels = encodings.iloc[:, 0]
encodings = encodings.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier
print("Encoding data shape before mask", encodings.shape)
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) > 0.90
encodings = encodings.iloc[mask, :]
res_labels = res_labels.iloc[mask]
print("Encoding data shape after mask", encodings.shape)

# Normalize encoding rows to unit norm (~43k points in 1024 dimensional space each distance 1 away from origin)
encodings = normalize(encodings, 'l2', axis=1)

kmeans = KMeans(n_clusters=100).fit(encodings)

# Draw points in 3D space with kmeans cluster labels
pca = PCA(n_components=3)
W_pca = pca.fit_transform(encodings)

# Filter points again ~43,000 -> ~4,300 to make plotting easier, kmeans data is not filtered
np.random.seed(0)
mask = np.random.rand(W_pca.shape[0]) > 0.90
W_pca = W_pca[mask, :]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(W_pca[:, 0], W_pca[:, 1], W_pca[:, 2], c=kmeans.labels_[mask])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

# Get distances for each point to its cluster centroid and return 10 closest points
distances_to_centroids = kmeans.transform(encodings)
bottom_k = np.argsort(distances_to_centroids, axis=0)[:10, :]
print(distances_to_centroids[bottom_k[:, 1], 1])

# Number of clusters to output, you may not always want to look at all 100 clusters
clusters = 10

pdb_info = pd.DataFrame(())
for c in range(clusters):
    pdb_info['identifier_' + str(c)] = [str(identifier) for identifier in res_labels.iloc[bottom_k[:, c]]]

print(pdb_info)
with open(output_pdb_list, 'w') as file:
    for c in range(clusters):
        for i in range(10):
                file.write(pdb_info['identifier_' + str(c)][i][:4] + '\n')

pdb_info.to_csv(output_pdb_info)
