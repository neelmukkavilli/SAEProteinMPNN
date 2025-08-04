import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/normalized_encodings_test.csv')#, usecols=range(1, 1025))
print(df.shape)
mask = np.random.rand(df.shape[0]) > 0.99 # Filters out 99% of data points ~430,000 -> ~4,300
res_labels = df.iloc[mask, 0]
df = df.iloc[mask, 1:]
print(df.shape)

if False:
    W = pd.read_csv('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/output_encodings_test.csv', usecols=range(1,1025))
    #W = pd.read_csv('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/data_labeling/new_dssp_summary.csv', usecols=range(1,4))
    #W_normalized = torch.nn.functional.normalize(W, p=2, dim=1)  # [out_dim, in_dim]
    
    pca = PCA(n_components=100, svd_solver='arpack')  # Keep 95% variance
    W_np = W.to_numpy()
    #W_np = W_np[~np.isnan(W_np).any(axis=1)]  # Convert to numpy for PCA
    pca.fit(W_np)
    plt.bar(range(1,101), pca.explained_variance_ratio_)
    plt.title("Explained variance ratio")
    plt.ylabel("'%' of variance explained")
    plt.xlabel("Top 100 dimensions")
    plt.tight_layout()
    plt.savefig("explained_variance_ratio.png")
    plt.show()
    print("Explained variance ratio:", pca.explained_variance_ratio_[:10])

    pca = PCA(n_components=2)
    W_pca = pca.fit_transform(W_np)
    plt.figure()
    plt.scatter(W_pca[:, 0], W_pca[:, 1], c='dodgerblue', edgecolor='k', alpha=0.7)
    plt.title("Neuron Weight Vectors (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    #plt.xlim(-10, 10)
    #plt.ylim(-10, 10)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("pca_projection.png")
    plt.show()


W_np = df.to_numpy()
kmeans = KMeans(n_clusters=40).fit(W_np)

pca = PCA(n_components=2)
W_pca = pca.fit_transform(W_np)

plt.figure()
plt.scatter(W_pca[:, 0], W_pca[:, 1], c=kmeans.labels_, edgecolor='k', alpha=0.7)
plt.title("Neuron Weight Vectors (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()

pca = PCA(n_components=3)
W_pca = pca.fit_transform(W_np)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(W_pca[:, 0], W_pca[:, 1], W_pca[:, 2], c=kmeans.labels_)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
distances_to_centroids = kmeans.transform(W_np)

cluster = 5
bottom_k = np.argpartition(distances_to_centroids[:, cluster], 10)[:10]
print(bottom_k)
print(res_labels.iloc[bottom_k])