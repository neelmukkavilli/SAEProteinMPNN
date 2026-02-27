import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import pandas as pd
import pickle

#iris = load_iris()
#X = iris.data
encodings_path = '/WAVE/bio/ML/SAE_train/SAEProteinMPNN/evaluation/created_data/encodings/output_256_40_2.pkl'

#encodings = pl.read_csv(encodings_path).to_pandas()
dfs = []
with open(encodings_path, "rb") as f:
    while True:
        try:
            dfs.append(pickle.load(f))
        except EOFError:
            break
    encodings = pd.concat(dfs, ignore_index=True)
encodings=encodings.iloc[:, 1:]
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < 0.01
X = encodings.iloc[mask, :]
print(X.shape)

#y = iris.target
#target_names = iris.target_names

# 2. Apply t-SNE
# t-SNE is a stochastic method, so results may vary between runs.
# It is recommended to first reduce the dimensions with PCA for high-dimensional data.
# The 'n_components' is typically set to 2 for 2D visualization.
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
X_2d = tsne.fit_transform(X)

# 3. Visualize the results
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_2d[:, 0], y=X_2d[:, 1],
    #hue=target_names[y],
    #palette=sns.color_palette("hsv", n_colors=len(target_names)),
    legend="full",
    alpha=0.8
)
plt.title('t-SNE visualization')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.legend(title='Species')
plt.show()

# 4. Use the t-SNE output for clustering (optional)
# t-SNE output (X_2d) can be used as input to a clustering algorithm like K-Means.
from sklearn.cluster import KMeans

# Determine the number of clusters (e.g., using the elbow method or domain knowledge)
# For the Iris dataset, we know there are 3 species.
k = 3 
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
clusters = kmeans.fit_predict(X_2d)

# Add cluster labels to a DataFrame for visualization
df = pd.DataFrame(X_2d, columns=['comp1', 'comp2'])
df['cluster_label'] = clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x='comp1', y='comp2',
    hue='cluster_label',
    palette=sns.color_palette("bright", n_colors=k),
    legend="full",
    alpha=0.8
)
plt.title('K-Means Clustering on t-SNE output')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()