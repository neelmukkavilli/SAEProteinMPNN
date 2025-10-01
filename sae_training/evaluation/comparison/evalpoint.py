from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

csv = 'dense'

point = pd.DataFrame(pd.read_csv('point.csv', header=0).iloc[:, 1])
encodings = pd.read_csv('../encodings/normalized_encodings_' + csv + '.csv')
encodings = encodings.dropna(axis='index')

output_pdb_list = '../data_labeling/uniprot/pdb_list.txt'
output_pdb_info = '../data_labeling/' + csv + '_testcheck_pdb_info.csv'

# Split up labels from encodings
res_labels = encodings.iloc[:, 0]
encodings = encodings.iloc[:, 1:]

# Mask 90% of data ~430,000 -> ~43,000 to make clustering easier (0.1 cluster mask level)
#print("Encoding data shape before mask", encodings.shape)
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < 0.1
encodings = encodings.iloc[mask, :]
res_labels = res_labels.iloc[mask]

#print("Encoding data shape after mask", encodings.shape)

# Normalize encoding rows to unit norm (~43k points in 1024 dimensional space each distance 1 away from origin)
encodings = normalize(encodings, 'l2', axis=1)
tree = KDTree(encodings)
ref_point = point.to_numpy().T
distance, index = tree.query(ref_point, k=25)
print(distance)
print(np.mean(distance))

pdb_info = pd.DataFrame(())
pdb_info['identifier_' + str(0)] = [str(identifier) for identifier in res_labels.iloc[pd.Series(index[0])]]

with open(output_pdb_list, 'w') as file:
    for i in range(25):
        file.write(pdb_info['identifier_' + str(0)][i][:4] + '\n')

pdb_info.to_csv(output_pdb_info)

'''
#kmeans = KMeans(n_clusters=args.num_clusters).fit(encodings)

#centroids = pd.read_csv('../' + csv + '_centroids.csv', header=0, index_col=0).T

#print(point.shape)
#print(centroids.shape)

all_points = pd.DataFrame(np.hstack([point.values, centroids.values])).T
#all_points = pd.DataFrame(centroids.values).T
#print(all_points.shape)

# newlsamples500_2
E_SS = [2, 6, 13, 16, 18, 23, 30, 31, 43, 47, 51, 58, 63, 65, 68, 70, 74, 78, 81, 92, 94]
low_ASA = [16, 43, 58]
phobic = [1, 14, 16, 19, 22, 23, 53, 58, 62, 78, 89, 92]

# Dense
#E_SS = [0, 7, 8, 11, 17, 20, 22, 24, 32, 33, 34, 37, 47, 52, 56, 57, 59, 62, 78, 80, 84, 85]
#low_ASA = [33, 57, 62]
#phobic = [8, 10, 24, 32, 33, 51, 57, 59, 78, 87, 98]
labels = [0] * 100

#for i in range(101):
#    if i in phobic:
#        labels[i] = 2

labels = [1] + labels

print(labels[:10])

#print(len(labels))
#print(labels[:5])
pca = PCA(n_components=3)
W_pca = pca.fit_transform(all_points)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(W_pca[:, 0], W_pca[:, 1], W_pca[:, 2], c=labels)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig('point_and_centroids.png')
plt.show()


print(point.shape)
print(centroids.shape)
centroids = centroids.to_numpy().T
print(centroids.shape)
tree = KDTree(centroids)
ref_point = point.to_numpy().T
print(ref_point.shape)
distance, index = tree.query(ref_point, k=1)
print(index)
nearest_point = centroids[index[0][0]]
#print(nearest_point)
print(distance[0][0])
'''
