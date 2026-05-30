import numpy as np
import pandas as pd
import csv
import scipy.stats as sci
import sklearn
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
#import polars as pl
from sklearn import metrics
import pickle
import sys

#kivanov: modified code to optionally take in an argument of the model name for automation
if len(sys.argv) == 2:
    model = sys.argv[1]
else:
    model = 'log15_edge_exp2_100e_2'  # Change this to the desired model name

# Load feature data and encodings
feature_path = '../created_data/features/test_edge_features.pkl'
encodings_path = '../created_data/encodings/ki_encodings/output_' + model + '.pkl'

# Open feature pkl
with open(feature_path, "rb") as pkl:
    feature_df = pickle.load(pkl)
print(feature_df)

interesting_dims = set()
# Open encoding pkl
def main(significant_features, num_dims_significant, interesting_dims, feature_df, neighbor, file_pickle=True):
    encoded_all = None
    labels_all = None
    
    if file_pickle:
        with open(encodings_path, "rb") as f:
            while True:
                try:
                    data = pickle.load(f)
                    if encoded_all is None:
                        encoded_all = data["encoded"][:, neighbor, :]
                        labels_all = data["labels"][:, neighbor]
                    else:
                        encoded_all = np.concatenate((encoded_all, data["encoded"][:, neighbor, :]), axis=0)
                        labels_all = np.concatenate((labels_all, data["labels"][:, neighbor]), axis=0)
                except EOFError:
                    break
        def build_df(encoded_all, labels_all):
            df = pd.DataFrame(encoded_all)
            df.insert(0, "identifier", labels_all)
            return df.sort_values(by='identifier')

        encoding_df = build_df(encoded_all, labels_all)

    print("read data")
    # Filter data to make sure there are entries for both sets of data and remove duplicates
    encoding_df = encoding_df.drop_duplicates(subset='identifier').dropna()
    feature_df = feature_df.drop_duplicates(subset='identifier').dropna()
    matching_ids = set(encoding_df['identifier']) & set(feature_df['identifier'])
    encoding_df = encoding_df[encoding_df['identifier'].isin(matching_ids)].sort_values('identifier').reset_index(drop=True)
    feature_df = feature_df[feature_df['identifier'].isin(matching_ids)].sort_values('identifier').reset_index(drop=True)
    print("filtered data")
    res_labels = encoding_df.iloc[:, 0]
    encodings=encoding_df.iloc[:, 1:]
    np.random.seed(0)
    #mask = np.random.rand(encodings.shape[0]) < 0.01
    #encodings = encodings.iloc[mask, :]
    #features = feature_df.iloc[mask, :]
    #res_labels = res_labels.iloc[mask]


    alphabet = list('ACDEFGHIKLMNPQRSTVWYX')  # Standard amino acids

    # Pearson correlation test
    def safe_pearsonr(dssp, encodings, feat, dim):
        x = dssp.to_numpy()
        y = encodings.to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if np.sum(x) == 0 or np.sum(y) == 0:
            #print("no data")
            return 0

        if np.std(x) == 0 or np.std(y) == 0:
            #print("no data")
            return 0

        r, p = sci.pearsonr(x, y)
        if r > 0.7 or r < -0.7:
            print(r, dim, feat)
            interesting_dims.add(dim)
            return 1#r
        else:
            #print(r)
            return 0#r

    def eval_roc_auc(activations, feature_act, feature, dim, thresh=0.75, reverse=False, print_always=False, recorddims=set()):
        # Ensure equal number of true postitives/negatives
        pos_idx = pd.Series(feature_act[feature_act == True].index.tolist())
        neg_idx = pd.Series(feature_act[feature_act == 0].index.tolist())
        n = min(len(pos_idx), len(neg_idx))
        #print(len(pos_idx), len(neg_idx))
        if n > 40:
            balanced_pos = pos_idx.sample(n, random_state=0).sort_values() # Randomly downsample
            balanced_neg = neg_idx.sample(n, random_state=0).sort_values()
            balanced_idx = balanced_pos.tolist() + balanced_neg.to_list()

            feature_act = feature_act.loc[balanced_idx]
            activations = activations.loc[balanced_idx]
            #print(feature_act[:10])
            #print(feature_act[-10:])
            #print(activations[:10])
            score = round(sklearn.metrics.roc_auc_score(feature_act, activations), 3)
            if score > thresh or score < (1-thresh):
                #print(score, dim, feature)
                if print_always:
                    #print(len(pos_idx), len(neg_idx))
                    return 1
                    #print(score, dim, feature)
                    #return 1
                else:
                    return 1
                    #recorddims.add(dim)
                    #return 1
            else:
                return 0
                #print(score, dim, feature)
                #return 0
            #feature_act = 1 - feature_act
            #score = sklearn.metrics.roc_auc_score(feature_act, activations)
            #if score > thresh:
            #    count += 1#print(score, dim, feature, "reversed")
                #if print_always:
                #    print(score, dim, feature, "reversed")                
        else:
            print(n, feature, dim)
            return 0

    type_one_hot = pd.get_dummies(feature_df['contact_type'])
    contact_types = type_one_hot.columns

    n_dims = encodings.shape[-1]
    counter = 0

    #print("Contact Type")
    #for ct in contact_types:
    #    for dim in range(n_dims):
    #        activation = encodings.iloc[:, dim]
    #        counter += eval_roc_auc(activation, type_one_hot[ct], ct, dim)
    #    print(ct)
    #    print(counter)
    #    significant_features.append(ct)
    #    num_dims_significant.append(counter)
    #    counter = 0
    #important_dims = [140]#[4, 8, 140, 141, 148, 22, 28, 29, 31, 35, 37, 168, 51, 180, 55, 183, 61, 189, 200, 203, 207, 209, 84, 91, 92, 237, 113, 120, 124]
    
    sig_dims_15_0 = [7, 10, 13, 144, 19, 147, 148, 153, 154, 28, 157, 31, 34, 169, 42, 172, 46, 50, 182, 54, 56, 60, 61, 193, 196, 71, 199, 77, 208, 81, 210, 84, 86, 219, 221, 223, 96, 225, 226, 97, 100, 249, 122, 231, 101, 233, 236, 112, 241, 113, 116, 117, 118, 247, 244, 121, 126, 245, 125, 254]
    sig_dims_15_1 = [128, 129, 0, 131, 133, 11, 15, 143, 152, 155, 28, 29, 160, 163, 36, 37, 38, 167, 42, 45, 46, 176, 179, 55, 57, 60, 188, 190, 63, 67, 197, 201, 74, 75, 206, 79, 82, 210, 85, 216, 224, 233, 108, 239, 120, 121, 251, 127]
    sig_dims_15_2 = [10, 29, 161, 39, 175, 176, 196, 79, 208, 80, 83, 214, 87, 89, 218, 92, 99, 229, 230, 103, 107, 117, 120, 249, 127]
    
    feature_act = feature_df['distance_bin']
    for dim in range(n_dims):
        activation = encodings.iloc[:, dim]
        counter += safe_pearsonr(activation, feature_act, 'distance_bin', dim)
        #if dim in plot_data:
            #plot_data[dim].append(safe_pearsonr(activation, feature_act, 'distance_bin', dim))
        #data[dim] = []: safe_pearsonr(activation, feature_act, 'distance_bin', dim) > 0:
        #else:
        #    plot_data[dim] = safe_pearsonr(activation, feature_act, 'distance_bin', dim)
            #counter += 1
    #print("distance bin")
    #print(counter)
    significant_features.append(neighbor)
    num_dims_significant.append(counter)
    #print(significant_features)
    #print(num_dims_significant)
    print(counter)
    counter = 0

    #feature_act = feature_df['contact_order']
    #for dim in range(n_dims):
    #    activation = encodings.iloc[:, dim]
    #    counter += safe_pearsonr(activation, feature_act, 'contact_order', dim)
    #print("contact order")
    #print(counter)
    #significant_features.append('contact order')
    #num_dims_significant.append(counter)
    counter = 0
    return significant_features, num_dims_significant
'''
neighbor = 5
encoded_all = None
labels_all = None
with open(encodings_path, "rb") as f:
    while True:
        try:
            data = pickle.load(f)
            if encoded_all is None:
                encoded_all = data["encoded"][:, neighbor, :]
                labels_all = data["labels"][:, neighbor]
            else:
                encoded_all = np.concatenate((encoded_all, data["encoded"][:, neighbor, :]), axis=0)
                labels_all = np.concatenate((labels_all, data["labels"][:, neighbor]), axis=0)
        except EOFError:
            break

def build_df(encoded_all, labels_all, neighbor):
    df = pd.DataFrame(encoded_all)
    df.insert(0, "identifier", labels_all)
    return df.sort_values(by='identifier')

encoding_df = build_df(encoded_all, labels_all, neighbor)

print("read data")
# Filter data to make sure there are entries for both sets of data and remove duplicates
encoding_df = encoding_df.drop_duplicates(subset='identifier').dropna()
feature_df = feature_df.drop_duplicates(subset='identifier').dropna()
matching_ids = set(encoding_df['identifier']) & set(feature_df['identifier'])
encoding_df = encoding_df[encoding_df['identifier'].isin(matching_ids)].sort_values('identifier').reset_index(drop=True)
feature_df = feature_df[feature_df['identifier'].isin(matching_ids)].sort_values('identifier').reset_index(drop=True)
print("filtered data")
res_labels = encoding_df.iloc[:, 0]
encodings=encoding_df.iloc[:, 1:]
np.random.seed(0)

feature_act = feature_df['distance_bin']
values = encoding_df.iloc[:, 200]
plt.scatter(feature_act, values)
plt.show()
'''

significant_features, num_dims_significant  = [], []
for i in range(48):
    print(f'Neighbor {i+1}')
    significant_features, num_dims_significant = main(significant_features, num_dims_significant, interesting_dims, feature_df, neighbor=i)
print(interesting_dims)

#bar_heights = [0, 7, 4, 10, 43, 0, 48]

#significant_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
#num_dims_significant = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 5, 3, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 8, 9, 8, 8, 9, 11, 10, 10, 10, 11, 13, 15]

bar_heights = num_dims_significant
bar_labels = significant_features
print(bar_heights, bar_labels)
#bar_labels = [1, 2, 3, 4, 5, 6, 7]
fig, ax = plt.subplots(figsize=(10, 6))

cmap = plt.cm.Blues

# Generate progressively darker colors
colors = cmap(np.linspace(0.3, 1, len(bar_labels)))  

print(bar_labels)
print(bar_heights)
# Horizontal Bar Plot
ax.barh(bar_labels, bar_heights, color=colors)

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

ax.invert_yaxis()

# Add Plot Title
ax.set_title("Number of Dimensions Related to Distance")
ax.set_ylabel("Neighbor")
ax.set_xlabel("Number of Dimensions")
#plt.savefig(f"{model} 5 r bar plot of feature correlations.png", bbox_inches='tight')
plt.show()
plt.clf()

#print(plot_data)
#print(plot_data.keys())
#for label, values in plot_data.items():
#    plt.plot(range(1, 49), values, label=label)
#plt.xlabel("Neighbors")
#plt.ylabel("r Value")
#plt.title("Neuron performance for different neighbors")
#plt.legend()
#plt.show()
#plt.savefig("test_plt.png")

