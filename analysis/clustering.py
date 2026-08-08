import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import pickle

def load_data(SAE_type, model, layer, mask_lvl, load_dssp=True):
    # Load feature data and encodings
    dssp_path = '../evaluation/created_data/features/node_features.csv'
    encodings_path = '../evaluation/created_data/encodings/' + SAE_type + '_' + model + '/output_' + model + '_' + str(layer) + '.pkl'

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
    encodings = StandardScaler().fit_transform(encodings)

    return encodings, res_labels, n_dims, dssp

def tSNE_reduction(encodings):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_graph = tsne.fit_transform(encodings)
    return tsne_graph

def plot_tSNE(tsne_graph, dssp, n_dims, layer, model, encodings, tsne_type):
    fig, ax = plt.subplots()
    ax.set_title(f"SAE with {n_dims} Neurons: Layer {layer+1}")
    ax.set_xlabel('tSNE Dimension 1')
    ax.set_ylabel('tSNE Dimension 2')
    if tsne_type == "structure":
        helix = (dssp['sec_struct'] == 'H') + (dssp['sec_struct'] == 'G') + (dssp['sec_struct'] == 'I')
        beta_strand = (dssp['sec_struct'] == 'E')
        unstruct = (dssp['sec_struct'] == 'B') + (dssp['sec_struct'] == 'T') + (dssp['sec_struct'] == 'S') + (dssp['sec_struct'] == '-')
        ax.scatter(tsne_graph[:, 0][helix], tsne_graph[:, 1][helix], label='Helix')
        ax.scatter(tsne_graph[:, 0][beta_strand], tsne_graph[:, 1][beta_strand], label='Beta Strand')
        ax.scatter(tsne_graph[:, 0][unstruct], tsne_graph[:, 1][unstruct], label='Turn or Bend')
        ax.legend(title='Secondary Structure', loc='best')
        plt.savefig(f'tSNE reduction with sec_struct for {model} layer{layer+1}')
    elif tsne_type == "idp":
        X = np.vstack((encodings, idp_encodings))
        tsne_graph = tSNE_reduction(X)
        ax.scatter(tsne_graph[:encodings.shape[0], 0], tsne_graph[:encodings.shape[0], 1], label='Ordinary Residues')
        ax.scatter(tsne_graph[encodings.shape[0]:, 0], tsne_graph[encodings.shape[0]:, 1], label='IDP Residues')
        ax.legend()
        plt.savefig(f'IDP tSNE reduction for {model} layer{layer+1}')
    plt.show()

tsne_type = "structure" # "structure" or "idp"
SAE_type = 'node'
model = 'log17_exp2'  #ex: dense, log17_exp2
layer = 1 # 0-indexed
cluster_mask_level = 0.01 # 0.01 for regular encodings, 0.5 for IDP encodings

if tsne_type == "structure":
    encodings, res_labels, n_dims, dssp = load_data(SAE_type, model, layer, cluster_mask_level=0.01, dssp=True)
    tsne_graph = tSNE_reduction(encodings)
    plot_tSNE(tsne_graph, dssp, n_dims, layer, model, encodings, tsne_type="structure")
elif tsne_type == "idp":
    encodings, res_labels, n_dims, dssp = load_data(SAE_type, model, layer, cluster_mask_level=0.01, dssp=False)
    idp_encodings, _, _, _ = load_data(SAE_type, model, layer, cluster_mask_level=0.5, dssp=False)
    plot_tSNE(None, None, n_dims, layer, model, encodings, tsne_type="idp")
