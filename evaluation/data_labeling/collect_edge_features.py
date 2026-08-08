import numpy as np
from protein_mpnn_utils import tied_featurize, parse_PDB, _S_to_seq, StructureDatasetPDB
from protein_mpnn_utils import CA_ProteinFeatures
from pathlib import Path
import torch
import copy
import pandas as pd
from natsort import natsorted
import pickle

output = "test_edge_features.pkl"

df_distance_bins = pd.DataFrame()
df_contact_order = pd.DataFrame()
df_contact_type = pd.DataFrame()

ca_only = False
batch_size = 1
max_length = 200000
pdb_path_chains = ""
fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict = None, None, None, None, None
BATCH_COPIES = batch_size
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))    
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
bias_AAs_np = np.zeros(len(alphabet))

def run_distance_collection(pdb_path):
    pdb_dict_list = parse_PDB(str(pdb_path), ca_only)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    designed_chain_list = all_chain_list
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    top_k = 48
    features = CA_ProteinFeatures(
        edge_features=128,
        node_features=128,
        top_k=top_k)

    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, __annotations__ = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=ca_only)
            Ca = X[:,:,1,:]
            D_neighbors, E_idx, _ = features._dist(Ca, mask)

    name = str(pdb_path)[-8:-4]
    labels = []
    for i in range(E_idx.shape[1]):
        for j in range(E_idx.shape[2]):
            labels.append(f'{name}_{E_idx[0,i,0]}-{E_idx[0,i,j]}')
    label_df = pd.DataFrame(labels, columns=['identifier'])    

    dmin = 2 - 4/6
    dmax = 22 + 4/6
    rbf = 16
    def count_distance_bins(D_neighbors): # Returns dict "{residue: {min dist: number of h bonds either acceptor or donor}, {min dist: ...}, }...}
        D_neighbors = D_neighbors[0,:,:].numpy()
        distances = np.linspace(dmin, dmax, rbf+1)
        distances[0], distances[-1] = 0, 100
        edge_dist_bins = np.digitize(D_neighbors, distances) -1 

        return edge_dist_bins
    
    distance_bins = count_distance_bins(D_neighbors)

    def count_positional_distance(E_idx):
        E_idx = E_idx[0,:,:].numpy()
        edge_pos_dist = np.zeros_like(E_idx, dtype=float)
        for i in range(E_idx.shape[0]):
            edge_pos_dist[i, :] = abs((E_idx[i, :] - E_idx[i, 0]))

        return edge_pos_dist
    
    contact_order = count_positional_distance(E_idx)
    def get_res_type(res1, res2):
        polar = ['S', 'T', 'N', 'Q', 'C']
        charge_pos = ['R', 'H', 'K']
        charge_neg = ['D', 'E']
        phobic = ['A', 'V', 'I' 'L', 'M', 'F', 'W', 'G', 'P', 'Y']

        if res1 in polar and res2 in polar:
            pair_type = "Po"
        elif (res1 in charge_pos and res2 in charge_neg) or (res1 in charge_neg and res2 in charge_pos):
            pair_type = "SB"
        elif res1 in phobic and res2 in phobic:
            pair_type = "Ph"
        else:
            pair_type = "XX"
        return pair_type
    
    def res_type_pairs(E_idx, S):
        res_type_arr = np.empty((E_idx.shape[1], E_idx.shape[2]), dtype=object)
        seq = _S_to_seq(S)
        for i in range(E_idx.shape[1]):
            res1 = seq[E_idx[0, i, 0]]
            for j in range(E_idx.shape[2]):
                res2 = seq[E_idx[0, i, j]]
                res_type_arr[i, j] = get_res_type(res1, res2)
        return res_type_arr
    
    contact_type = res_type_pairs(E_idx, S)
    mask = np.repeat(S != 20, top_k)
    mask = [bool(val) for val in mask]
    df_distance_bins, df_contact_order, df_contact_type = pd.DataFrame(np.reshape(distance_bins, (-1)), columns=['distance_bin']), pd.DataFrame(np.reshape(contact_order, (-1)), columns=['contact_order']), pd.DataFrame(np.reshape(contact_type, (-1)), columns=['contact_type'])
    
    return pd.concat([label_df, df_distance_bins, df_contact_order, df_contact_type], axis=1).iloc[mask, :]

input_pdb_dir = Path('../inputs')

df = pd.DataFrame()

pdb_files = list(input_pdb_dir.glob('*.pdb'))
pdb_files = [Path(p) for p in natsorted([str(p) for p in pdb_files])]
pdb_files = pdb_files  # Limit to first 100 PDB files for testing

for pdb_file in pdb_files:
    print(pdb_file.name)
    data = run_distance_collection(pdb_file)
    np.random.seed(0)
    df = pd.concat([df, data], axis=0)

with open(output, 'ab') as f:
    pickle.dump(df, f)
print(f"✅ Edge data saved to {output} with {len(df)} rows.")