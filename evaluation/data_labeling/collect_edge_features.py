import numpy as np
from protein_mpnn_utils_eval import tied_featurize, parse_PDB
from protein_mpnn_utils_eval import StructureDatasetPDB
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
    def _dist(X, mask, eps=1E-6):
            """ Pairwise euclidean distances """
            # Convolutional network on NCHW
            mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
            dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
            D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

            # Identify k nearest neighbors (including self)
            D_max, _ = torch.max(D, -1, keepdim=True)
            D_adjust = D + (1. - mask_2D) * D_max

            D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
            mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
            return D_neighbors, E_idx, mask_neighbors

    def gather_edges(edges, neighbor_idx):
        # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
        neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
        edge_features = torch.gather(edges, 2, neighbors)
        return edge_features

    with torch.no_grad():
            test_sum, test_weights = 0., 0.
            for ix, protein in enumerate(dataset_valid):
                batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=ca_only)
                Ca = X[:,:,1,:]
                D_neighbors, E_idx, mask_neighbors = _dist(Ca, mask)

    name = str(pdb_path)[-8:-4]
    labels = []
    for i in range(E_idx.shape[1]):
        for j in range(E_idx.shape[2]):
            labels.append(f'{name}_{E_idx[0,i,0]}-{E_idx[0,i,j]}')
    label_df = pd.DataFrame(labels, columns=['identifier'])    

    def _S_to_seq(S):
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        seq = [alphabet[c] for c in S.tolist()[0]]
        return seq

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

input_pdb_dir = Path('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/evaluation/inputs')

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
#f.to_csv("test_edge_features.csv", index=False)
print(f"✅ Edge data saved to edge_features.csv with {len(df)} rows.")