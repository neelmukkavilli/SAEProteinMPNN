# pull out uniprot IDs, find Uniprot features, and tie them to node summary data
import pandas as pd
import requests
import numpy as np

pdb_info_csv = '../data_labeling/pdb_info.csv'
node_features_csv = '../node_features.csv'
pdb_info = pd.read_csv(pdb_info_csv)
node_features = pd.read_csv(node_features_csv, index_col=0, header=0)

feature_types_pair = ['Disulfide bond', 'Cross-link'] # Feature types with that are for pairs of residues, not relevant to residues inbetween start and end values

# Use Uniprot ID to query and return features for that protein
def get_uniprot_features(uniprot_id):
        if uniprot_id != "Not Found":
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
            r = requests.get(url)
            r.raise_for_status()
            try:
                return r.json()['features']
            except KeyError:
                print(uniprot_id)
                return "Not Found"
        else:
            return "Not Found"

# For each feature, assign the feature to the corresponding residue, most features will include residues inbetween the start and end values so features are assigned to all of them
    # i.e. chain starts at res 2 and ends at res 48, the same feature is assigned for all intermediate residues.
def build_per_residue_annotation(features):
    per_residue = {}
    if features != "Not Found":
        for feat in features:
            if feat['type'] in feature_types_pair:
                start = int(feat['location']['start']['value'])
                end = int(feat['location']['end']['value'])
                if start not in per_residue:
                    per_residue[start] = {}
                per_residue[start][feat['type'] + ' val'] = True
                per_residue[start][feat['type'] + ' description'] = feat['description']
                if end not in per_residue:
                    per_residue[end] = {}
                per_residue[end][feat['type'] + ' val'] = True
                per_residue[end][feat['type'] + ' description'] = feat['description']
            else:
                try:
                    start = int(feat['location']['start']['value'])
                    end = int(feat['location']['end']['value'])
                    for i in range(start, end+1):
                        if i not in per_residue:
                            per_residue[i] = {}
                        per_residue[i][feat['type'] + ' val'] = True
                        per_residue[i][feat['type'] + ' description'] = feat['description']

                except ValueError:
                    continue
    return per_residue

for cluster in range(pdb_info.shape[1]): # Print out features for each cluster
    print(f'CLUSTER: {cluster}')
    node_cluster_features = node_features.loc[[str(identifier) for identifier in pdb_info['identifier_' + str(cluster)]]]
    pdb_ids = [str(identifier[:4]) for identifier in pdb_info['identifier_' + str(cluster)]]
    pdb_chains = [str(identifier[4]) for identifier in pdb_info['identifier_' + str(cluster)]]
    pdb_res_ids = [int(identifier[5:]) for identifier in pdb_info['identifier_' + str(cluster)]]
    uniprot_ids = []

    # Find Uniprot IDs from tsv files
    for i in range(pdb_info.shape[0]):
        found = False
        with open('../data_labeling/uniprot/out/' + str(pdb_ids[i])+'.tsv', 'r') as file:
            lines = file.readlines()
            for line in lines:
                try:
                    ids = line.split('\t')
                    chain = str(ids[1])
                    res_id = int(ids[3])
                    uniprot_id = str(ids[4])
                    if chain == pdb_chains[i] and res_id == pdb_res_ids[i]:
                        uniprot_ids.append(uniprot_id)
                        found = True
                except:
                    continue
            if found != True:
                uniprot_ids.append("Not Found")

    uniprot_features = {}
    # Collect Uniprot features for each identifier in the cluster
    for i in range(pdb_info.shape[0]):
        features = get_uniprot_features(uniprot_ids[i])
        feature_dict = build_per_residue_annotation(features)
        try:
            uniprot_features[pdb_info['identifier_' + str(cluster)][i]] = feature_dict[pdb_res_ids[i]]
        except KeyError:
            uniprot_features[pdb_info['identifier_' + str(cluster)][i]] = {} # If there are no Uniprot features for that residue, feature_dict[pdb_res_ids[i]] returns key error
        print(uniprot_features[pdb_info['identifier_' + str(cluster)][i]]) # To show easily, Uniprot features are printed separately

    uniprot_df = pd.DataFrame(uniprot_features).T
    all_features = pd.concat([node_cluster_features, uniprot_df], axis=1)
    print(all_features.iloc[:, :18]) # Only show first 18 columns of features, most contact distances are excluded
