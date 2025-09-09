# pull out uniprot IDs, find Uniprot features, and tie them to node summary data
import pandas as pd
import requests
import numpy as np
import argparse
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv", type=str, default='llb')
argparser.add_argument("--num_to_print", type=int, default=10)
argparser.add_argument("--automate", action='store_true')
args = argparser.parse_args()

args.csv = 'newlsamples500_2'
args.automate = False
pdb_info_csv = '../data_labeling/' + args.csv + '_testcheck_pdb_info.csv' ## Set to TESTCHECK
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
                try:
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
                except (ValueError, TypeError):
                    continue
            else:
                try:
                    start = int(feat['location']['start']['value'])
                    end = int(feat['location']['end']['value'])        
                    for i in range(start, end+1):
                        if i not in per_residue:
                            per_residue[i] = {}
                        per_residue[i][feat['type'] + ' val'] = True
                        per_residue[i][feat['type'] + ' description'] = feat['description']
                except (ValueError, TypeError):
                    print(feat['location']['start'])
                    print(feat['location']['end'])
                    continue
    return per_residue

# By function
function_dict = {}
function_dict['phobic'] = list('HAVLMI')
function_dict['philic'] = list('STCPNQ')
function_dict['pos'] = list('KRH')
function_dict['neg'] = list('DE')
function_dict['aro'] = list('FYW')
# By shape
shape_dict = {}
shape_dict['small'] = list('GA')
shape_dict['chain2'] = list('SC')
shape_dict['branch1'] = list('VTND')
shape_dict['branch2'] = list('ILQE')
shape_dict['long'] = list('MKR')
shape_dict['ring1'] = list('PHFY')
shape_dict['ring2'] = list('W')

def categorize_AA(feature_df, categorize_by):
    df = pd.DataFrame()
    if categorize_by == 'function':
        for i in function_dict.keys():
            df[i] = [feature_df[list(set(feature_df.index) & set(function_dict[i]))].sum()]
    if categorize_by == 'shape':
        for i in shape_dict.keys():
            df[i] = [feature_df[list(set(feature_df.index) & set(shape_dict[i]))].sum()]
    return df

def quantify_categorization(feature_df):
    quantized = pd.DataFrame()
    for feature in feature_df.columns:
        if feature == 'residue':
            one_hot = pd.get_dummies(feature_df.loc[:, feature], dtype=float)

            sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
            function_one_hot = categorize_AA(sum_AA, 'function')
            shape_one_hot = categorize_AA(sum_AA, 'shape')

            quantized['AA_name'] = [sum_AA.max()]
            quantized['AA_function'] = function_one_hot.max(axis=1)
            quantized['AA_shape'] = shape_one_hot.max(axis=1)
            if sum_AA.max() > 0.69:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster: {cluster} for {sum_AA.idxmax() + '_name'} at frequency: {sum_AA.max().item()*100}%')
            if function_one_hot.max(axis=1).item() > 0.69:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for {function_one_hot.idxmax(axis=1).item() + '_function'} at frequency: {function_one_hot.max(axis=1).item()*100}%')
            if shape_one_hot.max(axis=1).item() > 0.69:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for {shape_one_hot.idxmax(axis=1).item() + '_shape'} at frequency: {shape_one_hot.max(axis=1).item()*100}%')
        elif feature == 'sec_struct':
            one_hot = pd.get_dummies(feature_df.loc[:, feature], dtype=float)
            sum_SS = one_hot.sum(axis=0) / one_hot.shape[0]
            quantized['SS'] = [sum_SS.max()]
            if sum_SS.max() > 0.79:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for {sum_SS.idxmax() + '_SS'} at frequency: {sum_SS.max().item()*100}%')
        elif feature == 'ASA':
            avg = feature_df[feature].mean()
            if avg < 0.01:
                print(f'Cluster : {cluster} for low ASA at avg: {avg}')
        elif feature in ['ASA', 'phi', 'psi', 'bfactor', 'xy', 'xz', 'yz', 'philic', 'phobic', 'SB']:
            if feature_df[feature].mean() == 0:
                quantized[feature] = [0]
            else:
                quantized[feature] = [feature_df[feature].std() / feature_df[feature].mean()]
            #if quantized[feature].item() < 0.20:
            #    print(f'Cluster : {cluster} for {feature} at CV: {quantized[feature].item()}')
    return quantized

def categorize_cluster_encodings(feature_df, encodings_df, feature):
    if feature == 'sec_struct':
        one_hot = pd.get_dummies(feature_df.loc[:, feature], dtype=float)
        sum_feature = one_hot.sum(axis=0) / one_hot.shape[0]
        if sum_feature.max() > 0.74:
            print(cluster + 1)
            print(sum_feature.idxmax())
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])
    if feature == 'AA_function':
        one_hot = pd.get_dummies(feature_df.loc[:, 'residue'], dtype=float)
        sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
        function_one_hot = categorize_AA(sum_AA, 'function')
        if function_one_hot.max(axis=1).item() > 0.74:
            print(cluster + 1)
            print(function_one_hot)
            print(function_one_hot.idxmax())
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])
    if feature == 'AA_name':
        one_hot = pd.get_dummies(feature_df.loc[:, 'residue'], dtype=float)
        sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
        if sum_AA.max().item() > 0.45:
            print(cluster + 1)
            print(sum_AA)
            print(sum_AA.idxmax())
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])
    if feature == 'ASA':
        avg = feature_df.loc[:, feature].mean()
        if avg < 0.025:
            print(cluster + 1)
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])

cluster_dict = {}
# newlsamples interesting clusters [3, 4, 14, 16, 17, 19, 20, 21, 35, 38, 40, 45, 46, 53, 98]
#[0, 3, 12, 13, 14, 25, 39, 40, 42, 50, 55, 73, 78, 81, 86]
encoding_info = pd.read_csv('../encodings/normalized_encodings_' + args.csv + '.csv', index_col=0)
feature_sums = pd.DataFrame()
all_features = pd.DataFrame()
for cluster in range(pdb_info.shape[1] - 1): # Print out features for each cluster
    #cluster = 17
    if args.automate == False:
        print(f'CLUSTER: {cluster + 1}')
    node_cluster_features = node_features.loc[[str(identifier) for identifier in pdb_info['identifier_' + str(cluster)]]]
    node_cluster_encodings = encoding_info.loc[[str(identifier) for identifier in pdb_info['identifier_' + str(cluster)]]]
    pdb_ids = [str(identifier[:4]) for identifier in pdb_info['identifier_' + str(cluster)]]
    pdb_chains = [str(identifier[4]) for identifier in pdb_info['identifier_' + str(cluster)]]
    pdb_res_ids = [int(identifier[5:]) for identifier in pdb_info['identifier_' + str(cluster)]]
    uniprot_ids = []
    if args.automate: # Do not use Uniprot Ids
        #categorize_cluster_encodings(node_cluster_features, node_cluster_encodings, 'AA_name')
        df = quantify_categorization(node_cluster_features)
        df.index = [cluster + 1]
        feature_sums = pd.concat([feature_sums, df], axis=0)
        all_features = pd.concat([all_features, node_cluster_features], axis=0)
    else:
        # Find Uniprot IDs from tsv files
        for i in range(10):#pdb_info.shape[0]):
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
        for i in range(10):#pdb_info.shape[0]):
            features = get_uniprot_features(uniprot_ids[i])
            feature_dict = build_per_residue_annotation(features)
            try:
                uniprot_features[pdb_info['identifier_' + str(cluster)][i]] = feature_dict[pdb_res_ids[i]]
            except KeyError:
                uniprot_features[pdb_info['identifier_' + str(cluster)][i]] = {} # If there are no Uniprot features for that residue, feature_dict[pdb_res_ids[i]] returns key error
            print(uniprot_features[pdb_info['identifier_' + str(cluster)][i]]) # To show easily, Uniprot features are printed separately

        uniprot_df = pd.DataFrame(uniprot_features).T
        all_features = pd.concat([node_cluster_features, uniprot_df], axis=1)
        print(all_features.iloc[:, :18])
        
        plt.figure()
        plt.imshow(node_cluster_encodings.iloc[:, :128])
        plt.title("Node Encodings")
        plt.xlabel("Neuron")
        plt.ylabel("Residue")
        plt.savefig('node_encodings')
        plt.show()
        
if args.automate:
    print(feature_sums.mean(axis=0))#.to_string(index=False))
'''
for idx, col in enumerate(all_features.columns[2:12]):
    for j in range(pdb_info.shape[1] - 1): # number of clusters
        sample = all_features.iloc[(j*25):(j*25+24), idx+2]
        population = node_features.iloc[:, idx+2].mean().item()
        ttest_result = ttest_1samp(sample, population)
        if ttest_result.pvalue < 0.05:
            if j not in cluster_dict:
                 cluster_dict[j] = 1
            else:
                cluster_dict[j] += 1
            print(col, j, ttest_result.pvalue)
'''
#print(cluster_dict)

#Cluster 25(10), 0(7), 12(6), 13(6), 14(5), 40(6), 42(5), 50(6), 55(5), 81(6), 86(7), 3(6), 39(6), 73(6), 78(6)