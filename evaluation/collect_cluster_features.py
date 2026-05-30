# pull out uniprot IDs, find Uniprot features, and tie them to node summary data
import pandas as pd
import requests
import numpy as np
import argparse
#from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv", type=str, default='llb')
argparser.add_argument("--num_to_print", type=int, default=10)
argparser.add_argument("--automate", action='store_true')
args = argparser.parse_args()


pdb_info_csv = 'data_labeling/' + args.csv + '_pdb_info.csv' ## Set to TESTCHECK
node_features_csv = 'created_data/features/node_features.csv'
pdb_info = pd.read_csv(pdb_info_csv)
node_features = pd.read_csv(node_features_csv, index_col=0, header=0)

feature_types_pair = ['Disulfide bond', 'Cross-link'] # Feature types with that are for pairs of residues, not relevant to residues inbetween start and end values

# Return Uniprot ID from PDB ID from precalculated tsv files created using PDB2Uniprot
def find_uniprot_id(uniprot_ids, pdb_id, pdb_chain, pdb_res_id):
            found = False
            with open('data_labeling/uniprot/out/' + str(pdb_id)+'.tsv', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    try:
                        ids = line.split('\t')
                        chain = str(ids[1])
                        res_id = int(ids[3])
                        uniprot_id = str(ids[4])
                        if chain == pdb_chain and res_id == pdb_res_id:
                            uniprot_ids.append(uniprot_id)
                            found = True
                    except:
                        continue
                if found != True:
                    uniprot_ids.append("Not Found")
            return uniprot_ids

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
                    #per_residue[start][feat['type'] + ' val'] = True
                    per_residue[start][feat['type'] + ' description'] = feat['description']
                    if end not in per_residue:
                        per_residue[end] = {}
                    #per_residue[end][feat['type'] + ' val'] = True
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
                        #per_residue[i][feat['type'] + ' val'] = True
                        per_residue[i][feat['type'] + ' description'] = feat['description']
                except (ValueError, TypeError):
                    print(feat['location']['start'])
                    print(feat['location']['end'])
                    continue
    return per_residue

def categorize_AA(feature_df):
    # By function
    function_dict = {
        'phobic': list('HAVLMI'),
        'philic': list('HAVLMI'),
        'pos': list('KRH'),
        'neg': list('DE'),
        'aro': list('FYW')
    }

    # By shape
    shape_dict = {
        'small': list('GA'),
        'chain2': list('SC'),
        'branch1': list('VTND'),
        'branch2': list('ILQE'),
        'long': list('MKR'),
        'ring1': list('PHFY'),
        'ring2': list('W')
    }

    # Mapping for residue to function
    residue_to_function = {
        residue: category
        for category, residues in function_dict.items()
        for residue in residues
    }

    # Mapping for residue to shape
    residue_to_shape = {
        residue: category
        for category, residues in shape_dict.items()
        for residue in residues
    }

    feature_df['Function'] = feature_df['residue'].map(residue_to_function)
    feature_df['Shape'] = feature_df['residue'].map(residue_to_shape)


    #df = pd.DataFrame()
    #if categorize_by == 'function':
    #    for i in feature_df.index:
    #        df[i] = feature_df[i]
    #    feature_df['residue']
    #    for i in function_dict.keys():
    #        df[i] = [feature_df[list(set(feature_df.index) & set(function_dict[i]))].sum()]
    #if categorize_by == 'shape':
    #    for i in shape_dict.keys():
    #        df[i] = [feature_df[list(set(feature_df.index) & set(shape_dict[i]))].sum()]
    #return df


def counttop3(feature_df, feature):
    counts = feature_df[feature].value_counts()
    percents = feature_df[feature].value_counts(normalize=True) * 100

    top3 = pd.DataFrame({
        "Value": counts.index[:3],
        #"count": counts.values[:3],
        "% Occurence": [round(val, 2) for val in percents.values[:3]]
    }) 
    return top3

def cat_summarize_cluster(feature_df):
    
    # Add function and shape columns
    categorize_AA(feature_df)
    cluster_cat_summary = pd.concat(
        {
            "Res": counttop3(feature_df, 'residue'),
            "Function": counttop3(feature_df, 'Function'),
            "Shape": counttop3(feature_df, 'Shape'),
            "Sec Struct": counttop3(feature_df, 'sec_struct')
        }
    )
    print(cluster_cat_summary)

def quant_summarize_cluster(feature_df):
    cols = [2, 3, 4, 9, 10, 11, 12 ,13, 14, 15, 16, 17]

    summary_df = pd.DataFrame({
        'mean': feature_df.iloc[:, cols].mean(),
        'std': feature_df.iloc[:, cols].std()
    })

    print(summary_df)

# For each cluster, counts how many "significant" characteristics there are
def quantify_categorization(feature_df, aa_tresh=0.7, ss_thresh=0.8, asa_thresh=0.1):
    quantized = pd.DataFrame()
    for feature in feature_df.columns:
        # cluster_dict[cluster] = # of common (70%) features (amino acid, shape, functionality) in a cluster
        if feature == 'residue':
            one_hot = pd.get_dummies(feature_df.loc[:, feature], dtype=float)
            sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
            function_one_hot = categorize_AA(sum_AA, 'function')
            shape_one_hot = categorize_AA(sum_AA, 'shape')
            quantized['AA_name'] = [sum_AA.max()]
            quantized['AA_function'] = function_one_hot.max(axis=1)
            quantized['AA_shape'] = shape_one_hot.max(axis=1)
            
            cluster_dict[cluster]['res'] = ''
            
            if sum_AA.max() > aa_tresh:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster: {cluster} for {sum_AA.idxmax()}_name at frequency: {sum_AA.max().item()*100}%')
            if function_one_hot.max(axis=1).item() > aa_tresh:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for {function_one_hot.idxmax(axis=1).item()}_function at frequency: {function_one_hot.max(axis=1).item()*100}%')
            if shape_one_hot.max(axis=1).item() > aa_tresh:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for {shape_one_hot.idxmax(axis=1).item()}_shape at frequency: {shape_one_hot.max(axis=1).item()*100}%')
        
        # How many common (80%) secondary structures are there?
        elif feature == 'sec_struct':
            one_hot = pd.get_dummies(feature_df.loc[:, feature], dtype=float)
            sum_SS = one_hot.sum(axis=0) / one_hot.shape[0]
            quantized['SS'] = [sum_SS.max()]
            if sum_SS.max() > ss_thresh:
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for {sum_SS.idxmax()}_SS at frequency: {sum_SS.max().item()*100}%')
        
        # Is cluster avg ASA significantly high (0.9) or low (0.1)?
        elif feature == 'ASA':
            avg = feature_df[feature].mean()
            if avg < asa_thresh or avg > (1-asa_thresh):
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = 1
                else:
                    cluster_dict[cluster] += 1
                print(f'Cluster : {cluster} for ASA at avg: {avg}')
    return quantized

def categorize_cluster_encodings(feature_df, encodings_df, feature, aa_thresh=0.7, ss_thresh=0.8, asa_thresh=0.1):

    # Print if most common secondary structure in cluster is > 80%
    if feature == 'sec_struct':
        one_hot = pd.get_dummies(feature_df.loc[:, feature], dtype=float)
        sum_feature = one_hot.sum(axis=0) / one_hot.shape[0]
        if sum_feature.max() > ss_thresh:
            print(cluster + 1)
            print(sum_feature.idxmax())

            # ???
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])

    # Print if most common amino acid function in cluster is > 70%
    if feature == 'AA_function':
        one_hot = pd.get_dummies(feature_df.loc[:, 'residue'], dtype=float)
        sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
        shape_one_hot = categorize_AA(sum_AA, 'shape')
        if shape_one_hot.max(axis=1).item() > aa_thresh:
            print(cluster + 1)
            print(shape_one_hot)
            print(shape_one_hot.idxmax())
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])

    # Print if most common amino acid function in cluster is > 70%
    if feature == 'AA_shape':
        one_hot = pd.get_dummies(feature_df.loc[:, 'residue'], dtype=float)
        sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
        function_one_hot = categorize_AA(sum_AA, 'function')
        if function_one_hot.max(axis=1).item() > aa_thresh:
            print(cluster + 1)
            print(function_one_hot)
            print(function_one_hot.idxmax())
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])

    # Print if most common amino acid in cluster is > 50%        
    if feature == 'AA_name':
        one_hot = pd.get_dummies(feature_df.loc[:, 'residue'], dtype=float)
        sum_AA = one_hot.sum(axis=0) / one_hot.shape[0]
        if sum_AA.max().item() > aa_thresh:
            print(cluster + 1)
            print(sum_AA)
            print(sum_AA.idxmax())
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])

    # Print if avg ASA in cluster is < 0.1 or > 0.9
    if feature == 'ASA':
        avg = feature_df.loc[:, feature].mean()
        if avg < asa_thresh or avg > (1-asa_thresh):
            print(cluster + 1)
            encod_avg = encodings_df.mean(axis=0)
            print(encod_avg.index.values[encod_avg > 0.05])
            print(encod_avg[encod_avg > 0.05])


cluster_dict = {}
encoding_info = pd.read_csv('created_data/encodings/ki_encodings/output_' + args.csv + '.csv', index_col=0)
feature_sums = pd.DataFrame()
all_features = pd.DataFrame()

for cluster in range(pdb_info.shape[1] - 1): # Print out features for each cluster
    node_cluster_features = node_features.loc[[str(identifier) for identifier in pdb_info['identifier_' + str(cluster)]]]
    node_cluster_encodings = encoding_info.loc[[str(identifier) for identifier in pdb_info['identifier_' + str(cluster)]]]
    print(f'CLUSTER: {cluster + 1}')
    if args.automate: # Does not use Uniprot Ids, reports common categorical features and averages/stdev for quantitative features
        cat_summarize_cluster(node_cluster_features)
        quant_summarize_cluster(node_cluster_features)
    
    else:
        # Find Uniprot IDs from tsv files
        pdb_ids = [str(identifier[:4]) for identifier in pdb_info['identifier_' + str(cluster)]]
        pdb_chains = [str(identifier[4]) for identifier in pdb_info['identifier_' + str(cluster)]]
        pdb_res_ids = [int(identifier[5:]) for identifier in pdb_info['identifier_' + str(cluster)]]
        uniprot_ids = []
        for i in range(10): # Look only at top 10 Uniprot labels in cluster
            find_uniprot_id(uniprot_ids, pdb_ids[i], pdb_chains[i], pdb_res_ids[i])
        
        # Collect Uniprot features for each identifier in the cluster
        uniprot_features = {}
        for i in range(10):
            features = get_uniprot_features(uniprot_ids[i])
            feature_dict = build_per_residue_annotation(features)
            try:
                uniprot_features[pdb_info['identifier_' + str(cluster)][i]] = feature_dict[pdb_res_ids[i]]
            except KeyError:
                uniprot_features[pdb_info['identifier_' + str(cluster)][i]] = {} # If there are no Uniprot features for that residue, feature_dict[pdb_res_ids[i]] returns key error
            print(uniprot_features[pdb_info['identifier_' + str(cluster)][i]]) # To show easily, Uniprot features are printed separately

        # Combine Uniprot with MDAnalysis features and print
        uniprot_df = pd.DataFrame(uniprot_features).T
        all_features = pd.concat([node_cluster_features, uniprot_df], axis=1)
        print(all_features.iloc[:, :18]) # Don't reprint Uniprot features
        
        # Display encodings for cluster (Good clustering should show very similar encoding pattern)
        plt.figure()
        plt.imshow(node_cluster_encodings.iloc[:, :128])
        plt.title("Node Encodings")
        plt.xlabel("Neuron")
        plt.ylabel("Residue")
        plt.savefig('node_encodings')
        plt.show()
        
