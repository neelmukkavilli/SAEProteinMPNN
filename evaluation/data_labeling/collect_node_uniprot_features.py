import pandas as pd
import numpy as np
import requests
import os

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

feature_types_pair = ['Disulfide bond', 'Cross-link']

def match_resids(input, pdb_resid, uniprot_resid):
    indices = [i for i, v in enumerate(uniprot_resid) if int(v[5:]) == input]
    if len(indices) > 0:
        pdb_residues = [pdb_resid[index] for index in indices]
        return pdb_residues, len(indices)
    if len(indices) == 0:
        return None, None

def build_per_residue_annotation(features, pdb_resid, uniprot_resid):
    per_residue = {}
    if features != "Not Found":
        for feat in features:
            if feat['type'] in feature_types_pair:
                try:
                    start = int(feat['location']['start']['value'])
                    end = int(feat['location']['end']['value'])
                    start, num_start = match_resids(start, pdb_resid, uniprot_resid)
                    end, num_end = match_resids(start, pdb_resid, uniprot_resid)
                    if start != None:
                        for i in range(num_start):
                            if start[i] not in per_residue:
                                per_residue[start[i]] = {}
                            per_residue[start[i]][feat['type'] + ' val'] = True
                            per_residue[start[i]][feat['type'] + ' description'] = feat['description']
                    if end != None:
                        for i in range(num_end):
                            if end[i] not in per_residue:
                                per_residue[end[i]] = {}
                            per_residue[end[i]][feat['type'] + ' val'] = True
                            per_residue[end[i]][feat['type'] + ' description'] = feat['description']
                except TypeError: # Sometimes end value is a NoneType so I wrap in a try except clause, probably a better way to handle this
                    continue      
            else:
                try:
                    start = int(feat['location']['start']['value'])
                    end = int(feat['location']['end']['value'])
                    for val in range(start, end+1):
                        resid, num_id = match_resids(val, pdb_resid, uniprot_resid)
                        if resid != None:
                            for i in range(num_id):
                                if resid[i] not in per_residue:
                                    per_residue[resid[i]] = {}
                                per_residue[resid[i]][feat['type'] + ' val'] = True
                                per_residue[resid[i]][feat['type'] + ' description'] = feat['description']
                except TypeError: # Sometimes end value is a NoneType so I wrap in a try except clause, probably a better way to handle this
                     continue
    return per_residue

folder_path = 'uniprot/out/'

total_uniprot_features = pd.DataFrame()
for filename in os.listdir(folder_path):
    pdb_resid = []
    uniprot_resid = []
    file_path = folder_path + str(filename)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
                    ids = line.split('\t')
                    pdb_id = ids[0]
                    pdb_aa = ids[2]
                    chain = ids[1]
                    res_id = ids[3]
                    uniprot_id = ids[4]
                    uniprot_aa = ids[5]
                    up_res_id = ids[6][:-1]
                        
                    if pdb_aa == uniprot_aa and res_id != 'null':
                        pdb_resid.append((pdb_id+chain+res_id))
                        uniprot_resid.append((pdb_id+chain+up_res_id))

    features = get_uniprot_features(uniprot_id) # last uniprot id in file, probably fine
    feature_dict = build_per_residue_annotation(features, pdb_resid, uniprot_resid)

    uniprot_features = pd.DataFrame(feature_dict).T
    total_uniprot_features = pd.concat([total_uniprot_features, uniprot_features])

    print(uniprot_features)
output_path = '../created_data/features/uniprot_features.csv'
total_uniprot_features.to_csv(output_path, mode='w', header = True)
