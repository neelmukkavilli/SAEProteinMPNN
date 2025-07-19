import os
from pathlib import Path
import PeptideBuilder
from Bio.PDB import PDBParser, DSSP, PPBuilder
import Bio.PDB.Polypeptide as polypep
import pandas as pd
import warnings
from natsort import natsorted
import numpy as np
import glob

warnings.filterwarnings("ignore")

# === Update your paths ===
input_pdb_dir = Path('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/inputs')   # PDBs from .pt conversion
dssp_dir = "dssp_data"
#dssp_dir.mkdir(parents=True, exist_ok=True)
results = []
chain_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L']

def run_dssp(pdb_path):
    parser = PDBParser(QUIET=False)
    structure = parser.get_structure("prot", str(pdb_path))
    model = structure[0]
    dssp = DSSP(model, str(pdb_path), dssp='mkdssp')
    indexes = []
    old_index = 0
    chain = 0
    for key in dssp.keys():
        #current_idx = int(dssp[key][0])
        idx = key[1][1]
        indexes.append(idx)
        if old_index > idx:
            chain += 1
            old_index = idx
        else:
            old_index = idx
        didx, aa, ss, acc, phi, psi, NO1, ON1, NO2, ON2 = (
            dssp[key][0], dssp[key][1], dssp[key][2], dssp[key][3],
            dssp[key][4], dssp[key][5], dssp[key][7], dssp[key][9],
            dssp[key][11], dssp[key][13]
        )
        results.append({
            "identifier": f"{pdb_path.name[:-4]}{chain_letters[chain]}{idx}",
            "residue": aa,
            "sec_struct": ss,
            "ASA": acc,
            "phi": phi,
            "psi": psi,
            "NO1": NO1,
            "ON1": ON1,
            "NO2": NO2,
            "ON2": ON2
        })
    #print(len(indexes))

#run_dssp(Path('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/inputs/1abn.pdb'))
#print('test')
# === Run pipeline on all PDBs ===
pdb_files = list(input_pdb_dir.glob('*.pdb'))
#print(pdb_files)
pdb_files = [Path(p) for p in natsorted([str(p) for p in pdb_files])]
pdb_files = pdb_files
for pdb_file in pdb_files:
    print(pdb_file.name)
    run_dssp(pdb_file)

# === Save to CSV ===
def normalize(arr):
    arr = pd.to_numeric(arr, errors='coerce')
    min_val = arr.min(skipna=True)
    max_val = arr.max(skipna=True)

    if max_val != min_val:
        normalized_col = (arr - min_val)/(max_val - min_val)
    else:
        normalized_col = (arr - min_val)
    '''    
    for j in range(arr.shape[0]):
        if arr.iloc[j] == 'NA':
            #print(arr.iloc[j])
            #print(type(arr.iloc[j]))
            arr.iloc[j] = float('nan')
    max_val = arr.max()
    min_val = arr.min()

    if max_val != min_val:
        normalized_col = (arr - min_val)/(max_val - min_val)
    else:
        normalized_col = (arr - min_val)
    normalized_col = pd.to_numeric(normalized_col, errors='coerce').round(5)
    '''

    return normalized_col.round(5)

def categorize_angles(arr):
    arr = pd.to_numeric(arr, errors='coerce')
    arr = (arr +360) % 360
    arr = np.where(arr > 180, 360 - arr, arr)
    '''
    for j in range(arr.shape[0]):
        if not isinstance(arr.iloc[j], float):
            #print(arr.iloc[j])
            #print(type(arr.iloc[j]))
            arr.iloc[j] = np.nan
    for j in range(arr.shape[0]):
        arr += 360
        arr = arr % 360

        if float(arr.iloc[j]) > 180:
            arr.iloc[j] = 360 - float(arr.iloc[j])

    normalized_col = pd.to_numeric(arr/180, errors='coerce').round(5)
    '''
    return pd.Series(arr).round(5)

df = pd.DataFrame(results)

#for key in df.keys():
#    print(key)

arr = df['ASA']
norm_arr = normalize(arr)
df['ASA'] = norm_arr
print("ASA normalized")

arr = df['NO1']
norm_arr = normalize(arr)
df['NO1'] = norm_arr
print('NO1 normalized')

arr = df['ON1']
norm_arr = normalize(arr)
df['ON1'] = norm_arr
print('ON1 normalized')

arr = df['NO2']
norm_arr = normalize(arr)
df['NO2'] = norm_arr
print('NO2 normalized')

arr = df['ON2']
norm_arr = normalize(arr)
df['ON2'] = norm_arr
print('ON2 normalized')

arr = df['phi']
norm_arr = categorize_angles(arr)
df['phi'] = norm_arr
print('phi normalized')

arr = df['psi']
norm_arr = categorize_angles(arr)
df['psi'] = norm_arr
print('psi normalized')

df.to_csv("dssp_summary.csv", index=False)
print(f"✅ DSSP data saved to dssp_summary.csv with {len(df)} rows.")
