import os
from pathlib import Path
import PeptideBuilder
from Bio.PDB import PDBParser, DSSP, PPBuilder
import Bio.PDB.Polypeptide as polypep
import pandas as pd
import warnings
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from natsort import natsorted
import numpy as np

warnings.filterwarnings("ignore")

# === Update your paths ===
input_pdb_dir = Path("/home/neelm/data/ProteinMPNN-copy/sae_training/evaluation/inputs/")   # PDBs from .pt conversion
dssp_dir = Path("/home/neelm/data/ProteinMPNN-copy/sae_training/evaluation/data_labeling/dssp_data")
dssp_dir.mkdir(parents=True, exist_ok=True)

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
        didx, aa, ss, acc, phi, psi, h_bond_energy = (
            dssp[key][0], dssp[key][1], dssp[key][2], dssp[key][3],
            dssp[key][4], dssp[key][5], dssp[key][10]
        )
        results.append({
            "identifier": f"{pdb_path.name[:-4]}{chain_letters[chain]}{idx}",
            "residue": aa,
            "sec_struct": ss,
            "ASA": acc,
            "phi": phi,
            "psi": psi,
            "h_bond_energy": h_bond_energy
        })
    print(len(indexes))

# === Run pipeline on all PDBs ===
pdb_files = list(input_pdb_dir.glob("*.pdb"))
pdb_files = [Path(p) for p in natsorted([str(p) for p in pdb_files ])]
pdb_files = pdb_files
for pdb_file in pdb_files:
    print(pdb_file.name)
    run_dssp(pdb_file)

# === Save to CSV ===
def normalize(arr):
    for j in range(arr.shape[0]):
        if arr.iloc[j] == 'NA':
            print(arr.iloc[j])
            print(type(arr.iloc[j]))
            arr.iloc[j] = float('nan')
    max_val = arr.max()
    min_val = arr.min()

    if max_val != min_val:
        normalized_col = (arr - min_val)/(max_val - min_val)
    else:
        normalized_col = (arr - min_val)

    normalized_col = round(normalized_col, 5)

    return normalized_col

def categorize_angles(arr):
    for j in range(arr.shape[0]):
        if not isinstance(arr.iloc[j], float):
            print(arr.iloc[j])
            print(type(arr.iloc[j]))
            arr.iloc[j] = np.nan
    for j in range(arr.shape[0]):
        arr += 360
        arr = arr % 360

        if float(arr.iloc[j]) > 180:
            arr.iloc[j] = 360 - float(arr.iloc[j])

    normalized_col = arr/180

    return round(normalized_col, 5)

df = pd.DataFrame(results)

#for key in df.keys():
#    print(key)

arr = df['ASA']
norm_arr = normalize(arr)
df['ASA'] = norm_arr

arr = df['h_bond_energy']
norm_arr = normalize(arr)
df['h_bond_energy'] = norm_arr

arr = df['phi']
norm_arr = categorize_angles(arr)
df['phi'] = norm_arr

arr = df['psi']
norm_arr = categorize_angles(arr)
df['psi'] = norm_arr

df.to_csv("dssp_summary.csv", index=False)
print(f"✅ DSSP data saved to dssp_summary.csv with {len(df)} rows.")
