import numpy as np
from pathlib import Path
from natsort import natsorted
from Bio.PDB import PDBParser, DSSP
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore", 
    category=UserWarning
)

# For each PDB first MDAnalysis is used to collect spatial information
# Features stored in dict with keys "{pdb_name}{chain_ID}{resID (cannonical PDB number)}" i.e. 1ab8B1072
# Biopython.PDB.DSSP is used to collect residue information and stored in same dict

def run_dssp(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))
    model = structure[0]
    try:
        dssp = DSSP(model, str(pdb_path), dssp='mkdssp')
        for key in dssp.keys():
            try:
                chain = key[0]
                idx = key[1][1]
                ss = dssp[key][2]
                identifier = f"{str(pdb_path)[-8:-4]}{chain}{idx}"
                main_dict[identifier] = {}
                # Flag if residue lies on edge of secondary structure
                ss_edge = [False, False]
                try:
                    ss0 = dssp[(key[0], (key[1][0], key[1][1] - 1, key[1][2]))][2]
                    ss_edge[0] = (ss != ss0)
                except KeyError:
                    ss_edge[0] = True
                try:
                    ss2 = dssp[(key[0], (key[1][0], key[1][1] + 1, key[1][2]))][2]
                    ss_edge[1] = (ss != ss2)
                except KeyError:
                    ss_edge[1] = True

                if ss_edge[0] or ss_edge[1]:
                    main_dict[identifier]['ss_end'] = True
                else:
                    main_dict[identifier]['ss_end'] = False
            except (TypeError, KeyError, ):
                continue
    except Exception:
        print(f'skipped {str(pdb_path)}')

input_pdb_dir = Path('../inputs')
pdb_files = list(input_pdb_dir.glob('*.pdb'))
pdb_files = np.array([Path(p) for p in natsorted([str(p) for p in pdb_files])])
np.random.seed(0)
main_df = pd.DataFrame()
for pdb_file in pdb_files:
    main_dict = {}
    print("Working on: " + str(pdb_file)[-8:])
    run_dssp(pdb_file)
    column_names = ['ss_end']
    if main_dict != {}:
        df = pd.DataFrame(main_dict).T[column_names]
        main_df = pd.concat([main_df, df], axis=0)
    else:
        continue

main_df.index.name = 'identifier'
main_df.to_csv('ss_end_node_features.csv', index=True)
print("Node features saved")
