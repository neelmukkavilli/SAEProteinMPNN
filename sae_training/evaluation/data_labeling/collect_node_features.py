import MDAnalysis
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
#from pdbfixer import PDBFixer
#from openmm.app import PDBFile
from MDAnalysis.analysis import contacts
from pathlib import Path
from natsort import natsorted
from Bio.PDB import PDBParser, DSSP, PPBuilder
#import Bio.PDB.Polypeptide as polypep
import pandas as pd

main_dict = {}

# For each PDB first MDAnalysis is used to collect spatial information
# Features stored in dict with keys "{pdb_name}{chain_ID}{resID (cannonical PDB number)}" i.e. 1ab8B1072
# Biopython.PDB.DSSP is used to collect residue information and stored in same dict
# Some values then normalized and data stored in csv

def collect_B_factor(calphas, pdb_name): # Returns main_dict[ID]['bfactor] = val

    for atom in calphas:
        identifier = f"{pdb_name}{atom.chainID}{int(atom.resnum)}"
        if identifier not in main_dict:
            main_dict[identifier] = {}
        if 'bfactor' not in main_dict[identifier]:
            main_dict[identifier]['bfactor'] = round(float(atom.tempfactor), 3)

def find_angles(calphas, pdb_name): # Returns main_dict[ID]['xy'], main_dict[ID]['xz'], main_dict[ID]['yz'] = val1, val2, val3

    positions = np.zeros([len(calphas) - 2, 3, 3])
    for i in range(len(calphas) - 2):
        positions[i, 0, :] = calphas[i].position
        positions[i, 1, :] = calphas[i + 1].position
        positions[i, 2, :] = calphas[i + 2].position

    # Calculate 3 angles of CA-CA-CA plane
    for i in range(len(calphas) - 2):
        v1 = positions[i, 1] - positions[i, 0]
        v2 = positions[i, 2] - positions[i, 0]

        n = np.cross(v1, v2)
        normxy = [0, 0, 1]
        normxz = [0, 1, 0]
        normyz = [1, 0, 0]

        identifier = f"{pdb_name}{calphas[i+1].chainID}{int(calphas[i + 1].resnum)}"

        main_dict[identifier]['xy'] = round(np.rad2deg(np.arccos(np.dot(n, normxy) / np.dot(np.linalg.norm(n), np.linalg.norm(normxy)))).item(), 3)
        main_dict[identifier]['xz'] = round(np.rad2deg(np.arccos(np.dot(n, normxz) / np.dot(np.linalg.norm(n), np.linalg.norm(normxz)))).item(), 3)
        main_dict[identifier]['yz'] = round(np.rad2deg(np.arccos(np.dot(n, normyz) / np.dot(np.linalg.norm(n), np.linalg.norm(normyz)))).item(), 3)
    
    identifier = f"{pdb_name}{calphas[0].chainID}{int(calphas[0].resnum)}"
    main_dict[identifier]['xy'], main_dict[identifier]['xz'], main_dict[identifier]['yz'] = float('nan'), float('nan'), float('nan')
    identifier = f"{pdb_name}{calphas[-1].chainID}{int(calphas[-1].resnum)}"
    main_dict[identifier]['xy'], main_dict[identifier]['xz'], main_dict[identifier]['yz'] = float('nan'), float('nan'), float('nan')

def count_pairs(sel1, sel2, min_thresh, max_thresh, pdb_name, pair): # Returns main_dict[ID]['pair type'] = number of contacts within threshold
    distance_array = MDAnalysis.analysis.distances.distance_array(sel1.positions, sel2.positions)

    true_false_dist_array = (distance_array >= min_thresh) & (distance_array < max_thresh)
    pos_true_false = np.sum(true_false_dist_array, axis = 0)
    neg_true_false = np.sum(true_false_dist_array, axis = 1)

    for idx in range(len(sel2)):
        identifier = f"{pdb_name}{sel2[idx].chainID}{int(sel2[idx].resnum)}"
        if pair not in main_dict[identifier]:
            main_dict[identifier][pair] = pos_true_false[idx].item()
        else:
            main_dict[identifier][pair] += pos_true_false[idx].item()

    if pair == 'SB': # phobic and philic residues are double counted in rows/columns so only one axis is summed
        for idx in range(len(sel1)):
            identifier = f"{pdb_name}{sel1[idx].chainID}{int(sel1[idx].resnum)}"
            if pair not in main_dict[identifier]:
                main_dict[identifier][pair] = neg_true_false[idx].item()
            else:
                main_dict[identifier][pair] += neg_true_false[idx].item()

def count_distance_bins(sel1, sel2, dmin, dmax, rbf, pdb_name): # Returns main_dict[ID][minimum distance of bin] = number of contacts within that bin, number of contacts limited to closest 48 per atom type => 48 * 5 = 240 contacts per residue
    distance_array = MDAnalysis.analysis.distances.distance_array(sel1.positions, sel2.positions)
    sorted_array = np.sort(distance_array, axis=1)
    distance_array = sorted_array[:, :48]
    distances = np.linspace(dmin, dmax, rbf + 1) # Values outside of bins set to bin=0 or bin=len(bins)

    edge_dist_bins = np.digitize(distance_array, distances) -1 
    for i in range(edge_dist_bins.shape[0]):
        for j in range(edge_dist_bins.shape[1]):
            identifier = f"{pdb_name}{sel1[i].chainID}{sel1[i].resnum}"
            dist_bin = edge_dist_bins[i, j].item()
            if dist_bin not in main_dict[identifier]:
                main_dict[identifier][dist_bin] = 1
            else:
                main_dict[identifier][dist_bin] += 1
'''
def count_HB(pdb_path): # Returns dict {residue: number of h bonds either acceptor or donor, res1: a, res2: b,...}}
    fixer = PDBFixer(pdb_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open('output_h.pdb', 'w'), keepIds=True)
    u = MDAnalysis.Universe('output_h.pdb')
    hbonds = HBA(universe = u,
                 donors_sel = 'protein and name N NE2 NZ ND2 NE1',
                 hydrogens_sel = 'protein and name H*',
                 acceptors_sel = 'protein and name O OD1 OE1 OG OG1 OH')
    
    hbonds.run()
    hbond_count = hbonds.count_by_ids()
    hbond_count = np.delete(hbond_count, [2, 3], axis=1)
    #hbond_count_acceptor = np.zeros(hbond_count.shape[0])
    for i in range(hbond_count.shape[0]):
        for j in range(2):
            hbond_count[i, j] = (u.atoms[hbond_count[i, j]]).residue.resnum
    print(hbond_count.flatten()[:20])
    print(hbond_count.shape)
    HBcount_results = {}
    for res in hbond_count.flatten():
        if res not in HBcount_results:
            HBcount_results[int(res)] = 1
        else:
            HBcount_results[int(res)] += 1
'''
def find_node_features(pdb_path):
    pdb_name = pdb_path[-8:-4]
    u = MDAnalysis.Universe(str(pdb_path))

    calphas = u.select_atoms('protein and resname GLY ALA PRO VAL LEU ILE MET SER THR CYS ASN GLN PHE TYR TRP LYS ARG HIS ASP GLU and name CA')
    collect_B_factor(calphas, pdb_name)
    
    find_angles(calphas, pdb_name)
    u = calphas.residues.atoms
    sel1 = u.select_atoms("protein and resname GLY ALA PRO VAL LEU ILE MET SER THR CYS ASN GLN PHE TYR TRP LYS ARG HIS ASP GLU and not name H* C*")

    count_pairs(sel1, sel1, 0, 3.5, pdb_name, pair='philic')

    sel1 = u.select_atoms("protein and resname GLY ALA PRO VAL LEU ILE MET SER THR CYS ASN GLN PHE TYR TRP LYS ARG HIS ASP GLU and name H* C*")
    count_pairs(sel1, sel1, 2.5, 4.6, pdb_name, pair='phobic')

    sel1 = u.select_atoms("protein and resname ARG LYS HIS and name NZ ND1 NE2 NH1 NH2")
    sel2 = u.select_atoms("(protein and (resname ASP GLU and name OD1 OD2 OE1 OE2)) or (protein and name OXT)")
    count_pairs(sel1, sel2, 0, 4.0, pdb_name, pair='SB')

    sel1 = u.select_atoms("protein and resname GLY ALA PRO VAL LEU ILE MET SER THR CYS ASN GLN PHE TYR TRP LYS ARG HIS ASP GLU and name CA C O N CB")
    #sel2 = u.select_atoms("protein and name Ca C O N CB")
    dmin = 2 - 4/6
    dmax = 22 + 4/6
    rbf = 16
    count_distance_bins(sel1, sel1, dmin, dmax, rbf, pdb_name)

    for i in calphas: # If there are no contacts in that bin, write zero
        identifier = f"{pdb_name}{i.chainID}{i.resnum}"
        for j in ['bfactor', 'xy', 'xz', 'yz', 'phobic', 'philic', 'SB'] + list(range(16)):
            if not j in main_dict[identifier]:
                main_dict[identifier][j] = 0

def run_dssp(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))
    model = structure[0]
    dssp = DSSP(model, str(pdb_path), dssp='dssp')

    for key in dssp.keys():
        try:
            chain = key[0]
            idx = key[1][1]
            aa, ss, asa, phi, psi = (
                dssp[key][1], dssp[key][2], dssp[key][3],
                dssp[key][4], dssp[key][5]
            )
            identifier = f"{str(pdb_path)[-8:-4]}{chain}{idx}"
            main_dict[identifier]['residue'] = aa
            main_dict[identifier]['sec_struct'] = ss
            main_dict[identifier]['ASA'] = asa
            main_dict[identifier]['phi'] = phi
            main_dict[identifier]['psi'] = psi
        except KeyError:
            continue

input_pdb_dir = Path('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/inputs')
pdb_files = list(input_pdb_dir.glob('*.pdb'))
pdb_files = np.array([Path(p) for p in natsorted([str(p) for p in pdb_files])])
np.random.seed(0)
#mask = np.random.rand(len(pdb_files)) > 0
pdb_files = pdb_files#[mask]
print(len(pdb_files))
#print(set(pdb_files))
print(len(set(pdb_files)))
#addition += 1
pdb_files = pdb_files  # Limit to first few PDB files for testing
main_df = pd.DataFrame()
for pdb_file in pdb_files:
    find_node_features(str(pdb_file))
    print("Working on: " + str(pdb_file)[-8:])
    run_dssp(pdb_file)
    #column_names = ['bfactor', 'xy', 'xz', 'yz', 'philic', 'phobic', 'SB'] + list(range(16))
    column_names = ['residue', 'sec_struct', 'ASA', 'phi', 'psi', 'bfactor', 'xy', 'xz', 'yz', 'philic', 'phobic', 'SB'] + list(range(16))
    df = pd.DataFrame(main_dict).T[column_names]
    main_dict = {}
    main_df = pd.concat([main_df, df], axis=0)

# Normalization steps
def norm_angles(arr):
    #arr = arr.apply(lambda x: (x + 360) % 360 if pd.notnull(x) else x)
    arr = pd.to_numeric(arr, errors='coerce')
    arr = (arr + 360) % 360
    arr = np.where(arr > 180, 360 - arr, arr)

    return pd.Series(arr).round(5)

def norm_vals(arr):
    arr = pd.to_numeric(arr, errors='coerce')
    min_val = arr.min(skipna=True)
    max_val = arr.max(skipna=True)

    if max_val != min_val:
        normalized_col = (arr - min_val)/(max_val - min_val)
    else:
        normalized_col = (arr - min_val)

    return normalized_col.round(5)

arr = main_df['ASA']
norm_arr = norm_vals(arr)
main_df['ASA'] = norm_arr.values
print("ASA normalized")

arr = main_df['phi']
norm_arr = norm_angles(arr)
main_df['phi'] = norm_arr.values
print("phi normalized")

arr = main_df['psi']
norm_arr = norm_angles(arr)
main_df['psi'] = norm_arr.values
print("psi normalized")

arr = main_df['xy']
norm_arr = norm_angles(arr)
main_df['xy'] = norm_arr.values
print("xy normalized")

arr = main_df['xz']
norm_arr = norm_angles(arr)
main_df['xz'] = norm_arr.values
print("xz normalized")

arr = main_df['yz']
norm_arr = norm_angles(arr)
main_df['yz'] = norm_arr.values
print("yz normalized")

main_df.to_csv('node_features.csv', index=True)
print("Node features saved")
