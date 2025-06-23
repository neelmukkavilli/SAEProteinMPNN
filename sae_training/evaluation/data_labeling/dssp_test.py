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
from Bio.PDB import PDBParser, is_aa
warnings.filterwarnings("ignore")

# === Update your paths ===
input_pdb_dir = Path("/home/neelm/data/ProteinMPNN-copy/sae_training/evaluation/inputs/")   # PDBs from .pt conversion
dssp_dir = Path("/home/neelm/data/ProteinMPNN-copy/sae_training/evaluation/data_labeling/dssp_data")
dssp_dir.mkdir(parents=True, exist_ok=True)


required_atoms = {'N', 'CA', 'C', 'O'}

def check_integrity(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    model = structure[0]  # or select appropriate model
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            atoms = {atom.get_id() for atom in residue}
            missing = required_atoms - atoms
            if missing:
                print(f"Residue {residue.get_resname()} {residue.id} in chain {chain.id} is missing: {missing}")

pdb_files = list(input_pdb_dir.glob("*.pdb"))
pdb_files = [Path(p) for p in natsorted([str(p) for p in pdb_files ])]
pdb_files = pdb_files
for pdb_file in pdb_files:
    print(pdb_file.name)
    check_integrity(pdb_file)
