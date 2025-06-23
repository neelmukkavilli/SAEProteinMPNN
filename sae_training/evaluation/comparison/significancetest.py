import numpy as np
import pandas as pd
import csv
import scipy.stats as sci

dssp_csv = '/home/neelm/data/ProteinMPNN-copy/sae_training/evaluation/dssp_summary.csv'
encodings = '/home/neelm/data/ProteinMPNN-copy/sae_training/evaluation/normalized_encodings.csv'

features = ['identifier', 'residue','sec_struct','ASA','phi','psi','h_bond_energy']
feature = 'ASA'
neuron = 1

#labels = np.asarray(pd.read_csv(dssp_csv, usecols=[feature]).to_numpy()[:,0])
#activation = np.asarray(pd.read_csv(encodings, usecols=[neuron]).to_numpy()[:,0])

#print(labels.shape)
def ANOVA(feature_act, encodings, feature_list):
    encodings = encodings.iloc[:, 0]
    #print(encodings.shape)
    groups = []
    for cat in feature_list:
        mask = feature_act == cat
        mask = mask.iloc[:, 0]
        groups.append(encodings[mask])

    f_stat, p_val = sci.f_oneway(*groups)

    if p_val < 0.05/1025:
        print(neuron)
        print(f_stat, p_val)
        #tukey = sci.tukey_hsd(*groups)
        #print(tukey)

def safe_pearsonr(dssp_csv, encodings):
    # Feature Amplitude
    x = pd.read_csv(dssp_csv, usecols=[feature]).to_numpy()[:,0]
    # Model Activations
    y = pd.read_csv(encodings, usecols=[neuron]).to_numpy()[:,0]

    mask = np.isfinite(x) & np.isfinite(y)
    #print(x.shape)
    x = x[mask]
    y = y[mask]
    #print(x.shape)

    if np.sum(x) == 0 or np.sum(y) == 0:
        #print("Arrays are zero")
        return 0, 0
        #raise ValueError("Arrays are zero")

    if np.std(x) == 0 or np.std(y) == 0:
        #print("One of the arrays is constant; correlation is undefined")
        return 0, 0
        raise ValueError("One of the arrays is constant; correlation is undefined.")

    return sci.pearsonr(x, y)

# Usage
'''
for neuron in range(1, 1025):
    r, p = safe_pearsonr(dssp_csv, encodings)

    if p < 1e-4 and (r > 0.8 or r < -0.8):
        print(r, p)
        print(neuron)
        print(feature)
'''
for neuron in [5, 141, 216, 276, 559, 993]:
    activations = pd.read_csv(encodings, usecols=[neuron]).to_numpy()[:,0]
    identifier = pd.read_csv(encodings, usecols=[0]).to_numpy()[:,0]
    max_val = np.where(activations == 1)[0]
    print(neuron)
    print(identifier[max_val])

#x = pd.read_csv(dssp_csv, usecols=[feature])
#print(x)
#one_hot = pd.get_dummies(x)
#amino_acids = one_hot.columns
AA = list("ACDEFGHIKLMNPQRSTVWY")
SS = list("HBTS")
#print(x == AA[10])
#for neuron in range(1, 1025):
#    activations = pd.read_csv(encodings, usecols=[neuron])
#    ANOVA(x, activations, SS)
#neuron = 43
#activations = pd.read_csv(encodings, usecols=[neuron])
#ANOVA(x, activations, SS)


#for neuron in range(1, 2):
#    activations = pd.read_csv(encodings, usecols=[neuron])
#    ANOVA(x, activations)
#r, p = sci.pearsonr(labels, activation)
#print(r, p)
#print(labels)
#print(activation)


