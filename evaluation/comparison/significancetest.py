import numpy as np
import pandas as pd
import scipy.stats as sci
import sklearn
import matplotlib.pyplot as plt
import pickle

def load_data(SAE_type, model, layer, mask_lvl):
    # Load feature data and encodings
    dssp_path = '../created_data/features/node_features.csv'
    encodings_path = '../created_data/encodings/' + SAE_type + '_' + model + '/output_' + model + '_' + str(layer) + '.pkl'

    # Read and format pkl files
    dfs = []
    with open(encodings_path, "rb") as f:
        while True:
            try:
                dfs.append(pickle.load(f))
            except EOFError:
                break
        encodings = pd.concat(dfs, ignore_index=True)
    dfs = []
    np.random.seed(0)
    mask = np.random.rand(encodings.shape[0]) < mask_lvl
    encodings = encodings.iloc[mask, :]
    print('read encodings', encodings.shape)
    dssp = pd.read_csv(dssp_path)
    print("read data")
    # Filter data to make sure there are entries for both sets of data and remove duplicates
    encodings = encodings.drop_duplicates(subset='identifier')
    dssp = dssp.drop_duplicates(subset='identifier')
    encodings = encodings.dropna() # Drop rows with NA values
    dssp = dssp.dropna()
    encodings['identifier'] = encodings['identifier'].astype(str).str.strip().str.lower()
    dssp['identifier'] = dssp['identifier'].astype(str).str.strip().str.lower()
    matching_ids = set(encodings['identifier']) & set(dssp['identifier'])
    encodings = encodings[encodings['identifier'].isin(matching_ids)]
    encodings = encodings.sort_values('identifier').reset_index(drop=True)
    dssp = dssp[dssp['identifier'].isin(matching_ids)]
    dssp = dssp.sort_values('identifier').reset_index(drop=True)
    print("filtered data")

    # Isolate sample labels and set up mask (default 10% of samples -> ~43,000 samples)
    res_labels = encodings.iloc[:, 0]
    encodings = encodings.iloc[:, 1:]
    print(f'{res_labels.shape[0]} samples')
    
    print(f'{res_labels.shape[0]} samples')
    n_dims = encodings.shape[1]
    print(f'{n_dims} dimensions')

    return encodings, dssp, res_labels, n_dims
# ROC AUC Thresh = 0.7
# Pearson Thresh = 0.5

alphabet = list('ACDEFGHIKLMNPQRSTVWY')  # Standard amino acids
# Anova test, not used in favor of ROC AUC
def ANOVA(feature_act, encodings, aa, neuron):

    feauture_present = encodings[feature_act]
    feature_missing = encodings[feature_act==False]
    f_stat, p_val = sci.f_oneway(feauture_present, feature_missing)

    if p_val < 1e-4 and f_stat > 5:
        print(f"{f_stat},{neuron},{aa}")

# Mutual information, not used in favor of ROC AUC
def mi_class(dssp, encodings, feat, dim):
    y = dssp.to_numpy()
    x = encodings.to_numpy().reshape(-1, 1)

    #mask = np.isfinite(x) & np.isfinite(y)
    #x = x[mask]
    #y = y[mask]

    #if np.sum(x) == 0 or np.sum(y) == 0:
    #    return 0
    #if np.std(x) == 0 or np.std(y) == 0:
    #    return 0
    
    mi = mutual_info_classif(x, y)
    if mi > 0.2:
        print(mi, dim)
        return 1
    else:
        #print(mi, dim)
        return 0

# F1 scores, not used in favor of ROC AUC
def get_f1_scores(thresh, feature_act):

    true_pos = (feature_act == True).sum()
    false_pos = (feature_act == False).sum()
    false_neg = 0
    default_f1 = true_pos/((false_pos + false_neg)* 0.5 + true_pos)

    true_pos = ((thresh == True) & (feature_act == True)).sum()
    false_pos = ((thresh == True) & (feature_act == False)).sum()
    false_neg = ((thresh == False) & (feature_act == True)).sum()

    f1 = true_pos/((false_pos + false_neg)* 0.5 + true_pos)

    return f1 #(f1 - default_f1)/(1 - default_f1)

# Mutual information, not used in favor of Pearson
def mi_regress(dssp, encodings, feat, dim):
    y = dssp.to_numpy()
    x = encodings.to_numpy().reshape(-1, 1)

    #mask = np.isfinite(x) & np.isfinite(y)
    #x = x[mask]
    #y = y[mask]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0
    #if np.std(x) == 0 or np.std(y) == 0:
        return 0
    
    mi = mutual_info_regression(x, y)
    if mi > 0.2:
        print(mi, feat, dim)
        return 1
    else:
        return 0    

# Pearson correlation test
def safe_pearsonr(encodings, features, feature_name, dim, n_dims, thresh=0.5):
    activation = encodings.iloc[:, dim].to_numpy()
    features = features.to_numpy()
    mask = np.isfinite(features) & np.isfinite(activation)

    features = features[mask]
    activation = activation[mask]

    if np.std(features) == 0 or np.std(activation) == 0:
        return 0

    r, p = sci.pearsonr(features, activation)
    if (r > thresh or r < -1*thresh) and p < (0.05/n_dims):
        return 1
    else:
        return 0

def graph_roc_auc():
    fpr, tpr, _ = sklearn.metrics.roc_curve(feature_act, activations, dim, model, feat)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # Plot ROC
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC AUC Curve for Presence of {feat}: Dim {dim}')
    plt.legend(loc='upper left')
    plt.savefig(f'{model}_roc_auc_{dim}.png')
    plt.show()

def roc_auc(encodings, features, feature_name, dim, thresh=0.7, graph=False):
    activations = encodings.iloc[:, dim]

    # Ensure equal number of true postitives/negatives
    pos_idx = pd.Series(features[features == True].index.tolist())
    neg_idx = pd.Series(features[features == False].index.tolist())
    n = min(len(pos_idx), len(neg_idx))
    #print(len(pos_idx), len(neg_idx))
    if n > 40:
        balanced_pos = pos_idx.sample(n, random_state=0).sort_values() # Randomly downsample
        balanced_neg = neg_idx.sample(n, random_state=0).sort_values()
        balanced_idx = balanced_pos.tolist() + balanced_neg.to_list()

        features = features.loc[balanced_idx]
        activations = activations.loc[balanced_idx]
        score = round(sklearn.metrics.roc_auc_score(features, activations), 3)
        if score > thresh or score < (1-thresh):
            #print(feature_name, dim, score)
            return 1
        else:
            return 0
        if graph == True:
            graph_roc_auc(features, activations, dim, model, feature_name)
    else:
        return 0

# Add ['Function'] and ['Shape'] columns to feature_df
def categorize_AA(feature_df):
    # By function
    function_dict = {
        'phobic': list('GAVILM'),
        'philic': list('CPSTNQ'),
        'pos': list('KRH'),
        'neg': list('DE'),
        'aro': list('FYW')
    }

    # By polarity (Trinquier and Sanejouand 1998)
    pol_dict = {
        'phobic': list('WCMIFLV'),
        'mid': list('GRSTAP'),
        'philic': list('EDKNQHY')
    } 

    # By shape
    shape_dict = {
        'small': list('GA'),
        'sbranch': list('SCVT'),
        'branch1': list('DNIL'),
        'branch2': list('EQ'),
        'long': list('KM'),
        'lbranch': list('R'),
        'ring1': list('PHFY'),
        'ring2': list('W')
    }

    # By volume
    # Cateogrized by Van der Waals volume of side chain (Darby and Creighton 1993)
    vol_dict = {
        'vsmall': list('G'), # 48
        'small': list('A'), # 67
        'larger': list('S'), # 73
        'smedium': list('CPDTN'), # 86-96
        'medium': list('VEQH'), # 105-118
        'lmedium': list('ILM'), # 124
        'large': list('KFYR'), # 135-148
        'vlarge': list('W') # 163
    }
    # Mapping residues
    residue_to_function = {
        residue: category
        for category, residues in function_dict.items()
        for residue in residues
    }

    residue_to_polarity = {
        residue: category
        for category, residues in pol_dict.items()
        for residue in residues
    }

    # Mapping for residue to shape
    residue_to_shape = {
        residue: category
        for category, residues in shape_dict.items()
        for residue in residues
    }

    residue_to_vol = {
        residue: category
        for category, residues in vol_dict.items()
        for residue in residues
    }
    feature_df.insert(loc = 3, column = 'Function', value = feature_df['residue'].map(residue_to_function))
    feature_df.insert(loc = 4, column = 'Polarity', value = feature_df['residue'].map(residue_to_polarity))
    feature_df.insert(loc = 5, column = 'Shape', value = feature_df['residue'].map(residue_to_shape))
    feature_df.insert(loc = 6, column = 'Volume', value = feature_df['residue'].map(residue_to_vol))
    #feature_df['Function'] = feature_df['residue'].map(residue_to_function)
    #feature_df['Shape'] = feature_df['residue'].map(residue_to_shape)

def count_categorical(features, base_feature, n_dims, encodings, bar_labels, bar_heights, sum=True):
    one_hot = pd.get_dummies(features[base_feature])
    counter, prev_total = 0, 0
    for feat in one_hot.columns:
        for dim in range(n_dims):
            counter += roc_auc(encodings, one_hot[feat], feat, dim)
        #print(feat)
        #print(counter - prev_total)
        if sum != True:
            bar_labels.append(feat)
            bar_heights.append(counter)
            counter = 0
        else:
            prev_total = counter
    if sum:
        #print(base_feature)
        #print(counter)
        bar_labels.append(base_feature)
        bar_heights.append(counter)

def count_sig_dims(n_dims, encodings, features):

    bar_labels = []
    bar_heights = []
    count_categorical(features, 'sec_struct', n_dims, encodings, bar_labels, bar_heights, sum=False)
    count_categorical(features, 'Function', n_dims, encodings, bar_labels, bar_heights)
    count_categorical(features, 'Polarity', n_dims, encodings, bar_labels, bar_heights)
    count_categorical(features, 'Shape', n_dims, encodings, bar_labels, bar_heights)
    count_categorical(features, 'Volume', n_dims, encodings, bar_labels, bar_heights)

    for feat in ['ASA', 'phi', 'psi', '0', '1', '2', '3', '4', '5', '6']:
        counter = 0
        for dim in range(n_dims):
            counter += safe_pearsonr(encodings, features[feat], feat, dim, n_dims, thresh=0.5)
        print(feat)
        print(counter)
        bar_labels.append(feat)
        bar_heights.append(counter)

    return bar_labels, bar_heights


SAE_type = "node"
model = "log17_exp2"
layer = 0
mask_level = 0.1
encodings, dssp, res_labels, n_dims = load_data(SAE_type, model, layer, mask_level)
categorize_AA(dssp)
bar_labels, bar_heights = count_sig_dims(n_dims, encodings, dssp)

print(model, bar_heights)
