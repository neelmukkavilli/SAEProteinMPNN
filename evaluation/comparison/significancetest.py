import numpy as np
import pandas as pd
import csv
import scipy.stats as sci
import sklearn
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import matthews_corrcoef
#from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle

def load_data(SAE_type, model, layer, mask_lvl):
    # Load feature data and encodings
    dssp_path = '/home/neelm/SAEProteinMPNN/evaluation/created_data/features/node_features.csv'
    encodings_path = '/home/neelm/SAEProteinMPNN/evaluation/created_data/encodings/' + SAE_type + '_' + model + '/output_' + model + '_' + str(layer) + '.pkl'

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
def safe_pearsonr(encodings, features, feature_name, dim, thresh=0.5):
    activation = encodings.iloc[:, dim].to_numpy()
    features = features.to_numpy()
    mask = np.isfinite(features) & np.isfinite(activation)

    features = features[mask]
    activation = activation[mask]

    if np.std(features) == 0 or np.std(activation) == 0:
        return 0

    r, p = sci.pearsonr(features, activation)
    if r > thresh or r < -1*thresh:
        #print(r, p<(0.05/128), dim, feature_name)
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
            counter += safe_pearsonr(encodings, features[feat], feat, dim, thresh=0.5)
        print(feat)
        print(counter)
        bar_labels.append(feat)
        bar_heights.append(counter)

    return bar_labels, bar_heights


#SAE_type = "node"
#model = "log18_exp4"
#layer = 2
#mask_level = 0.1
#encodings, dssp, res_labels, n_dims = load_data(SAE_type, model, layer, mask_level)
#categorize_AA(dssp)
#bar_labels, bar_heights = count_sig_dims(n_dims, encodings, dssp)

def clean_data(heights_list, labels, models=['No SAE', 'SAE128', 'SAE256', 'SAE512', 'SAE1024']):
    rename = {
            'H': 'Alpha helix',
            'E': 'Beta strand',
            'B': 'Beta bridge',
            'G': '3-10 helix',
            'I': 'Pi helix',
            'T': 'Turn',
            'S': 'Bend',
            '-': 'None',
            'Function': 'Function',
            'Polarity': 'Polarity',
            'Shape': 'Shape',
            'Volume': 'Volume',
            'ASA': 'SASA',
            'phi': 'Phi angle',
            'psi': 'Psi angle',
            '0': '2.0 \u212B',
            '1': '3.3 \u212B',
            '2': '4.7 \u212B',
            '3': '6.0 \u212B',
            '4': '7.3 \u212B',
            '5': '8.7 \u212B',
            '6': '10.0 \u212B',
        }
    data = {}
    dims = [128, 128, 256, 512, 1024]
    for i, heights in enumerate(heights_list):
        height_dict = dict(zip(labels, heights))
        new_labels = []
        new_heights = []
        for label, new_label in rename.items():
            if label in height_dict:
                new_labels.append(new_label)
                new_heights.append(height_dict[label] * 100 / (dims[i])) 
        data[models[i]] = new_heights
    return data, new_labels

def clean_bar_graph(labels, data, layer, allmodels=True):
    #plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    fig, ax = plt.subplots(figsize=(9, 6))
    w, x = 0.15, np.arange(len(labels))
    for i, heights in enumerate(data.values()):
        print(i, heights)
        plt.barh((x - 2*w + i*w), heights, height=w, label=list(data.keys())[i], tick_label=labels)#(x - 2*w + i*w), width=w, height-heights, width=w)#, label=list(data.keys())[i])
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    ax.invert_yaxis()
    #ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    if allmodels:
        ax.set_xlabel('Percentage of Neurons Correlated')
        ax.set_title(f"Correlated Dimensions for Layer {layer} of SAE models")
        plt.savefig(f"all models layer {layer} bar plot of feature correlations.png", bbox_inches='tight', dpi=500)
    plt.show()
    plt.clf()
def old_bar_graph(labels, heights, allmodels=True):
    rename = {
        'H': 'Alpha helix',
        'E': 'Beta strand',
        'B': 'Beta bridge',
        'G': '3-10 helix',
        'I': 'Pi helix',
        'T': 'Turn',
        'S': 'Bend',
        '-': 'None',
        'Function': 'Function',
        'Polarity': 'Polarity',
        'Shape': 'Shape',
        'Volume': 'Volume',
        'ASA': 'Exposed area',
        'phi': 'Phi angle',
        'psi': 'Psi angle',
        '0': '2.0 \u212B',
        '1': '3.3 \u212B',
        '2': '4.7 \u212B',
        '3': '6.0 \u212B',
        '4': '7.3 \u212B',
        '5': '8.7 \u212B',
        '6': '10.0 \u212B',
    }




    height_dict = dict(zip(labels, heights))
    new_labels = []
    new_heights = []

    for label, new_label in rename.items():
        if label in height_dict:
            new_labels.append(new_label)
            new_heights.append(height_dict[label])

    #blues = plt.cm.Blues(np.linspace(0.9, 0.3, 8))
    #greens = plt.cm.Greens(np.linspace(0.9, 0.3, 4))
    #reds = plt.cm.Reds(np.linspace(0.9, 0.3, 3))
    #purples = plt.cm.Purples(np.linspace(0.9, 0.3, 7))
    #bar_colors = np.concatenate((blues, greens, reds, purples))
    
    
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(new_labels, new_heights, color=bar_colors)
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    ax.set_xlabel('Number of Neurons Correlated')
    ax.invert_yaxis()
    ax.set_title(f"SAE with {exp_size*128} Neurons: Layer {layer+1}")
    plt.savefig(f"{model} layer {layer+1} bar plot of feature correlations.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

'''
data_dict = {}
SAE_type = 'node'
models = ["dense", "log20_exp1", "log17_exp2", "log18_exp4", "log19_exp8"]
models = ["log19_exp8"]
#exp_size = 1
layer = 1 # 0-indexed
mask_lvl = 0.1

for model in models:
    encodings, dssp, res_labels, n_dims = load_data(SAE_type, model, layer, mask_lvl)
    categorize_AA(dssp)
    bar_labels, bar_heights = count_sig_dims(n_dims, encodings, dssp)
    print(model)
    print(bar_labels)
    print(bar_heights)
    #dict[model] = bar_heights
'''
# Layer 1
labels = ['-', 'B', 'E', 'G', 'H', 'I', 'S', 'T', 'Function', 'Polarity', 'Shape', 'Volume', 'ASA', 'phi', 'psi', '0', '1', '2', '3', '4', '5', '6']
dense_l1 = [0, 5, 12, 0, 0, 0, 0, 3, 0, 3, 1, 1, 52, 0, 0, 0, 0, 1, 35, 0, 31, 7]
log20_exp1_l1 = [23, 7, 17, 13, 83, 0, 32, 20, 0, 0, 1, 64, 0, 10, 10, 0, 1, 0, 0, 0, 0, 0]
log17_exp2_l1 = [41, 14, 39, 21, 143, 0, 54, 49, 0, 0, 1, 143, 0, 10, 6, 0, 5, 0, 0, 0, 0, 0]
log18_exp4_l1 = [75, 15, 50, 27, 224, 0, 103, 99, 0, 0, 0, 252, 1, 12, 5, 0, 4, 1, 0, 1, 0, 0]
log19_exp8_l1 = [139, 20, 60, 47, 409, 0, 196, 185, 0, 0, 7, 473, 0, 14, 8, 0, 17, 1, 0, 0, 0, 0]

heights_list_l1 = [dense_l1, log20_exp1_l1, log17_exp2_l1, log18_exp4_l1, log19_exp8_l1]

# Layer 2
dense_l2 = [0, 6, 12, 0, 0, 0, 0, 5, 0, 2, 2, 2, 54, 0, 0, 0, 0, 2, 41, 0, 37, 5]
log20_exp1_l2 = [9, 5, 14, 6, 56, 0, 6, 7, 0, 1, 2, 47, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0]
log17_exp2_l2 = [15, 7, 12, 10, 60, 0, 6, 12, 1, 2, 6, 46, 2, 2, 1, 0, 2, 1, 0, 0, 1, 0]
log18_exp4_l2 = [25, 14, 20, 19, 136, 0, 11, 23, 1, 0, 10, 95, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0]
log19_exp8_l2 = [76, 27, 14, 20, 251, 35, 35, 27, 0, 0, 3, 142, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0]

heights_list_l2 = [dense_l2, log20_exp1_l2, log17_exp2_l2, log18_exp4_l2, log19_exp8_l2]

# Layer 3
dense_l3 = [0, 8, 7, 0, 0, 0, 0, 6, 0, 3, 2, 2, 60, 0, 0, 0, 0, 4, 38, 0, 36, 6]
log20_exp1_l3 = [4, 0, 2, 4, 9, 0, 0, 2, 9, 5, 12, 30, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
log17_exp2_l3 = [2, 2, 5, 6, 12, 0, 0, 2, 10, 3, 16, 50, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
log18_exp4_l3 = [6, 3, 12, 5, 30, 0, 0, 7, 13, 2, 23, 82, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
log19_exp8_l3 = [25, 47, 7, 3, 69, 16, 3, 2, 6, 2, 30, 178, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

heights_list_l3 = [dense_l3, log20_exp1_l3, log17_exp2_l3, log18_exp4_l3, log19_exp8_l3]

cleaned_data, cleaned_labels = clean_data(heights_list_l3, labels)
clean_bar_graph(cleaned_labels, cleaned_data, 3)
#clean_bar_graph(bar_labels, bar_heights)

'''
features = dssp['3']
for dim in [11, 13, 22, 78, 84, 95]:
    activation = encodings.iloc[:, dim]
    fig, ax = plt.subplots()
    ax = plt.scatter(activation, features)
    plt.xlabel('Sparse Activation')
    plt.ylabel(f'{feat}')
    plt.title(f'{dim}')
    plt.annotate("r\u00B2 = {:.3f}".format(rscore[idx]), (0, 1))
    plt.savefig(f"{dim}_{feat}_{model}scatterplot.png", dpi=500)
    plt.show()
'''
