import numpy as np
import pandas as pd
import csv
import scipy.stats as sci
import sklearn
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
#import polars as pl
from sklearn import metrics
import pickle

model = 'slow_exp2_se4_2'  # Change this to the desired model name

# Load feature data and encodings
dssp_path = '/WAVE/bio/ML/SAE_train/SAEProteinMPNN/evaluation/created_data/features/node_features.csv'
encodings_path = '/WAVE/bio/ML/SAE_train/SAEProteinMPNN/evaluation/created_data/encodings/output_' + model + '.pkl'

#encodings = pl.read_csv(encodings_path).to_pandas()
dfs = []
with open(encodings_path, "rb") as f:
    while True:
        try:
            dfs.append(pickle.load(f))
        except EOFError:
            break
    encodings = pd.concat(dfs, ignore_index=True)

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
#print(dssp.columns)

res_labels = encodings.iloc[:, 0]
encodings=encodings.iloc[:, 1:]
np.random.seed(0)
mask = np.random.rand(encodings.shape[0]) < 0.01
encodings = encodings.iloc[mask, :]
dssp = dssp.iloc[mask, :]
res_labels = res_labels.iloc[mask]
'''
# ROC AUC Thresh = 0.7
# Pearson Thresh = 0.5

# llb100_1e-2 (0)
# Uniprot
# Disulfide bond: 0, Active site: 1, Lipidation: 48, Propeptide: 1, Coiled coil: 4, Initiator methionine: 53, Peptide: 5, Signal: 1
# DSSP
# ASA: 2, 3: 1, 5: 3, 6: 1

# llb100_1e-2 (1)
# Uniprot
# Disulfide bond: 3, Active site: 1, Lipidation: 59, Propeptide: 1, Coiled coil: 11, Initiator methionine: 88, Peptide: 8, Signal: 5
# DSSP
# H: 1, ASA: 16, 3: 10, 5: 7, 6: 5 

# llb100_1e-2 (1)
# Uniprot
# Disulfide bond: 0, Active site: 1, Lipidation: 39, Propeptide: 0, Coiled coil: 11, Initiator methionine: 54, Peptide: 3, Signal: 2
# DSSP
# E: 1, I: 1, ASA: 13, 3: 7, 5: 7, 6: 7, 7: 3

# lsamples500_2 (2)
# Uniprot
# Beta strand: 26, Helix: 13, Disulfide bond: 17, Turn: 1, Glycosylation: 4, Transmembrane: 5, Zinc finger: 7, Coiled coil: 45, Initiator methionine: 295, Cross-link: 49, Peptide: 17, Signal: 13
# DSSP
# -: 9, E: 66, G: 5, H: 46, I: 4, S: 6, T: 43, function: 54, shape: 17, ASA: 63, phi: 11, psi: 2, 2: 1, 3: 13, 5: 8, 6: 2, 
'''
alphabet = list('ACDEFGHIKLMNPQRSTVWYX')  # Standard amino acids
# Anova test, not used in favor of F1 scores
def ANOVA(feature_act, encodings, aa, neuron):

    feauture_present = encodings[feature_act]
    feature_missing = encodings[feature_act==False]
    f_stat, p_val = sci.f_oneway(feauture_present, feature_missing)

    if p_val < 1e-4 and f_stat > 5:
        print(f"{f_stat},{neuron},{aa}")

# Pearson correlation test
def safe_pearsonr(dssp, encodings, feat, dim):
    x = dssp.to_numpy()
    y = encodings.to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0

    if np.std(x) == 0 or np.std(y) == 0:
        return 0

    r, p = sci.pearsonr(x, y)
    if r > 0.5 or r < -0.5:
        print(r, p, dim, feat)
        return 1
    else:
        return 0

def mi_class(dssp, encodings, feat, dim):
    x = dssp.to_numpy().reshape(-1, 1)
    y = encodings.to_numpy()

    #mask = np.isfinite(x) & np.isfinite(y)
    #x = x[mask]
    #y = y[mask]

    #if np.sum(x) == 0 or np.sum(y) == 0:
    #    return 0
    #if np.std(x) == 0 or np.std(y) == 0:
    #    return 0
    
    mi = mutual_info_classif(x, y)
    if mi > 0.3:
        print(mi, dim)
        return 1
    else:
        #print(mi, dim)
        return 0

def mi_regress(dssp, encodings, feat, dim):
    x = dssp.to_numpy()
    y = encodings.to_numpy().reshape(-1, 1)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    
    mi = mutual_info_regression(x, y)
    if mi > 0.3:
        print(feat, dim)
        return 1
    else:
        return 0    
# F1 scores
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

cat_dict = {}

# Categorize AA by function
cat_dict['function'] = {}
cat_dict['function']['phobic'] = list('HAVLMI')
cat_dict['function']['philic'] = list('STCPNQ')
cat_dict['function']['pos'] = list('KRH')
cat_dict['function']['neg'] = list('DE')
cat_dict['function']['aro'] = list('FYW')

# Categorize AA by shape
cat_dict['shape'] = {}
cat_dict['shape']['small'] = list('GA')
cat_dict['shape']['chain_2'] = list('SC')
cat_dict['shape']['branch_1'] = list('VTND')
cat_dict['shape']['branch_2'] = list('ILQE')
cat_dict['shape']['long'] = list('MKR')
cat_dict['shape']['ring_1'] = list('PHFY')
cat_dict['shape']['ring_2'] = list('W')

thresholds = [0.15, 0.50, 0.60, 0.80]

def eval_threshold(encodings_filtered, feature_act, feature, aa, score_csv=None, print_to_csv=True):
    for neuron in range(1, 1025):
            # Activation threshold to convert normalized values to binary
            activations = encodings_filtered.iloc[:, neuron]
            max = 0
            for thresh in thresholds:
                true_false = activations > thresh
                f1 = get_f1_scores(true_false, feature_act)
                if f1 > max:
                    max = f1
            #aps_score = average_precision_score(feature_act, activations)
            
            if print_to_csv:
                if feature == 'sec_struct':
                    score_csv.loc[str(neuron), 'ss' + aa] = max#aps_score
                else:
                    score_csv.loc[str(neuron), aa] = max#aps_score

def eval_roc_auc(activations, feature_act, feature, dim, thresh=0.7, reverse=False, print_always=False, recorddims=set()):
    # Ensure equal number of true postitives/negatives
    pos_idx = pd.Series(feature_act[feature_act == True].index.tolist())
    neg_idx = pd.Series(feature_act[feature_act == 0].index.tolist())
    n = min(len(pos_idx), len(neg_idx))
    #print(len(pos_idx), len(neg_idx))
    if n > 40:
        balanced_pos = pos_idx.sample(n, random_state=0).sort_values() # Randomly downsample
        balanced_neg = neg_idx.sample(n, random_state=0).sort_values()
        balanced_idx = balanced_pos.tolist() + balanced_neg.to_list()

        feature_act = feature_act.loc[balanced_idx]
        activations = activations.loc[balanced_idx]
        #print(feature_act[:10])
        #print(feature_act[-10:])
        #print(activations[:10])
        score = round(sklearn.metrics.roc_auc_score(feature_act, activations), 3)
        if score > thresh or score < (1-thresh):
            #print(score, dim, feature)
            if print_always:
                #print(len(pos_idx), len(neg_idx))
                return 1
                #print(score, dim, feature)
                #return 1
            else:
                return 1
                #recorddims.add(dim)
                #return 1
        else:
            return 0
            #print(score, dim, feature)
            #return 0
        #feature_act = 1 - feature_act
        #score = sklearn.metrics.roc_auc_score(feature_act, activations)
        #if score > thresh:
        #    count += 1#print(score, dim, feature, "reversed")
            #if print_always:
            #    print(score, dim, feature, "reversed")                
    else:
        print(n, feature, dim)
        return 0

def uniprot_cat(uniprot_features, n):
    column_feat = uniprot_features.iloc[:, 2*n] == 'True'
    return column_feat

#print(dssp['Chain val'][0] == True)
#print(pd.Series((dssp['Chain val'] == True).index.tolist()))
#print(dssp['Chain val'] == "True")
#print(sum(int(dssp[dssp['Chain val'] != "True"])))

#interesting_dims = set()

interesting_dims = [1, 16, 21, 22, 23, 536, 24, 35, 549, 43, 58, 59, 60, 574, 64, 
67, 595, 85, 90, 98, 610, 613, 619, 620, 115, 631, 121, 636, 637, 129, 641, 644, 
648, 137, 142, 661, 149, 156, 672, 165, 679, 684, 175, 689, 177, 183, 185, 697, 
705, 706, 195, 196, 197, 715, 719, 214, 727, 216, 731, 229, 230, 743, 246, 249, 
251, 768, 772, 261, 267, 781, 782, 277, 278, 792, 282, 285, 288, 800, 293, 295, 
809, 812, 305, 820, 824, 828, 835, 840, 330, 335, 848, 337, 343, 856, 345, 859, 
351, 882, 883, 884, 374, 890, 379, 383, 897, 905, 398, 405, 406, 918, 921, 922, 
413, 927, 934, 423, 937, 428, 433, 956, 961, 964, 453, 455, 971, 974, 466, 986, 
474, 479, 482, 504, 1002, 491, 490, 1008, 501, 1014, 1016, 505, 507, 508, 511]

features = ['Beta strand val', 'Helix val', 'Disulfide bond val', 'Turn val', 'Glycosylation val', 'Transmembrane val', 'Zinc finger val', 
            'Coiled coil val', 'Cross-link val', 'Peptide val', 'Signal val']

keep_these_dims = set()
ndims = encodings.shape[-1]
#for idx, feat in enumerate(dssp.columns):
#    if idx%2 == 1:
'''
for dim in sorted(interesting_dims):
    features_corr = [dim]
    for feat in features:
        feat_act = dssp[feat]
        activation = encodings.iloc[:, dim]
        features_corr.append(eval_roc_auc(activation, feat_act, feat, dim, print_always=True, recorddims=keep_these_dims))
    print([f for f in features_corr])
        #print(count)
        #print(keep_these_dims)
'''
ss_one_hot = pd.get_dummies(dssp['sec_struct'])
secondary_structures = ss_one_hot.columns

aa_one_hot = pd.get_dummies(dssp['residue'])

dims = [648, 637, 345, 482, 859, 937]
#for dim in dims:
#    activation = encodings.iloc[:, dim]
#    #for ss in secondary_structures:
#    #    print(eval_roc_auc(activation, ss_one_hot[ss], ss, dim), dim, ss+'_ss')
#    for aa in ['I', 'V', 'L', 'G', 'P']:#aa_one_hot.columns:
#        print(eval_roc_auc(activation, aa_one_hot[aa], aa, dim), dim, aa)

'''
testing_dims = [374]

y_true = yes_B = dssp['sec_struct'] == 'E'
y_score = encodings.iloc[:, 374]
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='example estimator')
display.plot()
plt.title("Dimension 375 AUC For Beta Strands")
plt.show()
plt.savefig("roc display B strand")

y_true = yes_B = dssp['sec_struct'] != 'H'
y_score = encodings.iloc[:, 374]
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='example estimator')
display.plot()
plt.title("Dimension 375 AUC For not Alpha Helix")
plt.show()
plt.savefig("roc display beta bridge")
'''
'''
for dim in testing_dims:
    activation = encodings.iloc[:, dim]
    yes_B = dssp['sec_struct'] == 'E'
    yes_B_act = activation[yes_B]
    no_B = 1 - yes_B
    no_B_act = activation[no_B]

    fig, axs = plt.subplots()

    data = [yes_B_act, no_B_act]
    label = ["Beta Strand", "Not in a Beta Strand"]
    axs.hist(data, density=True, label=label, bins=10)
    plt.legend()
    plt.title("Activations for Dimension 375")
    plt.ylabel("Frequency")
    plt.xlabel("Activation Value")
    plt.show()
    plt.savefig(f"histograms{dim}")

    fig, axs = plt.subplots()
    axs.boxplot(data, tick_labels=label, showfliers=False)
    plt.title("Activations for Dimension 375")
    plt.ylabel("Activation Value")
    plt.show()
    plt.savefig(f"boxplots{dim}")
'''
'''
## normalized_encodings_newlsamples500_2, thresh auc = +-0.5, r = +-0.7
#H, B, E, T = 3, 0, 15, 0 -> 0.00293, 0, 0.0146, 0
#Function, Shape = ,  -> 0., 
#ASA, phi, bfactor = , , 0 -> 

## normalized_encodings_dense, thresh auc = +- 0.5, r = +- 0.7
#H, B, E, T = 6, 0, 3, 0 -> 
#Function, Shape = 13, 21 -> 
#ASA, phi, bfactor = 0, 0, 0 -> 0 0, 0

## normalized_encodings_dense, thresh auc = +- 0.7, r = +-0.5
#H, B, E, T = -> 33, 3, 24, 15 -> 0.258, 0.0234, 0.1875, 0.117 
#Function, Shape =  2, 0 -> 0.156, 0 
#ASA, phi, bfactor =  6, 1, 0 -> 0.0469, 0.00781, 0 

## normalized_encodings_lsamples500_2, thresh auc = +-0.7, r = +-0.5
#-, B, E, G, H, I, S, T = 7, 6, 81, 9, 48, 0, 8, 44
#Function, Shape = 55, 11
#ASA, phi, psi, bfactor, xy, xz, yz = 65, 9, 9, 0, 0, 0, 0
#philic, phobic, SB = 0, 0, 0
#0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 = 0, 1, 7, 26, 0, 9, 3, 0, 0, 0, 0, 0, 0, 0
'''
#plt.clf()

n_dims = encodings.shape[-1]
secondary_structures = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
counter = 0
num_dims_significant = []
significant_features = []

#for dim in range(n_dims):
#    activation = encodings.iloc[:, dim]
#    counter_ss += mi_class(activation, ss_one_hot['H'], 'H', dim)
#    counter_aa += mi_class(activation, aa_one_hot['A'], 'A', dim)
#print(counter_ss)
#print(counter_aa)


'''
for ss in secondary_structures:
    for dim in range(n_dims):
        activation = encodings.iloc[:, dim]
        counter += eval_roc_auc(activation, ss_one_hot[ss], ss, dim)
        #counter += mi_class(activation, ss_one_hot[ss], ss, dim)
    print(ss)
    print(counter)
    significant_features.append(ss)
    num_dims_significant.append(counter)
    counter = 0
for group in cat_dict['function'].keys():
    feature_act = aa_one_hot[cat_dict['function'][group]].any(axis='columns').astype(int)
    for dim in range(n_dims):
        activation = encodings.iloc[:, dim]
        counter += eval_roc_auc(activation, feature_act, group, dim)
        #counter += mi_class(activation, feature_act, group, dim)
print("function")
print(counter)
significant_features.append('AA chemistry')
num_dims_significant.append(counter)
counter = 0
for group in cat_dict['shape'].keys():
    feature_act = aa_one_hot[cat_dict['shape'][group]].any(axis='columns').astype(int)
    for dim in range(n_dims):
        activation = encodings.iloc[:, dim]
        #counter += mi_class(activation, feature_act, group, dim)
        counter += eval_roc_auc(activation, feature_act, group, dim)
print("shape")
print(counter)
significant_features.append('AA shape')
num_dims_significant.append(counter)
counter = 0
for feat in dssp.columns[3:]:
    feature_act = dssp[feat]
    for dim in range(n_dims):
        activation = encodings.iloc[:, dim]
        counter += safe_pearsonr(activation, feature_act, feat, dim)
        #counter += mi_regress(activation, feature_act, feat, dim)
    print(feat)
    print(counter)
    significant_features.append(feat)
    num_dims_significant.append(counter)
    counter = 0
'''
feat = 'ASA'
feature_act = dssp[feat]
'''
for dim in range(n_dims):
    activation = encodings.iloc[:, dim]
    safe_pearsonr(activation, feature_act, feat, dim)
'''
dims = [252]
rscore = [0.7882527056]
for idx, dim in enumerate(dims):
    activation = encodings.iloc[:, dim]
    ax = plt.scatter(activation, feature_act)
    plt.xlabel('Sparse Activation')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.ylabel('ASA')
    plt.title(f'ASA correlation for Dimension {dim} with r = {round(rscore[idx], 3)}')#) {:.3f}'.format(rscore[idx]))
    #plt.annotate("r\u00B2 = {:.3f}".format(rscore[idx]), (0, 1))
    plt.savefig("scatterplot.png", dpi=500)
    plt.show()

#bar_heights = [7, 6, 81, 9, 48, 0, 8, 44, 55, 11, 65, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 26, 0, 9, 3, 0, 0, 0, 0]
#bar_labels = ['No Structure', 'Beta bridge', 'Beta strand', '3-10 helix', 'Alpha helix', 'Pi helix', 'Bend', 'Turn', 
#            'AA chemistry', 'AA shape', 'Surface area', 'Phi angle', 'Psi angle', 'Bfactor',
#            'xy', 'xz', 'yz', 'Hydrophobic', 'Hydrophilic', 'Salt bridge', 
#            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
'''
bar_heights = num_dims_significant
bar_labels = significant_features

blue_8 = ['mediumblue', 'royalblue', 'dodgerblue', 'cornflowerblue', 'deepskyblue', 'lightskyblue', 'skyblue', 'powderblue']
red_2 = ['firebrick', 'indianred']
green_4 = ['darkgreen', 'seagreen', 'lightgreen', 'palegreen']
yellow_6 = ['yellow']*6
magenta_7 = ['blueviolet', 'indigo', 'rebeccapurple', 'purple', 'darkviolet', 'mediumorchid', 'orchid']
slate_4 = ['blue']*4
bar_colors = blue_8 + red_2 + green_4 + yellow_6 + magenta_7 + slate_4

fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal Bar Plot
ax.barh(bar_labels, bar_heights, color=bar_colors)

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

ax.invert_yaxis()

# Add Plot Title
ax.set_title("Number of Dimensions Related to Features")

plt.savefig(f"{model} bar plot of feature correlations", bbox_inches='tight')
plt.show()
plt.clf()
'''

'''
for dim_of_interest in range(373, 376):
    print(dim_of_interest)
    #print(np.argpartition(encodings.iloc[:, dim_of_interest].to_numpy(), -100))
    top_k_ind = encodings.iloc[:, dim_of_interest].nlargest(25).index.to_numpy()
    #print(top_k_ind)
    top_identifiers = np.array(res_labels[top_k_ind])
    #print(top_identifiers)
    dim_375_feats = dssp.loc[top_k_ind]
    #print(dim_375_feats)
    #print(dim_375_feats['sec_struct'])

    yes_B = np.sum(dssp['sec_struct'] == 'E')
    no_B = np.sum(dssp['sec_struct'] != 'E')
    print(yes_B)
    yes375_B = np.sum(dim_375_feats['sec_struct'] == 'E')
    no375_B = np.sum(dim_375_feats['sec_struct'] != 'E')
    print(yes375_B)
    
    print(yes_B/(yes_B + no_B))
    print(yes375_B/(yes375_B + no375_B))
'''
def score_cat(dssp_filtered, feature, encodings_filtered, sector, score_csv):
    # One hot encoding of features
    one_hot = pd.get_dummies(dssp_filtered[feature])
    amino_acids = one_hot.columns
    if sector != False:
        for cat_key in cat_dict[sector].keys():
            print(cat_key)
            feature_act = one_hot.loc[:, cat_dict[sector][cat_key]].any(axis='columns')
            eval_roc_auc(encodings_filtered, feature_act, feature, cat_key, score_csv)
    else:
        for aa in amino_acids:
            print(aa)
            feature_act = one_hot[aa]
            eval_roc_auc(encodings_filtered, feature_act, feature, aa, score_csv)
