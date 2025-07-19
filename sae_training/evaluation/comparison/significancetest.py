import numpy as np
import pandas as pd
import csv
import scipy.stats as sci
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

model = 'lsamples500_2'

# Load feature data and encodings
dssp_path = '/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/dssp_summary.csv'
encodings_path = '/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/encodings/normalized_encodings_' + model + '.csv'

features = ['identifier', 'residue','sec_struct','ASA','phi','psi','NO1','ON1','NO2','ON2']

encodings = pd.read_csv(encodings_path)
dssp = pd.read_csv(dssp_path)

# Filter data to make sure there are entries for both sets of data and remove duplicates
encodings = encodings.drop_duplicates(subset='identifier')
dssp = dssp.drop_duplicates(subset='identifier')

encodings['identifier'] = encodings['identifier'].astype(str).str.strip().str.lower()
dssp['identifier'] = dssp['identifier'].astype(str).str.strip().str.lower()

matching_ids = set(encodings['identifier']) & set(dssp['identifier'])

encodings_filtered = encodings[encodings['identifier'].isin(matching_ids)]
encodings_filtered = encodings_filtered.sort_values('identifier').reset_index(drop=True)

dssp_filtered = dssp[dssp['identifier'].isin(matching_ids)]
dssp_filtered = dssp_filtered.sort_values('identifier').reset_index(drop=True)

# Anova test, not used in favor of F1 scores
def ANOVA(feature_act, encodings, aa, neuron):

    feauture_present = encodings[feature_act]
    feature_missing = encodings[feature_act==False]
    f_stat, p_val = sci.f_oneway(feauture_present, feature_missing)

    if p_val < 1e-4 and f_stat > 5:
        print(f"{f_stat},{neuron},{aa}")

# Pearson correlation test
def safe_pearsonr(dssp, encodings, neuron, feature):

    # Filters out NaN and infinite values

    # Feature Amplitude
    x = pd.to_numeric(dssp[feature], errors='coerce').to_numpy()
    # Model Activations
    y = pd.to_numeric(encodings.iloc[:, neuron], errors='coerce').to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0, 0

    if np.std(x) == 0 or np.std(y) == 0:
        return 0, 0

    return sci.pearsonr(x, y)
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

# Usage

# Used for categorical variables (residues and secondary structure)
# Threshold F1 score of 0.5

#amino_acids = list('ACDEFGHIKLMNPQRSTVWXY')
#SS = list('HBEGITS-')
#SS = ['ssH', 'ssB', 'ssE', 'ssG', 'ssI', 'ssT', 'ssS', 'ss-']
#feature_cols = ['neuron'] + amino_acids + SS + features[3:]

scores = pd.DataFrame([])

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

def score_cat(dssp_filtered, feature, encodings_filtered, sector, score_csv):
    # One hot encoding of features
    one_hot = pd.get_dummies(dssp_filtered[feature])
    amino_acids = one_hot.columns
    if sector != False:
        for cat_key in cat_dict[sector].keys():
            print(cat_key)
            feature_act = one_hot.loc[:, cat_dict[sector][cat_key]].any(axis='columns')
            eval_threshold(encodings_filtered, feature_act, feature, cat_key, score_csv)
    else:
        for aa in amino_acids:
            print(aa)
            feature_act = one_hot[aa]
            eval_threshold(encodings_filtered, feature_act, feature, aa, score_csv)


#aps_scores = {}
#mat_scores = {}
improvedf1_scores = {}
'''
one_hot = pd.get_dummies(dssp_filtered['residue'])
amino_acids = one_hot.columns
for cat_key in cat_dict['function'].keys():
    print(cat_key)
    feature_act = one_hot.loc[:, cat_dict['function'][cat_key]].any(axis='columns')
    aps_scores[cat_key] = []
    #mat_scores[cat_key] = []
    improvedf1_scores[cat_key] = []
    eval_threshold(encodings_filtered, feature_act, 'residue', cat_key)
    for neuron in range(1, 1025):
        activations = encodings_filtered.iloc[:, neuron]
        aps_scores[cat_key].append(average_precision_score(feature_act, activations))
        #mat_scores[cat_key].append(matthews_corrcoef(feature_act, activations))
'''


'''
# APS scores
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = list(range(1, 1025))
for cat_key in cat_dict['function'].keys():
    ax1.scatter(x, aps_scores[cat_key], label=cat_key)
plt.legend()
plt.title(model + ' APS scores')
plt.show()

# mat scores
'''
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = list(range(1, 1025))
for cat_key in cat_dict['function'].keys():
    ax1.scatter(x, mat_scores[cat_key], label=cat_key)
plt.legend()
plt.title(model + ' Mat scores')
plt.show()
'''
'''
# Improved f1 scores
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = list(range(1, 1025))
for cat_key in cat_dict['function'].keys():
    ax1.scatter(x, improvedf1_scores[cat_key], label=cat_key)
plt.title(model + ' Improved F1 scores')
plt.legend()
plt.show()
'''

score_cat(dssp_filtered, 'residue', encodings_filtered, 'function', scores)
score_cat(dssp_filtered, 'residue', encodings_filtered, 'shape', scores)
score_cat(dssp_filtered, 'residue', encodings_filtered, False, scores)
score_cat(dssp_filtered, 'sec_struct', encodings_filtered, False, scores)

for i in range(3, 10):
    feature = features[i]
    print(feature)
    for neuron in range(1, 1025):
        r, p = safe_pearsonr(dssp_filtered, encodings_filtered, neuron, feature)
        scores.loc[str(neuron), feature] = r
        if p < 1e-4 and (r > 0.8 or r < -0.8):
            print(f"r value of {r}, p value of {p} at neuron {neuron} with feature {feature}")

scores.to_csv('scores_' + model + '.csv')



'''
for i in range(3, 10):
    feature = features[i]
    print(feature)
    for neuron in range(1, 1025):
        r, p = safe_pearsonr(dssp_filtered, encodings_filtered, neuron, feature)
        scores.loc[str(neuron), feature] = r
        if p < 1e-4 and (r > 0.8 or r < -0.8):
            print(f"r value of {r}, p value of {p} at neuron {neuron} with feature {feature}")
scores.to_csv('scores_' + model + '.csv')
'''


'''
for i in range(1, 3):
    feature = features[i]
    # One hot encoding of features
    one_hot = pd.get_dummies(dssp_filtered[feature])
    amino_acids = one_hot.columns
    for aa in amino_acids:
        feature_act = one_hot[aa]
        for neuron in range(1, 1025):
            # Activation threshold to convert normalized values to binary
            activations = encodings_filtered.iloc[:, neuron]
            thresh_dict = {}
            thresh_dict[0] = activations > 0
            thresh_dict[15] = activations > 0.15
            thresh_dict[50] = activations > 0.5
            thresh_dict[60] = activations > 0.60
            thresh_dict[80] = activations > 0.80
            max = 0
            for key in thresh_dict.keys():
                f1 = get_f1_scores(thresh_dict[key], feature_act)
                if f1 > max:
                    max = f1
                if f1 > 0.5:
                    print(feature, f1, key, neuron, aa)
            if feature == 'sec_struct':
                scores.loc[str(neuron), 'ss' + aa] = max
            else:
                scores.loc[str(neuron), aa] = max

# Used for quantitative variables
# Threshold p val of 1e-4 and r value less than -0.8 or above 0.8

for i in range(3, 10):
    feature = features[i]
    print(feature)
    for neuron in range(1, 1025):
        r, p = safe_pearsonr(dssp_filtered, encodings_filtered, neuron, feature)
        scores.loc[str(neuron), feature] = r
        if p < 1e-4 and (r > 0.8 or r < -0.8):
            print(f"r value of {r}, p value of {p} at neuron {neuron} with feature {feature}")
print("done")
'''
'''
for neuron in [5, 141, 216, 276, 559, 993]:
    activations = pd.read_csv(encodings, usecols=[neuron]).to_numpy()[:,0]
    identifier = pd.read_csv(encodings, usecols=[0]).to_numpy()[:,0]
    max_val = np.where(activations == 1)[0]
    print(neuron)
    print(identifier[max_val])
'''