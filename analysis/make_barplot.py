import matplotlib.pyplot as plt
import numpy as np

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