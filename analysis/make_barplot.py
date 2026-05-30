
import matplotlib.pyplot as plt
import numpy as np
model = 'log17_node_exp2_100e_0'
'''
bar_heights = [14, 2, 5, 3, 6, 4, 1, 3, 
               12, 13, 1, 0, 0, 
               0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0]
bar_labels = ['Alpha Helix', 'Beta bridge', 'Beta strand', '3-10 helix', 'Pi helix', 'Turn', 'Bend', 'None',
            'AA chemistry', 'AA shape', 'Surface area', 'Phi angle', 'Psi angle', 'Bfactor',
            'xy', 'xz', 'yz', 'Hydrophobic', 'Hydrophilic', 'Salt bridge', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
'''
# Layer 3
bar_heights = [14, 2, 5, 3, 6, 4, 1, 3, 
               12, 13, 1, 0, 0, 
               0, 0, 0,
               0, 0, 0, 0, 0, 0, 0]
# Layer 2
bar_heights = [65, 4, 12, 7, 17, 13, 7, 15, 
               0, 2, 2, 3, 1, 
               0, 0, 0, 
               0, 2, 1, 0, 0, 0, 0]
# Layer 1
bar_heights = [148, 7, 35, 17, 27, 50, 47, 51, 
               0, 0, 0, 9, 7,
               0, 0, 0, 
               0, 2, 0, 0, 0, 0, 0]
total_feats = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-', 
 'AA chemistry', 'AA shape', 'ASA', 'phi', 'psi', 'bfactor', 
 'xy', 'xz', 'yz', 'philic', 'phobic', 'SB', 
 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
bar_labels = ['Alpha Helix', 'Beta bridge', 'Beta strand', '3-10 helix', 'Pi helix', 'Turn', 'Bend', 'None',
            'AA Chemistry', 'AA Shape', 'Surface area', 'Phi angle', 'Psi angle',
            'Hydrophobic', 'Hydrophilic', 'Salt bridge', 
            '2.0 \u212B', '3.3 \u212B', '4.7 \u212B', '6.0 \u212B', '7.3 \u212B', '8.7 \u212B', '10.0 \u212B']#, '11.3 \u212B', '12.7 \u212B', '14.0 \u212B', '15.3 \u212B']

blues = plt.cm.Blues(np.linspace(0.9, 0.3, 8))
reds = plt.cm.Reds(np.linspace(0.9, 0.3, 5))
greens = plt.cm.Greens(np.linspace(0.9, 0.3, 3))
purples = plt.cm.Purples(np.linspace(0.9, 0.3, 7))

bar_colors = np.concatenate((blues, reds, greens, purples))

fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal Bar Plot
ax.barh(bar_labels, bar_heights, color=bar_colors)

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

ax.invert_yaxis()

# Add Plot Title
ax.set_title("Number of Dimensions Related to Features: Layer 1")

plt.savefig(f"{model} bar plot of feature correlations", bbox_inches='tight', dpi=300)
plt.show()
plt.clf()