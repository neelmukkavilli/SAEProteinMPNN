
import matplotlib.pyplot as plt

model = 'slow_exp2_se4_l3'

bar_heights = [0, 0, 12, 0, 1, 2, 0, 0, 55, 11, 65, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 26, 0, 9, 3, 0, 0, 0, 0]
bar_labels = ['Alpha Helix', 'Beta bridge', 'Beta strand', '3-10 helix', 'Pi helix', 'Turn', 'Bend', 'None',
            'AA chemistry', 'AA shape', 'Surface area', 'Phi angle', 'Psi angle', 'Bfactor',
            'xy', 'xz', 'yz', 'Hydrophobic', 'Hydrophilic', 'Salt bridge', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

bar_heights = [0, 12, 1, 2, 
               61, 0, 0, 0, 
               0, 0, 0, 
               0, 0, 5, 43, 1, 42, 13, 0, 0, 0, 0]
bar_labels = ['Alpha Helix',  'Beta strand', 'Pi helix', 'Turn',
            'Surface area', 'Phi angle', 'Psi angle', 'Bfactor',
            'Hydrophobic', 'Hydrophilic', 'Salt bridge', 
            '2.0 \u212B', '3.3 \u212B', '4.7 \u212B', '6.0 \u212B', '7.3 \u212B', '8.7 \u212B', '10.0 \u212B', '11.3 \u212B', '12.7 \u212B', '14.0 \u212B', '15.3 \u212B']

blue_4 = ['royalblue',  'cornflowerblue', 'lightskyblue', 'powderblue']
red_4 = ['firebrick', 'indianred'] + ['yellow']*2
yellow_3 = ['yellow']*3
#green_4 = ['darkgreen', 'seagreen', 'lightgreen', 'palegreen']
magenta_10 = ['blueviolet', 'indigo', 'rebeccapurple', 'purple', 'darkviolet', 'mediumorchid', 'orchid'] + ['yellow']*4
bar_colors = blue_4 + red_4 + yellow_3 + magenta_10

fig, ax = plt.subplots(figsize=(10, 6))

# Horizontal Bar Plot
ax.barh(bar_labels, bar_heights, color=bar_colors)

ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

ax.invert_yaxis()

# Add Plot Title
ax.set_title("Number of Dimensions Related to Features")

plt.savefig(f"{model} bar plot of feature correlations", bbox_inches='tight', dpi=300)
plt.show()
plt.clf()