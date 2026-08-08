import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Dict to make creating plots easier
values = {
    'scfv_baseline': ('Baseline', ['scfv_native', 'scfv_evo', 'scfv_mpnn']),
    'scfv_sae': ('Candidate', ['scfv_mod0', 'scfv_mod05', 'scfv_mod1', 'scfv_mod2']),
    'scfv_native': 'Native',
    'scfv_evo': 'GeoEvo',
    'scfv_mpnn': 'ProteinMPNN',
    'scfv_mod0': 'SAE Mod=0',
    'scfv_mod05': 'SAE Mod=0.5',
    'scfv_mod1': 'SAE Mod=1',
    'scfv_mod2': 'SAE Mod=2',
    '298': 10, # Y-axis limit for temperatures
    '373': 10,
    '433': 25
}

def load_rmsd(rmsd_data, protein, temp, method="min_max"):
    rmsd_df = pd.read_pickle(f"rmsd_data//{protein}/rmsd_{temp}.pkl")
    time = rmsd_df.index.values
    if method == "min_max":
        mean_rmsd = rmsd_df.mean(axis=1).rolling(25, center=True).mean()
        max_rmsd = rmsd_df.max(axis=1).rolling(25, center=True).mean()
        min_rmsd = rmsd_df.min(axis=1).rolling(25, center=True).mean()
        rmsd_data[protein] = (mean_rmsd, max_rmsd, min_rmsd)
    elif method == "avg":
        mean_rmsd = rmsd_df.mean(axis=1).rolling(25, center=True).mean()
        max_rmsd = mean_rmsd
        min_rmsd = mean_rmsd
    else:
        raise ValueError("Invalid method. Choose 'min_max' or 'avg'.")
    return time, rmsd_data

def graph_rmsd(time, rmsd_data, proteins, protein_folder, temp, method="min_max"):
    fig, ax = plt.subplots()
    for prot in proteins:
        mean_rmsd, max_rmsd, min_rmsd = rmsd_data[prot]
        plt.plot(time, mean_rmsd, '-', label=f'{values[prot]} Mean RMSD')
        if method == "min_max":
            plt.fill_between(time, min_rmsd, max_rmsd, alpha=0.3)
    plt.xlabel('Time (ns)')
    plt.ylabel(r'C$\alpha$ RMSD ($\AA$)')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0, top=values[temp])
    plt.title(f'{values[protein_folder][0]} Protein RMSD at {temp} K')
    plt.legend(loc='upper right')
    plt.savefig(f'rmsd_band_{values[protein_folder][0]}_{temp}_plot.png', dpi=300)
    plt.show()

def calc_rmsf(temp, rmsf_data):
    for prot in list(values.keys())[2:8]:
        rmsd_df = pd.read_pickle(f"rmsd_data/{prot}/rmsd_{temp}.pkl")
        mean_rmsf = rmsd_df.mean().mean()
        se = stats.sem(rmsd_df.mean())
        confidence = 0.95
        n = 500
        dof = n - 1
        confidence_interval = stats.t.interval(confidence, dof, mean_rmsf, se)
        err = (confidence_interval[1] - confidence_interval[0])/2
        rmsf_data[values[prot]] = (mean_rmsf, err)
        print((mean_rmsf, err))
    return rmsf_data

def graph_rmsf(rmsf_data):
    heights = [tup[0] for tup in rmsf_data.values()]
    err = [tup[1] for tup in rmsf_data.values()]
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.bar(rmsf_data.keys(), heights, yerr=err, capsize=4)
    plt.ylabel(r"Mean C$\alpha$ RMSF")
    plt.title(f"Mean RMSF for scFv Proteins at {temp} K")
    plt.savefig(f'rmsf_bar_plot_{temp}.png', dpi=300)
    plt.show()

protein_folder = "scfv_baseline"
temp = "373"
method = "min_max"
rmsd_data = {}
rmsf_data = {}

for prot in values[protein_folder][1]:
    time, rmsd_data = load_rmsd(rmsd_data, prot, temp, method="min_max")
graph_rmsd(time, rmsd_data, values[protein_folder][1], method=method)

rmsf_data = calc_rmsf(temp, rmsf_data)
print([val[0] for val in rmsf_data.values()])
graph_rmsf(rmsf_data)