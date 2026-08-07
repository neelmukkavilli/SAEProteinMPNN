import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from scipy import stats

#folder = Path(f"/WAVE/bio/MD/{protein_folder}")


#rmsd_results = {}
#data = {}

def load_rmsd(rmsd_data, protein_folder, protein, temp, method="min_max"):
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

def graph_rmsd(time, rmsd_data, proteins, method="min_max"):
    fig, ax = plt.subplots()
    for prot in proteins:
        mean_rmsd, max_rmsd, min_rmsd = rmsd_data[prot]
        plt.plot(time, mean_rmsd, '-', label=f'{values[prot]} Mean RMSD')
        plt.fill_between(time, min_rmsd, max_rmsd, alpha=0.3)
    plt.xlabel('Time (ns)')
    plt.ylabel(r'C$\alpha$ RMSD ($\AA$)')
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0, top=values[temp])
    plt.title(f'{values[protein_folder][0]} Protein RMSD at {temp} K')
    plt.legend(loc='upper right')
    plt.savefig(f'rmsd_band_{values[protein_folder][0]}_{temp}_plot.png', dpi=300)
    plt.show()

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

#protein_folder = "scfv_baseline"
#temp = "373"
#method = "min_max"
#rmsd_data = {}
#for prot in values[protein_folder][1]:
#    time, rmsd_data = load_rmsd(rmsd_data, protein_folder, prot, temp, method="min_max")
#graph_rmsd(time, rmsd_data, values[protein_folder][1], method=method)



#print(list(values.keys()))
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

temp = 433
rmsf_data = {}
rmsf_data = calc_rmsf(temp, rmsf_data)
print([val[0] for val in rmsf_data.values()])
graph_rmsf(rmsf_data)

'''
for prot in ['scfv_native', 'scfv_mod0', 'scfv_mod05', 'scfv_mod1', 'scfv_mod2']:
    for i in range(1, 6):
        if prot == 'scfv_native':
            folder = Path(f"/WAVE/bio/MD/senior_design_neelm/{prot}/")
        else:
            folder = Path(f"/WAVE/bio/MD/senior_design_neelm/scfv_sae/{prot}/")
        top = folder / f"{prot}.psf"
        dcds = sorted((folder / f"{temp}/{i}").glob("md*.dcd"))
        u = mda.Universe(top, dcds)

        R = rms.RMSD(
            u,u,
            select = ('resid 1:110 or resid 134:250 and name CA'),
            ref_frame=0,
            superposition=True).run(step=100)
        rmsd_results[f"Rep {i}"] = R.results.rmsd[:,2]    
        print(i, prot)
    results = final_array = np.column_stack(list(rmsd_results.values()))
    mean_rmsd = pd.Series(np.mean(results, axis=1)).rolling(25, center=True).mean()
    max_rmsd = pd.Series(np.max(results, axis=1)).rolling(25, center=True).mean()
    min_rmsd = pd.Series(np.min(results, axis=1)).rolling(25, center=True).mean()
    data[prot] = (mean_rmsd, max_rmsd, min_rmsd)


fig, ax = plt.subplots()
x_range = R.results.rmsd[:, 0]/1000
plt.ylim(bottom=0, top=5)
plt.xlim(0, 100)
#ax.plot(x_range, data['scfv_native'][0], '-', color='blue', label='Native')
#ax.fill_between(x_range, data['scfv_native'][2], data['scfv_native'][1], alpha=0.2, color='blue')
#ax.plot(x_range, data['scfv_mpnn'][0], '-', color='red', label='MPNN')
#ax.fill_between(x_range, data['scfv_mpnn'][2], data['scfv_mpnn'][1], alpha=0.2, color='red')
#ax.plot(x_range, data['scfv_evo'][0], '-', color='green', label='GeoEvo')
#ax.fill_between(x_range, data['scfv_evo'][2], data['scfv_evo'][1], alpha=0.2, color='green')

ax.plot(x_range, data['scfv_native'][0], '-', color='blue', label='Native')
ax.plot(x_range, data['scfv_mod0'][0], '-', color='red', label='SAE mod=0')
ax.plot(x_range, data['scfv_mod05'][0], '-', color='green', label='SAE mod=0.5')
ax.plot(x_range, data['scfv_mod1'][0], '-', color='purple', label='SAE mod=1')
ax.plot(x_range, data['scfv_mod2'][0], '-', color='darkorange', label='SAE mod=2')



plt.xlabel('Time (ns)')
plt.title(f'Modified Protein RMSD at {temp} K')
plt.legend(loc='upper right')
plt.savefig(f'mod_rmsd_band_allprots_{temp}_plot.png', dpi=300)
'''
