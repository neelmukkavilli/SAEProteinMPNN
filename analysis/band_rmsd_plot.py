import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

#protein_folder = "mpnn_md/enh_mpnn_4"
#protein = 'scfv_native'
temp = "298"

#folder = Path(f"/WAVE/bio/MD/{protein_folder}")


rmsd_results = {}
data = {}

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
plt.ylabel(r'C$\alpha$ RMSD ($\AA$)')
plt.title(f'Modified Protein RMSD at {temp} K')
plt.legend(loc='upper right')
plt.savefig(f'mod_rmsd_band_allprots_{temp}_plot.png', dpi=300)