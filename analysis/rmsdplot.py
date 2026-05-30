import MDAnalysis as mda
from MDAnalysis.analysis import diffusionmap, align, rms
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle

#protein_folder = "mpnn_md/enh_mpnn_4"
protein = 'scfv_native'
temp = "433"
name = "Native"
trials = 5
#folder = Path(f"/WAVE/bio/MD/{protein_folder}")
folder = Path(f"/WAVE/bio/MD/senior_design_neelm/{protein}/")
top = folder / f"{protein}.psf"

rmsd_results = {}

def distance_matrix(protein, temp, name, trials, folder):
    for i in range(1, trials +1):
        dcds = sorted((folder / f"{temp}/{i}").glob("md*.dcd"))
        #dcds = f"/WAVE/bio/MD/senior_design_neelm/scfv_sae/{protein}/{temp}/{i}/aligned.dcd"
        u = mda.Universe(top, dcds)
        align.AlignTraj(u, mda.Universe(top, dcds[0]), select="protein and name CA", filename='aligned.dcd').run(step=1000)
        aligned = mda.Universe(top, 'aligned.dcd')
        matrix = diffusionmap.DistanceMatrix(aligned.atoms, select=('resid 1:110 or resid 134:250 and name CA'), superposition=True).run()
        plt.imshow(matrix.results.dist_matrix, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('Frame Index')
        plt.ylabel('Frame Index')
        plt.title(f'{name} scFv Distance Matrix at {temp} K')
        #plt.pcolor(matrix.results.dist_matrix, vmin=0, vmax=15)
        plt.savefig(f'graphics/aligned_distance_matrix_{protein}_{temp}_rep{i}.png')
        #plt.show()
        plt.clf()
        print(f'aligned {protein} at {temp} for rep: {i}: {np.average(matrix.results.dist_matrix)}')

#distance_matrix(protein, temp, name, trials, folder)


for i in range(1, trials +1):
    #dcds = f"/WAVE/bio/MD/senior_design_neelm/scfv_sae/{protein}/{temp}/{i}/aligned.dcd"
    dcds = sorted((folder / f"{temp}/{i}").glob("md*.dcd"))
    u = mda.Universe(top, dcds)
    #align.AlignTraj(u, mda.Universe(top, dcds[0]), select="protein and name CA", filename='aligned.dcd').run(step=100)
    #aligned = mda.Universe(top, 'aligned.dcd')
    R = rms.RMSD(
        u,
        select = ('resid 1:110 or resid 134:250 and name CA'),
        ref_frame=0,
        superposition=True).run(step=100)
    

    rmsd_results[f"Rep {i}"] = R.results.rmsd[:,2]
    print(R.results.rmsd.shape)

plt.figure(figsize=(8, 4))
for label, rmsd_vals in rmsd_results.items():
    plt.plot(R.results.rmsd[:, 0]/1000, rmsd_vals, label=label, linewidth=2)
plt.ylim(bottom=0, top=15)
plt.xlim(0, 100)
plt.xlabel('Time (ns)')
plt.ylabel(r'C$\alpha$ RMSD ($\AA$)')
plt.title(f'{name} scFv RMSD at {temp} K')
plt.legend(loc='upper right')
plt.savefig(f'graphics/rmsd_plots/rmsd_{protein}_{temp}_plot.pdf', bbox_inches='tight')
plt.show()

