import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle

#protein_folder = "mpnn_md/enh_mpnn_4"
protein = 'scfv_native'
temp = "373"

#folder = Path(f"/WAVE/bio/MD/{protein_folder}")
folder = Path(f"/WAVE/bio/MD/senior_design_neelm/{protein}/")
top = folder / f"{protein}.psf"

rmsd_results = {}
for i in range(1, 6):
    dcds = sorted((folder / f"{temp}/{i}").glob("md*.dcd"))
    #dcds = '/WAVE/bio/MD/senior_design_neelm/scfv_mpnn/298/1/aligned_clean.dcd'
    u = mda.Universe(top, dcds)
    #align_sel = "protein and name CA"
    #ref = mda.Universe(top, dcds[0])

    #aligner = align.AlignTraj(u, ref, select=align_sel).run()

    #with mda.Writer("aligned.dcd", u.atoms.n_atoms) as W:
    #    for ts in u.trajectory[::100]:
    #        W.write(u.atoms)

    #aligned = mda.Universe(top, 'aligned.dcd')
    
    #ca = u.select_atoms(('resid 1:110' or 'resid 134:250') and 'name CA')

    R = rms.RMSD(
        u,u,
        select = ('resid 1:110 or resid 134:250 and name CA'),
        ref_frame=0,
        superposition=True).run(step=100)
    

    rmsd_results[f"Rep {i}"] = R.results.rmsd[:,2]
    print(R.results.rmsd.shape)

plt.figure(figsize=(8, 4))
for label, rmsd_vals in rmsd_results.items():
    plt.plot(R.results.rmsd[:, 0]/1000, rmsd_vals, label=label, linewidth=2)
plt.ylim(bottom=0, top=4)
plt.xlim(0, 100)
plt.xlabel('Time (ns)')
plt.ylabel(r'C$\alpha$ RMSD ($\AA$)')
plt.title(f'Native scFv RMSD at {temp} K')
plt.legend(loc='upper right')
plt.savefig(f'rmsd_{protein}_{temp}_plot.png', dpi=300)
plt.show()

