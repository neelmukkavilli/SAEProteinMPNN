import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle

protein = "scfv0"
temp = "298"


folder = Path("/WAVE/bio/MD/senior_design_neelm/scfv_test/")
top = folder / f"{protein}.psf"

rmsd_results = {}
for i in range(1, 4):
    dcds = sorted((folder / f"{temp}/{i}").glob("md*.dcd"))
    u = mda.Universe(top, *dcds)

    sel = u.select_atoms('name CA')

    R = rms.RMSD(sel,
                sel,
                ref_frame=0)
    
    R.run(step=50)
    rmsd_results[f"Rep {i}"] = R.results.rmsd[:,2]

plt.figure(figsize=(8, 4))
for label, rmsd_vals in rmsd_results.items():
    plt.plot(R.results.rmsd[:, 0]/1000, rmsd_vals, label=label, linewidth=2)
plt.ylim(bottom=0)
plt.xlabel('Time (ns)')
plt.ylabel(r'C$\alpha$ RMSD ($\AA$)')
plt.title('RMSD over Time')
plt.legend()
plt.savefig('rmsd_plot.png', dpi=300)
plt.show()