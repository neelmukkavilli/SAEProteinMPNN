#!/bin/bash 
# 
#SBATCH --job-name=edge_sig 
#SBATCH --output=kivanov_run_edge_sig.out 
# 
#SBATCH --partition=cmp,amd
#
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00 
#
#SBATCH --mail-user=kivanov@scu.edu
#SBATCH --mail-type=END

#Automate edge_signifiancetest.py for all edge encoding .pkl files

module load Anaconda3
#if fails, try running conda init bash in terminal
source ~/.bashrc #conda activate command seems to not work if this line is absent
conda activate pytorchenv
which python

for log in 15; do
    for SAE in 0 1 2; do
        model="log${log}_edge_exp2_100e_${SAE}"
        echo "----- Processing ${model} -----"
        python edge_significancetest.py $model
    done
done