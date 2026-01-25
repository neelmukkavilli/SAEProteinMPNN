#!/bin/bash



python ../protein_mpnn_run.py \
        --SAE_level 'node' \
        --reinsert_SAE False \
        --path_to_model_weights "../training/exp_020/model_weights/largeruns/longlargebatch" \
        --model_name "epoch10" \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 8 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
