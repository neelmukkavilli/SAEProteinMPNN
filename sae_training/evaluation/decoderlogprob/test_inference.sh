#!/bin/bash
path_to_PDB="../inputs/1a7w.pdb"

output_dir="outputs/training_test_output"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design=""

python decoder_mpnn_eval.py \
        --path_to_model_weights "../../training/exp_020" \
        --model_name "v_48_020" \
        --SAE_level "node" \
        --csv_output "edge_test" \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 1 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
