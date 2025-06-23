input_dir="inputs"

output_dir="../outputs/training_test_output"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design = ""

for path_to_PDB in "$input_dir"/*.pdb; do
    echo "Processing $path_to_PDB..."

    python protein_mpnn_eval.py \
            --path_to_model_weights "../training/exp_020/model_weights" \
            --model_name "epoch_s100_e1000" \
            --pdb_path $path_to_PDB \
            --pdb_path_chains "$chains_to_design" \
            --out_folder $output_dir \
            --num_seq_per_target 1 \
            --sampling_temp "0.1" \
            --seed 37 \
            --batch_size 1
done
