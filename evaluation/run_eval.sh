input_dir="inputs"

output_dir="../outputs/training_test_output"

output_csv="slow_exp2_se4"

touch "created_data/encodings/output_slow_exp2_se4_0.pkl"

touch "created_data/encodings/output_slow_exp2_se4_1.pkl"

touch "created_data/encodings/output_slow_exp2_se4_2.pkl"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design = ""

for path_to_PDB in "$input_dir"/*.pdb; do
    echo "Processing $path_to_PDB..."

    python protein_mpnn_eval.py \
            --path_to_model_weights "../training/exp_020/model_weights/bps_final_models/slow_exp_2_s0001" \
            --model_name "epoch50" \
            --SAE_level "node" \
            --pdb_path $path_to_PDB \
            --pdb_path_chains "$chains_to_design" \
            --out_folder $output_dir \
            --csv_output $output_csv
done 

#export CSV=$output_csv
#python normalize_encodings.py  
