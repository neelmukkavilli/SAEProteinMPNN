input_dir="inputs"

output_dir="../outputs/training_test_output"

output_csv="logprobs"

touch "encodings/output_$output_csv.csv"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design = ""

for path_to_PDB in "$input_dir"/*.pdb; do
    echo "Processing $path_to_PDB..."

    python protein_mpnn_eval.py \
            --path_to_model_weights "../training/exp_020" \
            --model_name "v_48_020" \
            --SAE_level "node" \
            --return_log_probs \
            --pdb_path $path_to_PDB \
            --pdb_path_chains "$chains_to_design" \
            --out_folder $output_dir \
            --csv_output $output_csv
done 

#export CSV=$output_csv

#python normalize_encodings.py  
