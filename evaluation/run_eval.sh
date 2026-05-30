input_dir="inputs"

output_dir="../outputs/training_test_output"

output_csv="node_log18"

touch "created_data/encodings/output_${output_csv}_0.pkl"

touch "created_data/encodings/output_${output_csv}_1.pkl"

touch "created_data/encodings/output_${output_csv}_2.pkl"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design = ""

for path_to_PDB in "$input_dir"/*.pdb; do
    echo "Processing $path_to_PDB..."

    python protein_mpnn_eval.py \
            --path_to_model_weights "../training/exp_020/model_weights/node_log18_NM" \
            --model_name "epoch_last" \
            --SAE_level "node" \
            --pdb_path $path_to_PDB \
            --pdb_path_chains "$chains_to_design" \
            --out_folder $output_dir \
            --csv_output $output_csv
done 

#export CSV=$output_csv
#python normalize_encodings.py  
