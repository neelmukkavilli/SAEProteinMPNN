input_dir="inputs"

output_dir="../outputs/training_test_output"

output_csv="llb100_1e-2"

touch "encodings/output_llb100_1e-2_0.csv"

touch "encodings/output_llb100_1e-2_1.csv"

touch "encodings/output_llb100_1e-2_2.csv"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design = ""

for path_to_PDB in "$input_dir"/*.pdb; do
    echo "Processing $path_to_PDB..."

    python protein_mpnn_eval.py \
            --path_to_model_weights "../training/exp_020/model_weights/largeruns/reinit_class/llb100_1e-2" \
            --model_name "epoch_last" \
            --SAE_level "node" \
            --pdb_path $path_to_PDB \
            --pdb_path_chains "$chains_to_design" \
            --out_folder $output_dir \
            --csv_output $output_csv
done 

#export CSV=$output_csv
#python normalize_encodings.py  
