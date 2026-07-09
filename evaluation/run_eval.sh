input_dir="idp_inputs"
output_dir="../outputs/training_test_output"
SAE_type="node"
model="log17_exp2"

touch "created_data/encodings/idp_${SAE_type}_${model}/output_${model}_0.pkl"
touch "created_data/encodings/idp_${SAE_type}_${model}/output_${model}_1.pkl"
touch "created_data/encodings/idp_${SAE_type}_${model}/output_${model}_2.pkl"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

for path_to_PDB in "$input_dir"/*.pdb; do
    echo "Processing $path_to_PDB..."
    python protein_mpnn_eval.py \
            --path_to_model_weights "../training/exp_020/model_weights/${SAE_type}_${model}" \
            --model_name "epoch_last" \
            --SAE_level $SAE_type \
            --pdb_path $path_to_PDB \
            --pdb_path_chains "$chains_to_design" \
            --out_folder $output_dir \
            --csv_output $model
done   
