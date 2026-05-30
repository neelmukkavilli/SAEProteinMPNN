#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training.py \
	--SAE_level "node" \
	--reinit "classic" \
	--reinsert_SAE \
	--learning_rate 0.00000 \
	--sparse_weight 0.001 \
	--reinit_every_n_steps 5000 \
	--reservoir_size 100000 \
	--path_for_outputs "/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/training/exp_020" \
	--path_for_training_data "pdb_2021aug02_sample" \
	--previous_checkpoint "model_weights/largeruns/reinit_class/llb1e-2/epoch_last.pt" \
	--num_examples_per_epoch 10000 \
	--num_epochs 10 \
	--batch_size 10000

# Reservoir size is multiplied to be 48 times larger for SAE_level = "edge"

# Other arguments
#	--reinsert_SAE True

# Path for WAVE training 
#	-- path_for_outputs "/WAVE/users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020"
# test
# --previous_checkpoint "model_weights/epoch_last.pt"
