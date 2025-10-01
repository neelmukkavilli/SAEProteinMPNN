#!/bin/bash

python training.py \
	--SAE_level "node" \
	--learning_rate 0.0001 \
	--sparse_weight 0.1 \
	--mse_weight 1.0 \
	--reinit_every_n_steps 20 \
	--reservoir_size 100000 \
	--path_for_outputs "../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020" \
	--path_for_training_data "pdb_2021aug02" \
	--previous_checkpoint "" \
	--num_examples_per_epoch 10000 \
	--num_epochs 100 \
	--batch_size 10000

# Other arguments
#	--reinsert_SAE True

# Path for WAVE training 
#	-- path_for_outputs "../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020"