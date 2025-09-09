#!/bin/bash

python training.py \
	--SAE_level "node" \
	--reinsert_SAE False \
	--learning_rate 0.0001 \
	--sparse_weight 0.1 \
	--mse_weight 1.0 \
	--reinit_every_n_steps 5000 \
	--reservoir_size 10000 \
	--path_for_outputs "../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020" \
	--path_for_training_data "pdb_2021aug02" \
	--previous_checkpoint "" \
	--num_examples_per_epoch 10000 \
	--num_epochs 200 \
	--batch_size 10000