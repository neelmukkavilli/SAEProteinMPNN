import argparse
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc
import os
import itertools
from itertools import islice

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import torch.distributed as dist
    import queue
    import copy
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, SAE_loss, get_std_opt, ProteinMPNN, store_inputs_and_losses, reinit_anthropic, remove_parallel_grads, per_sample_SAE_loss, reinit_classic

    sparse_weight, mse_weight, reinit_every_n_steps = args.sparse_weight, args.mse_weight, args.reinit_every_n_steps
    
    print("Using GPU" if (torch.cuda.is_available()) else "Using CPU")

    device = torch.device(f"cuda")
    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)


    # Write log file
    PATH = args.previous_checkpoint
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
            f.write('sparse weight: {}, mse weight: {}, reinit every n steps: {}, lr: {}\n reservoir size: {}, num examples per epoch {}, batch size {}\n'.format(sparse_weight, mse_weight, reinit_every_n_steps, args.learning_rate, args.reservoir_size, args.num_examples_per_epoch, args.batch_size))
    if PATH:
        PATH = base_folder + PATH

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }
    
    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 1} # default = 4

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000
    
    train, valid, test = build_training_clusters(params, args.debug)
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    #valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    #valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)
    
    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        reinit_steps = checkpoint['reinit_step']
        activity_mask = checkpoint['activity_mask']
        reservoir_inputs = checkpoint['reservoir_inputs']
        reservoir_losses = checkpoint['reservoir_losses'] 
    else:
        model.load_state_dict(torch.load(base_folder + "v_48_020.pt", weights_only=False), strict=False)
        total_step = 0
        reinit_steps = 0
        epoch = 0
        activity_mask = torch.tensor(np.zeros((1024)))
        reservoir_inputs = []
        reservoir_losses = []

    optimizer = torch.optim.Adam([
        {"params": [param for name, param in model.sae_layers.named_parameters() if "0.W" in name]},
        {"params": [param for name, param in model.sae_layers.named_parameters() if "1.W" in name]},
        {"params": [param for name, param in model.sae_layers.named_parameters() if "2.W" in name]},
    ], args.learning_rate)
    

    if True:
        print("Parsing proteins")
        train_pdbs_gen = get_pdbs(train_loader, max_length=args.max_protein_length)
        print("Slicing dataset")
        dataset_train = StructureDataset(islice(train_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
        print("Cutting into batches")
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        '''
        valid_pdbs_gen = get_pdbs(valid_loader, max_length=args.max_protein_length)
        dataset_valid = StructureDataset(islice(valid_pdbs_gen, 1000), truncate=1000, max_length=args.max_protein_length) # args.num_examples_per_epoch -> 1000 
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        '''
        reload_c = 0
        print("Starting training...")
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.eval() # Now whole model is trained with model.eval()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            epoch_activity_mask = torch.tensor(np.zeros((1024))) # Which neurons are active in each epoch
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    train_pdbs_gen = get_pdbs(train_loader, max_length=args.max_protein_length)
                    dataset_train = StructureDataset(islice(train_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    '''
                    dataset_valid = StructureDataset(islice(valid_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
                    valid_pdbs_gen = get_pdbs(valid_loader, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    '''
                reload_c += 1
            # Used to collect info for reinitialization
            if args.SAE_level == 'edge':
                reservoir_size = args.reservoir_size #* 48 # Edges have 48 times more inputs/losses, size automatically multiplied for consistency
            else:
                reservoir_size = args.reservoir_size
            
            train_sparse_loss, train_mse_loss, specificity = 0, 0 ,0
            
            for batch_idx, batch in enumerate(loader_train):
                count = 0
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch

                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
            
                log_probs, original, encoded, decoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, args.SAE_level, args.reinsert_SAE)
                
                # Find active neurons
                if args.SAE_level == 'edge':
                    fired = (model.encoded_act[2].abs().sum(dim=[0,1,2]) > 0).int().cpu() # Sum batches, samples, and neighbors -> [1024]
                else:
                    fired = (model.encoded_act[2].abs().sum(dim=[0,1]) > 0).int().cpu() # Sum batches and samples -> [1024]
                activity_mask += fired.float()
                epoch_activity_mask += fired.float()
                
                total_loss, sparse_loss, mse_loss = SAE_loss(model, args.sparse_weight, mask_for_loss.bool())
                total_loss.backward()
                train_sparse_loss += torch.sum(sparse_loss * mask_for_loss).cpu().data.numpy()
                train_mse_loss += torch.sum(mse_loss * mask_for_loss).cpu().data.numpy()
                
                # Specificity is the average % of samples a neuron will activate for (# of samples > 0 / # of samples)
                specificity += (torch.mean((model.encoded_act[2] > 0).float()) / len(loader_train)).cpu().data.numpy()
                
                # Ensure gradient descent doesn't change dictionary vector length
                remove_parallel_grads(model.sae_layers)
                
                optimizer.step()

                total_step += 1
                reinit_steps += 1
                
                # Collect all inputs and losses for all batches in loader for at most 100 reinit steps
                if args.reinit == "anthropic":
                    if reinit_steps > (reinit_every_n_steps - 100):
                        reservoir_inputs, reservoir_losses = store_inputs_and_losses(count, reservoir_inputs, reservoir_losses, reservoir_size, original, encoded, decoded, mask, chain_M, sparse_weight)
                if reinit_steps > reinit_every_n_steps:
                    if args.reinit == "anthropic": # Anthropic reinitialization defined in X paper
                        with torch.no_grad():
                            dead_neurons = (activity_mask == 0).nonzero(as_tuple=True)[0]
                            alive_neurons = (activity_mask > 0).nonzero(as_tuple=True)[0]
                            #per_epoch_dead_neurons = (epoch_activity_mask == 0).nonzero(as_tuple=True)[0]
                            print(f"Number of dead neurons: {len(dead_neurons)}")
                            if len(dead_neurons) > 0:
                                print(f'L2 norm before reinitialization {torch.linalg.vector_norm(model.sae_layers[2].WS1.weight[dead_neurons], dim=1).mean().item()}')
                                reinit_anthropic(model, optimizer, reservoir_inputs, reservoir_losses, alive_neurons, dead_neurons, device)
                                print(f'L2 norm after reinitialization {torch.linalg.vector_norm(model.sae_layers[2].WS1.weight[dead_neurons], dim=1).mean().item()}')
                            else:
                                print("No dead neurons, not reinitializing")
                            activity_mask.zero_()
                            reservoir_losses = []
                            reservoir_inputs = []
                            reinit_steps = 0
                            count = 0
                    elif args.reinit == "classic": # Reinitialize random fraction of dead neurons to kaiming_uniform_
                        with torch.no_grad():
                            dead_neurons = (activity_mask == 0).nonzero(as_tuple=True)[0]
                            alive_neurons = (activity_mask > 0).nonzero(as_tuple=True)[0]
                            if len(dead_neurons) > 0:
                                print(f'L2 norm before reinitialization {torch.linalg.vector_norm(model.sae_layers[2].WS1.weight[dead_neurons], dim=1).mean().item()}')
                                reinit_classic(model, optimizer, dead_neurons, device, fraction_reinit=1)
                                print(f'L2 norm before reinitialization {torch.linalg.vector_norm(model.sae_layers[2].WS1.weight[dead_neurons], dim=1).mean().item()}')
                            else:
                                print("No dead neurons, not reinitializing")
                            activity_mask.zero_()
                            reservoir_losses = []
                            reservoir_inputs = []
                            reinit_steps = 0
                            count = 0
            '''
                # Calculate training loss
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            
            model.eval()
            print("Reinit steps:", reinit_steps)
            with torch.no_grad():
                validation_sum, validation_weights, validation_sparse_loss, validation_mse_loss, validation_norm_loss = 0., 0., 0., 0., 0.
                validation_acc = 0.
                specificity = 0
                for batch_idx, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs, original, encoded, decoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, args.SAE_level, args.reinsert_SAE)
                    
                    # specificity is the average % of samples a neuron will activate for (# of samples > 0 / # of samples)
                    specificity += torch.mean((model.encoded_act[2] > 0).float()) / len(loader_valid)
                    
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    total_loss, sparse_loss, mse_loss = SAE_loss(model, sparse_weight, mask_for_loss.bool())

                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    validation_sparse_loss += torch.sum(sparse_loss * mask_for_loss).cpu().data.numpy()
                    validation_mse_loss += torch.sum(mse_loss * mask_for_loss).cpu().data.numpy()
                    #validation_norm_loss += torch.sum(loss_av_smoothed * mask_for_loss).cpu().data.numpy()
            

            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
            epoch_activity_mask.zero_()
            norm_loss_ = np.format_float_positional(np.float32(validation_norm_loss), unique=False, precision=3)
            '''

            specificity_ = np.format_float_positional(np.float32(specificity.item()), unique=False, precision=3)
            sparse_loss_ = np.format_float_positional(np.float32(train_sparse_loss), unique=False, precision=3)
            mse_loss_ = np.format_float_positional(np.float32(train_mse_loss), unique=False, precision=3)
            dead_neurons = len((activity_mask == 0).nonzero(as_tuple=True)[0])
            per_epoch_dead_neurons = len((epoch_activity_mask == 0).nonzero(as_tuple=True)[0])

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, time: {dt}, sparse loss: {sparse_loss_}, mse loss: {mse_loss_}, dead neurons: {dead_neurons}, per epoch dead neurons: {per_epoch_dead_neurons}, specificity: {specificity_}\n')#, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, time: {dt}, reinit step: {reinit_steps}, sparse loss: {sparse_loss_}, mse loss: {mse_loss_}, dead neurons: {dead_neurons}, per epoch dead neurons: {per_epoch_dead_neurons}, specificity: {specificity_}')#, valid_acc: {validation_accuracy_}')
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            print("Len of res inputs:", len(reservoir_inputs))
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'reinit_step': reinit_steps,
                        'activity_mask': activity_mask,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'reservoir_inputs': reservoir_inputs,
                        'reservoir_losses': reservoir_losses,
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                print("saving")
                checkpoint_filename = base_folder +'model_weights/epoch{}.pt'.format(e+1)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'reinit_step': reinit_steps,
                        'activity_mask': activity_mask,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'reservoir_inputs': reservoir_inputs,
                        'reservoir_losses': reservoir_losses,
                        }, checkpoint_filename)

            torch.cuda.empty_cache()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## Hyperparameters to update for better training results
    argparser.add_argument("--SAE_level", type=str, default='node', help='node or edge')
    argparser.add_argument("--reinsert_SAE", action="store_true", help="if the decoded embedding is fed back into the model")
    argparser.add_argument("--learning_rate", type=float, default=0.0001, help="range from 1e-2 to 1e-6")
    argparser.add_argument("--sparse_weight", type=float, default=1e-3, help='range from 10 to 1e-5')
    argparser.add_argument("--mse_weight", type=float, default=1.0, help='keep at 1.0 and change sparse_weight')
    argparser.add_argument("--reinit_every_n_steps", type=int, default=10000, help='for default at 10k, starts storing dead neurons for previous n/2 = 5k steps')
    argparser.add_argument("--reinit", type=str, default='anthropic', help='Choose "anthropic", "classic", or "none"')
    argparser.add_argument("--reservoir_size", type=int, default=10000, help='number of proteins to collect input and losses for reinitialization')
    ## Double check the first time you run to make sure the path is correct
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")  # pdb_2021aug02_sample for debugging
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights") # Outputs cannot write to this WAVE/bio/ML folder without write access ../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt") # default = model_weights/epoch_last.pt
    ## Only change if running out of memory or training is too slow
    argparser.add_argument("--num_epochs", type=int, default=300, help="number of epochs to train for") # default = 200
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=20, help="reload training data every n epochs") # default = 20, should be more often if num_examples_per_epoch is low
    argparser.add_argument("--num_examples_per_epoch", type=int, default=10000, help="number of training example to load for one epoch") # default = 1,000,000
    argparser.add_argument("--batch_size", type=int, default=5000, help="number of tokens for one batch") # default = 10,000, number of tokens not examples, larger tensors are more efficient but require more memory
    argparser.add_argument("--max_protein_length", type=int, default=1000, help="maximum length of the protein complex") # default = 10,000
    # Normal ProteinMPNN parameters, do not change
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=False, help="train with mixed precision") # default = True
    args = argparser.parse_args()    
    main(args)