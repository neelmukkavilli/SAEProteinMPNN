import argparse
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    def orthogonality_loss(act):
        orth = 0
        for i in range(act.shape[0]):
            act_n = torch.nn.functional.normalize(act[i, : :], dim=0)
            AT_A = torch.matmul(act_n, act_n.T)
            identity = torch.eye(AT_A.shape[0], device=act.device)
            orth += torch.norm(AT_A - identity, p='fro')**2

        return orth / act.shape[0]
        '''
        for layer in model.encoder_layers:
            for sublayer_name in ['WS1', 'WS2']:
                sublayer = getattr(layer, sublayer_name)
                W = sublayer.weight
                W_normalized = torch.nn.functional.normalize(W, p=2, dim=1)
                WT_W = torch.matmul(W_normalized, W_normalized.T)
                identity = torch.eye(WT_W.shape[0], device=W.device)

                ortho_loss += torch.norm(WT_W - identity, p='fro')**2

        return ortho_loss
        '''    
    sparse_weight, mse_weight, norm_weight, ortho_weight, KL_weight = 1, 1, 1e1, 2e-4, 1e-4
    fraction_to_reinit = 0.05
    dead_floor = 200
    start_reinit_after = 50
    rho_target = 0.55

    def KL_divergence(rho, encoded):
        rho_hat = torch.mean(F.sigmoid(encoded), dim=2)
        rho = torch.full(rho_hat.shape, rho).to(device)
        kl = torch.sum(rho * torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat)))
        return kl

    def KL_stdv(rho, encoded):
        rho_hat = torch.mean(F.sigmoid(encoded), dim=2)
        rho = torch.full(rho_hat.shape, rho).to(device)
        kl = torch.sum(rho * torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat)))
        return kl
    
    def SAE_loss(x, x_reconstructed, encoded, model, S, log_probs, mask_for_loss):
        mse_loss = 0
        sparse_loss = 0
        KL_loss = 0
        ortho_loss = 0
        _, loss_smooth = loss_smoothed(S, log_probs, mask_for_loss)

        for i, act in enumerate(model.encoded_act):
            mse_loss += mse_weight * torch.nn.functional.mse_loss(model.output_act[i], model.input_act[i])
            KL_loss += KL_weight * KL_divergence(rho_target, act)
            sparse_loss += sparse_weight * torch.mean(torch.abs(act))
            ortho_loss += ortho_weight * orthogonality_loss(act)
        
        #ortho_loss = ortho_weight * orthogonality_loss(model)
            
        total_loss = ortho_loss + KL_loss + norm_weight * loss_smooth#sparse_loss + ortho_loss + norm_weight * loss_smooth
        return total_loss, sparse_loss.detach() / sparse_weight, mse_loss.detach(), loss_smooth.detach(), ortho_loss.detach() / ortho_weight, KL_loss.detach()
    
    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Using GPU" if (torch.cuda.is_available()) else "Using CPU")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    base_folder = "../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020/"
    
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    
    print("Check 1")

    PATH = args.previous_checkpoint
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
            f.write('sparse weight: {}, mse weight: {}, norm weight: {}, ortho weight: {}, KL weight: {}\n'.format(sparse_weight, mse_weight, norm_weight, ortho_weight, KL_weight))
            f.write('rho target: {}, fraction to reinitiate: {}, dead floor: {}, start reinit after: {}\n'.format(rho_target, fraction_to_reinit, dead_floor, start_reinit_after))
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

    print("Check 2")

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 0} # default = 4

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    print("Check 3")

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
    else:
        model.load_state_dict(torch.load(base_folder + "v_48_020.pt", weights_only=False), strict=False)
        total_step = 0
        epoch = 0

    print("Check 4")
    SAE_params = [param for name, param in model.named_parameters()]# if "WS1" in name or "WS2" in name]
    
    optimizer = torch.optim.Adam([
        {"params": SAE_params}], lr=1e-4)

    process_pool = False
    if process_pool:
        with ProcessPoolExecutor(max_workers=1) as executor: # default = 12
            q = queue.Queue(maxsize=1) # default = 3
            p = queue.Queue(maxsize=1) # default = 3
            for i in range(1): # default = 3
                q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            pdb_dict_train = q.get().result()
            pdb_dict_valid = p.get().result()
        
            dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
            dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
            
            loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
            loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
    else:
        #train_pdbs_gen = list(itertools.islice(get_pdbs(train_loader), args.num_examples_per_epoch))
        #valid_pdbs_gen = list(itertools.islice(get_pdbs(valid_loader), args.num_examples_per_epoch))
        #pdb_dict_train = get_pdbs(train_loader, 1, args.max_protein_length, args.num_examples_per_epoch)
        #pdb_dict_valid = get_pdbs(valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch)

        train_pdbs_gen = get_pdbs(train_loader, max_length=args.max_protein_length)
        valid_pdbs_gen = get_pdbs(valid_loader, max_length=args.max_protein_length)

        print("Check 4.1")

        dataset_train = StructureDataset(islice(train_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
        dataset_valid = StructureDataset(islice(valid_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
        
        print("Check 4.2")

        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        print("Check 4.3")

        reload_c = 0
        reinitiate = False 
        print("Check 5")
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    train_pdbs_gen = get_pdbs(train_loader, max_length=args.max_protein_length)
                    valid_pdbs_gen = get_pdbs(valid_loader, max_length=args.max_protein_length)

                    dataset_train = StructureDataset(islice(train_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
                    dataset_valid = StructureDataset(islice(valid_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
                    
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    '''
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    '''
                # Reinitialize dead weights and update sparsity every 20 epochs
                #if e % args.reload_data_every_n_epochs == 0 and reload_c != 0:
                reinitiate = True
                reload_c += 1

            for batch_idx, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
            
                if args.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        log_probs, h_V_from_encoder, h_V_original, encoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                        total_loss, sparse_loss, mse_loss, loss_av_smoothed, ortho_loss, KL_loss = SAE_loss(h_V_original, h_V_from_encoder, encoded, model, S, log_probs, mask_for_loss)
                    scaler.scale(loss_av_smoothed).backward()
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs, h_V_from_encoder, h_V_original, encoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    total_loss, sparse_loss, mse_loss, loss_av_smoothed, ortho_loss, KL_loss = SAE_loss(h_V_original, h_V_from_encoder, encoded, model, S, log_probs, mask_for_loss)
                    total_loss.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights, validation_sparse_loss, validation_mse_loss, validation_norm_loss, validation_ortho_loss, validation_KL_loss = 0., 0., 0., 0., 0., 0., 0.
                validation_acc = 0.
                dead_columns = {}
                average_active = 0
                counter = 0
                for batch_idx, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs, h_V_from_encoder, h_V_original, encoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    if reinitiate == True: 
                        # Report fraction of neurons active and change sparsity
                        for layer, act in enumerate(model.encoded_act):
                            fraction_active = (act > 0).float().sum(dim=2) / encoded.shape[2]
                            average_active += fraction_active.mean() / 3

                        if batch_idx == 0:
                            # Check for dead neurons    
                            for layer, act in enumerate(model.encoded_act):
                                # Returns True where all activations are 0
                                dead_columns[layer] = torch.sum(act > 0, dim=(0,1)) == 0
                                
                        else:
                            for layer, act in enumerate(model.encoded_act):
                                dead_columns[layer] = torch.logical_and(dead_columns[layer], (torch.sum(act > 0, dim=(0,1)) == 0))
                    counter += 1   
                if reinitiate == True and e > start_reinit_after:            
                    for layer, act in enumerate(model.encoded_act):
                        dead_indices = torch.where(dead_columns[layer])[0]
                        num_dead = len(dead_indices)
                        if num_dead > dead_floor:
                            num_to_reinit = max(1, int(fraction_to_reinit * num_dead))
                            selected_indices = dead_indices[torch.randperm(num_dead)[:num_to_reinit]]
                            weight = model.encoder_layers[layer].WS1.weight
                            bias = model.encoder_layers[layer].WS1.bias
                            reinit_weight = torch.empty_like(weight)
                            nn.init.kaiming_uniform_(reinit_weight, nonlinearity='leaky_relu')
                            weight[selected_indices] = reinit_weight[selected_indices]
                            bias[selected_indices] = 0.0
                    reinitiate = False
                    average_active_ = np.format_float_positional(np.float32(average_active.item())/counter, unique=False, precision=3)

                    with open(logfile, 'a') as f:
                        f.write(f'Average active neurons: {average_active_}, dead_columns: {torch.sum(dead_columns[0]).item(), torch.sum(dead_columns[1]).item(), torch.sum(dead_columns[2]).item()} \n')
                    print(f'Average active neurons: {average_active_}, dead_columns: {torch.sum(dead_columns[0]).item(), torch.sum(dead_columns[1]).item(), torch.sum(dead_columns[2]).item()}')
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                total_loss, sparse_loss, mse_loss, loss_av_smoothed, ortho_loss, KL_loss = SAE_loss(h_V_original, h_V_from_encoder, encoded, model, S, log_probs, mask_for_loss)

                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                validation_sparse_loss += torch.sum(sparse_loss * mask_for_loss).cpu().data.numpy()
                validation_mse_loss += torch.sum(mse_loss * mask_for_loss).cpu().data.numpy()
                validation_norm_loss += torch.sum(loss_av_smoothed * mask_for_loss).cpu().data.numpy()
                validation_ortho_loss += torch.sum(ortho_loss * mask_for_loss).cpu().data.numpy()
                validation_KL_loss += torch.sum(KL_loss * mask_for_loss).cpu().data.numpy()

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
            sparse_loss_ = np.format_float_positional(np.float32(validation_sparse_loss), unique=False, precision=3)
            mse_loss_ = np.format_float_positional(np.float32(validation_mse_loss), unique=False, precision=3)
            norm_loss_ = np.format_float_positional(np.float32(validation_norm_loss), unique=False, precision=3)
            ortho_loss_ = np.format_float_positional(np.float32(validation_ortho_loss), unique=False, precision=3)
            KL_loss_ = np.format_float_positional(np.float32(validation_KL_loss), unique=False, precision=3)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, sparse loss: {sparse_loss_}, mse loss: {mse_loss_}, ortho loss: {ortho_loss_}, KL loss: {KL_loss_}, norm loss: {norm_loss_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, sparse loss: {sparse_loss_}, mse loss: {mse_loss_}, ortho loss: {ortho_loss_}, KL_loss: {KL_loss_}, norm loss: {norm_loss_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                print("saving")
                checkpoint_filename = base_folder +'model_weights/epoch{}.pt'.format(e+1)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt") # default = ./exp_020/model_weights/epoch_last.pt
    # ../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020/model_weights/epoch_last.pt
    argparser.add_argument("--num_epochs", type=int, default=250, help="number of epochs to train for") # default = 200
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=50, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=10, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000, help="number of training example to load for one epoch") # default = 1,000,000
    argparser.add_argument("--batch_size", type=int, default=5000, help="number of tokens for one batch") # default = 10,000, number of tokens not examples
    argparser.add_argument("--max_protein_length", type=int, default=500, help="maximum length of the protein complex") # default = 10,000
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
