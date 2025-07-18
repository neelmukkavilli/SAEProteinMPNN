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

    def SAE_loss(x, x_reconstructed, encoded, sparsity):
        mse_loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        sparsity_loss = sparsity * torch.mean(torch.abs(encoded))
        return mse_loss + sparsity_loss
    
    scaler = torch.amp.GradScaler('cuda')
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Using GPU" if (torch.cuda.is_available()) else "Using CPU")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

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

    #get_std_opt(model.parameters(), args.hidden_dim, total_step)
    #for name, param in model.named_parameters():
    #    param.requires_grad = False

    #for name, param in model.named_parameters():
    #    if "WS1" in str(name):
    #        #print(name)
    #        param.requires_grad = True
    #    if "WS2" in str(name):
    #        #print(name)
    #        param.requires_grad = True
    SAE_params = [param for name, param in model.named_parameters() if "WS1" in name or "WS2" in name]
    
    optimizer = torch.optim.Adam([
        {"params": SAE_params}], lr=1e-4)

    #if PATH:
    #    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

        dataset_train = StructureDataset(islice(train_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
        dataset_valid = StructureDataset(islice(valid_pdbs_gen, args.num_examples_per_epoch), truncate=args.num_examples_per_epoch, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        reload_c = 0 
        for e in range(args.num_epochs):
            #print(str(e) + " epoch")
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
                reload_c += 1
            for batch_idx, batch in enumerate(loader_train):
                #print(str(batch_idx) + " number batch")
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                #print("featurize finished")
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                #print("grad finished")
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        log_probs, h_V_from_encoder, h_V_original, encoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        #print("model finished")
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                        sparse_loss = SAE_loss(h_V_original, h_V_from_encoder, encoded, sparsity=1e-3)
                        #print("loss calculated")
                    scaler.scale(loss_av_smoothed).backward()
                    #print("scaler went backward")
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    #print("scaler updated and stepped")
                else:
                    log_probs, h_V_from_encoder, h_V_original, encoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    #print("model ran")
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    sparse_loss = SAE_loss(h_V_original, h_V_from_encoder, encoded, sparsity=1e-3)
                    #print("calculated loss")
                    #print(str(sparse_loss) + " sparse loss")
                    #print("h_V_from encoder during training", h_V_from_encoder.requires_grad)
                    #print("h_V_original during training", h_V_original.requires_grad)
                    #print("sparse_loss", sparse_loss)
                    #print(sparse_loss.requires_grad)
                    #print(sparse_loss.grad_fn)
                    sparse_loss.backward()
                    #for name, param in model.named_parameters():
                    #    if param.requires_grad:
                    #        print(name, param.grad)

                    #print("went backward")
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                    #print("optimizer steps")

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1
                #print("program steps")

            model.eval()
            #print("evaluation mode")
            with torch.no_grad():
                validation_sum, validation_weights, validation_sparse_loss = 0., 0., 0.
                validation_acc = 0.
                for batch_idx, batch in enumerate(loader_valid):
                    #print(str(batch_idx) + " batch number")
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    #print("val featurize done")
                    log_probs, h_V_from_encoder, h_V_original, encoded = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    #print("model ran")
                    sparse_loss = SAE_loss(h_V_original, h_V_from_encoder, encoded, sparsity=1e-3)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                    validation_sparse_loss += torch.sum(sparse_loss * mask_for_loss).cpu().data.numpy()
            
            for param in SAE_params:
                print(np.format_float_positional(np.float32(torch.sum(param).detach().numpy()), unique=False, precision=3))

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

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, sparse loss: {sparse_loss_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            #print(f'WS1 Layer 0: {WS1_0}, WS1 Layer 1: {WS1_1}, WS1 Layer 2: {WS1_2}')
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
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_filename)

            # ChatGPT said to delete these after every epoch
            #del train_pdbs_gen, valid_pdbs_gen
            #gc.collect()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="./exp_020/model_weights/epoch_last.pt", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=20, help="number of epochs to train for") # default = 200
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
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
