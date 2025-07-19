import argparse
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc
import seaborn as sns
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import itertools
from itertools import islice
model_name = "lsamples500_2"  # Change this to the model you want to analyze
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

    '''
    def orthogonality_loss(model, ortho_weight=1e-3):
        ortho_loss = 0.0
        for layer in model.encoder_layers:
            for sublayer_name in ['WS1', 'WS2']:
                sublayer = getattr(layer, sublayer_name)
                W = sublayer.weight

                W_normalized = torch.nn.functional.normalize(W, p=2, dim=1)

                WT_W = torch.matmul(W_normalized, W_normalized.T)
                identity = torch.eye(WT_W.shape[0], device=W.device)

                ortho_loss += torch.norm(WT_W - identity, p='fro')**2

        return ortho_weight * ortho_loss

    def SAE_loss(x, x_reconstructed, encoded, model, loss, sparsity=5e-3, ortho_weight=5e-3):
        mse_loss = torch.nn.functional.mse_loss(x_reconstructed, x)
        sparsity_loss = sparsity * torch.mean(torch.abs(encoded))
        ortho_loss = orthogonality_loss(model, ortho_weight)
        total_loss = mse_loss + sparsity_loss + ortho_loss + 0.1*loss
        return total_loss, sparsity_loss/sparsity
    
    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("Using GPU" if (torch.cuda.is_available()) else "Using CPU")

    PATH = args.previous_checkpoint

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
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    '''
    #W = model.encoder_layers[2].WS1.weight.detach().cpu()

    W = pd.read_csv('/WAVE/bio/ML/SAE_train/SAEProteinMPNN/sae_training/evaluation/encodings/normalized_encodings_' + model_name + '.csv', usecols=range(1,1025))
    #W_normalized = torch.nn.functional.normalize(W, p=2, dim=1)  # [out_dim, in_dim]
    
    #pca = PCA(n_components=100, svd_solver='arpack')  # Keep 95% variance
    W_np = W.to_numpy()  # Convert to numpy for PCA
    pca = PCA(n_components=100, svd_solver='arpack')
    pca.fit(W_np)
    plt.bar(range(1,101), pca.explained_variance_ratio_)
    plt.title("Explained variance ratio")
    plt.ylabel("'%' of variance explained")
    plt.xlabel("Top 100 dimensions")
    plt.tight_layout()
    plt.savefig("explained_variance_ratio.png")
    plt.show()
    print("Explained variance ratio:", pca.explained_variance_ratio_[:10])
    
    threshold = 0.01  # Set a threshold for non-zero counts

    nonzero_counts = (W_np > threshold).sum(axis=1)
    plt.hist(nonzero_counts)
    plt.title("Histogram of Non-Zero Counts per Activation")
    plt.xlabel("Number of Non-Zero Counts")
    plt.ylabel("Frequency")
    plt.savefig("nonzero_counts_histogram.png")
    plt.show()

    per_unit_usage = (W_np > threshold).sum(axis=0) / W_np.shape[0]  # Calculate usage frequency per neuron
    plt.figure(figsize=(10, 5))
    plt.title("Per Unit Usage of Neurons")
    plt.xlabel("Neuron Index")
    plt.ylabel("Usage Frequency")
    plt.hist(per_unit_usage)
    plt.savefig("per_unit_usage.png")
    plt.show()


    pca = PCA(n_components=2, svd_solver='arpack')
    W_pca = pca.fit_transform(W_np)
    plt.figure()
    plt.scatter(W_pca[:, 0], W_pca[:, 1], c='dodgerblue', edgecolor='k', alpha=0.7)
    plt.title("Neuron Weight Vectors (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("pca_projection.png")
    plt.show()



    # Get weight matrix from model (change index to whichever layer you want)
    '''
    for i in range(3):
        W = model.encoder_layers[i].WS1.bias.detach().cpu()
        print(np.histogram(W))  # shape: [out_dim, in_dim]
        plt.imshow(W)
        plt.savefig('Weights'+str(i)+'.png')
    '''
    #W = model.encoder_layers[2].WS1.weight.detach().cpu()
    #W_normalized = torch.nn.functional.normalize(W, p=2, dim=1).numpy()  # [out_dim, in_dim]
    

    
    #cosine_sim = torch.matmul(W_normalized, W_normalized.T)  # shape: [out_dim, out_dim]
    #cosine_sim_np = cosine_sim.numpy()

    #mask = np.eye(cosine_sim_np.shape[0], dtype=bool)
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cosine_sim_np, 
        cmap='coolwarm', 
        center=0, 
        mask = mask,
        xticklabels=False, 
        yticklabels=False,
        vmin=-0.2, vmax=0.2,
        cbar_kws={'label': 'Cosine Similarity'})
    plt.title("Pairwise Cosine Similarity Between Neurons")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.tight_layout()
    plt.savefig("cosine_similarity_heatmap.png")
    plt.show()
    '''
    #W_np = W_normalized.numpy()

    #pca = PCA(n_components=2)
    #W_pca = pca.fit_transform(W_np)

    #W = model.encoder_layers[0].WS1.weight.detach().cpu()

    #pca = PCA(n_components=100)
    #pca.fit(W_np)

    #print("Explained variance ratio:", pca.explained_variance_ratio_)
    '''
    plt.bar(range(1,101), pca.explained_variance_ratio_)
    plt.title("Explained variance ratio")
    plt.ylabel("'%' of variance explained")
    plt.xlabel("Top 100 dimensions")
    plt.ylim(0.005, 0.02)
    plt.tight_layout()
    plt.savefig("explained_variance_ratio.png")
    plt.show()
    
    
    plt.figure(figsize=(6, 6))
    plt.scatter(W_pca[:, 0], W_pca[:, 1], c='dodgerblue', edgecolor='k', alpha=0.7)
    plt.title("Neuron Weight Vectors (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig("pca_projection.png")
    plt.show()
    '''

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="exp_020/model_weights/" + model_name + "/epoch_last.pt", help="path for previous model weights, e.g. file.pt") # default = ./exp_020/model_weights/epoch_last.pt
    # ../../../../../../users2/unix/nmukkavilli/ProteinMPNN/sae_training/training/exp_020/model_weights/epoch_last.pt
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for") # default = 200
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
