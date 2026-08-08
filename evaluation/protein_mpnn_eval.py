import argparse
import os.path
import pandas as pd

def main(args):
    import os
    import numpy as np
    import pickle
    import torch
    from torch import manual_seed
    import copy
    import random
    import os.path
    from protein_mpnn_utils import tied_featurize, parse_PDB, create_labels
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
  
    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'

    checkpoint_path = model_folder_path + f'{args.model_name}.pt'
    
    BATCH_COPIES = args.batch_size
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    if args.pdb_path:
        pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
        designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
    else:
        dataset_valid = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length, verbose=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer expansion size of latent space
    size = int(checkpoint['model_state_dict']['sae_layers.0.WS1.weight'].shape[0] / 128)

    model = ProteinMPNN(num_letters=21,
                        node_features=hidden_dim,
                        edge_features=hidden_dim, 
                        hidden_dim=hidden_dim,
                        expansion=size,
                        num_encoder_layers=num_layers, 
                        num_decoder_layers=num_layers, 
                        augment_eps=args.backbone_noise, 
                        k_neighbors=checkpoint['num_edges'])
    model.to(device)    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    protein_name = args.pdb_path[-8:-4]
    model_name = '/'.join(args.path_to_model_weights.split('/')[-1:])
    if args.show_graphs:
        graph_info = [model_name, protein_name, args.SAE_level]
    else:
        graph_info = False
    
    csv_outputs = []
    if args.pdb_path == 'inputs':
        prefix = ''
    elif args.pdb_path == 'idp_inputs':
        prefix = 'idp_'
    elif args.pdb_path == 'input_test':
        prefix = 'test_'
    for i in range(3):
        csv_outputs.append(f'created_data/encodings/{prefix}{args.SAE_level}_{args.csv_output}/output_{args.csv_output}_{i}.pkl')

    # Validation epoch
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, chain_M, chain_encoding_all, chain_M_pos, residue_idx = tied_featurize(batch_clones, device, chain_id_dict)
                
            if args.SAE_level == 'node' and args.return_log_probs != True:
                error, res_labels = create_labels(args.pdb_path, args.SAE_level)
                if error != True:
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    h_V, h_E, original, encoded, decoded, E_idx = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, graph_info, args.return_log_probs, SAE_level=args.SAE_level)
                    if args.show_graphs != True:
                        for i in range(3):
                            mask_for_empty = np.asarray((S[0] != 20).cpu())
                            encoded_ = np.round(model.encoded_act[i].cpu().numpy()[0,:,:], decimals=5)[mask_for_empty]
                            res_df = pd.DataFrame(res_labels, columns = ['identifier'])
                            encoded_df = pd.DataFrame(encoded_, columns = range(1, (128*size + 1)))
                            encoded_df = pd.concat([res_df, encoded_df], axis=1)
                            with open(csv_outputs[i], "ab") as f:
                                pickle.dump(encoded_df, f)
                else:
                    print("Error was had")
                    print(args.pdb_path)
            elif args.SAE_level == 'edge' and args.return_log_probs != True:
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                h_V, h_E, original, encoded, decoded, E_idx = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, None, return_log_probs=False, SAE_level=args.SAE_level)
                name = args.pdb_path[-8:-4]
                labels = []
                for i in range(E_idx.shape[1]):
                    for j in range(E_idx.shape[2]):
                        labels.append(f'{name}_{E_idx[0,i,0]}-{E_idx[0,i,j]}')
                mask_for_empty = (S[0] != 20).cpu()
                labels = np.asarray(labels).reshape(-1, 48)[mask_for_empty, :]
                for i in range(3):
                    encoded = np.round(model.encoded_act[i].cpu().numpy()[0, mask_for_empty, :, :], decimals = 5)
                    encoded = encoded.reshape(-1, 48, model.encoded_act[i].shape[3])
                    mask = np.random.rand(encoded.shape[0]) < 1/48
                    encoded = encoded[mask, :, :]
                    labels_ = labels[mask, :]
                    data = {"encoded": encoded,
                            "labels": labels_}
                    with open(csv_outputs[i], "ab") as f:
                        pickle.dump(data, f)
            elif args.return_log_probs:
                error, res_labels = create_labels(args.pdb_path, args.SAE_level)
                if error != True:
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, args.show_graphs, args.return_log_probs, SAE_level=args.SAE_level)
                    mask_for_empty = np.asarray(S[0] != 20)
                    log_probs_ = np.round(log_probs.numpy()[0,:,:], decimals=5)[mask_for_empty]
                    res_df = pd.DataFrame(res_labels, columns = ['identifier'])
                    log_probs_df = pd.DataFrame(log_probs_, columns = [alphabet[num] for num in range(21)])
                    log_probs_df = pd.concat([res_df, log_probs_df], axis=1)
                    write_header = os.path.getsize(csv_outputs[0]) == 0
                    log_probs_df.to_csv(csv_outputs[0], mode='a', header = write_header, index=False)
                else:
                    print("Error was had")
                    print(args.pdb_path)
            else: # Collect dense encodings
                error, res_labels = create_labels(args.pdb_path, args.SAE_level)
                if error != True:
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    h_V, h_E, original, encoded, decoded, E_idx = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1, SAE_level=args.SAE_level)
                    mask_for_empty = np.asarray(S[0] != 20)
                    for i in range(3):
                        original_ = np.round(model.input_act[i].cpu().numpy()[0,:,:], decimals=5)[mask_for_empty]
                        res_df = pd.DataFrame(res_labels, columns = ['identifier'])
                        original_df = pd.DataFrame(original_, columns = range(1, 129))
                        original_df = pd.concat([res_df, original_df], axis=1)
                        with open(csv_outputs[i], "ab") as f:
                            pickle.dump(original_df, f)
                else:
                    print("Error was had")
                    print(args.pdb_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--SAE_level", type=str, default="node", help="SAE at either node or edge")
    argparser.add_argument("--show_graphs", action="store_true", default=False, help="Display and/or save to png the original, encoded, and decoded heatmaps")
    argparser.add_argument("--return_log_probs", action="store_true", default=False, help="Doesn't store encodings, model only returns log probs")
    argparser.add_argument("--expansion", type=int, default=8, help="Factor by which latent space increases")
    argparser.add_argument("--pdb_path", type=str, default='', help="inputs, idp_inputs, or input_test")
    argparser.add_argument("--csv_output", type=str, default="test")
    argparser.add_argument("--suppress_print", type=int, default=1, help="0 for False, 1 for True")
    argparser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)")   
    argparser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")  
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    args = argparser.parse_args()    
    main(args)   
