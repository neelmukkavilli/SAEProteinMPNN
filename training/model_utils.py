from __future__ import print_function
import numpy as np
import torch
import torch.utils
import torch.utils.checkpoint

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch.nn as nn
import torch.nn.functional as F
import random

def featurize(batch, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        '''
        for step, letter in enumerate(all_chains):
            # ChatGPT edits to skip over broken chains
            seq_key = f'seq_chain_{letter}'
            coords_key = f'coords_chain_{letter}'
            
            # Skip if essential keys are missing
            if seq_key not in b or coords_key not in b:
                print(f"[WARNING] Missing keys for chain {letter} in entry: {b.get('name', 'unknown')}. Skipping chain.")
            else:
                print("fine")
            
            chain_seq = b[seq_key]
            chain_length = len(chain_seq)
            chain_coords = b[coords_key]

            try:
                x_chain = np.stack([
                    chain_coords[f'N_chain_{letter}'],
                    chain_coords[f'CA_chain_{letter}'],
                    chain_coords[f'C_chain_{letter}'],
                    chain_coords[f'O_chain_{letter}']
                ], axis=1)  # [chain_length, 4, 3]
            except KeyError as e:
                print(f"[WARNING] Missing atom coordinate {e} for chain {letter} in entry {b.get('name', 'unknown')}. Skipping chain.")
                continue
        '''
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains:
                #print(b.keys()) 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

## Loss functions
def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed 
    return loss, loss_av

def SAE_loss(model, sparse_weight, mask):
        # Now calculates all losses but only prints the last one
        total_loss = 0
        for i in range(3):
            original, encoded, decoded = model.input_act[i], model.encoded_act[i], model.output_act[i]
            mse_loss = torch.mean(torch.nn.functional.mse_loss(decoded, original, reduction='none')[mask])
            sparse_loss = torch.mean(torch.abs(encoded)[mask])
            total_loss += (sparse_weight * sparse_loss) + mse_loss
        return total_loss, sparse_loss.detach(), mse_loss.detach()

# KL sparse loss as alternative to L1 loss
def KL_divergence(rho, encoded, device):
        rho_hat = torch.mean(F.sigmoid(encoded), dim=2)
        rho = torch.full(rho_hat.shape, rho).to(device)
        kl = torch.sum(rho * torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat)))
        return kl

# Penalizes non-orthogonal vectors, not currently used
def orthogonality_loss(act):
        orth = 0
        for i in range(act.shape[0]):
            act_n = torch.nn.functional.normalize(act[i, : :], dim=0)
            AT_A = torch.matmul(act_n, act_n.T)
            identity = torch.eye(AT_A.shape[0], device=act.device)
            orth += torch.norm(AT_A - identity, p='fro')**2

        return orth / act.shape[0]

## The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def remove_parallel_grads(weight):
    for i in range(3):
        if weight[i].WS2.weight.grad != None:
                W = weight[i].WS2.weight
                W_normed = W / W.norm(dim=0, keepdim=True)
                proj = (W.grad * W_normed).sum(dim=0, keepdim=True) * W_normed
                W.grad -= proj
        else:
            print("grad = None")
        
## See Anthropic Neuron Resampling procedure in "Towards Monosemanticity..."
# also see https://github.com/shehper/sparse-dictionary-learning/blob/main/autoencoder/autoencoder.py for code
def store_inputs_and_losses(count, reservoir_inputs, reservoir_losses, reservoir_size, original, encoded, decoded, mask, chain_M, sparse_weight, expansion):
    '''
    mask_for_loss = torch.reshape((mask * chain_M).detach().bool(), (-1, 1))
    # Package [B, L, D] or [B, L, K, D] -> [B*L, D] or [B*L*K, D]
    input = torch.reshape(original.detach(), (-1, 128)) # Size of dense encoded dimensions
    encoded = torch.reshape(encoded.detach(), (-1, 1024)) # 8x hidden dim
    output = torch.reshape(decoded.detach(), (-1, 128))
    '''
    mask_for_loss = (mask*chain_M).detach().bool()
    input = original.detach()
    encoded = encoded.detach()
    output = decoded.detach()
    
    batch_loss = per_sample_SAE_loss(input, encoded, output, sparse_weight, mask_for_loss, expansion)

    for i in range(batch_loss.shape[0]): # B*L or B*L*K
        count += 1
        if len(reservoir_inputs) < reservoir_size:
            reservoir_inputs.append(torch.reshape(input, (-1, 128))[i, :].cpu()) # Reshape [B, L, D] -> [B*L, D] or [B, L, K, D] -> [B*L*K, D]
            reservoir_losses.append(batch_loss[i].cpu())
        else:
            j = random.randint(0, count-1)
            if j < reservoir_size:
                reservoir_inputs[j] = torch.reshape(input, (-1, 128))[i, :].cpu()
                reservoir_losses[j] = batch_loss[i].cpu()
    return reservoir_inputs, reservoir_losses

def per_sample_SAE_loss(input, encoded, output, sparse_weight, mask, expansion):
    #mse_loss = torch.mean(torch.nn.functional.mse_loss(output, input, reduction='none')[mask.squeeze()], dim=1)#.detach()
    #sparse_loss = torch.mean(torch.abs(encoded)[mask.squeeze()], dim=1)#.detach()

    # Calculates loss of shape [B, L, K, D] -> mask applied -> Reshaped to [B*L*K, D] -> mean loss for each sample -> [B*L*K] 
    mse_loss = torch.mean(torch.reshape((torch.nn.functional.mse_loss(output, input, reduction='none')[mask]), (-1, 128)), dim=1)
    sparse_loss = torch.mean(torch.reshape((torch.abs(encoded)[mask]), (-1, 128*expansion)), dim=1)
    return mse_loss + sparse_weight * sparse_loss

def reinit_anthropic(model, optimizer, reservoir_inputs, reservoir_losses, alive_neurons, dead_neurons, device):
    inputs = torch.stack(reservoir_inputs)
    losses = torch.stack(reservoir_losses)

    resampling_indices = sample_inputs_from_losses(inputs, losses, dead_neurons, device)
    adjust_weights(model, alive_neurons, dead_neurons, resampling_indices)
    reset_optimizer(optimizer, dead_neurons)

def sample_inputs_from_losses(inputs, losses, dead_neurons, device):
    probs = losses ** 2
    probs /= probs.sum()
    idx = torch.multinomial(probs, num_samples=len(dead_neurons), replacement=True)
    resampling_indices = inputs[idx].to(device)
    return resampling_indices

def adjust_weights(model, alive_neurons, dead_neurons, resampling_indices):
    # Adjusts the 3rd SAE section since encoding step is repeated 3 times

    # Find average "length" of encoding vector
    avg_enc_norm = torch.linalg.vector_norm(model.sae_layers[2].WS1.weight[alive_neurons], dim=1).mean()
    # Normalize inputs to unit L2 (make vector length=1)
    examples_unit_norm = F.normalize(resampling_indices, dim=1) * avg_enc_norm * 1e-3
    # Set decoder weights to dictionary vector (inputs)
    model.sae_layers[2].WS2.weight[:, dead_neurons] = examples_unit_norm.T
    # Multiply inputs by average encoded length and 0.2 to make them weakly activate
    adjusted_examples = examples_unit_norm * avg_enc_norm * 0.2
    # Set encoder weights and set encoder biases to 0
    model.sae_layers[2].WS1.weight[dead_neurons, :] = adjusted_examples
    model.sae_layers[2].WS1.bias[dead_neurons] = 0

def reset_optimizer(optimizer, dead_neurons, device):
    for i, param in enumerate(optimizer.param_groups[0]['params']):
        param_state = optimizer.state[param]
        if i in [0, 1]:
            param_state['exp_avg'][dead_neurons] = 0
            param_state['exp_avg_sq'][dead_neurons] = 0
        elif i == 2:
            param_state['exp_avg'][:, dead_neurons] = 0
            param_state['exp_avg_sq'][:, dead_neurons] = 0

def reinit_classic(model, layer, optimizer, dead_neurons, device, fraction_reinit=1):
    num_to_reinit = max(1, int(fraction_reinit * len(dead_neurons)))

    indices = dead_neurons[torch.randperm(len(dead_neurons))[:num_to_reinit]]

    WS1 = model.sae_layers[layer].WS1
    WS2 = model.sae_layers[layer].WS2
    idx = indices.to(WS1.weight.device)

    # 1. Generate full random weight tensors (same shape)
    new_W1 = torch.empty_like(WS1.weight)
    new_W2 = torch.empty_like(WS2.weight)

    nn.init.kaiming_uniform_(new_W1)
    nn.init.kaiming_uniform_(new_W2)

    # 2. Replace only the dead neurons
    WS1.weight[idx, :] = new_W1[idx, :]
    WS2.weight[:, idx] = new_W2[:, idx]

    # 3. Reset biases
    WS1.bias[idx] = 0
    #WS2.bias[idx] = 0

    '''
    WS1_bias, WS1_weight, WS2_weight = model.sae_layers[2].WS1.bias.data, model.sae_layers[2].WS1.weight.data, model.sae_layers[2].WS2.weight.data
    #WS1_bias[indices, :] = 0
    nn.init.kaiming_uniform_(WS1_weight[indices, :])
    nn.init.kaiming_uniform_(WS2_weight[:, indices])
    WS1_bias[indices] = 0
    #WS2_bias[indices] = 0
    '''
    reset_optimizer(optimizer, indices, device)
        
## Define ProteinMPNN
class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, SAE_level='node', reinsert_SAE=False):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        
        # Node Layers
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        '''
        # SAE Layers
        self.SAE_act = nn.ReLU()
        self.WS1 = nn.Linear(num_hidden, num_hidden*8, bias=True) # Dense encodings expanded by 8 times i.e. [B, L, 128] -> [B, L, 1024]
        self.WS2 = nn.Linear(num_hidden*8, num_hidden, bias=True)

        nn.init.kaiming_uniform_(self.WS1.weight)
        nn.init.kaiming_uniform_(self.WS2.weight)

        self.normalize_decoder()
        '''

        # Edge Layers
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
    '''
    def normalize_decoder(self):
        with torch.no_grad():
            W = self.WS2.weight
            W /= W.norm(dim=0, keepdim=True)
    '''
    def forward(self, h_V, h_E, E_idx, SAE_level='node', reinsert_SAE=False, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV))))).detach()

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh)).detach() # 
        if mask_V is not None:
             mask_V = mask_V.unsqueeze(-1)
             h_V = mask_V * h_V

        '''
        ## Node SAE
        # SAE is always applied after norm and dropout layers which will change if we want to probe previous MLP layers
        if SAE_level == 'node':
            encoded = self.SAE_act(self.WS1(h_V - self.WS2.bias))
            if reinsert_SAE == True: # Use to test accuracy of encoding space, otherwise not necessary
                h_V, original = self.WS2(encoded), h_V
                decoded = h_V
            else:
                decoded = self.WS2(encoded)
        '''
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message)).detach()
        
        '''
        ## Edge SAE
        if SAE_level == 'edge':
            encoded = self.SAE_act(self.WS1(h_E - self.WS2.bias))
            if reinsert_SAE == True:
                h_E, original = self.WS2(encoded), h_E
                decoded = h_E
            else:
                decoded = self.WS2(encoded)
        '''
        return h_V, h_E#, original, encoded, decoded

class SAELayer(nn.Module):
    def __init__(self, num_in, expansion=8):
        super(SAELayer, self).__init__()
        self.num_hidden = num_in
        self.expansion = expansion

        self.WS1 = nn.Linear(num_in, num_in*self.expansion, bias=True) # Dense encodings expanded by 8 times i.e. [B, L, 128] -> [B, L, 1024]
        self.WS2 = nn.Linear(num_in*self.expansion, num_in, bias=True)
        self.act = nn.ReLU()
        nn.init.kaiming_uniform_(self.WS1.weight)
        nn.init.kaiming_uniform_(self.WS2.weight)

        #self.normalize_decoder()

    # Make sure decoder weights columns are unit norm, remove other gradients
    def normalize_decoder(self):
        with torch.no_grad():
            W = self.WS2.weight
            W  = W / W.norm(dim=0, keepdim=True)
            self.WS2.weight.copy_(W)

    def forward(self, X):
        encoded = self.act(self.WS1(X - self.WS2.bias)) # Use pre-encoder bias tied to second set of weights
        decoded = self.WS2(encoded)
    
        return encoded, decoded

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1, expansion=8):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.sae_layers = nn.ModuleList([
            SAELayer(hidden_dim, expansion)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
    
    def sae_grad_calc(self, log_probs, S, encoded, mask, chain_M):
        grad_avgs = torch.zeros_like(encoded)
        grad_avgs = np.round(grad_avgs.cpu().data.numpy(), 3)
        mask_for_loss = (mask * chain_M)
        for r in range(log_probs.shape[1]):
            S = S.clone()
            S[:, r] = 0
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            encoded.retain_grad()
            loss_av.backward(retain_graph=True)
            grad_avgs = np.round(grad_avgs, 3)
            grad_avgs = grad_avgs + (np.round(encoded.grad.cpu().data.numpy(), 3) / (log_probs.shape[1]*log_probs.shape[0]))
            encoded.grad = None
        print(grad_avgs)
        print(np.min(grad_avgs))
        print(np.max(grad_avgs))

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, SAE_level, reinsert_SAE=False):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        if True:
            E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
            h_E = self.W_e(E)
            # Encoder is unmasked self-attention
            mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
        
            self.input_act= []
            self.encoded_act = []
            self.output_act = []
            for layer in self.encoder_layers:
                #h_V, h_E, original, encoded, decoded = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, SAE_level, reinsert_SAE, mask, mask_attend, use_reentrant=True)
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                    #h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=True)
                    if SAE_level == 'edge':
                        self.input_act.append(h_E)#torch.reshape(h_E, (1, -1, h_E.shape[3])))
                    else:
                        self.input_act.append(h_V)

        for idx, layer in enumerate(self.sae_layers):
            encoded, decoded = layer(self.input_act[idx])
            self.encoded_act.append(encoded)
            self.output_act.append(decoded)
        
        if reinsert_SAE == True:
            if SAE_level == 'node':
                h_V = self.output_act[2]
            elif SAE_level == 'edge':
                h_E = self.output_act[2]
        
        if True: # Don't collect gradients for decoder
            # Concatenate sequence embeddings for autoregressive decoder

            h_S = self.W_s(S)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            chain_M = chain_M*mask #update chain_M to include missing regions
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder

            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask, use_reentrant=True)
            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)

        #self.sae_grad_calc(log_probs, S, encoded, mask, chain_M)
            
        return log_probs, self.input_act[2], self.encoded_act[2], self.output_act[2]
        

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
