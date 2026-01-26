# SAEProteinMPNN

![Sparse Autoencoder graphic showing heatmaps of original, encoded, and decoded latent spaces](/analysis/graphics/SAE_graphic.png)

## Organization:

The code for SAEProteinMPNN is organized into four folders:

1. /training:
First the model is trained using the pretrained ProteinMPNN weights with the new SAE layers

2. /evaluation:
Then the trained model and be tested and the neurons labeled

3. /steering:
Finally the SAEProteinMPNN can be used to build new sequences

4. /analysis:
A supplementary folder for analyzing Molecular Dynamics simulations and generating other figures

## Sparse Autoencoders:

Sparse Autoencoders (SAEs) are a simple but powerful method for converting compact, dense latent spaces into larger, sparse latent spaces which can be more easily interpreted. While originally designed for large language models (LLMs), we find a new use for them in creaing interpretable protein design models which similarly rely on graph based neural networks. 

The trained model creates sparse representations (sometimes referred to as features) of dense activations which can then be labeled based on their correlations to the true features in the evaluation phase.

Our SAE uses 2 loss functions: MSE Reconstruction Loss (L2) and Sparse Loss (L1 Regularization)

**MSE Reconstruction Loss**\
MSE Loss = $\sum_{i}^{j}{(x_{i}-\hat{x}_i)^2}$

**Sparse Loss**\
Sparse Loss = $\sum_{i}^{k}\left|z_i \right|$

- k = # of neurons in the sparse layer
- j = # of neurons in the orginal and decoded layers
- x = original activations
- z = sparse encoded activations
- $\hat{x}$ = decoded activations

## SAE Architecture

ProteinMPNN is a message passing graphical neural network. The graph consists of vectors storing information of nodes (residues) and edges (connections between residues). Information from the edges is consolidated into a "message", passed through three MLP layers, and used to update the nodes. Information from nodes similarly updates the edges. There are three rounds of message passing and after each round, the information is said to be stored in a "latent space" These are the dense inputs used for the SAE layers placed after each round of encodings.

![ProteinMPNN encoding scheme for node and edge activations](/analysis/graphics/ProtMPNN_graphic.png)

The outputs of the SAE (the decoded dense activations) are not fed back into the model except if 'args.reinsert_SAE == True', in which case the last SAE's output is used as the input for the ProteinMPNN decoder.

The SAE encoded latent space is by default, 8 times larger than the input latent space.

The SAE is defined as follows:

**SAE Encoder**\
$z_i = ReLU(W_1\cdot(x_i - b_2) +b_1)$

**SAE Decoder**\
$\hat{x_i} = W_2 \cdot z_i + b_2$


The input is first substracted by a pre-encoder bias which is tied to the second layer's bias as suggested by [Anthropic](https://transformer-circuits.pub/2023/monosemantic-features).
