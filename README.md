# NeRD: a multichannel neural network to predict cellular response of drugs by integrating multidimensional data
## Data

## Source codes
preprocess.py: load data and convert to pytorch format  
train.py: train the model and make predictions  
functions.py: some custom functions  
simles2graph: convert SMILES sequence to graph  
AE.py: dimensionality reduction for ultra-high dimensional features  
NeRD_Net.py: multi-channel neural network model 
## Dependencies 
torcch1.4  
torch_geometric(install torch_cluster, torch_scatter, torch_sparse, torch_spline_conv before installation) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
## Operation steps
