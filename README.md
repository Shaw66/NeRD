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
torch1.4  
torch_geometric (install torch_cluster, torch_scatter, torch_sparse, torch_spline_conv before installation) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html  
matplotlib  
scipy  
sklearn  
rdkit  
networkx
## Operation steps
1. Run AE.py to reduce the dimensionality of the copynumber feature.  
2. Run preprocess.py to convert label data and feature data into pytorch format.  
3. Run train.py for training and prediction.
