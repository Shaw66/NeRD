# NeRD: a multichannel neural network to predict cellular response of drugs by integrating multidimensional data
## Data

## Source codes
1.preprocess.py: load data and convert to pytorch format  
2.train.py: train the model and make predictions  
3.functions.py: some custom functions  
4.simles2graph: convert SMILES sequence to graph  
5.AE.py: dimensionality reduction for ultra-high dimensional features  
6.NeRD_Net.py: multi-channel neural network model 
## Operation steps
1. Run AE.py to reduce the dimensionality of the copynumber feature.  
2. Run preprocess.py to convert label data and feature data into pytorch format.  
3. Run train.py for training and prediction.
## Dependencies 
torch1.4  
torch_geometric (install torch_cluster, torch_scatter, torch_sparse, torch_spline_conv before installation) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html  
matplotlib  
scipy  
sklearn  
rdkit  
networkx

