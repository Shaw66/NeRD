# NeRD
A multichannel neural network to predict cellular response of drugs by integrating multidimensional data
## Data
1. drug_response_data: drug response data from the PRISM database  
2. 388-cell-line-list: list of cell lines we use  
3. 470-734dim-miRNA: feature of miRNA  
4. 461-23316dim-copynumber: feature of copy number  
5. 1448-269dim-physicochemical: list of drugs and physicochemical properties  
6. 1448-881dim-fingerprint: feature of molecular fingerprint  
7. drug_smiles：SMILES (Simplified molecular input line entry system) of drugs 
## Source codes
1. preprocess.py: load data and convert to pytorch format  
2. train.py: train the model and make predictions  
3. functions.py: some custom functions  
4. simles2graph.py: convert SMILES sequence to graph  
5. AE.py: dimensionality reduction for ultra-high dimensional features  
6. NeRD_Net.py: multi-channel neural network model 
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

