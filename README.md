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
## Step-by-step instructions
1. Unzip the data. Due to the large amount of data, part of the data is compressed when uploading.  
2. Install dependencies, including torch1.4, torch_geometric (you need to install torch_cluster, torch_scatter, torch_sparse, and torch_spline_conv before installation), matplotlib, scipy, sklearn, rdkit, and networkx.  
3. Run AE.py to reduce the dimensionality of the copynumber feature.
4. Run preprocess.py to convert label data and feature data into pytorch format.
5. Run train.py for training and prediction.
## How to use it with other data instances?
1. Process the drug response data you want to use into csv format, with each entry containing the drug id, cell line id, and response value.
2. Organize all cell line IDs and drug IDs into two lists and store them in two csv files respectively.
3. Download the features of the drugs in the list, including SMILES and molecular fingerprints, from …. Then process it into the format of the feature data we uploaded.
4. Download the features of the cell lines in the list, including miRNA and copy number, from …. Then process it into the format of the feature data we uploaded.
5. Run the program as “Step-by-step instructions” (2)-(5).
## Dependencies 
torch1.4  
torch_geometric (install torch_cluster, torch_scatter, torch_sparse, torch_spline_conv before installation) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html  
matplotlib  
scipy  
sklearn  
rdkit  
networkx
