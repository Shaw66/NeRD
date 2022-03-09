from torch_geometric.data import InMemoryDataset
import os
import torch
from torch_geometric import data as DATA
from math import sqrt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='set', xdf=None, xds=None,
                 xcm=None, xcc=None, y=None,
                 transform=None, pre_transform=None, smile_graph=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xdf, xds, xcm, xcc, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xdf, xds, xcm, xcc, y, smile_graph):
        assert (len(xdf) == len(xds) and len(xds) == len(xcm)
                and len(xcm) == len(xcc) and len(xcc) == len(y)), "The five lists must be the same length!"
        data_list = []
        data_len = len(xds)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            finger = xdf[i]
            smiles = xds[i]
            miRNA = xcm[i]
            copynumber = xcc[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            processedData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))

            processedData.miRNA = torch.FloatTensor([miRNA])
            processedData.finger = torch.FloatTensor([finger])
            processedData.copynumber = torch.FloatTensor([copynumber])

            processedData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(processedData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp


def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def draw_loss(train_losses, test_losses, title, path):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(path+".png")  # should before show method


def draw_pearson_r2(pearsons, r2, title, path):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.plot(r2, label='test R²')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson&R²')
    plt.legend()
    # save image
    plt.savefig(path+".png")  # should before show method
