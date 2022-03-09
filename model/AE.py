import csv
import torch.nn as nn
from sklearn import preprocessing
import torch
import torch.utils.data as Data
from scipy.stats import pearsonr
import pickle


EPOCH = 500
BATCH_SIZE = 388
LR = 1e-4


def read_cell_line_list(filename):  # load cell lines and build a dictionary
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    cell_line_dict = {}
    index = 0
    for line in reader:
        cell_line_dict[line[0]] = index
        index += 1
    return cell_line_dict


def read_cell_line_copynumber(filename, cell_line_dict):  # load one of the features of cell line - copynumber
    f = open(filename, 'r')
    reader = csv.reader(f)
    for i in range(5):
        reader.__next__()
    copynumber = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            copynumber[cell_line_dict[line[0]]] = line[1:]
    return copynumber


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(23316, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 23316),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_data():
    cell_line_dict = read_cell_line_list('../data/cell_line/388-cell-line-list.csv')
    copynumber = read_cell_line_copynumber('../data/cell_line/461-23316dim-copynumber.csv', cell_line_dict)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    copynumber = min_max_scaler.fit_transform(copynumber)
    train_data = torch.FloatTensor(copynumber)

    train_size = int(train_data.shape[0] * 0.8)
    data_train = train_data[:train_size]
    data_test = train_data[train_size:]

    return data_train, data_test, train_data


def train(train_data, test_data, data_all):
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    autoencoder = AutoEncoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    best_loss = 1
    res = [[0 for col in range(512)] for row in range(388)]

    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):
            encoded, decoded = autoencoder(data)
            loss = loss_func(decoded, data)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

        test_en, test_de = autoencoder(test_data)
        test_loss = loss_func(test_de, test_data)
        pearson = pearsonr(test_de.view(-1).tolist(), test_data.view(-1))[0]
        if test_loss < best_loss:
            best_loss = test_loss
            res, _ = autoencoder(data_all)
            pickle.dump(res.data.numpy(), open('../data/cell_line/512dim_copynumber.pkl', 'wb'))
            print("best_loss: ", best_loss.data.numpy())
            print("pearson: ", pearson)

    return


if __name__ == "__main__":
    train_data, test_data, all_data = load_data()
    train(train_data, test_data, all_data)
