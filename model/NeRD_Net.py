import torch
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import torch.nn as nn


class NeRD_Net(torch.nn.Module):
    def __init__(self, n_filters=4, num_features_xd=78, output_dim=128, dropout=0.5):

        super(NeRD_Net, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # molecular graph
        self.ds_conv1 = GCNConv(num_features_xd, num_features_xd)
        # self.ds_bn1 = nn.BatchNorm1d(num_features_xd)
        self.ds_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        # self.ds_bn2 = nn.BatchNorm1d(num_features_xd * 2)
        self.ds_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        # self.ds_bn3 = nn.BatchNorm1d(num_features_xd * 4)
        self.ds_fc1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.ds_bn4 = nn.BatchNorm1d(1024)
        self.ds_fc2 = torch.nn.Linear(1024, output_dim)
        self.ds_bn5 = nn.BatchNorm1d(output_dim)

        # drug finger
        self.df_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.df_bn1 = nn.BatchNorm1d(n_filters)
        self.df_pool1 = nn.MaxPool1d(3)
        self.df_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.df_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.df_pool2 = nn.MaxPool1d(3)
        self.df_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.df_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.df_pool3 = nn.MaxPool1d(3)
        self.df_fc1 = nn.Linear(464, 512)  # 3712
        self.df_bn4 = nn.BatchNorm1d(512)
        self.df_fc2 = nn.Linear(512, output_dim)  # 2944
        self.df_bn5 = nn.BatchNorm1d(output_dim)

        # miRNA
        self.cm_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.cm_bn1 = nn.BatchNorm1d(n_filters)
        self.cm_pool1 = nn.MaxPool1d(3)
        self.cm_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.cm_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.cm_pool2 = nn.MaxPool1d(3)
        self.cm_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.cm_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.cm_pool3 = nn.MaxPool1d(3)
        self.cm_fc1 = nn.Linear(368, 512)
        self.cm_bn4 = nn.BatchNorm1d(512)
        self.cm_fc2 = nn.Linear(512, output_dim)
        self.cm_bn5 = nn.BatchNorm1d(output_dim)

        # copynumber
        self.cc_fc1 = nn.Linear(256, 1024)
        self.cc_bn1 = nn.BatchNorm1d(1024)
        self.cc_fc2 = nn.Linear(1024, 256)
        self.cc_bn2 = nn.BatchNorm1d(256)
        self.cc_fc3 = nn.Linear(256, output_dim)
        self.cc_bn3 = nn.BatchNorm1d(output_dim)

        # fusion layers
        self.comb_fc1 = nn.Linear(4 * output_dim, 1024)
        self.comb_bn1 = nn.BatchNorm1d(1024)
        self.comb_fc2 = nn.Linear(1024, 128)
        self.comb_bn2 = nn.BatchNorm1d(128)
        self.comb_out = nn.Linear(128, 1)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        miRNA = data.miRNA
        miRNA = miRNA[:, None, :]
        copynumber = data.copynumber
        finger = data.finger
        finger = finger[:, None, :]

        # smiles
        x = self.ds_conv1(x, edge_index)
        x = self.relu(x)
        x = self.ds_conv2(x, edge_index)
        x = self.relu(x)
        x = self.ds_conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.ds_fc1(x)
        x = self.ds_bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ds_fc2(x)
        x = self.ds_bn5(x)
        x = self.dropout(x)

        # drug finger
        xdf = self.df_conv1(finger)
        xdf = self.df_bn1(xdf)
        xdf = self.relu(xdf)
        xdf = self.df_pool1(xdf)
        xdf = self.df_conv2(xdf)
        xdf = self.df_bn2(xdf)
        xdf = self.relu(xdf)
        xdf = self.df_pool2(xdf)
        xdf = self.df_conv3(xdf)
        xdf = self.df_bn3(xdf)
        xdf = self.relu(xdf)
        xdf = self.df_pool3(xdf)
        xdf = xdf.view(-1, xdf.shape[1] * xdf.shape[2])
        xdf = self.df_fc1(xdf)
        xdf = self.df_bn4(xdf)
        xdf = self.relu(xdf)
        xdf = self.dropout(xdf)
        xdf = self.df_fc2(xdf)
        xdf = self.df_bn5(xdf)

        # miRNA
        xcm = self.cm_conv1(miRNA)
        xcm = self.cm_bn1(xcm)
        xcm = self.relu(xcm)
        xcm = self.cm_pool1(xcm)
        xcm = self.cm_conv2(xcm)
        xcm = self.cm_bn2(xcm)
        xcm = self.relu(xcm)
        xcm = self.cm_pool2(xcm)
        xcm = self.cm_conv3(xcm)
        xcm = self.cm_bn3(xcm)
        xcm = self.relu(xcm)
        xcm = self.cm_pool3(xcm)
        xcm = xcm.view(-1, xcm.shape[1] * xcm.shape[2])
        xcm = self.cm_fc1(xcm)
        xcm = self.cm_bn4(xcm)
        xcm = self.cm_fc2(xcm)
        xcm = self.cm_bn5(xcm)

        # cell copynumber
        xcc = self.cc_fc1(copynumber)
        xcc = self.cc_bn1(xcc)
        xcc = self.relu(xcc)
        xcc = self.cc_fc2(xcc)
        xcc = self.cc_bn2(xcc)
        xcc = self.relu(xcc)
        xcc = self.cc_fc3(xcc)
        xcc = self.cc_bn3(xcc)

        # concat
        xfusion = torch.cat((x, xdf, xcm, xcc), 1)

        # fusion
        xfusion = self.comb_fc1(xfusion)
        xfusion = self.comb_bn1(xfusion)
        xfusion = self.relu(xfusion)
        xfusion = self.dropout(xfusion)
        xfusion = self.comb_fc2(xfusion)
        xfusion = self.comb_bn2(xfusion)
        xfusion = self.relu(xfusion)
        xfusion = self.dropout(xfusion)

        out = self.comb_out(xfusion)
        out = self.sigmoid(out)

        return out

