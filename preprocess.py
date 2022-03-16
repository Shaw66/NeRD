import csv
from smiles2graph import smile_to_graph
import pickle
from sklearn import preprocessing
import random
import numpy as np
from functions import TestbedDataset


def read_drug_list_and_properties(filename):  # load drugs and their physicochemical properties from files
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_properties = []
    drug_dict = {}
    index = 0
    for line in reader:
        drug_properties.append(line[1:])
        drug_dict[line[0]] = index  # build a dictionary to save the index of samples
        index += 1
    return drug_dict, drug_properties


def read_drug_finger(filename, drug_dict):  # load drugs' molecular fingerprints
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_finger = [list() for i in range(len(drug_dict))]
    for line in reader:
        drug_finger[drug_dict[line[0]]] = list(map(int, line[1:]))  # use the index in dictionary
    return drug_finger


def read_drug_smiles(filename, drug_dict):  # load drugs' SMILES
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_smiles = [list() for i in range(len(drug_dict))]
    for line in reader:
        drug_smiles[drug_dict[line[0]]] = line[1]  # use the index in dictionary
    return drug_smiles


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


def read_cell_line_miRNA(filename, cell_line_dict):  # load one of the features of cell line - miRNA
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    miRNA = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            miRNA[cell_line_dict[line[0]]] = line[1:]
    return miRNA


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


def min_max_nomalization(list, min, max):
    res = []
    for item in list:
        temp = (item - min) / (max - min)
        res.append(temp)
    return res


def get_all_graph(drug_smiles):
    smile_graph = {}
    for smile in drug_smiles:
        if len(smile) > 0:
            graph = smile_to_graph(smile)
            smile_graph[smile] = graph

    return smile_graph


def read_response_data_and_process(filename):
    # load features
    drug_dict, drug_properties = read_drug_list_and_properties('data/drug/1448-269dim-physicochemical.csv')
    finger = read_drug_finger('data/drug/1448-881dim-fingerprint.csv', drug_dict)
    smile = read_drug_smiles('data/drug/drug_smiles.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('data/cell_line/388-cell-line-list.csv')
    miRNA = read_cell_line_miRNA('data/cell_line/470-734dim-miRNA.csv', cell_line_dict)
    copynumber = pickle.load(open('data/cell_line/256dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE

    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    miRNA = min_max_scaler.fit_transform(miRNA)
    copynumber = min_max_scaler.fit_transform(copynumber)

    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[3]
        cell_line = line[0]
        ic50 = float(line[7])
        data.append((drug, cell_line, ic50))
    random.shuffle(data)

    # match features and labels
    drug_smile = []
    drug_finger = []
    cell_miRNA = []
    cell_copy = []
    label = []
    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            drug_smile.append(smile[drug_dict[drug]])
            drug_finger.append(finger[drug_dict[drug]])
            cell_miRNA.append(miRNA[cell_line_dict[cell_line]])
            cell_copy.append(copynumber[cell_line_dict[cell_line]])
            label.append(ic50)
    label = min_max_nomalization(label, min(label), max(label))

    # split data
    drug_smile, drug_finger = np.asarray(drug_smile), np.asarray(drug_finger)
    cell_miRNA, cell_copy = np.asarray(cell_miRNA), np.asarray(cell_copy)
    label = np.asarray(label)

    for i in range(5):
        total_size = drug_smile.shape[0]
        size_0 = int(total_size * 0.2 * i)
        size_1 = size_0 + int(total_size * 0.1)
        size_2 = int(total_size * 0.2 * (i + 1))
        # features of drug fingers
        drugfinger_train = drug_finger[size_0:size_1]
        drugfinger_val = drug_finger[size_1:size_2]
        drugfinger_test = np.concatenate((drug_finger[:size_0], drug_finger[size_2:]), axis=0)
        # features of drug smiles
        drugsmile_train = drug_smile[size_0:size_1]
        drugsmile_val = drug_smile[size_1:size_2]
        drugsmile_test = np.concatenate((drug_smile[:size_0], drug_smile[size_2:]), axis=0)
        # features of cell miRNA
        cellmiRNA_train = cell_miRNA[size_0:size_1]
        cellmiRNA_val = cell_miRNA[size_1:size_2]
        cellmiRNA_test = np.concatenate((cell_miRNA[:size_0], cell_miRNA[size_2:]), axis=0)
        # features of cell copynumber
        cellcopy_train = cell_copy[size_0:size_1]
        cellcopy_val = cell_copy[size_1:size_2]
        cellcopy_test = np.concatenate((cell_copy[:size_0], cell_copy[size_2:]), axis=0)
        # label
        label_train = label[size_0:size_1]
        label_val = label[size_1:size_2]
        label_test = np.concatenate((label[:size_0], label[size_2:]), axis=0)

        TestbedDataset(root='data', dataset='train_set{num}'.format(num=i), xdf=drugfinger_train,
                       xds=drugsmile_train,
                       xcm=cellmiRNA_train, xcc=cellcopy_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_set{num}'.format(num=i), xdf=drugfinger_val, xds=drugsmile_val,
                       xcm=cellmiRNA_val, xcc=cellcopy_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_set{num}'.format(num=i), xdf=drugfinger_test, xds=drugsmile_test,
                       xcm=cellmiRNA_test, xcc=cellcopy_test,
                       y=label_test, smile_graph=smile_graph)

    return


def process_blind_cell(filename):
    # load features
    drug_dict, drug_properties = read_drug_list_and_properties('data/drug/1448-269dim-physicochemical.csv')
    finger = read_drug_finger('data/drug/1448-881dim-fingerprint.csv', drug_dict)
    smile = read_drug_smiles('data/drug/drug_smiles.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('data/cell_line/388-cell-line-list.csv')
    miRNA = read_cell_line_miRNA('data/cell_line/470-734dim-miRNA.csv', cell_line_dict)
    copynumber = pickle.load(open('data/cell_line/256dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE

    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    miRNA = min_max_scaler.fit_transform(miRNA)
    copynumber = min_max_scaler.fit_transform(copynumber)

    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[3]
        cell_line = line[0]
        ic50 = float(line[8])
        data.append((drug, cell_line, ic50))
    random.shuffle(data)

    dict_drug_cell = {}

    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            if cell_line in dict_drug_cell:
                dict_drug_cell[cell_line].append((drug, ic50))
            else:
                dict_drug_cell[cell_line] = [(drug, ic50)]

    for i in range(5):
        total_size = len(dict_drug_cell)
        size = int(total_size * i * 0.2)
        size1 = size + int(total_size * 0.1)
        size2 = int(total_size * (i + 1) * 0.2)

        drugsmile_train = []
        drugfinger_train = []
        cellmiRNA_train = []
        cellcopy_train = []
        label_train = []

        drugsmile_val = []
        drugfinger_val = []
        cellmiRNA_val = []
        cellcopy_val = []
        label_val = []

        drugsmile_test = []
        drugfinger_test = []
        cellmiRNA_test = []
        cellcopy_test = []
        label_test = []

        pos = 0
        for cell, values in dict_drug_cell.items():
            pos += 1
            for v in values:
                drug, ic50 = v
                if pos >= size and pos < size1:
                    drugsmile_test.append(smile[drug_dict[drug]])
                    drugfinger_test.append(finger[drug_dict[drug]])
                    cellmiRNA_test.append(miRNA[cell_line_dict[cell]])
                    cellcopy_test.append(copynumber[cell_line_dict[cell]])
                    label_test.append(ic50)
                elif pos >= size1 and pos < size2:
                    drugsmile_val.append(smile[drug_dict[drug]])
                    drugfinger_val.append(finger[drug_dict[drug]])
                    cellmiRNA_val.append(miRNA[cell_line_dict[cell]])
                    cellcopy_val.append(copynumber[cell_line_dict[cell]])
                    label_val.append(ic50)
                else:
                    drugsmile_train.append(smile[drug_dict[drug]])
                    drugfinger_train.append(finger[drug_dict[drug]])
                    cellmiRNA_train.append(miRNA[cell_line_dict[cell]])
                    cellcopy_train.append(copynumber[cell_line_dict[cell]])
                    label_train.append(ic50)

        drugsmile_train, drugfinger_train = np.asarray(drugsmile_train), np.asarray(drugfinger_train)
        cellmiRNA_train, cellcopy_train = np.asarray(cellmiRNA_train), np.asarray(cellcopy_train)
        label_train = np.asarray(label_train)

        drugsmile_val, drugfinger_val = np.asarray(drugsmile_val), np.asarray(drugfinger_val)
        cellmiRNA_val, cellcopy_val = np.asarray(cellmiRNA_val), np.asarray(cellcopy_val)
        label_val = np.asarray(label_val)

        drugsmile_test, drugfinger_test = np.asarray(drugsmile_test), np.asarray(drugfinger_test)
        cellmiRNA_test, cellcopy_test = np.asarray(cellmiRNA_test), np.asarray(cellcopy_test)
        label_test = np.asarray(label_test)

        TestbedDataset(root='data', dataset='train_blind_cell{num}'.format(num=i), xdf=drugfinger_train,
                       xds=drugsmile_train,
                       xcm=cellmiRNA_train, xcc=cellcopy_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_blind_cell{num}'.format(num=i), xdf=drugfinger_val, xds=drugsmile_val,
                       xcm=cellmiRNA_val, xcc=cellcopy_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_blind_cell{num}'.format(num=i), xdf=drugfinger_test, xds=drugsmile_test,
                       xcm=cellmiRNA_test, xcc=cellcopy_test,
                       y=label_test, smile_graph=smile_graph)

    return


def process_blind_drug(filename):
    # load features
    drug_dict, drug_properties = read_drug_list_and_properties('data/drug/1448-269dim-physicochemical.csv')
    finger = read_drug_finger('data/drug/1448-881dim-fingerprint.csv', drug_dict)
    smile = read_drug_smiles('data/drug/drug_smiles.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('data/cell_line/388-cell-line-list.csv')
    miRNA = read_cell_line_miRNA('data/cell_line/470-734dim-miRNA.csv', cell_line_dict)
    copynumber = pickle.load(open('data/cell_line/256dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE

    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    miRNA = min_max_scaler.fit_transform(miRNA)
    copynumber = min_max_scaler.fit_transform(copynumber)

    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[3]
        cell_line = line[0]
        ic50 = float(line[8])
        data.append((drug, cell_line, ic50))
    random.shuffle(data)

    dict_drug_cell = {}

    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell_line, ic50))
            else:
                dict_drug_cell[drug] = [(cell_line, ic50)]

    for i in range(5):
        total_size = len(dict_drug_cell)
        size = int(total_size * i * 0.2)
        size1 = size + int(total_size * 0.1)
        size2 = int(total_size * (i + 1) * 0.2)

        drugsmile_train = []
        drugfinger_train = []
        cellmiRNA_train = []
        cellcopy_train = []
        label_train = []

        drugsmile_val = []
        drugfinger_val = []
        cellmiRNA_val = []
        cellcopy_val = []
        label_val = []

        drugsmile_test = []
        drugfinger_test = []
        cellmiRNA_test = []
        cellcopy_test = []
        label_test = []

        pos = 0
        for drug, values in dict_drug_cell.items():
            pos += 1
            for v in values:
                cell, ic50 = v
                if pos >= size and pos < size1:
                    drugsmile_test.append(smile[drug_dict[drug]])
                    drugfinger_test.append(finger[drug_dict[drug]])
                    cellmiRNA_test.append(miRNA[cell_line_dict[cell]])
                    cellcopy_test.append(copynumber[cell_line_dict[cell]])
                    label_test.append(ic50)
                elif pos >= size1 and pos < size2:
                    drugsmile_val.append(smile[drug_dict[drug]])
                    drugfinger_val.append(finger[drug_dict[drug]])
                    cellmiRNA_val.append(miRNA[cell_line_dict[cell]])
                    cellcopy_val.append(copynumber[cell_line_dict[cell]])
                    label_val.append(ic50)
                else:
                    drugsmile_train.append(smile[drug_dict[drug]])
                    drugfinger_train.append(finger[drug_dict[drug]])
                    cellmiRNA_train.append(miRNA[cell_line_dict[cell]])
                    cellcopy_train.append(copynumber[cell_line_dict[cell]])
                    label_train.append(ic50)

        drugsmile_train, drugfinger_train = np.asarray(drugsmile_train), np.asarray(drugfinger_train)
        cellmiRNA_train, cellcopy_train = np.asarray(cellmiRNA_train), np.asarray(cellcopy_train)
        label_train = np.asarray(label_train)

        drugsmile_val, drugfinger_val = np.asarray(drugsmile_val), np.asarray(drugfinger_val)
        cellmiRNA_val, cellcopy_val = np.asarray(cellmiRNA_val), np.asarray(cellcopy_val)
        label_val = np.asarray(label_val)

        drugsmile_test, drugfinger_test = np.asarray(drugsmile_test), np.asarray(drugfinger_test)
        cellmiRNA_test, cellcopy_test = np.asarray(cellmiRNA_test), np.asarray(cellcopy_test)
        label_test = np.asarray(label_test)

        TestbedDataset(root='data', dataset='train_blind_drug{num}'.format(num=i), xdf=drugfinger_train,
                       xds=drugsmile_train,
                       xcm=cellmiRNA_train, xcc=cellcopy_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_blind_drug{num}'.format(num=i), xdf=drugfinger_val, xds=drugsmile_val,
                       xcm=cellmiRNA_val, xcc=cellcopy_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_blind_drug{num}'.format(num=i), xdf=drugfinger_test, xds=drugsmile_test,
                       xcm=cellmiRNA_test, xcc=cellcopy_test,
                       y=label_test, smile_graph=smile_graph)

    return


if __name__ == "__main__":
    read_response_data_and_process('data/label/drug_response_data.csv')
    # process_blind_cell('data/label/drug_response_data.csv')
    # process_blind_drug('data/label/drug_response_data.csv')
