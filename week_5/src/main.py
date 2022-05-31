import os
import sys
import pandas as pd
import numpy as np
import json
import random
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold


font = {'family': 'serif', 'size': 24}
plt.rc('font', **font)

local_path = "../week_5/"
cancer_names = ["blca", "brca", "coad", "hnsc", "ucec"]


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(32)
        self.conv1 = GCNConv(12, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, 2)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        out = self.classifier(h)
        return out, h


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping


def read_files():
    final_path = cancer_names[0] + "/"
    driver = pd.read_csv(final_path + "drivers", header=None)
    gene_features = pd.read_csv(final_path + "gene_features", header=None)
    links = pd.read_csv(final_path + "links", header=None)
    passengers = pd.read_csv(final_path + "passengers", header=None)

    print(driver)
    print("----")
    print(gene_features)
    print("----")
    print(links)
    print("----")
    print(passengers)
    print("----")

    driver_gene_list = driver[0].tolist()
    passenger_gene_list = passengers[0].tolist()
    
    x, mapping = load_node_csv(final_path + "gene_features", 0)
    y = torch.zeros(x.shape[0], dtype=torch.long)
    # assign all labels to -1
    y[:] = -1
    driver_ids = driver.replace({0: mapping})
    passenger_ids = passengers.replace({0: mapping})
    # driver = 1, passenger = 0
    y[driver_ids[0].tolist()] = 1
    y[passenger_ids[0].tolist()] = 0

    #print(y, y.shape)

    print("Saving mapping...")
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping))

    print("replacing gene ids")
    # set number of edges
    links = links[:5000]
    # replace gene names with ids
    re_links = links.replace({0: mapping})
    re_links = re_links.replace({1: mapping})
    print(re_links)
    # create data object
    x = x.loc[:, 1:].to_numpy()
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(re_links.to_numpy(), dtype=torch.long)
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())

    driver_gene_list.extend(passenger_gene_list)
    compact_data.y = y

    print(compact_data)

    model = GCN()
    print(model)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

    print("Replacing gene ids...")
    all_gene_ids = gene_features.replace({0: mapping})
    k_folds = 5
    
    driver_ids_list = driver_ids[0].tolist()
    passenger_ids_list = passenger_ids[0].tolist()
    driver_ids_list.extend(passenger_ids_list)
    random.shuffle(driver_ids_list)

    driver_ids_list = np.reshape(driver_ids_list, (len(driver_ids_list), 1))
    kfold = KFold(n_splits=k_folds, shuffle=True)
    tr_loss_epo = list()
    te_acc_epo = list()
    n_epo = 500
    for epoch in range(n_epo):
        tr_loss_fold = list()
        te_acc_fold = list()
        for fold, (train_ids, test_ids) in enumerate(kfold.split(driver_ids_list)):

            tr_genes = driver_ids_list[train_ids]
            tr_genes = tr_genes.reshape((tr_genes.shape[0]))

            te_genes = driver_ids_list[test_ids]
            te_genes = te_genes.reshape((te_genes.shape[0]))

            tr_mask = all_gene_ids[0].isin(tr_genes)
            te_mask = all_gene_ids[0].isin(te_genes)

            tr_mask = torch.tensor(tr_mask, dtype=torch.bool)
            te_mask = torch.tensor(te_mask, dtype=torch.bool)

            #print(te_mask[])
            print("Setting tr and te masks...")
            compact_data.train_mask = tr_mask
            compact_data.test_mask = te_mask

            print("Training epoch {}, fold {}/{} ...".format(str(epoch+1), str(fold+1), str(k_folds)))
            tr_loss, h = train(compact_data, optimizer, model, criterion)
            tr_loss_fold.append(tr_loss.detach().numpy())

            # predict on test fold
            model.eval()
            out = model(compact_data.x, compact_data.edge_index)

            pred = out[0].argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[compact_data.test_mask] == compact_data.y[compact_data.test_mask]  # Check against ground-truth labels.
            #print(test_correct)
            test_acc = int(test_correct.sum()) / int(compact_data.test_mask.sum())  # Derive ratio of correct predictions.
            print("Epoch: {}, Fold: {}/{}, Train loss: {}, test accuracy: {}".format(str(epoch+1), str(fold+1), str(k_folds), str(tr_loss), str(test_acc)))
            te_acc_fold.append(test_acc)
            print()
        tr_loss_epo.append(np.mean(tr_loss_fold))
        te_acc_epo.append(np.mean(te_acc_fold))
        if epoch % 20 == 0:
            print("Training Loss after {} epochs: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
            print("Test Loss after {} epochs: {}".format(str(epoch+1), str(np.mean(te_acc_fold))))
    print()
    print("Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("Test Loss after {} epochs: {}".format(str(n_epo), str(np.mean(te_acc_epo))))
    plot_loss_acc(n_epo, tr_loss_epo, te_acc_epo)


def train(data, optimizer, model, criterion):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


def plot_loss_acc(n_epo, tr_loss, te_acc):
    x_val = np.arange(n_epo)
    plt.plot(x_val, tr_loss)
    plt.ylabel("Training loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Training loss vs epochs")
    plt.show()
    
    plt.plot(x_val, te_acc)
    plt.ylabel("Test accuracy")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Test accuracy vs epochs")
    plt.show()
    



if __name__ == "__main__":
    read_files()
