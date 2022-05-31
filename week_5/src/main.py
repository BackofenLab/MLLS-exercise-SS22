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


local_path = "../week_5/"
cancer_names = ["blca", "brca", "coad", "hnsc", "ucec"]

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7998488/


def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


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

    print(y, y.shape)

    print("Saving mapping...")
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping))

    print("replacing gene ids")
    # set number of edges
    links = links[:500]
    # replace gene names with ids
    re_links = links.replace({0: mapping})
    re_links = re_links.replace({1: mapping})
    print(re_links)
    # create data object
    x = x.loc[:, 1:].to_numpy()
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(re_links.to_numpy(), dtype=torch.long)
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())

    #print(gene_features[0])
    #print("create mask...")
    driver_gene_list.extend(passenger_gene_list)
    #tr_mask_drivers = gene_features[0].isin(driver_gene_list)
    #tr_mask_drivers = torch.tensor(tr_mask_drivers, dtype=torch.bool)
    #compact_data.train_mask = tr_mask_drivers
    compact_data.y = y

    print(compact_data)

    # plot original graph
    #print("plot original graph")
    #G = to_networkx(data, to_undirected=True)
    #visualize_graph(G, color=data.y)

    model = GCN()
    print(model)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

    print("Replacing gene ids...")
    all_gene_ids = gene_features.replace({0: mapping})
    #print(all_gene_ids)
    k_folds = 5
    
    driver_ids_list = driver_ids[0].tolist()
    passenger_ids_list = passenger_ids[0].tolist()
    driver_ids_list.extend(passenger_ids_list)
    random.shuffle(driver_ids_list)
    driver_ids_list = np.reshape(driver_ids_list, (len(driver_ids_list), 1))
    #print(driver_ids_list.shape)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for epoch in range(200):
        tr_acc_fold = list()
        te_acc_fold = list()
        for fold, (train_ids, test_ids) in enumerate(kfold.split(driver_ids_list)):
            #print(fold, len(train_ids), driver_ids_list[train_ids])
            #print(fold, len(test_ids), driver_ids_list[test_ids])
            #print("----")
            tr_genes = driver_ids_list[train_ids]
            tr_genes = tr_genes.reshape((tr_genes.shape[0]))

            te_genes = driver_ids_list[test_ids]
            #print(te_genes)
            te_genes = te_genes.reshape((te_genes.shape[0]))
            #print(te_genes[0])
            #print(tr_genes, te_genes, all_gene_ids)
            #print("----")
            tr_mask = all_gene_ids[0].isin(tr_genes)
            te_mask = all_gene_ids[0].isin(te_genes)

            #print(tr_genes[0], tr_genes[1])
            #print(tr_mask[tr_genes[0]], tr_mask[tr_genes[1]])

            #print(te_genes[0], te_genes[1])
            #print(te_mask[te_genes[0]], te_mask[te_genes[1]])
            
            tr_mask = torch.tensor(tr_mask, dtype=torch.bool)
            te_mask = torch.tensor(te_mask, dtype=torch.bool)

            #print(te_mask[])
            print("Setting tr and te masks...")
            compact_data.train_mask = tr_mask
            compact_data.test_mask = te_mask

            print("Training epoch {}, fold {}/{} ...".format(str(epoch), str(fold), str(k_folds)))
            tr_loss, h = train(compact_data, optimizer, model, criterion)
            tr_acc_fold.append(tr_loss)

            # predict on test fold
            model.eval()
            out = model(compact_data.x, compact_data.edge_index)
            #print(out)
            
            pred = out[0].argmax(dim=1)  # Use the class with highest probability.
            #print(pred.shape)
            #print(compact_data.test_mask[te_genes[0]])
            test_correct = pred[compact_data.test_mask] == compact_data.y[compact_data.test_mask]  # Check against ground-truth labels.
            #print(test_correct)
            test_acc = int(test_correct.sum()) / int(compact_data.test_mask.sum())  # Derive ratio of correct predictions.
            print("Epoch: {}, Fold: {}/{}, Train loss: {}, test accuracy: {}".format(str(epoch), str(fold), str(k_folds), str(tr_loss), str(test_acc)))
            te_acc_fold.append(test_acc)
            print()
        if epoch % 20 == 0:
            print("Training Loss after {} epochs: {}".format(str(epoch), str(torch.mean(tr_acc_fold))))
            print("Test Loss after {} epochs: {}".format(str(epoch), str(torch.mean(te_acc_fold))))

    #sys.exit()
    '''
    model.eval()
      out = model(data.x)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    '''


def train(data, optimizer, model, criterion):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


if __name__ == "__main__":
    read_files()
