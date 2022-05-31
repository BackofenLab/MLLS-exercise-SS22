import os
import sys
import pandas as pd
import json
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


local_path = "../week_5/"
cancer_names = ["blca", "brca", "coad", "hnsc", "ucec"]

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7998488/


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
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
    #print(x)
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
    #print(driver_gene_list)
    passenger_gene_list = passengers[0].tolist()
    

    x, mapping = load_node_csv(final_path + "gene_features", 0)
    y = torch.zeros(x.shape[0], dtype=torch.long)
    

    driver_ids = driver.replace({0: mapping})
    passenger_ids = passengers.replace({0: mapping})

    '''print(driver_ids, driver_ids[0].tolist())
    print()
    print(passenger_ids, passenger_ids[0].tolist())'''

    y[driver_ids[0].tolist()] = 1
    y[passenger_ids[0].tolist()] = 0
    #print(y[:, 737])
    #print(y[:, 13255])
    print(y, y.shape)

    #sys.exit()
    # driver = 1, passenger = 0

    '''driver_mask = gene_features[gene_features[0].isin(driver_gene_list)]
    #print(driver_mask)
    driver_y = [1 for item, i in driver_mask.iterrows()]
    #print("----")
    passenger_mask = gene_features[gene_features[0].isin(passenger_gene_list)]
    passenger_y = [0 for item, i in passenger_mask.iterrows()]
    #print(passenger_mask)
    #print("----")
    driver_y.extend(passenger_y)
    #print("---------------")

    y = torch.tensor(driver_y, dtype=torch.float)'''

    print("Saving mapping...")
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping))

    print("replacing gene ids")
    #links = links[:50000]
    re_links = links.replace({0: mapping})
    re_links = re_links.replace({1: mapping})
    print(re_links)

    '''driver_ids = driver.replace({0: mapping})
    passenger_ids = passengers.replace({0: mapping})

    #print(driver_ids, driver_ids[0].tolist())
    #print()
    #print(passenger_ids, passenger_ids[0].tolist())

    #print(re_links)
    re_links = re_links[re_links[0].isin(driver_ids[0].tolist()) & re_links[1].isin(passenger_ids[0].tolist())]
    print(re_links)'''
    #sys.exit()

    links_mat = re_links.to_numpy()

    # create data object
    edge_index = torch.tensor(links_mat, dtype=torch.long) 
    '''combined_tr = pd.concat([driver_mask, passenger_mask])
    print(combined_tr)
    combined_tr = combined_tr.loc[:, 1:]
    print(combined_tr)
    combined_tr = combined_tr.to_numpy()'''
    x = x.loc[:, 1:]
    x = x.to_numpy()
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())

    #print(gene_features[0])
    print("create mask...")
    #tr_mask_drivers = gene_features[0].isin(total_gene_list) #gene_features[0].isin(driver_gene_list) | gene_features[0].isin(passenger_gene_list)
    #tr_mask_drivers = torch.tensor(tr_mask_drivers, dtype=torch.bool)
    driver_gene_list.extend(passenger_gene_list)
    tr_mask_drivers = gene_features[0].isin(driver_gene_list) #gene_features[0].isin(driver_gene_list) | gene_features[0].isin(passenger_gene_list)
    tr_mask_drivers = torch.tensor(tr_mask_drivers, dtype=torch.bool)
    data.train_mask = tr_mask_drivers
    data.y = y

    print(data)
    model = GCN()
    print(model)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

    for epoch in range(10):
        loss, h = train(data, optimizer, model, criterion)
        if epoch % 10 == 0:
            print("Loss after {} epochs: {}".format(str(epoch), str(loss)))



def train(data, optimizer, model, criterion):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


if __name__ == "__main__":
    read_files()
