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
    # driver = 1, passenger = 0

    driver_mask = gene_features[gene_features[0].isin(driver_gene_list)]
    print(driver_mask)
    driver_y = [1 for item, i in driver_mask.iterrows()]
    print("----")
    passenger_mask = gene_features[gene_features[0].isin(passenger_gene_list)]
    passenger_y = [0 for item, i in passenger_mask.iterrows()]
    print(passenger_mask)
    print("----")
    driver_y.extend(passenger_y)
    print("---------------")

    y = torch.tensor(driver_y, dtype=torch.float)

    print("Saving mapping...")
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping))

    print("replacing gene ids")
    links = links[:1000]
    re_links = links.replace({0: mapping})
    re_links = re_links.replace({1: mapping})
    links_mat = re_links.to_numpy()

    # create data object
    edge_index = torch.tensor(links_mat, dtype=torch.long) 
    combined_tr = pd.concat([driver_mask, passenger_mask])
    print(combined_tr)
    combined_tr = combined_tr.loc[:, 1:]
    print(combined_tr)
    combined_tr = combined_tr.to_numpy()
    x = torch.tensor(combined_tr, dtype=torch.float)
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

    # plot network
    #G = to_networkx(data, to_undirected=True)
    #visualize_graph(G, color=data.y)

    


if __name__ == "__main__":
    read_files()
