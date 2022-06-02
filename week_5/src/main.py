import os
import sys
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, RocCurveDisplay


# matplotlib settings
font = {'family': 'serif', 'size': 24}
plt.rc('font', **font)

# data path
local_path = "../week_5/"

# data parameters
cancer_names = ["blca", "brca", "coad", "hnsc", "ucec"]
cancer_type = cancer_names[0]

# neural network parameters
SEED = 32
n_epo = 50
k_folds = 5
batch_size = 64
num_classes = 2
gene_dim = 12
learning_rate = 0.001
n_edges = 200000


class GCN(torch.nn.Module):
    '''
    Neural network with graph convolution network (GCN)
    '''
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(gene_dim, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h


def read_files():
    '''
    Read raw data files and create Pytorch dataset
    '''
    final_path = cancer_type + "/"
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

    print("Saving mapping...")
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping))

    print("replacing gene ids")
    # set number of edges
    links = links[:n_edges]
    # replace gene names with ids
    re_links = links.replace({0: mapping})
    re_links = re_links.replace({1: mapping})
    # create data object
    x = x.loc[:, 1:].to_numpy()
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(re_links.to_numpy(), dtype=torch.long)
    # set up Pytorch geometric dataset
    compact_data = Data(x=x, edge_index=edge_index.t().contiguous())
    driver_gene_list.extend(passenger_gene_list)
    # set up true labels
    compact_data.y = y
    return compact_data, driver_ids, passenger_ids, gene_features, mapping


def create_training_proc(compact_data, driver_ids, passenger_ids, gene_features, mapping):
    '''
    Create network architecture and assign loss, optimizers ...
    '''
    print(compact_data)
    # initialize model
    model = GCN()
    print(model)
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # set integer ids to genes
    print("Replacing gene ids...")
    all_gene_ids = gene_features.replace({0: mapping})
    # balanced data size for each class
    equal_size = int(batch_size / float(num_classes))
    
    driver_ids_list = driver_ids[0].tolist()
    passenger_ids_list = passenger_ids[0].tolist()

    random.shuffle(driver_ids_list)
    random.shuffle(passenger_ids_list)

    driver_ids_list = np.reshape(driver_ids_list, (len(driver_ids_list), 1))
    passenger_ids_list = np.reshape(passenger_ids_list, (len(passenger_ids_list), 1))

    # create cross validation (CV) fold object
    kfold = KFold(n_splits=k_folds, shuffle=True)

    tr_loss_epo = list()
    te_acc_epo = list()
    dr_cls_acc_epo = list()
    dr_prec_epo = list()
    dr_recall_epo = list()
    dr_fpr_epo = list()
    dr_tpr_epo = list()
    
    # loop over epochs
    for epoch in range(n_epo):
        tr_loss_fold = list()
        te_acc_fold = list()
        dr_cls_acc_fold = list()

        dr_fpr = list()
        dr_tpr = list()
        dr_prec = list()
        dr_recall = list()
        # loop over folds
        # extract training and test for driver and passenger genes separately and then combine
        for fold, (dr_tr, pass_tr) in enumerate(zip(kfold.split(driver_ids_list), kfold.split(passenger_ids_list))):
            dr_tr_ids, dr_te_ids = dr_tr
            pass_tr_ids, pass_te_ids = pass_tr
            n_batches = int((len(dr_tr_ids) + len(pass_tr_ids) + 1) / float(batch_size))
            # combine te genes
            dr_te_ids = np.reshape(dr_te_ids, (dr_te_ids.shape[0]))
            pass_te_ids = np.reshape(pass_te_ids, (pass_te_ids.shape[0]))

            # extract gene ids
            dr_te_genes = driver_ids_list[dr_te_ids]
            pass_te_genes = passenger_ids_list[pass_te_ids]

            # reshape
            dr_te_genes = list(np.reshape(dr_te_genes, (dr_te_genes.shape[0])))
            pass_te_genes = list(np.reshape(pass_te_genes, (pass_te_genes.shape[0])))

            # create test masks
            # set test mask using test genes for drivers and passengers
            compact_data.test_mask = create_masks(all_gene_ids, dr_te_genes, pass_te_genes)

            batch_tr_loss = list()
            # loop over batches
            for bat in range(n_batches):
                random.shuffle(dr_tr_ids)
                random.shuffle(pass_tr_ids)
                batch_dr_tr_genes = driver_ids_list[dr_tr_ids]
                batch_dr_tr_genes = list(batch_dr_tr_genes.reshape((batch_dr_tr_genes.shape[0])))
                # balance train batches for both classes
                if len(batch_dr_tr_genes) < equal_size:
                    # oversample by choosing with replacement
                    batch_dr_tr_genes = list(np.random.choice(batch_dr_tr_genes, size=equal_size))
                else:
                    batch_dr_tr_genes = batch_dr_tr_genes[:int(batch_size / float(2))]
                batch_pass_tr_genes = passenger_ids_list[pass_tr_ids]
                batch_pass_tr_genes = batch_pass_tr_genes.reshape((batch_pass_tr_genes.shape[0]))
                if len(batch_pass_tr_genes) < equal_size:
                    # oversample by choosing with replacement
                    batch_pass_tr_genes = list(np.random.choice(batch_pass_tr_genes, size=equal_size))
                else:
                    batch_pass_tr_genes = batch_pass_tr_genes[:int(batch_size / float(2))]
                # set training mask using drivers and passengers genes with data balancing for each batch
                compact_data.train_mask = create_masks(all_gene_ids, batch_dr_tr_genes, batch_pass_tr_genes)
                # training for each batch
                tr_loss, h = train(compact_data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))

            # create list of driver and passenger genes
            test_driver_genes = np.reshape(driver_ids_list[dr_te_ids], (len(driver_ids_list[dr_te_ids]))).tolist()
            test_passenger_genes = np.reshape(passenger_ids_list[pass_te_ids], (len(passenger_ids_list[pass_te_ids]))).tolist()
            # predict using trained model
            dr_cls_acc, test_acc, _, _, _, _, _ = predict_data(model, compact_data, test_driver_genes, test_passenger_genes)
            dr_cls_acc_fold.append(dr_cls_acc)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}/{}, Fold: {}/{}, test accuracy: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(test_acc)))
            print("Epoch: {}/{}, Fold: {}/{}, test per class accuracy, Driver: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(dr_cls_acc)))
            te_acc_fold.append(test_acc)

        print("-------------------")
        tr_loss_epo.append(np.mean(tr_loss_fold))
        te_acc_epo.append(np.mean(te_acc_fold))
        dr_cls_acc_epo.append(np.mean(dr_cls_acc_fold))

        print()
        print("Epoch {}: Training Loss: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Epoch {}: Test accuracy: {}".format(str(epoch+1), str(np.mean(te_acc_fold))))
        print("Epoch {}: Test per class accuracy, Driver: {}".format(str(epoch+1), str(np.mean(dr_cls_acc_fold))))
        print()

    # prepare test mask using all driver genes (# of driver genes are less)
    compact_data.test_mask = create_masks(all_gene_ids, driver_ids[0].tolist(), passenger_ids[0].tolist())

    # compute accuracy on all driver genes using trained model
    dr_com_acc, te_acc, dr_com_fpr, dr_com_tpr, dr_com_prec, dr_com_rec, dr_roc_auc_score = predict_data(model, compact_data, driver_ids[0].tolist(), passenger_ids[0].tolist())

    print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("CV Test driver accuracy after {} epochs: {}".format(str(n_epo), str(np.mean(dr_cls_acc_epo))))

    # create plots
    plot_dr_prec_recall(dr_com_fpr, dr_com_tpr, dr_com_prec, dr_com_rec, dr_roc_auc_score)
    plot_loss_acc(n_epo, tr_loss_epo, dr_cls_acc_epo)

    # predict labels of unlabeled nodes
    # TODO:


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping


def agg_per_class_acc(prob_scores, pred, data, driver_ids, passenger_ids):
    '''
    Compute precision, recall, auc scores
    '''
    dr_tot = 0
    dr_corr = 0
    pass_tot = 0
    pass_corr = 0

    prob_scores = prob_scores.detach().numpy()
    # extract probability scores for driver (positive class, repr as 1) and passenger (negative class, repr as 0) genes 
    dr_prob_scores = prob_scores[driver_ids]
    pass_prob_scores = prob_scores[passenger_ids]

    # compute precision for driver genes
    for driver_id in driver_ids:
        if data.test_mask[driver_id] == torch.tensor(True):
             dr_pred = pred[driver_id]
             dr_tot += 1
             if dr_pred == torch.tensor(1):
                dr_corr += 1
    
    dr_pred_acc = dr_corr / float(dr_tot)
    # 1 as driver genes labels
    dr_label = torch.ones(dr_prob_scores.shape[0], dtype=torch.long)
    # 0 as passenger genes labels
    pass_label = torch.zeros(pass_prob_scores.shape[0], dtype=torch.long)
    # concatenate labels and their associated prob scores for driver and passenger genes
    dr_pass_label = torch.cat((dr_label, pass_label), 0)
    dr_pass_prob_scores = torch.cat((torch.tensor(dr_prob_scores), torch.tensor(pass_prob_scores)), 0)
    # compute precision, recall, tpr, fpr, auc scores
    dr_precision, dr_recall, _ = precision_recall_curve(dr_pass_label, dr_pass_prob_scores[:, 1], pos_label=1)
    dr_fpr, dr_tpr, _ = roc_curve(dr_pass_label, dr_pass_prob_scores[:, 1], pos_label=1)
    dr_roc_auc_score = roc_auc_score(dr_pass_label, dr_pass_prob_scores[:, 1])
    
    print("Driver prediction: Precision {}, # correctly predicted/total samples {}/{}".format(dr_pred_acc, dr_corr, dr_tot))
    print()
    return dr_pred_acc, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score


def create_masks(all_gene_ids, dr_te_genes, pass_te_genes):
    dr_te_genes.extend(pass_te_genes)
    te_mask = all_gene_ids[0].isin(dr_te_genes)
    te_mask = torch.tensor(te_mask, dtype=torch.bool)
    return te_mask


def predict_data(model, compact_data, test_driver_genes, test_passenger_genes):
    '''
    Predict using trained model and test data
    '''
    # predict on test fold
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    pred = out[0].argmax(dim=1)
    test_correct = pred[compact_data.test_mask] == compact_data.y[compact_data.test_mask]
    test_acc = int(test_correct.sum()) / int(compact_data.test_mask.sum())
    dr_cls_acc, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score = agg_per_class_acc(out[0], pred, compact_data, test_driver_genes, test_passenger_genes)
    return dr_cls_acc, test_acc, dr_fpr, dr_tpr, dr_precision, dr_recall, dr_roc_auc_score


def train(data, optimizer, model, criterion):
    '''
    Training step
    '''
    # Clear gradients
    optimizer.zero_grad()
    # forward pass
    out, h = model(data.x, data.edge_index)
    # compute error using training mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    # compute gradients
    loss.backward()
    # optimize weights
    optimizer.step()
    return loss, h


########################
# Plotting methods
########################
def plot_dr_prec_recall(fpr_epo, tpr_epo, dr_prec_epo, dr_recall_epo, dr_roc_auc_score):

    # plot precision, recall curve
    disp = PrecisionRecallDisplay(precision=dr_prec_epo, recall=dr_recall_epo, pos_label="Driver").plot()
    plt.title("Cancer {}, Driver prediction Precision-Recall (for all drivers), ROC AUC: {}".format(cancer_type, str(np.round(dr_roc_auc_score, 2))))
    plt.grid(True)
    plt.show()

    # plot ROC
    roc_auc = sklearn.metrics.auc(fpr_epo, tpr_epo)
    roc_display = RocCurveDisplay(fpr=fpr_epo, tpr=tpr_epo, roc_auc=roc_auc).plot()
    plt.title("Cancer {}, ROC curve (for all drivers)".format(cancer_type))
    plt.grid(True)
    plt.show()


def plot_loss_acc(n_epo, tr_loss, te_acc):
    # plot training loss
    x_val = np.arange(n_epo)
    plt.plot(x_val, tr_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Cancer {}, {} fold CV training loss vs epochs".format(cancer_type, str(k_folds)))
    plt.show()

    # plot driver gene precision vs epochs
    x_val = np.arange(n_epo)
    plt.plot(x_val, te_acc)
    plt.ylabel("Precision")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Cancer {}, {} fold CV Precision: Driver prediction vs epochs".format(cancer_type, str(k_folds)))
    plt.show()


if __name__ == "__main__":
    compact_data, driver_ids, passenger_ids, gene_features, mapping = read_files()
    create_training_proc(compact_data, driver_ids, passenger_ids, gene_features, mapping)
