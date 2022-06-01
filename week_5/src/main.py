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

import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, RocCurveDisplay


font = {'family': 'serif', 'size': 24}
plt.rc('font', **font)

local_path = "../week_5/"
cancer_names = ["blca", "brca", "coad", "hnsc", "ucec"]
SEED = 32
n_epo = 200
k_folds = 5
batch_size = 32
num_classes = 2
gene_dim = 12
n_edges = 10000


class GCN(torch.nn.Module):
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
        h = h.tanh()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        out = self.classifier(h)
        return out, h


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, header=None)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = df.iloc[:, 0:]
    return x, mapping


def agg_per_class_acc(prob_scores, pred, data, driver_ids, passenger_ids):
    dr_tot = 0
    dr_corr = 0
    pass_tot = 0
    pass_corr = 0
    prob_scores = prob_scores.detach().numpy()
    dr_prob_scores = prob_scores[driver_ids]
    pass_prob_scores = prob_scores[passenger_ids]
    #print(dr_prob_scores, dr_prob_scores.shape)
    #print(pass_prob_scores, pass_prob_scores.shape)
    #print(driver_ids)
    for driver_id in driver_ids:
        #print(data.test_mask[driver_id])
        if data.test_mask[driver_id] == torch.tensor(True):
             dr_pred = pred[driver_id]
             dr_tot += 1
             if dr_pred == torch.tensor(1):
                dr_corr += 1
    dr_label = torch.ones(dr_prob_scores.shape[0], dtype=torch.long)
    #print(dr_label)
    #dr_label_one_hot = torch.nn.functional.one_hot(dr_label, num_classes=2)
    #print(dr_label_one_hot)
    #print(dr_label, dr_prob_scores[:, 1])
    
    #dr_precision, dr_recall, dr_threshold = roc_curve(dr_label, dr_prob_scores[:, 1])
    dr_auc = 0 #roc_auc_score(dr_label_one_hot, dr_prob_scores)
    dr_avg_precision = dr_auc #average_precision_score(dr_label, dr_prob_scores[:, 1])
    dr_pred_acc = dr_corr / float(dr_tot)
    print("Driver prediction: ", dr_pred_acc, dr_corr, dr_tot)
    #print("Driver prediction: ", dr_pred_acc, dr_precision, dr_recall, dr_threshold, dr_avg_precision, dr_corr, dr_tot)

    '''for pass_id in passenger_ids:
        if data.test_mask[pass_id] == torch.tensor(True):
             pass_pred = pred[pass_id]
             pass_tot += 1
             if pass_pred == torch.tensor(0):
                pass_corr += 1
    pass_pred_acc = pass_corr / float(pass_tot)'''

    pass_label = torch.zeros(pass_prob_scores.shape[0], dtype=torch.long)
    #pass_label_one_hot = torch.nn.functional.one_hot(pass_label, num_classes=2)
    #pass_precision, pass_recall, pass_threshold = precision_recall_curve(pass_label, pass_prob_scores[:, 0])
    #pass_precision, pass_recall, pass_threshold = roc_curve(pass_label, pass_prob_scores[:, 0])
    #pass_auc = 0 #roc_auc_score(pass_label_one_hot, pass_prob_scores)
    #pass_avg_precision = pass_auc #average_precision_score(pass_label, pass_prob_scores[:, 0], pos_label=0)
    #print("Passenger prediction: ", pass_pred_acc, pass_precision, pass_recall, pass_threshold, pass_avg_precision, pass_corr, pass_tot)
    #print("Passenger prediction: ", pass_pred_acc, pass_corr, pass_tot)

    # combined ROC score
    dr_pass_label = torch.cat((dr_label, pass_label), 0)
    #print(dr_pass_label.shape, dr_label.shape, pass_label.shape)
    #print(dr_prob_scores.shape, pass_prob_scores.shape)
    dr_pass_prob_scores = torch.cat((torch.tensor(dr_prob_scores), torch.tensor(pass_prob_scores)), 0)

    dr_precision, dr_recall, _ = precision_recall_curve(dr_pass_label, dr_pass_prob_scores[:, 1], pos_label=1)
    dr_fpr, dr_tpr, _ = roc_curve(dr_pass_label, dr_pass_prob_scores[:, 1], pos_label=1)
    #dr_new_auc = roc_auc_score(dr_pass_label, dr_pass_prob_scores[:, 1])
    #pass_new_precision, pass_new_recall, _ = roc_curve(dr_pass_label, dr_pass_prob_scores[:, 0], pos_label=0)
    #pass_new_auc = roc_auc_score(dr_pass_label, dr_pass_prob_scores[:, 0])
    #print("==========")
    '''print(dr_new_precision)
    print(dr_new_recall)
    print()
    print(pass_new_precision)
    print(pass_new_recall)'''
    #print(dr_new_auc, pass_new_auc)
    #print("==========")
    return dr_pred_acc, dr_fpr, dr_tpr, dr_precision, dr_recall


def read_files():
    #print(torch.tensor(False))
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

    print("Saving mapping...")
    with open('gene_mapping.json', 'w') as outfile:
        outfile.write(json.dumps(mapping))

    print("replacing gene ids")
    # set number of edges
    links = links[:n_edges]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Define optimizer.

    print("Replacing gene ids...")
    all_gene_ids = gene_features.replace({0: mapping})

    
    equal_size = int(batch_size / float(num_classes))
    
    driver_ids_list = driver_ids[0].tolist()
    passenger_ids_list = passenger_ids[0].tolist()
    #driver_ids_list.extend(passenger_ids_list)
    random.shuffle(driver_ids_list)

    #driver_ids_list = np.reshape(driver_ids_list, (len(driver_ids_list), 1))
 
    driver_ids_list = np.reshape(driver_ids_list, (len(driver_ids_list), 1))
    passenger_ids_list = np.reshape(passenger_ids_list, (len(passenger_ids_list), 1))

    kfold = KFold(n_splits=k_folds, shuffle=True)
    tr_loss_epo = list()
    te_acc_epo = list()
    dr_cls_acc_epo = list()
    pass_cls_acc_epo = list()

    dr_prec_epo = list()
    dr_recall_epo = list()
    dr_fpr_epo = list()
    dr_tpr_epo = list()

    pass_prec_epo = list()
    pass_recall_epo = list()
    
    for epoch in range(n_epo):
        tr_loss_fold = list()
        te_acc_fold = list()
        dr_cls_acc_fold = list()
        pass_cls_acc_fold = list()

        dr_fpr = list()
        dr_tpr = list()
        dr_prec = list()
        dr_recall = list()
        pass_prec = list()
        pass_recall = list()

        for fold, (dr_tr, pass_tr) in enumerate(zip(kfold.split(driver_ids_list), kfold.split(passenger_ids_list))):
            dr_tr_ids, dr_te_ids = dr_tr
            pass_tr_ids, pass_te_ids = pass_tr
            n_batches = int((len(dr_tr_ids) + len(pass_tr_ids) + 1) / float(batch_size))

            '''tr_genes = driver_ids_list[train_ids]
            tr_genes = tr_genes.reshape((tr_genes.shape[0]))

            te_genes = driver_ids_list[test_ids]
            te_genes = te_genes.reshape((te_genes.shape[0]))

            tr_mask = all_gene_ids[0].isin(tr_genes)
            te_mask = all_gene_ids[0].isin(te_genes)'''

            # combine te genes
            dr_te_ids = np.reshape(dr_te_ids, (dr_te_ids.shape[0]))
            pass_te_ids = np.reshape(pass_te_ids, (pass_te_ids.shape[0]))

            dr_te_genes = driver_ids_list[dr_te_ids]
            pass_te_genes = passenger_ids_list[pass_te_ids]

            dr_te_genes = list(np.reshape(dr_te_genes, (dr_te_genes.shape[0])))
            pass_te_genes = list(np.reshape(pass_te_genes, (pass_te_genes.shape[0])))

            dr_te_genes.extend(pass_te_genes)
            te_genes = dr_te_genes
            te_mask = all_gene_ids[0].isin(te_genes)
            te_mask = torch.tensor(te_mask, dtype=torch.bool)

            #compact_data.test_mask = None
            compact_data.test_mask = te_mask

            batch_tr_loss = list()
            for bat in range(n_batches):
                random.shuffle(dr_tr_ids)
                random.shuffle(pass_tr_ids)
                batch_dr_tr_genes = driver_ids_list[dr_tr_ids]
                batch_dr_tr_genes = list(batch_dr_tr_genes.reshape((batch_dr_tr_genes.shape[0])))
                if len(batch_dr_tr_genes) < equal_size:
                    batch_dr_tr_genes = list(np.random.choice(batch_dr_tr_genes, size=equal_size))
                else:
                    batch_dr_tr_genes = batch_dr_tr_genes[:int(batch_size / float(2))]
                batch_pass_tr_genes = passenger_ids_list[pass_tr_ids]
                batch_pass_tr_genes = batch_pass_tr_genes.reshape((batch_pass_tr_genes.shape[0]))
                if len(batch_pass_tr_genes) < equal_size:
                    batch_pass_tr_genes = list(np.random.choice(batch_pass_tr_genes, size=equal_size))
                else:
                    batch_pass_tr_genes = batch_pass_tr_genes[:int(batch_size / float(2))]
                batch_dr_tr_genes.extend(batch_pass_tr_genes)
                tr_mask = all_gene_ids[0].isin(batch_dr_tr_genes)
                tr_mask = torch.tensor(tr_mask, dtype=torch.bool)
                ## Reset mask
                #compact_data.train_mask = None
                compact_data.train_mask = tr_mask
                tr_loss, h = train(compact_data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))
            # predict on test fold
            model.eval()
            out = model(compact_data.x, compact_data.edge_index)

            '''print(driver_ids[0].tolist(), len(driver_ids[0].tolist()))
            print()
            print(driver_ids_list[dr_tr_ids], len(driver_ids_list[dr_tr_ids]))
            print()
            print(driver_ids_list[dr_te_ids], len(driver_ids_list[dr_te_ids]))'''

            pred = out[0].argmax(dim=1) #Use the class with highest probability
            # dr_pred_acc, pass_pred_acc, dr_precision, dr_recall, pass_precision, pass_recall 
            # TODO: 
            #dr_cls_acc, pass_cls_acc, dr_prec_fold, dr_rec_fold, pass_prec_fold, pass_rec_fold = agg_per_class_acc(out[0], pred, compact_data, driver_ids[0].tolist(), passenger_ids[0].tolist())
            print()
            test_driver_genes = np.reshape(driver_ids_list[dr_te_ids], (len(driver_ids_list[dr_te_ids]))).tolist()
            test_passenger_genes = np.reshape(passenger_ids_list[pass_te_ids], (len(passenger_ids_list[pass_te_ids]))).tolist()
            dr_cls_acc, dr_fpr_fold, dr_tpr_fold, dr_prec_fold, dr_rec_fold = agg_per_class_acc(out[0], pred, compact_data, test_driver_genes, test_passenger_genes)

            '''dr_fpr.append(dr_fpr_fold)
            dr_tpr.append(dr_tpr_fold)
            dr_prec.append(dr_prec_fold)
            dr_recall.append(dr_rec_fold)'''
 
            dr_fpr = dr_fpr_fold
            dr_tpr = dr_tpr_fold
            dr_prec = dr_prec_fold
            dr_recall = dr_rec_fold

            #pass_prec.append(pass_prec_fold)
            #pass_recall.append(pass_rec_fold)

            dr_cls_acc_fold.append(dr_cls_acc)
            #pass_cls_acc_fold.append(pass_cls_acc)

            test_correct = pred[compact_data.test_mask] == compact_data.y[compact_data.test_mask]  #Check against ground-truth labels.

            test_acc = int(test_correct.sum()) / int(compact_data.test_mask.sum())  #Derive ratio of correct predictions.
            print("Epoch {}, fold {}/{} average training loss: {}".format(str(epoch+1), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}, Fold: {}/{}, test accuracy: {}".format(str(epoch+1), str(fold+1), str(k_folds), str(test_acc)))
            print("Epoch: {}, Fold: {}/{}, test per class accuracy, Driver: {}".format(str(epoch+1), str(fold+1), str(k_folds), str(dr_cls_acc)))
            te_acc_fold.append(test_acc)

        print("-------------------")
        tr_loss_epo.append(np.mean(tr_loss_fold))
        te_acc_epo.append(np.mean(te_acc_fold))

        dr_cls_acc_epo.append(np.mean(dr_cls_acc_fold))
        #pass_cls_acc_epo.append(np.mean(pass_cls_acc_fold))

        #print(np.array(dr_prec).shape, np.mean(np.array(dr_prec), axis=0).shape)
        #print(pass_prec)
        #print(np.array(pass_prec), np.array(pass_prec).shape)
        #print(np.mean(np.array(pass_prec).shape, axis=0))

        '''dr_fpr_epo.append(np.mean(np.array(dr_fpr), axis=0))
        dr_tpr_epo.append(np.mean(np.array(dr_tpr), axis=0))

        dr_prec_epo.append(np.mean(np.array(dr_prec), axis=0))
        dr_recall_epo.append(np.mean(np.array(dr_recall), axis=0))'''

        dr_fpr_epo = dr_fpr
        dr_tpr_epo = dr_tpr

        dr_prec_epo = dr_prec
        dr_recall_epo = dr_recall

        #dr_prec_epo = dr_prec
        #dr_recall_epo = dr_recall
        #pass_prec_epo.append(np.mean(np.array(pass_prec), axis=0))
        #pass_recall_epo.append(np.mean(np.array(pass_recall), axis=0))

        print()
        print("Training Loss after {} epochs: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Test accuracy after {} epochs: {}".format(str(epoch+1), str(np.mean(te_acc_fold))))
        print("After {} epochs, test per class accuracy, Driver: {}".format(str(epoch+1), str(np.mean(dr_cls_acc_epo))))
        print()

    '''dr_prec_epo = np.mean(np.array(dr_prec_epo), axis=0)
    dr_recall_epo = np.mean(np.array(dr_recall_epo), axis=0)
    fpr_epo = np.mean(np.array(dr_fpr_epo), axis=0)
    tpr_epo = np.mean(np.array(dr_tpr_epo), axis=0)'''

    dr_prec_epo = dr_prec_epo
    dr_recall_epo = dr_recall_epo
    fpr_epo = dr_fpr_epo
    tpr_epo = dr_tpr_epo
    
    #dr_prec_epo = dr_prec_epo #np.mean(np.array(), axis=0)
    #dr_recall_epo = dr_recall_epo #np.mean(np.array(), axis=0)
    #pass_prec_epo = np.mean(np.array(pass_prec_epo), axis=0)
    #pass_recall_epo = np.mean(np.array(pass_recall_epo), axis=0)
    print(dr_prec_epo)
    print(dr_recall_epo)
    print()
    print("Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("Test accuracy after {} epochs: {}".format(str(n_epo), str(np.mean(te_acc_epo))))
    #plot_loss_acc(n_epo, tr_loss_epo, te_acc_epo)
    plot_dr_prec_recall(fpr_epo, tpr_epo, dr_prec_epo, dr_recall_epo, n_epo, dr_cls_acc_epo)

    # predict labels of unlabeled nodes
    '''driver_ids_list = driver_ids_list.reshape((driver_ids_list.shape[0]))
    final_tr_mask = all_gene_ids[0].isin(driver_ids_list)
    final_te_mask = not final_tr_mask.all()
    compact_data.test_mask = final_te_mask
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    pred = out[0].argmax(dim=1) # Use the class with highest probability.
    print(out[0], pred, pred.shape)'''


def plot_dr_prec_recall(fpr_epo, tpr_epo, dr_prec_epo, dr_recall_epo, n_epo, dr_cls_acc_epo):
    #dr_recall_epo = sorted(dr_recall_epo, reverse=True)
    plt.plot(dr_recall_epo, dr_prec_epo)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.grid(True)
    plt.title("Driver prediction Precision-Recall curve")
    plt.show()

    #print(fpr_epo)
    #print(tpr_epo)
    #print(dr_prec_epo)
    #print(dr_recall_epo)

    roc_auc = sklearn.metrics.auc(fpr_epo, tpr_epo)
    roc_display = RocCurveDisplay(fpr=dr_prec_epo, tpr=dr_recall_epo, roc_auc=roc_auc).plot()
    plt.title("ROC: true positive rate vs false positive rate")
    plt.show()
    
    roc_auc_prc = sklearn.metrics.auc(dr_prec_epo, dr_recall_epo)
    print("Precision-recall AUC: {}".format(roc_auc_prc))
    #pr_display = PrecisionRecallDisplay(precision=dr_prec_epo, recall=dr_recall_epo).plot()
    #plt.title("Precision-Recall curve")
    #plt.show()

    '''x_val = np.arange(n_epo)
    plt.plot(x_val, dr_cls_acc_epo)
    plt.ylabel("Driver prediction precision")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Driver prediction precision vs epochs")
    plt.show()'''


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
    '''from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = fetch_openml(data_id=1464, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    clf.fit(X_train, y_train)

    from sklearn.metrics import roc_curve
    from sklearn.metrics import RocCurveDisplay

    y_score = clf.decision_function(X_test)
    print(y_test)
    print(y_score)

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    print(fpr)
    print(tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()'''

    read_files()
