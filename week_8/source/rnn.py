import os
import sys
import pandas as pd
import numpy as np
from itertools import product

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import Linear

import sklearn
from sklearn.metrics import roc_auc_score


train_path = "../data/train.tsv"
test_path = "../data/dev.tsv"
vocab = "AGCT"
batch_size = 32
hidden_dim = 100
fc_hidden_dim = 64
n_layers = 2
learning_rate = 2e-4
n_epochs = 50
dropout = 0.1
embedding_dim = 128
bidirectional = True
vocab_size = 4096 + 1
output_dim = 1


class RNNSeqClassifier(torch.nn.Module):
    
    def __init__(self):
        
        super(RNNSeqClassifier, self).__init__()
        # embedding layer: vector representation of each kmer index
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        # LSTM: recurrent layer that learns connections
        self.lstm = torch.nn.LSTM(embedding_dim,
            hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            dropout = dropout,
            batch_first = True
        )
        # dense layer for assigning class to each sequence of kmers
        self.fc_hidden = torch.nn.Linear(hidden_dim * 2, fc_hidden_dim)
        self.fc_relu = torch.nn.ReLU()
        self.fc_out = torch.nn.Linear(fc_hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden_state, cell_state) = self.lstm(embedded)
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        fc_hidden = self.fc_hidden(hidden)
        fc_hidden = self.fc_relu(fc_hidden)
        dense_outputs = self.fc_out(fc_hidden)
        outputs = self.sigmoid(dense_outputs)
        return outputs
    

def get_all_possible_words(vocab, kmer_size=3):
    '''
    Get an exhaustive list of combinations of "AGCT" as kmers of size defined by kmer_size
    '''
    all_com = [''.join(c) for c in product(vocab, repeat=kmer_size)]
    kmer_f_dict = {i + 1: all_com[i] for i in range(0, len(all_com))}
    kmer_r_dict = {all_com[i]: i + 1  for i in range(0, len(all_com))}
    return kmer_f_dict, kmer_r_dict


def convert_seq_2_integers(dataframe, r_dict):
    seq_mat = list()
    for index, item in dataframe.iterrows():
        kmers = item["sequence"].split(" ")
        kmers_integers = [r_dict[k] for k in kmers]
        seq_mat.append(kmers_integers)
    labels = dataframe["label"].tolist()
    return seq_mat, labels


def train_model(X_train, y_train, X_test, y_test):
    '''
    Train model using train sequences and their corresponding labels.
    Evaluate trained model using test sequences and their labels
    '''
    model = RNNSeqClassifier()
    print("RNN classifier architecture: \n")
    print(model)
    print()
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    training_set = DatasetMaper(X_train, y_train)
    test_set = DatasetMaper(X_test, y_test)
		
    loader_training = DataLoader(training_set, batch_size=batch_size)
    loader_test = DataLoader(test_set)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_epo = list()
    for epoch in range(n_epochs):
        loss_bat = list()
        predictions = []
        # training mode
        model.train()
        for x_batch, y_batch in loader_training:
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)
            y_pred = model(x)
            y_pred = torch.reshape(y_pred, (y_pred.shape[0], ))
            loss = F.binary_cross_entropy(y_pred, y)
            loss_bat.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_epo.append(np.mean(loss_bat))
        # evaluate model after each epoch of training on test data
        pred = evaluate_model(model, loader_test)
        auc_score, acc = calculate_accuray(y_test, pred)
        mean_batch_loss = np.mean(np.round(loss_bat, 2))
        # output training and evaluation results
        print("Training: Loss at epoch {} : {}".format(epoch+1, np.round(mean_batch_loss, 4)))
        print("Test: Accuracy at epoch {} : {}".format(epoch+1, acc))
        print("Test: ROC AUC score at epoch {} : {}".format(epoch+1, auc_score))
        print()
    print("Training: Loss after {} epochs: {}".format(epoch+1, np.mean(loss_epo)))


def evaluate_model(model, loader_test):
    predictions = []
    model.eval()
    # no gradient update while in test mode
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)
            y_pred = model(x)
            predictions += list(y_pred.detach().numpy())
    return predictions


def calculate_accuray(ground_truth, predictions):
    true_positives = 0
    true_negatives = 0
    for true, pred in zip(ground_truth, predictions):
        if (pred > 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    auc_score = np.round(roc_auc_score(ground_truth, predictions), 4)
    accuracy = np.round((true_positives + true_negatives) / len(ground_truth), 4)
    return auc_score, accuracy


def preprocess_data():
    '''
    Read raw data files and create Pytorch dataset
    '''
    f_dict, r_dict = get_all_possible_words(vocab, 6)
    train_df = pd.read_csv(train_path, sep="\t")
    train_mat, train_labels = convert_seq_2_integers(train_df, r_dict)
    test_df = pd.read_csv(test_path, sep="\t")
    test_mat, test_labels = convert_seq_2_integers(test_df, r_dict)
    return train_mat, train_labels, test_mat, test_labels


class DatasetMaper(Dataset):
	'''
	Handles batches of dataset
	'''
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess_data()
    train_model(X_train, y_train, X_test, y_test)
