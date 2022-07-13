import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from  torch.utils.data import DataLoader
from net import LSTM
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, optim
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from torch_geometric.data import Data
import itertools
import sys
import random
import uuid
import subprocess as sp
import os
import math


random.seed(0)
np.random.seed(0)


"""

MyDataset read in as a graph or not as a graph


"""
class Weisfeiler_Lehman_Graph():
	def __init__(self):
		self.label_dict = {}
	
	
	def propergate_class_labels(self, seq1, connections1):


		new_labels_seq = []
	
		#### add label for linear connection

		for num, seq in enumerate(seq1):
	
			if num == 0:
				new_labels_seq.append(str(seq) + str(seq1[num+1]))
		
		
			elif num == len(seq1) -1:
				new_labels_seq.append(str(seq) + str(seq1[num-1]))
		
			else:

				new_labels_seq.append(str(seq) + str(seq1[num-1]) + str(seq1[num+1]))
			

		##### add label for rest connections
		for connection in connections1:
	
			new_labels_seq[connection[0]] += seq1[connection[1]]
			new_labels_seq[connection[1]] += seq1[connection[0]]

		

		###### do label compression #######
		self.label_hashing(new_labels_seq)
		labels = self.label_compression(new_labels_seq)
		
		seq1.extend(labels)
		return seq1
		


	def label_compression(self, labels):
	
	
		labels_compressed = [self.label_dict[l] for l in labels]
	
		return labels_compressed


	def label_hashing(self, seq):
	
	
		label_keys = self.label_dict.keys()
		last_num = len(label_keys) - 1
		
		for label in seq:
		
			if label not in label_keys:
			
				self.label_dict[label] = str(last_num + 1)
				last_num = last_num + 1
		return


	def label_count(self, labels, max_label):
	
	
		labels_int = [int(l) for l in labels]
		count_list = [0] * (max_label +1)

		for label in labels_int:
		
			count_list[label] += 1
	
		return count_list


	def graph_similiarity(self, label1, label2):
	
		similarity = np.dot(label1, label2)	
	
		return similarity


	def calculate_weisfeiler_lehman_graph(self, seq1, connections1, seq2, connections2, num_iterations = 1):


		self.label_hashing(seq1)
		self.label_hashing(seq2)
		
		label_seq1 = self.label_compression(seq1)
		label_seq2 = self.label_compression(seq2)
		

		for n in range(num_iterations):


			label_seq1 = self.propergate_class_labels(label_seq1, connections1)
			label_seq2 = self.propergate_class_labels(label_seq2, connections2)



		max_val = max(max([int(l) for l in label_seq1]), max([int(l) for l in label_seq2]))

		count_list1 = self.label_count(label_seq1, max_val)
		count_list2 = self.label_count(label_seq2, max_val)
		
		
		#base_similarity = self.graph_similiarity(count_list2, count_list2)
		similarity = self.graph_similiarity(count_list1, count_list2)


		return similarity



def RNAfold(rnafold_cmd, fasta_file, constraint):

    
    fasta_file_preffix = fasta_file.rsplit('.', 1)[0]
    output_pdf = fasta_file_preffix + '_proteins.fa'
    log_file = fasta_file_preffix + '_RNAfold.log'
    rnafold_cmd += ' {input_fasta} --filename-full'
    if constraint == True: rnafold_cmd += " -C" 
        
    
    rnafold_cmd = rnafold_cmd.format(input_fasta=fasta_file)
    

    with open(log_file, 'w') as lf:
    	sp.call(rnafold_cmd.split(), stdout = lf)
    	
    os.remove(log_file)
        

    return


def extract_graph(file_name):


	file_ = open(file_name)
	lines = file_.readlines()
	
	start_ind = lines.index("/pairs [\n")
	end_ind = start_ind + lines[start_ind:].index("] def\n")

	connected = lines[start_ind + 1:end_ind]
	connected = [[int(c[:-1].split(" ")[0][1:])-1, int(c[:-1].split(" ")[1][:-1])-1] for c in connected]
	file_.close()



	return connected



def create_fasta_constraints(seq, dms_seq, file_name, constraints = False, divide_by = 1):


	constraints = ""
	


	for dms in dms_seq:
		if dms/divide_by < 0.04:
			constraints += "."
		elif dms/divide_by >= 0.04:
			constraints += "x"
			
	

	rand_id = str(uuid.uuid1())
	file_ = open(file_name, "w")
	
	file_.write(">" + rand_id)
	file_.write("\n")
	file_.write(seq)
	if constraints: file_.write("\n")
	if constraints: file_.write(constraints)	
	file_.close()



	return constraints, file_name



def create_graph(input_file, constraint):


	RNAfold("RNAfold", input_file, constraint)
	


	os.remove(input_file)
	current_files = glob.glob("./*.ps")
	connected = extract_graph(current_files[0])
	os.remove(current_files[0])

	return connected
	

class MyDataset(Dataset):


	def __init__(self, dataset_folder, kmer, scale_factor):
		self.scale_factor = scale_factor
		self.dataset_folder = dataset_folder
		self.kmer_list = self.get_kmer_list(kmer)
		
		value_list = self.read_dataset(self.dataset_folder)

		self.dataset, self.x_values, self.y_values, self.max_length, self.seq_list = self.construct_dataset(value_list, self.kmer_list)

		self.dataset = self.dataset
		self.x_values = self.x_values
		self.y_values = self.y_values
		self.seq_list = self.seq_list
		
		

		
	def concat(self, x_str):
	
		x_concat = ""
		for x in x_str:
			x_concat += x
	
		return x_concat
		
	def num_words(self):
	
		return len(self.kmer_list)
		
		
	def get_kmer_list(self, kmer):
	
		kmer_list = list(itertools.product(["A", "T", "G", "C"], repeat=kmer))
		kmer_list = [self.concat(x) for x in kmer_list]

		return kmer_list
		

	def read_dataset(self, file_):
	
		value_list = []
		id_list = []
		struct_list = []
		files  = open(file_)
		file_lines = files.readlines()
		
		
		for i in range(0,len(file_lines),3):
			value_list.append([file_lines[i], file_lines[i + 1], file_lines[i+2]])
		
		
		files.close()

		return value_list
		
		
	def construct_dataset(self, value_list, kmer_list):
	
		print(kmer_list)
	
		x_values = []
		y_values = []
		data_list = []
		seq_list = []
		max_length = 0
		

		for sample in value_list:
		
			seq = sample[1][:-1]
			seq_list.append(seq)
			dms = [float(dms) for dms in sample[2][:-2].split(" ")]
			kmer_length = len(kmer_list[0])
			
			
			
			
			x_encoded = []
			y_encoded = []
			edge_list = []

			
			for num,step_size in enumerate(range(0, len(seq) - (kmer_length-1))):


				kmer_dms = dms[step_size :step_size + kmer_length]
				kmer_dms = [dms for num, dms in enumerate(kmer_dms)]
				


				kmer_bases = seq[step_size:step_size+kmer_length]
				listofzeros = kmer_list.index(kmer_bases)
				
				
				

				x_encoded.append(listofzeros)
				edge_list.append([num, num+1])
				edge_list.append([num+1, num])

				#y_encoded.append(np.mean(kmer_dms))
				y_encoded.append(np.max(kmer_dms))


				
			### dms only affects A,C rest of values from wrong strand
			y_encoded = [y*self.scale_factor if seq[num] == "A" or seq[num] == "C" else 0 for num, y in enumerate(y_encoded)]
			
			
			if len(x_encoded) > max_length: max_length = len(x_encoded) 
			
			
			max_length = len(kmer_list)
			
			x_values.append(x_encoded)
			y_values.append(y_encoded)
			
			edge_index = torch.tensor(edge_list, dtype=torch.long)
			x = torch.tensor(x_encoded, dtype=torch.float)
			y = torch.tensor(y_encoded, dtype=torch.float)
	
			data = Data(x=x, y=y, edge_index=edge_index.t().contiguous(), id = str(sample[0]))
			data_list.append(data)
		
		 
			
		return data_list, x_values, y_values, max_length, seq_list
		
		
		
		
	def reconstruct_input(self, input_seq):
	
	
		seq = ""
	
	
	
		for enum, val in enumerate(input_seq[0]):
		
			
		
			
			kmer = self.kmer_list[val]
			#seq += kmer[0]
			seq += kmer
	
		return seq
        	
        
	def __len__(self):
		return len(self.y_values)

	def __getitem__(self, idx):

		return self.x_values[idx], self.y_values[idx]
		
	def return_x(self):
	
		return self.x_values
		
	def return_y(self):
	
		return self.y_values
		


def my_collate(batch):


    data = [item[0] for item in batch]
    target = [torch.FloatTensor(item[1]) for item in batch]
    target = target
    return [data, target]



def pearson_loss(output, target_data, input_data):


	vx = output - torch.mean(output)
	vy = target_data - torch.mean(target_data)
	
	loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
	

	
	return loss.item()


def train_model(trainloader, net, criterion, optimizer, epoch, teacher_forcing_ratio = 0.25):

	torch.autograd.set_detect_anomaly(True)
	running_loss = 0.0
	net.zero_grad()
	
	
	for i, data in enumerate(trainloader):
		input_data, target_data = data
		loss = 0
		hidden = net.initHidden()
		optimizer.zero_grad()
		output, hidden = net(torch.IntTensor(input_data), hidden)
		
		#output, hidden = net(input_data, hidden)

		#loss = pearson_loss(output,target_data)
		loss = criterion(output.squeeze(),target_data[0])
		
		
		loss.backward()
		optimizer.step()
		running_loss = running_loss + loss

	return net, running_loss/len(trainloader)




def eval_model(dataloader, net, criterion, epoch, pearson =False):


	running_loss = 0
	input_data_list = []
	output_data_list = []
	target_data_list = []
	
	with torch.no_grad():
		for data in dataloader:
			input_data, target_data = data
			loss = 0
			hidden = net.initHidden()
			output, hidden = net(torch.IntTensor(input_data), hidden)
		
			
			
			
			if pearson == False:
				loss = criterion(output.squeeze(), target_data[0])
			else:
			
				loss = pearson_loss(output.squeeze(), target_data[0], input_data)
			
			
			running_loss = running_loss + loss
			
			input_data_list.append(input_data)
			output_data_list.append(output.squeeze())
			target_data_list.append(target_data[0])
	    
	return running_loss/len(dataloader), output_data_list, input_data_list, target_data_list
	
	
	

if __name__ == "__main__":

	scale_factor  = 100
	dataset = MyDataset("./dms_dataset.fa", kmer=1, scale_factor = scale_factor)
	kfold_loss = []
	fold_res = []wei
	graph_sim_loss = []
	
	kfold = KFold(n_splits=5, shuffle=False, random_state=None)


	for enum, indeces in enumerate(kfold.split( dataset.return_x(), list(range(len(dataset.return_y()))))):
		train_indices, test_val_ind = indeces[0], indeces[1]
	
	
		split2 = int(np.floor(0.5 * len(test_val_ind)))
		val_indices,test_indices = test_val_ind[split2:], test_val_ind[:split2]
		
		
		g = torch.Generator()
		g.manual_seed(0)

		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)
		test_sampler = SubsetRandomSampler(test_indices)

		train_loader = DataLoader(dataset, batch_size=1, sampler = train_sampler, collate_fn=my_collate, generator=g)
		validation_loader = DataLoader(dataset, batch_size = 1, sampler = valid_sampler, collate_fn=my_collate, generator=g)
		test_loader = DataLoader(dataset, batch_size = 1, sampler = test_sampler, collate_fn=my_collate, generator=g)
	
		hidden_size = 50
		net = LSTM(dataset.num_words(),hidden_size, max_length=dataset.max_length)
		saved_net =  LSTM(dataset.num_words(),hidden_size, max_length=dataset.max_length)
		optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

		criterion = nn.L1Loss()
		best_validation = np.inf
		epoch_range = 10

		for epoch in range(0, epoch_range):
		
			#encoder, decoder, running_loss = train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=dataset.max_length)
			
			net, running_loss = train_model(train_loader, net = net, criterion = criterion,  optimizer = optimizer, epoch = epoch)
			running_loss_val, output, input_data, target_data = eval_model(validation_loader, net, criterion = criterion, epoch = epoch)
			print(f'train-loss in epoch {epoch}: {running_loss}')
	
			if running_loss_val <= best_validation and epoch > int(epoch_range/2):
				best_validation = running_loss_val
				saved_net.load_state_dict(net.state_dict())
				
		
		print(f'best val-loss:{best_validation}')
		#### final test using pearson correlation #####
		print("##############################################################################################")
		
		
		running_loss, output, input_data, target_data_list = eval_model(test_loader, saved_net, criterion = criterion, epoch = epoch, pearson = True)
		
		print(f'pearson-correlation in fold {enum}: {running_loss}')
		
		file_name = "./test.fa"

		graph_similarity = []
		
		for en,dms_predicted in enumerate(output):
		
			seq = dataset.reconstruct_input(input_data[en])
			
		
			constraints, input_file = create_fasta_constraints(seq, dms_predicted, file_name, constraints = True, divide_by = scale_factor)
			
			
			first_connections = create_graph(input_file, constraint = True)
			
			constraints, input_file = create_fasta_constraints(seq, dms_predicted, file_name, constraints = False, divide_by = scale_factor)
			second_connections = create_graph(input_file, constraint = False)
			
			kernel = Weisfeiler_Lehman_Graph()
			graph_sim = kernel.calculate_weisfeiler_lehman_graph(seq, first_connections, seq, second_connections, num_iterations = 1)
			graph_similarity.append(graph_sim)

		kfold_loss.append(running_loss)
		graph_sim_loss.append(graph_similarity)
		
		
		print(f'graph similarity in fold {enum}: {np.mean(graph_similarity)}')
		
		
	for num, auprc in enumerate(kfold_loss):
		print(f'pearson-correlation in fold {num}: {{kfold_loss[num]}}')
		print(f'Graph-similarity in fold {num}: {np.mean(graph_sim_loss[num])}')

	
	print(f'Average Loss over all folds: {np.mean(kfold_loss)}')
	print(f'Average Graph-similarity over all folds: {np.mean(graph_sim_loss)}')
	
	
	
