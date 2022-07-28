import glob
import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from  torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import BatchSampler
from sklearn.utils.class_weight import compute_class_weight
import random
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn import tree
import sys
from matplotlib import pyplot as plt
import modules_lrp as modules


def randperm(N,b):

        assert(b <= N) # if this fails no valid solution can be found.
        I = np.arange(0)
        while I.size < b:
            I = np.unique(np.append(I,np.random.randint(0,N,[b-I.size,])))
        return np.array(I)
            
        
class MyDataset(Dataset):


	def __init__(self, dataset_file, dataset_file_k562):
	
		self.dataset_hek = pd.read_csv(dataset_file)
		self.x_value_hek = self.encode_data(np.array(self.dataset_hek["Guide-Seq"]), np.array(self.dataset_hek["false_seq"]))
		self.guide_hot_one_hek, self.target_hot_one_hek = self.hot_one_encoding(np.array(self.dataset_hek["Guide-Seq"]), np.array(self.dataset_hek["false_seq"]))
		self.y_value_hek = self.dataset_hek["label"]
		self.y_value_class_hek = self.y_value_hek
		
		self.dataset_k562 = pd.read_csv(dataset_file_k562)
		self.x_value_k562 = self.encode_data(np.array(self.dataset_k562["Guide-Seq"]), np.array(self.dataset_k562["false_seq"]))
		self.guide_hot_one_k562, self.target_hot_one_k562 = self.hot_one_encoding(np.array(self.dataset_k562["Guide-Seq"]), np.array(self.dataset_k562["false_seq"]))
		self.y_value_k562 = self.dataset_k562["label"]
		self.y_value_cass_k562 = [y+2 for y in self.y_value_k562]
		
		self.full_dataset = []

		self.full_dataset.extend(self.dataset_hek)
		self.full_dataset.extend(self.dataset_k562)

		self.full_x_value = []
		self.full_x_value.extend(self.x_value_hek)
		self.full_x_value.extend(self.x_value_k562)


		self.full_y_value = []
		self.full_y_value.extend(self.y_value_hek)
		self.full_y_value.extend(self.y_value_k562)



		self.full_y_value_class = []
		self.full_y_value_class.extend(self.y_value_class_hek)
		self.full_y_value_class.extend(self.y_value_cass_k562)

		
		self.full_guide_hot_one = []
		self.full_guide_hot_one.extend(self.guide_hot_one_hek)
		self.full_guide_hot_one.extend(self.guide_hot_one_k562)
		
		self.full_target_hot_one = []
		self.full_target_hot_one.extend(self.target_hot_one_hek)
		self.full_target_hot_one.extend(self.target_hot_one_k562)
		


		self.shuffle_all()

		self.combined = []
		
		
		#self.combined = torch.cat([self.full_guide_hot_one[0], self.full_target_hot_one[0]], dim = 0).unsqueeze(dim=0)
		cat_list = []
		
		for idx in range(0,len(self.full_guide_hot_one)):
		
			#if idx > 1000: break
			#cat =  torch.cat([self.full_guide_hot_one[idx], self.full_target_hot_one[idx]], dim = 0).unsqueeze(dim=0)
			cat =  torch.cat([self.full_guide_hot_one[idx], self.full_target_hot_one[idx]], dim = 0)
			#self.combined = torch.cat([self.combined, cat], dim = 0)
			
			cat_list.append(cat)



		self.combined = torch.stack(cat_list, dim= 0)
		self.combined = self.combined.unsqueeze(dim=-1)

		
		
	def __len__(self):
		return len(self.full_x_value)
		
		
		
	def shuffle_all(self):
	
		shuffle_ind = list(range(len(self.full_y_value)))
		random.shuffle(shuffle_ind)
		
		#self.full_dataset = np.array(self.full_dataset)[shuffle_ind].tolist()
		self.full_x_value = torch.FloatTensor(np.array(self.full_x_value)[shuffle_ind].tolist())
		self.full_y_value = torch.LongTensor(np.array(self.full_y_value)[shuffle_ind].tolist())
		

		self.full_y_value_class = np.array(self.full_y_value_class)[shuffle_ind].tolist()
		

		self.full_guide_hot_one = np.array(self.full_guide_hot_one)[shuffle_ind].tolist()
		self.full_target_hot_one = np.array(self.full_target_hot_one)[shuffle_ind].tolist()


		return
		

	def __getitem__(self, idx):
	
	
		### change here
		x_val = torch.cat([self.full_guide_hot_one[idx], self.full_target_hot_one[idx]], dim = 0)

    
		return (self.full_x_value[idx], x_val), self.full_y_value[idx]
		
		
	def return_class_y(self, indeces = []):
	
		y_val = self.full_y_value_class
		
		if len(indeces) >0:
			y_val = list(np.array(y_val)[indeces])
	
		return y_val
		
	def seperate_by_class(self, indeces):
	
	
		y_class_val = self.return_class_y(indeces = indeces)
		indeces_hek = [indeces[en] for en, z in enumerate(y_class_val) if z == 0 or z == 1]
		indeces_k562 = [indeces[en] for en, z in enumerate(y_class_val) if z == 2 or z==3]
	
	
		return indeces_hek, indeces_k562
		
		
		
	def return_x(self):
	
		return self.full_x_value.tolist()
		
		
	def return_x_by_ind(self, indeces = []):
	

		return_x = torch.index_select(self.combined, 0, torch.tensor(indeces))
		
		
		return  np.array(return_x)
		
		
	def return_y_by_ind(self, indeces = []):
		
		y_val = self.full_y_value

		y_val = torch.index_select(y_val, 0, torch.tensor(indeces))

	
		return np.array(torch.tensor(y_val).unsqueeze(dim=-1))
		
		
		
	def return_y(self, indeces = []):
		
		y_val = self.full_y_value.tolist()
		
		if len(indeces) >0:
			y_val = list(np.array(y_val)[indeces])
	
		return y_val
		
		
		
	def hot_one(self, seq):
		hot_one_dict = {"A":0, "T":1, "G":2, "C":3}
		hot_one = [[0,0,0,0] for s in seq]
		for enum, s in enumerate(seq):
			hot_one[enum][hot_one_dict[s]] = 1
	
		return hot_one
		
	
	def hot_one_encoding(self, guide_seqs, seqs):
	
		hot_ones_target = []
		hot_ones_guide = []
	
		for en, guide in enumerate(guide_seqs):
			guide_hot_one = self.hot_one(guide)
			target_hot_one = self.hot_one(seqs[en])
			
			hot_ones_guide.append(guide_hot_one)
			hot_ones_target.append(target_hot_one)
			
		return torch.FloatTensor(hot_ones_guide), torch.FloatTensor(hot_ones_target)
		
		
	def encode_data(self, guide_seqs, seqs):
	
		encoded_seqs = []
	
		for seq_num, seq in enumerate(seqs):
		
			encoded_seqs.append([float(1) if nuc1!=guide_seqs[seq_num][en] else float(0) for en, nuc1 in enumerate(seq)])
			
		return encoded_seqs




def train(trainloader, net, criterion, optimizer, epoch):


	running_loss = 0.0
	for i, data in enumerate(trainloader):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)

		labels = labels.view(-1).float()
		loss = criterion(outputs.squeeze(), labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	return net, running_loss
	
	

def evaluate(dataloader, net,criterion, epoch):

	correct = 0
	y_values = []
	running_loss = 0.0
	predicted_list = []
	evals = []
	evals_auroc = []

	with torch.no_grad():
	    for data in dataloader:
	        inputs, labels = data
	        labels_save = labels
	        outputs = net(inputs)
	        labels = labels.view(-1).float()
	        loss = criterion(outputs.squeeze(), labels)
	        #_, predicted = torch.max(outputs.data, 1)
	        predicted = [0 if p < 0.5 else 1 for p in outputs.data]
	        
	        predicted_list.extend(outputs.data)
	        
	        y_values.extend(labels.tolist())
	        

        	running_loss += loss.item()
        	
        	eval_ = average_precision_score(labels.tolist(), outputs.data)
        	eval_auroc = 0
        	if len(np.unique(labels.tolist()))>1:eval_auroc = roc_auc_score(labels.tolist(), outputs.data)
        	evals.append(eval_)
        	evals_auroc.append(eval_auroc)
        
	
	return y_values, predicted_list, running_loss, np.mean(evals), np.mean(evals_auroc)

def get_undersample_mask(indeces, dataset):


	full_list = []
	y_val = dataset.return_y(indeces)
	pos_indeces = indeces[np.array([en for en, y in enumerate(y_val) if y == 1])]
	neg_indeces = indeces[np.array([en for en, y in enumerate(y_val) if y == 0])]
	num_to_sample = len(pos_indeces)
	y_samples = neg_indeces[:num_to_sample]
	
	full_list.extend(y_samples)
	full_list.extend(pos_indeces)
	random.shuffle(full_list)
	

	return full_list

def get_oversample_mask(indeces, dataset):


	full_list = []
	y_val = dataset.return_y(indeces)
	pos_indeces = indeces[np.array([en for en, y in enumerate(y_val) if y == 1])]

	neg_indeces = indeces[np.array([en for en, y in enumerate(y_val) if y == 0])]
	num_to_sample = len(neg_indeces)

	y_samples = []
	for i in range(math.ceil(num_to_sample/len(pos_indeces))):
		y_samples.extend(pos_indeces)
	
	y_samples = y_samples[:num_to_sample]
	
	full_list.extend(y_samples)
	full_list.extend(neg_indeces)
	random.shuffle(full_list)
	

	return full_list
	
	
	
	
	
class LIME():

	def __init__(self):
		self.tree = tree.DecisionTreeClassifier(max_depth = 5)
	
	
	def randomize_sample(self,matrix_input, num_sample = 500):
	

		matrix = matrix_input
		matrix_shape = np.shape(matrix)
		vec_len = matrix_shape[1]
		matrix_len = matrix_shape[0]
		
		### choose random base to modify ###
		data = []
		
		for _ in range(num_sample):
		

			matrix_mod = matrix.copy().squeeze(axis=-1)
			row = random.randint(0, matrix_len-1)
			base_list = [0,1,2,3]
			current_base = np.argmax(matrix_mod[row])

			base_list.remove(current_base)
			base = random.choice(base_list)
			hot_one = torch.zeros(vec_len)
			hot_one[base] = 1
			matrix_mod[row] = hot_one
			matrix_mod = np.expand_dims(matrix_mod, axis=-1)
			data.append(torch.tensor(matrix_mod))
			

		matrix_return = torch.stack(data, dim = 0)

		return np.array(matrix_return)
		
		
	def transform_categorical(self, x_data):
	
		new_x_data = []

		
		for matrix in x_data:
		
			
			transformd_features = [0] * len(matrix)
			
			for en, hot_one in enumerate(matrix):
				transformd_features[en] = np.argmax(hot_one).item()
		
			new_x_data.append(transformd_features)
	
		return new_x_data
		
	def fit_tree(self,x_data, y_prediction, tree_path):

		
		matrix = self.transform_categorical(x_data)
		

		self.tree.fit(matrix, y_prediction)
		tree_rules = export_text(self.tree, feature_names=None)
		fig, ax = plt.subplots(figsize=(10, 13))
		tree.plot_tree(self.tree, filled=True)
		plt.savefig(tree_path)
		plt.close()

		return tree_rules



if __name__ == "__main__":



	dataset = MyDataset("./hg19_samples_hek.csv", "./hg19_samples_k562.csv")
	
	indices = list(range(len(dataset)))
	split = int(np.floor(0.3 * len(dataset)))
	np.random.shuffle(indices)
	average_precision_score_folds = []
	fold_res = []
	
	kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)


	cv_list1_auprc_k562 = []
	cv_list2_auprc_k562 = []
	cv_list3_auprc_k562 = []

	cv_list1_auprc_hek = []
	cv_list2_auprc_hek = []
	cv_list3_auprc_hek = []

	"""
	net = modules.Sequential([  modules.Convolution(filtersize=(2,2,1,6),stride = (1,1)),\
                                modules.Rect(),\
                                #modules.Convolution(filtersize=(2,2,6,16),stride = (1,1)),\
                                #modules.Rect(),\
                                modules.Flatten(),\
                                modules.Linear(810, 400),\
                                modules.Rect(),\
                                modules.Linear(400, 20),\
                                modules.Rect(),\
                                modules.Linear(20, 1)
                            ])
                            
        """           

                            


	for enum, indeces in enumerate(kfold.split(dataset.return_x(), dataset.return_class_y())):



		net = modules.Sequential([  modules.Convolution(filtersize=(2,2,1,6),stride = (1,1)),\
                                modules.Rect(),\
                                modules.Convolution(filtersize=(2,2,6,16),stride = (1,1)),\
                                modules.Rect(),\
                                modules.Flatten(),\
                                modules.Linear(1408, 600),\
                                modules.Rect(),\
                                modules.Linear(600, 100),\
                                modules.Rect(),\
                                modules.Linear(100, 1)
                            ])
                            


		train_indices, test_val_ind = indeces[0], indeces[1]
	
		split2 = int(np.floor(0.5 * len(test_val_ind)))
		val_indices,test_indices = test_val_ind[split2:], test_val_ind[:split2]
		
		train_oversample_mask = get_oversample_mask(train_indices, dataset)


		x_values = dataset.return_x_by_ind(train_oversample_mask)
		y_values = dataset.return_y_by_ind(train_oversample_mask)
		
		
		x_values_validate =  dataset.return_x_by_ind(val_indices)
		y_values_validate = dataset.return_y_by_ind(val_indices)
		
		
		x_values_test =  dataset.return_x_by_ind(test_indices)
		y_values_test = dataset.return_y_by_ind(test_indices)
		

		#net.train(X = x_values, Y= y_values, Xval = x_values_validate, Yval = y_values_validate, iters = 10000, lrate=0.000005, batchsize=124)

		net.train(X = x_values, Y= y_values, Xval = x_values_validate, Yval = y_values_validate, iters = 10000, lrate=0.0001, batchsize=124)
		
		
		
		
		
		######## choose some test matrices #############
		batchsize = 30
		N = x_values_test.shape[0]
		samples = randperm(N,batchsize)
            	
		y = net.forward(x_values_test[samples,:])
		R = net.lrp(y)
		
		
		
		
		for sample_num, sample in enumerate(R):
		
			fig, ax = plt.subplots(figsize=(3, 13))
		
			pos = ax.imshow(R[sample_num])
			fig.colorbar(pos, ax=ax)
			plt.title("relevance map")
			plt.tight_layout()


			lime = LIME()
			samples = lime.randomize_sample(x_values_test[sample_num])
			y_samples = net.forward(samples)
			y_samples = np.array(torch.sigmoid(torch.tensor(y_samples)))
			
			pred = [1 if x > 0.5 else 0 for x in y_samples]
			
			if len(np.unique(pred)) > 1:
				
				plt.savefig("./plots/relevance_map_" + str(enum) + "_" + str(sample_num) +".svg")
				plt.close()
				
				
				rules = lime.fit_tree(samples, pred, "./plots/decision_tree_" + str(enum) + "_"+ str(sample_num) + ".svg")
				
				print("rules")
				print(rules)

			
			
		

	
