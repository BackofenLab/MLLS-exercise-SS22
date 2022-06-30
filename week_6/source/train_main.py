import glob
import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from  torch.utils.data import DataLoader
from net import cnn_net, siamese_cnn_net, sequence_cnn_net
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import BatchSampler
from sklearn.utils.class_weight import compute_class_weight
import random
import matplotlib.pyplot as plt


    
    
    
        
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
    
		return (self.full_x_value[idx], self.full_guide_hot_one[idx], self.full_target_hot_one[idx]), self.full_y_value[idx]
		
		
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
	        
	        #predicted_list.extend(predicted.tolist())
	        

	        
	        predicted_list.extend(outputs.data)
	        
	        y_values.extend(labels.tolist())
	        
	        
	        #correct += (predicted == labels_save).sum().item()
        	running_loss += loss.item()
        	
        	
        	eval_ = average_precision_score(labels.tolist(), outputs.data)
        	eval_auroc = roc_auc_score(labels.tolist(), outputs.data)
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
	
	
	
def plot_metric(epoch, predicted_list1_full, predicted_list2_full, predicted_list3_full, labels):


	fig, ax = plt.subplots(figsize=(3, 3))

	ax.set_title('AUPRC on validation set', color='C0')

	ax.plot(list(range(epoch+1)), predicted_list1_full, 'C1', label=labels[0])
	ax.plot(list(range(epoch+1)), predicted_list2_full, 'C2', label=labels[1])
	ax.plot(list(range(epoch+1)), predicted_list3_full, 'C3', label=labels[2])
	ax.set_xlabel("epochs")
	ax.set_ylabel("AUPRC")
	ax.legend()
	
	plt.show()



	return
	
	


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
	


	for enum, indeces in enumerate(kfold.split(dataset.return_x(), dataset.return_class_y())):
		train_indices, test_val_ind = indeces[0], indeces[1]
	
	
	
		split2 = int(np.floor(0.5 * len(test_val_ind)))
		val_indices,test_indices = test_val_ind[split2:], test_val_ind[:split2]
		
		
		train_oversample_mask = get_oversample_mask(train_indices, dataset)


		train_sampler = SubsetRandomSampler(train_oversample_mask)
		valid_sampler = SubsetRandomSampler(val_indices)
		

		test_indices_hek, test_indices_k562 = dataset.seperate_by_class(test_indices)		
		
		test_sampler_hek = SubsetRandomSampler(test_indices_hek)
		test_sampler_k562 = SubsetRandomSampler(test_indices_k562)

		train_loader = DataLoader(dataset, batch_size=30, sampler = train_sampler)
		validation_loader = DataLoader(dataset, batch_size = len(val_indices), sampler = valid_sampler)
		
		test_loader_hek = DataLoader(dataset, batch_size = len(test_indices), sampler = test_sampler_hek)
		test_loader_k562 = DataLoader(dataset, batch_size = len(test_indices), sampler = test_sampler_k562)
		
		
		#weights = compute_class_weight("balanced", classes = [0,1], y = dataset.return_y(train_indices))
		
	
		net = cnn_net()
		saved_cnn_net = cnn_net()
		optimizer1 = torch.optim.Adam(net.parameters(), lr=0.0001)
		criterion1 =  nn.BCELoss()
		siamese_net = siamese_cnn_net()
		saved_siamese_net = siamese_cnn_net()
		optimizer2 = torch.optim.Adam(siamese_net.parameters(), lr=0.0001)
		criterion2 = nn.BCELoss()
		sequence_net = sequence_cnn_net()
		saved_sequence_cnn_net = sequence_cnn_net()
		optimizer3 = torch.optim.Adam(sequence_net.parameters(), lr=0.0001)
		criterion3 = nn.BCELoss()
		
		y_value_list = []
		predicted_list1_full = []
		predicted_list2_full = []
		predicted_list3_full = []
		
		predicted_list1_auroc = []
		predicted_list2_auroc = []
		predicted_list3_auroc = []
		
		
		best_validation_cnn_net = np.inf
		best_validation_siamese_cnn_net = np.inf
		best_validation_sequence_cnn_net = np.inf

		for epoch in range(0, 10):	
			net, running_loss_train1 = train(train_loader, net = net, criterion = criterion1,  optimizer = optimizer1, epoch = epoch)
			y_values1, predicted_list1, running_loss1,eval_1, eval_auroc_1 = evaluate(validation_loader, net, criterion = criterion1,  epoch = epoch)

			siamese_net, running_loss_train2 = train(train_loader, net = siamese_net, criterion = criterion2,  optimizer = optimizer2, epoch = epoch)
			y_values2, predicted_list2, running_loss2,eval_2, eval_auroc_2 = evaluate(validation_loader, siamese_net, criterion = criterion2,  epoch = epoch)
			
			sequence_net, running_loss_train3 = train(train_loader, net = sequence_net, criterion = criterion3,  optimizer = optimizer3, epoch = epoch)
			y_values3, predicted_list3, running_loss3,eval_3, eval_auroc_3 = evaluate(validation_loader, sequence_net, criterion = criterion3,  epoch = epoch)
			
			y_value_list.extend(y_values1)
			predicted_list1_full.append(eval_1)
			predicted_list2_full.append(eval_2)
			predicted_list3_full.append(eval_3)
			predicted_list1_auroc.append(eval_auroc_1)
			predicted_list2_auroc.append(eval_auroc_2)
			predicted_list3_auroc.append(eval_auroc_3)
			

			if running_loss1 <= best_validation_cnn_net:
				best_validation_cnn_net = running_loss1
				saved_cnn_net.load_state_dict(net.state_dict())
				
			if running_loss2 <= best_validation_siamese_cnn_net:
				best_validation_siamese_cnn_net = running_loss2
				saved_siamese_net.load_state_dict(siamese_net.state_dict())
				
			if running_loss3 <= best_validation_sequence_cnn_net:
				best_validation_sequence_cnn_net = running_loss3
				saved_sequence_cnn_net.load_state_dict(sequence_net.state_dict())
				

			print(f'cnn-net: AUPRC in epoch {epoch}: {np.mean(eval_1)},  AUROC: {np.mean(eval_auroc_1)}, val-Loss:  {running_loss1 / len(validation_loader):.3f}, train-loss: {running_loss_train1 / len(train_loader):.3f}')
			print(f'siamese-net: AUPRC in epoch {epoch}: {np.mean(eval_2)},  AUROC: {np.mean(eval_auroc_2)}, val-Loss:  {running_loss2 / len(validation_loader):.3f}, train-loss: {running_loss_train2 / len(train_loader):.3f}')
			print(f'sequence_cnn_net: AUPRC in epoch {epoch}: {np.mean(eval_3)}, AUROC: {np.mean(eval_auroc_3)}, val-Loss:  {running_loss3 / len(validation_loader):.3f}, train-loss: {running_loss_train3 / len(train_loader):.3f}')
			print("##############################################################################################")
			
			
			
		#plot_metric(epoch, predicted_list1_full, predicted_list2_full, predicted_list3_full, ["cnn-net","siamese-net","sequence_cnn_net"])


	
		#### final test #####
		y_values1, predicted_list1, running_loss1,eval_1, eval_auroc_1 = evaluate(test_loader_hek, saved_cnn_net, criterion = criterion1,  epoch = epoch)
		y_values2, predicted_list2, running_loss2,eval_2, eval_auroc_2 = evaluate(test_loader_hek, saved_siamese_net, criterion = criterion2,  epoch = epoch)
		y_values3, predicted_list3, running_loss3,eval_3, eval_auroc_3 = evaluate(test_loader_hek, saved_sequence_cnn_net, criterion = criterion3,  epoch = epoch)
		
		cv_list1_auprc_hek.append(eval_1)
		cv_list2_auprc_hek.append(eval_2)
		cv_list3_auprc_hek.append(eval_3)
		print("##############################################################################################")
		print("TEST ON HEK CELL LINE")
		print("##############################################################################################")
		print(f'cnn-net: AUPRC in fold {enum}: {np.mean(eval_1)} , Test-Loss:  {running_loss1 / len(test_loader_hek):.3f}')
		print(f'siamese-net: AUPRC in fold {enum}:  {np.mean(eval_2)}, Test-Loss:  {running_loss2 / len(test_loader_hek):.3f}')
		print(f'sequence_cnn_net: AUPRC in fold {enum}:  {np.mean(eval_3)}, Test-Loss:  {running_loss3 / len(test_loader_hek):.3f}')
		print("##############################################################################################")
		
		
		#### final test #####
		y_values1, predicted_list1, running_loss1,eval_1, eval_auroc_1 = evaluate(test_loader_k562, saved_cnn_net, criterion = criterion1,  epoch = epoch)
		y_values2, predicted_list2, running_loss2,eval_2, eval_auroc_2 = evaluate(test_loader_k562, saved_siamese_net, criterion = criterion2,  epoch = epoch)
		y_values3, predicted_list3, running_loss3,eval_3, eval_auroc_3 = evaluate(test_loader_k562, saved_sequence_cnn_net, criterion = criterion3,  epoch = epoch)
		
		cv_list1_auprc_k562.append(eval_1)
		cv_list2_auprc_k562.append(eval_2)
		cv_list3_auprc_k562.append(eval_3)
		print("##############################################################################################")
		print("TEST ON K562 CELL LINE")
		print("##############################################################################################")
		print(f'cnn-net: AUPRC in fold {enum}: {np.mean(eval_1)} , Test-Loss:  {running_loss1 / len(test_loader_k562):.3f}')
		print(f'siamese-net: AUPRC in fold {enum}:  {np.mean(eval_2)}, Test-Loss:  {running_loss2 / len(test_loader_k562):.3f}')
		print(f'sequence_cnn_net: AUPRC in fold {enum}:  {np.mean(eval_3)}, Test-Loss:  {running_loss3 / len(test_loader_k562):.3f}')
		print("##############################################################################################")



	for num, auprc in enumerate(cv_list1_auprc_hek):
		print("##############################################################################################")
		print("TEST ON HEK CELL LINE")
	
		print("##############################################################################################")
		print(f'cnn-net: AUPRC in fold {num}: {np.mean(auprc):.3f}')
		print(f'siamese-net: AUPRC in fold {num}:  {np.mean(cv_list2_auprc_hek[num]):.3f}')
		print(f'sequence_cnn_net: AUPRC in fold {num}:  {np.mean(cv_list3_auprc_hek[num]):.3f}')
		print("##############################################################################################")

		print("##############################################################################################")
		print("TEST ON K562 CELL LINE")
		
		print("##############################################################################################")
		print(f'cnn-net: AUPRC in fold {num}: {np.mean(cv_list1_auprc_k562[num]):.3f}')
		print(f'siamese-net: AUPRC in fold {num}:  {np.mean(cv_list2_auprc_k562[num]):.3f}')
		print(f'sequence_cnn_net: AUPRC in fold {num}:  {np.mean(cv_list3_auprc_k562[num]):.3f}')
		print("##############################################################################################")
	
	
	
	print("##############################################################################################")
	print("TEST ON HEK CELL LINE")
	print("##############################################################################################")
	print(f'cnn-net: Average AUPRC over all folds: {np.mean(cv_list1_auprc_hek):.3f}')
	print(f'siamese-net: Average AUPRC over all folds: {np.mean(cv_list2_auprc_hek):.3f}')
	print(f'sequence_cnn_net: Average AUPRC over all folds: {np.mean(cv_list3_auprc_hek):.3f}')
	print("##############################################################################################")
	
	print("##############################################################################################")
	print("TEST ON K562 CELL LINE")
	print("##############################################################################################")
	print(f'cnn-net: Average AUPRC over all folds: {np.mean(cv_list1_auprc_k562):.3f}')
	print(f'siamese-net: Average AUPRC over all folds: {np.mean(cv_list2_auprc_k562):.3f}')
	print(f'sequence_cnn_net: Average AUPRC over all folds: {np.mean(cv_list3_auprc_k562):.3f}')
	print("##############################################################################################")
	
	
