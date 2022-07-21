import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from  torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
	
	def __init__(self, dataset_path, label_path):
	
		self.x_values, self.y_values = self.get_dataset(dataset_path, label_path)
		self.input_shape = np.shape(self.x_values)[1]
		self.distinct_labels= len(np.unique(self.y_values))
		

	def get_dataset(self,dataset_path, label_path):
	
		dataset = np.load(dataset_path)
		
		num_genes = np.shape(dataset)[1]

		label_file = open(label_path)
		labels = [l[:-1] for l in label_file.readlines()]
		
		gene_sum = np.sum(dataset, axis=0)

		non_empty_indece = [enum for enum, _ in enumerate(range(num_genes)) if gene_sum[enum] != 0]

		dataset = np.take(dataset, non_empty_indece, axis = 1)
		

		return dataset, labels
		
		
		
	def return_y(self):
		return self.y_values
		
	def return_x(self):
		return self.x_values
		
		
	def __len__(self):
		return len(self.y_values)

	def __getitem__(self, idx):
		return self.x_values[idx], self.y_values[idx]


def my_collate(batch):


    data = [item[0] for item in batch]
    target = [torch.FloatTensor(item[1]) for item in batch]
    target = target
    return [data, target]

    	
    	
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        

        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=1280
        )
                
        self.batch_norm_in = nn.BatchNorm1d(kwargs["input_shape"])
        

        
        self.batch_norm1 = nn.BatchNorm1d(1280)
        
        self.encoder_output_layer = nn.Linear(
            in_features=1280, out_features=20
                    )
                    
        self.batch_norm2 = nn.BatchNorm1d(20)
                    
        self.decoder_hidden_layer = nn.Linear(
            in_features=20, out_features=1280
        )
        
        self.batch_norm3 = nn.BatchNorm1d(1280)
        
        self.decoder_output_layer = nn.Linear(
            in_features=1280, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
    

        activation = self.batch_norm_in(features)
        
        activation = self.encoder_hidden_layer(activation)

        

        activation = torch.relu(activation)
        activation = self.batch_norm1(activation)
        
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        code = self.batch_norm2(code)
        
        
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        
        activation = self.batch_norm3(activation)
        
        reconstructed = self.decoder_output_layer(activation)

        
        return reconstructed, code
    	
    	
    	
def train_model(trainloader, net, criterion, optimizer, epoch):

	torch.autograd.set_detect_anomaly(True)
	running_loss = 0.0
	net.zero_grad()
	
	
	for i, data in enumerate(trainloader):
		input_data, target_data = data
		loss = 0
		

		optimizer.zero_grad()
		output, encoded = net(input_data)
		
		loss = criterion(output,input_data)
		
		loss.backward()
		running_loss = running_loss + loss

		optimizer.step()


	return net, running_loss/len(trainloader)


def get_embedding(dataloader, net):

	encoding_list = []

	with torch.no_grad():
		for data in dataloader:
			input_data, target_data = data
			output, encoding = net(input_data)
			
			
			encoding_list.extend(encoding.tolist())
			
			
			
	return encoding_list
	



def eval_model(dataloader, net, criterion, epoch):


	running_loss = 0
	pearson_running_loss2 = 0
	input_data_list = []
	output_data_list = []
	target_data_list = []
	
	with torch.no_grad():
		for data in dataloader:
			input_data, target_data = data
			loss = 0

			output, hidden = net(input_data)



			loss = criterion(output,input_data)

			
			running_loss = running_loss + loss
			
			input_data_list.append(input_data)
			output_data_list.append(output.squeeze())

			target_data_list.append(input_data)
			

	return running_loss, output_data_list, input_data_list, target_data_list, pearson_running_loss2
	
	
def kmean_clustering(data, num_labels):

	kmeans = KMeans(n_clusters=num_labels, random_state=0).fit(data)

	return kmeans.labels_
	
	
	
def spectral_clustering(data, num_labels):

	matrix = similarity_matrix(data)
	clustering = SpectralClustering(n_clusters=num_labels, assign_labels='kmeans', random_state=0, affinity = "precomputed").fit(matrix)

	return clustering.labels_
	
	
	
	
def similarity_matrix(input):


	similarity_matrix = np.zeros(shape=(len(input),len(input)))
	

	for column_ind,data in enumerate(input):

		vx = data - np.mean(data)
		
		for row_index, data2 in enumerate(input):
		
			vy = data2 - np.mean(data2)
	
			similarity = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
			similarity_matrix[column_ind][row_index] = max(similarity, 0)

	similarity_matrix = [np.nan_to_num(sim) for sim in similarity_matrix]
	
	
	return similarity_matrix	
	
	

def main():


	dataset = MyDataset("./gene_expression/matrix.npy", "./gene_expression/labels.txt")
	

	kfold_loss = []
	graph_sim_loss = []
	
	strat_split = StratifiedShuffleSplit(test_size = 0.1, random_state = 1)
	
	
	X_train, X_test, y_train, y_test  = train_test_split(range(len(dataset.return_x())), range(len(dataset.return_y())),test_size=0.2, random_state=42)
	
	train_indices, test_val_ind = X_train, X_test
	
	
	
	net = AE(input_shape = dataset.input_shape)
	saved_net = AE(input_shape = dataset.input_shape)
	optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.00009)
	criterion = nn.L1Loss()

	split2 = int(np.floor(0.5 * len(test_val_ind)))
	val_indices,test_indices = test_val_ind[split2:], test_val_ind[:split2]
		
		
	g = torch.Generator()
	g.manual_seed(0)

	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)
	test_sampler = SubsetRandomSampler(test_indices)


	full_dataloader = DataLoader(dataset, batch_size = 64)
	train_loader = DataLoader(dataset, batch_size=64, sampler = train_sampler, generator=g)
	validation_loader = DataLoader(dataset, batch_size = 64, sampler = valid_sampler, generator=g)
	test_loader = DataLoader(dataset, batch_size = 64, sampler = test_sampler, generator=g)
	best_validation = np.inf
		
		
	epoch_range = 25
	
	for epoch in range(epoch_range):
	
		net, running_loss = train_model(train_loader, net, criterion, optimizer, epoch)
		running_loss_val, output_data_list, input_data_list, target_data_list, pearson_running_loss2 = eval_model(validation_loader, net, criterion, epoch)
		print(f"In Epoch: {epoch}, Training-Loss: {running_loss}, Validation-Loss: {running_loss_val}")
			
			
		if running_loss_val <= best_validation and epoch > int(epoch_range/3):
			
			best_validation = running_loss_val
			saved_net.load_state_dict(net.state_dict())
				
				
	running_loss_test, output_data_list, input_data_list, target_data_list, pearson_running_loss2 = eval_model(test_loader, saved_net, criterion, epoch)
			
	print("#######################################################################")
	print(f"Test-Loss: {running_loss_test}")
		

		
	embedding = get_embedding(full_dataloader, saved_net)
		

		
	cluster_labels = kmean_clustering(embedding, dataset.distinct_labels)
	cluster_labels_spectral = spectral_clustering(embedding, dataset.distinct_labels)
		
		
	rand_score_eval = rand_score(dataset.return_y(),cluster_labels)
	adj_rand_score = adjusted_rand_score(dataset.return_y(), cluster_labels)
		
	rand_score_eval_spectral = rand_score(dataset.return_y(),cluster_labels_spectral)
	adj_rand_score_spectral = adjusted_rand_score(dataset.return_y(), cluster_labels_spectral)
		
		
	print("#######################################################################")
	print(f"rand score for kmeans-clustering: {rand_score_eval}, adjusted rand score for kmeans-clustering: {adj_rand_score}")
	print(f"rand score for spectral-clustering: {rand_score_eval_spectral}, adjusted rand score for spectral-clustering: {adj_rand_score_spectral}")
	print("#######################################################################")
		
	


if __name__ == "__main__":
	main()


