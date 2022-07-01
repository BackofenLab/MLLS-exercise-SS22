import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class cnn_net(nn.Module):

    def __init__(self):
        super(cnn_net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 6, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 3)
        self.fc1 = nn.Linear(304, 60) 
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 1)
        self.softmax = torch.nn.Softmax(dim = 1)
    def forward(self, input_):
        x, guide, target = input_[0], input_[1],  input_[2]
    
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        #x = F.max_pool1d(x, 2)
        #x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        

        x = F.relu(self.conv2(x))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)



class siamese_cnn_net(nn.Module):

    def __init__(self):
        super(siamese_cnn_net, self).__init__()
        self.conv1_head1 = nn.Conv1d(in_channels = 1, out_channels = 6, kernel_size = 3)
        self.conv2_head1 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 3)
        
        
        self.conv1_head2 = nn.Conv1d(in_channels = 4, out_channels = 6, kernel_size = 3)
        self.conv2_head2 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 3)
        
        self.fc1 = nn.Linear(608, 60) 
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, input_):
    
    
        mismatches, guide, target = input_[0], input_[1],  input_[2]
    
    	
    
        ### head one 
        #guide = guide.permute(0, 2, 1)
        #guide = F.relu(self.conv1_head1(guide))
        #guide = F.max_pool1d(guide, 2)
        #guide = F.max_pool1d(F.relu(self.conv2_head1(guide)), 2)
        #guide = F.relu(self.conv2_head1(guide))

        mismatches = mismatches.unsqueeze(1)
        mismatches = F.relu(self.conv1_head1(mismatches))
        mismatches = F.relu(self.conv2_head1(mismatches))

        ### head two
    
        target = target.permute(0, 2, 1)
        target = F.relu(self.conv1_head2(target))
        #target = F.max_pool1d(target, 2)
        #target = F.max_pool1d(F.relu(self.conv2_head2(target)), 2)
        

        target = F.relu(self.conv2_head2(target))
        
        
        x = torch.cat([mismatches, target], dim = 1)


        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


class sequence_cnn_net(nn.Module):

    def __init__(self):
        super(sequence_cnn_net, self).__init__()
        self.conv1_head1 = nn.Conv1d(in_channels = 4, out_channels = 6, kernel_size = 3)
        self.conv2_head1 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 3)
        
        
        self.conv1_head2 = nn.Conv1d(in_channels = 4, out_channels = 6, kernel_size = 3)
        self.conv2_head2 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 3)
        
        
        self.conv1_head3 = nn.Conv1d(in_channels = 1, out_channels = 6, kernel_size = 3)
        self.conv2_head3 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 3)
        
        self.fc1 = nn.Linear(912, 60) 
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, input_):
    
        mismatches, guide, target = input_[0], input_[1],  input_[2]
    
    	

        ### head one 

        guide = guide.permute(0, 2, 1)
        guide = F.relu(self.conv1_head1(guide))
        #guide = F.max_pool1d(guide, 2)
        #guide = F.max_pool1d(F.relu(self.conv2_head1(guide)), 2)
       

        guide =F.relu(self.conv2_head1(guide))
               
        ### head two

        target = target.permute(0, 2, 1)
        target = F.relu(self.conv1_head2(target))
        #target = F.max_pool1d(target, 2)
        #target = F.max_pool1d(F.relu(self.conv2_head2(target)), 2)

        target = F.relu(self.conv2_head2(target))
        
        ### head three

        
        mismatches = mismatches.unsqueeze(1)
        mismatches = F.relu(self.conv1_head3(mismatches))
        #mismatches = F.max_pool1d(mismatches, 2)
        #mismatches = F.max_pool1d(F.relu(self.conv2_head3(mismatches)), 2)
        

        mismatches = F.relu(self.conv2_head3(mismatches))

        x = torch.cat([guide, target, mismatches], dim = 1)


        
        x = torch.flatten(x, 1)

        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


