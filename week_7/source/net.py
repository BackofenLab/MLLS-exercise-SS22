import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, max_length):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.LSTM = nn.LSTM(max_length, hidden_size, batch_first = True)
        self.out1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.out2 = nn.Linear(self.hidden_size//2, 1)
        
    
    def forward(self, input, hidden):
    
        hot_one = self.return_hotone(input)
        
        output, hidden = self.LSTM(hot_one, hidden)
        output = F.relu(output)
        output = F.relu(self.out1(output))
        output = self.out2(output)
        return output, hidden
    

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        
    
    
    def return_hotone(self, input):
    
    	hot_one_vec = []
    	for batch in input:
    		for val in batch:

    			hot_one = [0] * self.max_length
    			hot_one[val] = 1
    			hot_one_vec.append(hot_one)
    	
    	return torch.FloatTensor([hot_one_vec])     

