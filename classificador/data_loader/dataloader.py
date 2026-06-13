import torch
from torch.utils import data
import os
import numpy as np

# This class is thought of being used for 1 subject
class fMRICNNcustomDataset(data.Dataset):
    def __init__(self, data_folder):
        self.path_list = []
        self.root_folder = data_folder
    
        # Walk through the directory and collect paths to all .p files
        for path, subdirs, files in os.walk(self.root_folder):
            for name in files:
                if name.endswith('.p'): 
                    self.path_list.append(os.path.join(path, name))

    def __len__(self):
        # Simply return the number of samples in the dataset
        'Denotes the total number of samples'
        return len(self.path_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        if torch.is_tensor(index):
            index = index.tolist() 
        
        # Load the sample
        sample = torch.load(self.path_list[index])
        x_data = np.array(sample['X'])
        y_cluster = np.array(sample['y'])
        
        return x_data, y_cluster