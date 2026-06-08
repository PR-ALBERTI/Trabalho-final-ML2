import torch
from torch.utils import data
import os
import numpy as np

class fMRICNNcustomDataset(data.Dataset):
    def __init__(self, data_folder):
        self.path_list = []
        self.root_folder = data_folder
    
        for path, subdirs, files in os.walk(self.root_folder):
            for name in files:
                if name.endswith('.p'): 
                    self.path_list.append(os.path.join(path, name))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.path_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Correção do bug: a variável era 'idx' mas não estava definida
        if torch.is_tensor(index):
            index = index.tolist() 
        
        # Carrega os dados
        sample = torch.load(self.path_list[index])
        x_data = np.array(sample['X'])
        y_cluster = np.array(sample['y'])
        
        return x_data, y_cluster