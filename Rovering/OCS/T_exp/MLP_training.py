import os 
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
folders = [os.path.join(current_dir, d) for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

def load_data_from_folders(folders):
    X_data = []
    Y_dist_data = []
    Y_norm_data = []
    
    for folder in folders:
        X_path = os.path.join(folder, 'X_processing.npy')
        Y_dist_path = os.path.join(folder, 'Y_dist.npy')
        Y_norm_path = os.path.join(folder, 'Y_norm.npy')
        
        if os.path.exists(X_path) and os.path.exists(Y_dist_path) and os.path.exists(Y_norm_path):
            X_data.append(np.load(X_path))
            Y_dist_data.append(np.load(Y_dist_path))
            Y_norm_data.append(np.load(Y_norm_path))
    
    return np.concatenate(X_data), np.concatenate(Y_dist_data), np.concatenate(Y_norm_data)

X, Y_dist, Y_norm = load_data_from_folders(folders)
print(X.shape, Y_dist.shape, Y_norm.shape)
# # Convert to PyTorch tensors
# X_tensor = torch.from_numpy(X).float()
# Y_dist_tensor = torch.from_numpy(Y_dist).float()
# Y_norm_tensor = torch.from_numpy(Y_norm).float()

# # Create dataset and dataloader
# dataset = TensorDataset(X_tensor, Y_dist_tensor, Y_norm_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)