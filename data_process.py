from tqdm import tqdm

import numpy as np
import pandas as pd

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class Dataset_2048(Dataset):
    def __init__(self, file_path):
        # Since the record csv is too large, chunksize is set here to avoid overflow
        dt = pd.read_csv(file_path, chunksize = 1000000)
        self.data, self.T = [ ], ToTensor()
        # Here tqdm's bar cannot be displayed correctly
        # However this does not effect the process of data loading
        for _, df in enumerate(tqdm(dt)):
            data = df.values
            m = np.shape(data)[0]
            for i in range(m):
                board = np.array(data[i, 0 : 16], dtype = int)
                dir_ = np.array(data[i, 16], dtype = int)
                self.data.append((board, dir_))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X, y = self.data[index]
        X = self.T(data_recover(X))
        return (X, y)

# Recover a 4 × 4 board from a flattened board
def data_recover(board):
    ans = np.zeros((4, 4, 12))
    for i in range(4):
        for j in range(4):
            ans[i, j, board[4 * i + j]] = 1
    return ans