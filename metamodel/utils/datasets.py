import torch
from torch.utils.data import Dataset
import numpy as np


class WindowedPredictionDatasetContinuous(Dataset):
    def __init__(
        self, input_params, cell_states, 
        window_size, tnf_vector = None, init_cell_states = None
    ):
        self.input_params = torch.from_numpy(input_params)
        self.cell_states = torch.from_numpy(cell_states)
        self.tnf_vector = torch.from_numpy(tnf_vector)\
            if tnf_vector is not None else None
        self.init_cell_states = torch.from_numpy(init_cell_states)\
            if init_cell_states is not None else None

        self.window_size = window_size

    def __getitem__(self, index):
        # Ensure that the index is within the range of the dataset
        if not (0 <= index < len(self)):
            raise ValueError("Index out of range")

        window_slice = (index*self.window_size, index*self.window_size+self.window_size)
        cell_states = self.cell_states[:,:, window_slice[0]:window_slice[1], :]
        
        if self.tnf_vector is not None:
            tnf_vector = self.tnf_vector[:,:, window_slice[0]:window_slice[1], :]

        labels = self.cell_states[:,:, window_slice[0]+1:window_slice[1]+1]

        if self.init_cell_states is not None:
            return self.input_params, self.init_cell_states, cell_states, labels
        if self.tnf_vector is not None:
            return self.input_params, self.input_params, cell_states, tnf_vector, labels
        return self.input_params, self.input_params, cell_states, labels

    def __len__(self):
        return int(np.floor((self.cell_states.shape[2]-1) / self.window_size))
  