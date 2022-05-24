import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class TrainDataLoader(Dataset):
    def __init__(self, data, c):
        self.data = torch.Tensor(data)
        self.c = c

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.c[index]


def load_random_new_data(batch_size):
    data = np.random.normal(size=(2000, 2074))
    c = np.random.randint(0, 99, size=(2000,))
    train_set = TrainDataLoader(data, c)
    train_dataset = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_dataset
