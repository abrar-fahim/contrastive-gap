'''
read a tsv file onto a pandas dataframe
and convert it to a pytorch dataset
''' 


import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import json
import random
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

class WitDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path, sep='\t')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample





