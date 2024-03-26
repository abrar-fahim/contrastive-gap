import torchvision.datasets as dset
import clip
import torch
from src.config import *
import random
from torch.utils.data import DataLoader, Subset
from src.utils import  get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
import os
from clips.hf_clip import HFClip
import numpy as np

from torchvision.datasets import Food101


class Food101Processor(DatasetProcessorParent):

    def __init__(self) -> None:
        self.root = './datasets/food101'
        super().__init__()
        self.name = 'Food 101'
        self.keyname = 'food101'
        
        self.print_dataset_stats()



    def load_val_dataset(self):
        self.val_dataset = Food101(root=self.root, split='test', download=True, transform=self.preprocess)

        self.classes = self.val_dataset.classes
    
        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]

    def load_train_dataset(self):

        self.train_dataset = Food101(root=self.root, split='train', download=True, transform=self.preprocess)

        self.classes = self.val_dataset.classes
        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]
        return
    





