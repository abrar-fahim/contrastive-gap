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
        super().__init__()


    def load_val_dataset(self):
        self.val_dataset = Food101(root='./datasets/food101', split='test', download=True, transform=self.preprocess)

        self.name = 'Food 101'

        

        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]

    def load_train_dataset(self):

        self.train_dataset = Food101(root='./datasets/food101', split='train', download=True, transform=self.preprocess)

        self.classes = self.val_dataset.classes
        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]
        return
    





