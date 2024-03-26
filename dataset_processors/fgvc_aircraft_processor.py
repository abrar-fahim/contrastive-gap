import torchvision.datasets as dset
import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
from src.utils import  get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
import os
from clips.hf_clip import HFClip
import numpy as np
import wandb

from torchvision.datasets import FGVCAircraft


class FGVCAircraftProcessor(DatasetProcessorParent):

    def __init__(self) -> None:
        self.root = './datasets/fgvc_aircraft'
        super().__init__()

        self.name = 'FGVC Aircraft'
        self.keyname = self.name.replace(' ', '').lower()
        self.print_dataset_stats()
        


    def load_val_dataset(self):
        self.val_dataset = FGVCAircraft(root=self.root, download=True, split='val', transform=self.preprocess)

        self.classes = self.val_dataset.classes

        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]

    def load_train_dataset(self):
        self.train_dataset = FGVCAircraft(root=self.root, download=True, split='train', transform=self.preprocess)





