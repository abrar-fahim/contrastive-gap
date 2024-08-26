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

from torchvision.datasets import CIFAR10

from tqdm import tqdm


class CIFAR10Processor(DatasetProcessorParent):

    def __init__(self, root = '.') -> None:
        self.root = f'{root}/datasets/cifar10'
        super().__init__()

        self.name = 'CIFAR 10'
        self.keyname = self.name.replace(' ', '').lower()
        dataset_config = eval(open(f"{self.root}/classes.py", "r").read())

        # print('dataset config ', dataset_config)
        classes, templates = dataset_config["classes"], dataset_config["templates"]

        self.templates = templates
        self.print_dataset_stats()

        


    def load_val_dataset(self):
        self.val_dataset = CIFAR10(root=self.root, train=False, download=True, transform=self.preprocess)

        self.classes = self.val_dataset.classes

        # add 'photo of ' to the beginning of each class name


        # self.classes = ['photo of ' + class_name for class_name in self.classes]


    def set_class_embeddings(self, class_embeddings: torch.Tensor):
        self.class_embeddings = class_embeddings

    def load_train_dataset(self):
        self.train_dataset = CIFAR10(root=self.root, train=True, download=True, transform=self.preprocess)





