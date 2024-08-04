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

from torchvision.datasets import VOCDetection


class VocProcessor(DatasetProcessorParent):

    def __init__(self) -> None:
        self.root = './datasets/voc2007'
        super().__init__()

        self.name = 'Voc 2007'
        self.keyname = 'voc2007'
        dataset_config = eval(open(f"{self.root}/classes.py", "r").read())

        classes, templates = dataset_config["classes"], dataset_config["templates"]
        self.templates = templates
        self.print_dataset_stats()
        


    def load_val_dataset(self):
        self.val_dataset = VOCDetection(root=self.root, year='2007', download=True, image_set='val', transform=self.preprocess)

        self.classes = self.val_dataset.classes

        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]

    def load_train_dataset(self):
        self.val_dataset = VOCDetection(root=self.root, year='2007', download=True, image_set='train', transform=self.preprocess)





