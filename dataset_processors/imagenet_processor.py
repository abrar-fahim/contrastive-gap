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

from torchvision.datasets import ImageNet

from tqdm import tqdm


class ImageNet1k(DatasetProcessorParent):

    def __init__(self) -> None:
        self.root = './datasets/imagenet1k'
        super().__init__()

        self.name = 'ImageNet1k2'
        self.keyname = self.name.replace(' ', '').lower()
        dataset_config = eval(open(f"{self.root}/classes.py", "r").read())

        # print('dataset config ', dataset_config)
        classes, templates = dataset_config["classes"], dataset_config["templates"]

        self.templates = templates
        self.print_dataset_stats()

        


    def load_val_dataset(self):
        self.val_dataset = ImageNet(root=self.root, split='val', transform=self.preprocess)

        self.classes = self.val_dataset.classes

        # add 'photo of ' to the beginning of each class name


        # self.classes = ['photo of ' + class_name for class_name in self.classes]


    def set_class_embeddings(self, class_embeddings: torch.Tensor):
        self.class_embeddings = class_embeddings

    def load_train_dataset(self):

        # throw exception
        raise Exception('ImageNet1k does not have a train dataset yet')
        self.train_dataset = ImageNet(root=self.root, split='train', transform=self.preprocess)





