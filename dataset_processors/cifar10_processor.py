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

from torchvision.datasets import CIFAR10


class CIFAR10Processor(DatasetProcessorParent):

    def __init__(self) -> None:

        self.val_dataset = None
        self.classes = None
        
        self.device = training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=self.device)
        # set seed
        torch.manual_seed(training_hyperparameters['seed'])
        random.seed(training_hyperparameters['seed'])
        self.load_val_dataset()


    def load_val_dataset(self):
        self.val_dataset = CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=self.preprocess)

        self.classes = self.val_dataset.classes

        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]

    def load_train_dataset(self):
        return
    
    def print_dataset_stats(self):
        print('CIFAR10 dataset stats')
        print('num classes ', len(self.classes))
        print('classes ', self.classes)
        print('num val samples ', len(self.val_dataset))



