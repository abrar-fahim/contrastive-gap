'''
Abstract class for dataset processors
'''
from abc import ABC, abstractmethod
import torch
import clip

from src.config import training_hyperparameters
import random


class DatasetProcessorParent(ABC):

    def __init__(self) -> None:
        self.val_dataset: torch.utils.data.Dataset = None
        self.classes: list = None
        
        self.device = training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=self.device)
        # set seed
        torch.manual_seed(training_hyperparameters['seed'])
        random.seed(training_hyperparameters['seed'])
        self.load_train_dataloader()
        self.load_val_dataloader()
        pass

    @abstractmethod
    def load_train_dataset(self) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def load_val_dataset(self) -> torch.utils.data.Dataset:
        pass

    def load_val_dataloader(self) -> None:
        self.load_val_dataset()
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=training_hyperparameters['batch_size'], shuffle=True, num_workers=training_hyperparameters['num_workers'])

        pass

    def load_train_dataloader(self) -> None:
        self.load_train_dataset()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=training_hyperparameters['batch_size'], shuffle=True, num_workers=training_hyperparameters['num_workers'])
        pass

    def print_dataset_stats(self):

        print(f'{self.name} dataset stats')
        print('num classes ', len(self.classes))
        print('classes ', self.classes)
        print('num val samples ', len(self.val_dataset))
        print('num train samples ', len(self.train_dataset))


