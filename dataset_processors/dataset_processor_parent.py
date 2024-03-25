'''
Abstract class for dataset processors
'''
from abc import ABC, abstractmethod
import torch
import clip
import random
import wandb


class DatasetProcessorParent(ABC):

    def __init__(self) -> None:
        self.val_dataset: torch.utils.data.Dataset = None
        self.classes: list = None
        
        self.device = wandb.config['cuda_device'] if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load(wandb.config['openai_clip_model'], device=self.device)
        # set seed
        torch.manual_seed(wandb.config['seed'])
        random.seed(wandb.config['seed'])
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
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=wandb.config['num_workers'])

        pass

    def load_train_dataloader(self) -> None:
        self.load_train_dataset()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=wandb.config['num_workers'])
        pass

    def print_dataset_stats(self):

        print(f'{self.name} dataset stats')
        print('num classes ', len(self.classes))
        print('classes ', self.classes)
        print('num val samples ', len(self.val_dataset))
        print('num train samples ', len(self.train_dataset))


