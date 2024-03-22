import torch
from torch import nn

from abc import ABC, abstractmethod

from src.config import training_hyperparameters
import importlib


class ClipParent(nn.Module):
    def __init__(self):
        super().__init__()

        assert torch.initial_seed() == training_hyperparameters['seed'], "Seed not set properly"

        pass

    @abstractmethod
    def encode_image(self, image):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass

    
    def project_image(self, image):
        pass

    def project_text(self, text):
        pass
    

    @abstractmethod
    def forward(self, image, text, scale=True):
        pass






