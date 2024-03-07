import torch

from src.config import *
from torch import nn


 
class Encoder(nn.Module):
    '''
    Parent class for Image and Text Encoders.
    Includes projection layer.
    '''

    def __init__(self):
        
        super().__init__()

        assert torch.initial_seed() == training_hyperparameters['seed'], "Seed not set properly"


        
    def forward(self, input):
        '''
        Encode method.
        Input can be batch of tokenized texts or preprocessed images
        '''

        pass

