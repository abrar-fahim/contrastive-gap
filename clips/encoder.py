import torch

from src.config import training_hyperparameters
from torch import nn
import importlib

 
class Encoder(nn.Module):
    '''
    Parent class for Image and Text Encoders.
    Includes projection layer.
    '''

    def __init__(self):
        
        super().__init__()

        self.hidden_size: int

        assert torch.initial_seed() == training_hyperparameters['seed'], "Seed not set properly"


        
    def forward(self, input):
        '''
        Encode method.
        Input can be batch of tokenized texts or preprocessed images
        '''

        pass

    def pool_hidden_state(self, hidden_state: torch.FloatTensor, input_ids: torch.Tensor) -> torch.FloatTensor:
        '''
        Pool hidden states
        '''

        pass

