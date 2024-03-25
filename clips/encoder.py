import torch


from torch import nn
import importlib

import wandb

 
class Encoder(nn.Module):
    '''
    Parent class for Image and Text Encoders.
    Includes projection layer.
    '''

    def __init__(self):
        
        super().__init__()

        self.hidden_size: int

        assert torch.initial_seed() == wandb.config['seed'], "Seed not set properly"


        
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

