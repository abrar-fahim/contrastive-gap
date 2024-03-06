import torch

from src.config import *
from torch import nn


 
class Encoder(nn.Module):
    '''
    Parent class for Image and Text Encoders.
    Includes projection layer.
    '''
    def forward(self, input):
        '''
        Encode method.
        Input can be batch of tokenized texts or preprocessed images
        '''

        pass

