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


        self.W = None

        if wandb.config['W_layer_gap'] >= 0:
            self.W: torch.FloatTensor = torch.empty(512, 512)

            self.W_set = False


        
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

    def setW(self, W: torch.FloatTensor):
        '''
        Set W for alignment
        '''

        assert wandb.config['W_layer_gap'] >= 0, "W_layer_gap must be >= 0"
        

        assert W.shape == (512, 512), f"self.W.shape = {self.W.shape}"

        self.W = W.to(self.device)

        self.W_set = True


    
    def align_embeddings(self, embeds: torch.FloatTensor):
        '''
        Aligns embeddings to the first encoder's embeddings
        '''

        assert self.W_set, "W not set"

        # normalize embeds 
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)

        if self.W is None:
            return embeds

        return embeds @ self.W.T
    

