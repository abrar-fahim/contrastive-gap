import torch
from torch import nn

'''

'''




class ProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = 512
        self.output_dim = 512

        self.name = 'projection layer'

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.output_dim),
        )

        self.init_weights()



    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def __repr__(self):
        return f'{self.name} - input_dim: {self.input_dim}, output_dim: {self.output_dim}'




class MultiLayerProjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = 512
        self.output_dim = 512

        self.name = 'projection layer'

        self.network = nn.Sequential(
            nn.ReLU(), # needs to start with ReLU because last layer just before this is nn.Linear
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim),
        )

        self.init_weights()



    def forward(self, x):

        residual = x
        y = self.network(x)
        out = y + residual
        return out

    def init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def __repr__(self):
        return f'{self.name} - input_dim: {self.input_dim}, output_dim: {self.output_dim}'