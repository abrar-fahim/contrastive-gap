import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import torch

from torch.optim import SGD

from torch.distributions.multivariate_normal import MultivariateNormal

# generate cluster of points around a point in unit sphere

d = 3

n = 1000

# generate random point in unit sphere

a = MultivariateNormal(torch.tensor([0, 1, 0]), torch.eye(d))


b = MultivariateNormal(torch.tensor([0, -1, 0]), torch.eye(d))

