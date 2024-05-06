import torch

import sys
import os

# import pca from skleran
from sklearn.decomposition import PCA
import numpy as np
# import matplotlib plt
import matplotlib.pyplot as plt




# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# randomly generate points
np.random.seed(0)
torch.manual_seed(0)
T = 0.01


logit_scale = np.log(1/T)

print('np log(100)' , logit_scale)

print('recovered T ', 1 / np.exp(logit_scale))
