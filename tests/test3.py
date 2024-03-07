import torch

import numpy as np
import random

def get_seed():
    print(torch.randint(0, 5, (10,)))
    print('done')