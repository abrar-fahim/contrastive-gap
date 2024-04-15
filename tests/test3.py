import torch

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.my_ce_loss import MyCEAlignmentLoss, MyCrossEntropyLoss




# identity matrix of size n
def identity(n):
    return torch.eye(n)



a = identity(32) * 100

print(a)

labels = torch.arange(32)

print('my ce loss ', MyCrossEntropyLoss()(a,labels))

print('my ce alignment loss ', MyCEAlignmentLoss()(a, labels))



