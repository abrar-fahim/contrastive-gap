import torch

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


a = torch.tensor([1, 2, 3], dtype=torch.float32)

b = torch.tensor([4, 5, 6], dtype=torch.float32)

a = a / a.norm(dim=0)
b = b / b.norm(dim=0)

# cosine sim between a and b
cosine_sim = torch.dot(a, b)

print('cosine_sim ', cosine_sim)