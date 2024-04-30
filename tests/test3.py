import torch

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



a = torch.arange(0, 8, dtype=torch.float32)

b = torch.randn(6,8)

U, S, Vh = torch.linalg.svd(b)


print('s ', S)


bucketed_a = torch.chunk(S, 8)

if len(bucketed_a) < 8:
    bucketed_a += tuple([torch.tensor([0.0]) for _ in range(8 - len(bucketed_a))])

print('new bucketed S ', bucketed_a)

# find means
means = [torch.mean(bucket) for bucket in bucketed_a]

print(bucketed_a)

print(means)





