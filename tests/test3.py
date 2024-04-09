import torch

import random

# seed
torch.manual_seed(42)
random.seed(42)

a = torch.tensor([1, 2, 3, 4, 5])

b = [1, 2, 3, 4, 5]

# shift b by one
c = [b[-1]] + b[:-1]
# c = b[1:] + [b[0]]

print(c)


# shift a by one
d = torch.roll(a, shifts=1, dims=0)

print(d)