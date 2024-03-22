

import torch

a = torch.tensor([1, 2, 3])


b = []

c = torch.tensor([4, 5, 6])

b.extend(a.tolist())
b.extend(c.tolist())

print(b)
