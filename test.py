import numpy as np
import torch
a = np.arange(16).reshape(4, 4)

print('a ', a)

# print upper triangle of a
print('np.triu(a) ', a[np.triu_indices(4, k=1)])

print('torch.triu(a) ', a[torch.triu(torch.ones(4, 4), diagonal=1).bool()])
