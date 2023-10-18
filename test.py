import torch

a = torch.tensor([[1,2], [3,4]], dtype=torch.float32)

b = torch.tensor([[1,2], [3,4]], dtype=torch.float32)

print('torch norm ', torch.norm(a, dim=1, keepdim=True))

a = a / torch.norm(a, dim=1, keepdim=True)
b = b / torch.norm(b, dim=1, keepdim=True)

print(a @ b.t())