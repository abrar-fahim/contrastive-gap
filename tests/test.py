import torch

a = torch.tensor([1, 2, 3], dtype=torch.float32)

print(torch.norm(a, dim=0))
print(torch.norm(a, dim=-1))