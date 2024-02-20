import torch


a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
b = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=torch.float32)

print(torch.norm(a-b, dim=-1)) # tensor([37.4162, 74.8322, 112.2483])
