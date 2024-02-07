import torch

# tensor a
a = torch.tensor([1, 2, 3], dtype=torch.float32)

# normalize a
a = a / a.norm()

# tensor b
b = torch.tensor([6, 4, 2], dtype=torch.float32)

# normalize b
b = b / b.norm()

# cosine similarity between a and b

# dot product of a and b
print(a @ b.t())

print(a.dot(b))
