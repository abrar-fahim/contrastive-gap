import torch


# set random seed
torch.manual_seed(42)

subset_indices = torch.randint(0, 10 , (2,)) 



print(subset_indices)

print(12 in subset_indices)
