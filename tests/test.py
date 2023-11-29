import torch

a = torch.tensor([11, 15])

# b = torch.tensor([[19.2484, 11.7084, 11.1596, 11.4631, 12.3431, 15.8071, 16.7049, 12.6975, 15.4289, 18.6648, 
#          14.3783, 
#          17.9233, 
#          11.9665, 14.8254, 15.0897, 10.9909],
#         [18.6673, 15.3969, 27.8116, 13.1985, 15.7663,  8.8110, 17.5427, 18.7815,
#          10.8143, 13.5417, 12.7912, 12.5115, 15.3884,  5.0697,  9.0324, 
#          26.9915]])
b = torch.tensor([1, 2,3,4,5,6,7])
b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# a = a.unsqueeze(1)

# print(torch.gather(b, 1, a))

c = torch.tensor([False,  True, False, False, False, False, False])

print([item for keep, item in zip(c, b) if keep])

# print(b[c])

# print(b[:, a])