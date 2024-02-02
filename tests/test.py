import torch



def loss(t):
    a = torch.tensor([0.7, 0.1, 0.1, 0.1])
    a = a / t
    label_a = torch.tensor(1)
    # contrastive loss of a
    a_loss = -torch.log(torch.exp(a[label_a]) / torch.exp(a).sum())
    # return a_loss * torch.exp(torch.tensor(t))
    # return a_loss * t
    return a_loss



print('loss(0.01): ', loss(0.01))
print('loss(0.1): ', loss(0.1))
print('loss(0.5): ', loss(0.5))
print('loss(1): ', loss(1))
