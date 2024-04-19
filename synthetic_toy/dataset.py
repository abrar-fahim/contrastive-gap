import torch

from torch.distributions.multivariate_normal import MultivariateNormal
# make dataloader for custom data


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, d=3):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = n
        self.d = d

        a_mean = torch.zeros(d, dtype=torch.float32)
        a_mean[1] = 1

        b_mean = torch.zeros(d, dtype=torch.float32)
        b_mean[1] = -1

        a = MultivariateNormal(a_mean, torch.eye(d) * 0.01)
        b = MultivariateNormal(b_mean, torch.eye(d) * 0.01)

        a_points = a.sample((n,))
        b_points = b.sample((n,))  

        # normalize
        a_points = a_points / a_points.norm(dim=1).view(-1, 1)
        b_points = b_points / b_points.norm(dim=1).view(-1, 1)

        self.ab = torch.stack([a_points, b_points], dim=0)
        # shape: (2, n, d)

        self.ab.requires_grad = True

       




    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # return self.ab[idx], self.labels[idx]
        return self.ab[0, idx], self.ab[1, idx]
