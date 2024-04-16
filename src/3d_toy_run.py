import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import torch

from torch.optim import SGD

from torch.distributions.multivariate_normal import MultivariateNormal

from src.my_ce_loss import MyCrossEntropyLoss

import matplotlib.pyplot as plt

from tqdm import tqdm

# generate cluster of points around a point in unit sphere

def plot(a_points, b_points):

    # normalize
    a_points = a_points / a_points.norm(dim=1).view(-1, 1)
    b_points = b_points / b_points.norm(dim=1).view(-1, 1)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    for i in range(n):
        ax.scatter(a_points[i, 0], a_points[i, 1], a_points[i, 2], c='r')
        ax.scatter(b_points[i, 0], b_points[i, 1], b_points[i, 2], c='b')

    # use fixed scale

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

d = 3

n = 30

T = 0.01

n_epochs = 100000

batch_size = 10

lr = 0.01

torch.manual_seed(0)

# generate random point in unit sphere

a = MultivariateNormal(torch.tensor([0, 1, 0], dtype=torch.float32, requires_grad=True), torch.eye(d) * 0.01)


b = MultivariateNormal(torch.tensor([0, -1, 0], dtype=torch.float32, requires_grad=True), torch.eye(d) * 0.01)

# generate the points

a_points = a.sample((n,))
b_points = b.sample((n,))

# normalize
a_points = a_points / a_points.norm(dim=1).view(-1, 1)
b_points = b_points / b_points.norm(dim=1).view(-1, 1)
        
print('a loc', a.loc)
# visualize the points

plot(a_points, b_points)

loss = MyCrossEntropyLoss()

# optimize the positions of the points

ab = torch.stack([a_points, b_points], dim=0)

ab.requires_grad = True


# print('a and b', ab)

# - optimizer -
sgd = SGD([ab], lr=lr)




epochs = tqdm(range(n_epochs))

loss_value = torch.tensor(0.0)

for epoch in epochs:
    epochs.set_description(f'Epoch {epoch}, loss: {loss_value.item()}')

    for i in range(n // batch_size): # for each point

        sgd.zero_grad()

        

        # select batch_size points randomly from a and b
        indices = torch.randint(0, n, (batch_size,))
        a_batch = ab[0, indices]
        b_batch = ab[1, indices]

        # normalize
        a_batch = a_batch / a_batch.norm(dim=1).view(-1, 1)
        b_batch = b_batch / b_batch.norm(dim=1).view(-1, 1)
        

        

        # - loss -

        # find similarity between a_batch and b_batch
        # the similarity is the dot product of a_batch and b_batch

        logits = torch.matmul(a_batch, b_batch.t()) # shape (batch_size, batch_size)

        # scale with T
        scaled_logits = logits / T

        # labels are the diagonal of the matrix
        labels = torch.arange(batch_size)

        # compute loss
        loss_value = loss(scaled_logits, labels)

        # print('loss', loss_value.item())

        # - backward -
        loss_value.backward()

        # - step -
        sgd.step()


# visualize the points
        
plot(ab[0].detach(), ab[1].detach())


