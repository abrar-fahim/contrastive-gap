

import torch
import numpy as np

import matplotlib.pyplot as plt
# import pca
from sklearn.decomposition import PCA



def findW(x, y):
    yx = y.T @ x

    u, s, v = torch.svd(yx)

    w = u @ v.T

    return w


def modality_gap(x, y):
    x = x / torch.norm(x, p=2, dim=1, keepdim=True)
    y = y / torch.norm(y, p=2, dim=1, keepdim=True)

    x_centroid = x.mean(dim=0)
    y_centroid = y.mean(dim=0)

    euclidean_distance = torch.norm(x_centroid - y_centroid, p=2)

    return euclidean_distance







# set seed
torch.manual_seed(42)

x = torch.randint(0, 5, (100, 512), dtype=torch.float32)


y = torch.randint(0, 5, (100, 512), dtype=torch.float32)


# make last half dimensions of y far from x
y[:, 255:] = y[:, 255:] + 1000

x[:, 255:] = x[:, 255:] -1000



x = x / torch.norm(x, p=2, dim=1, keepdim=True)

y = y / torch.norm(y, p=2, dim=1, keepdim=True)





# PCA dim reduce x and y to 2D

# pca = PCA(n_components=2)
# x_2d = pca.fit_transform(x)
# y_2d = pca.fit_transform(y)

x_2d = x
y_2d = y

# plt.scatter(x_2d[:, 0], x_2d[:, 1], c='r')
# plt.scatter(y_2d[:, 0], y_2d[:, 1], c='b')
# plt.show()

# exit()





print('original modality gap ', modality_gap(x, y))

y_to_x_direction = (x.mean(dim=0) - y.mean(dim=0))

# plot the gaps

gaps = []

for delta in np.arange(0, 1, 0.1):

    phantom_y = y + delta * y_to_x_direction

    phantom_y = phantom_y / torch.norm(phantom_y, p=2, dim=1, keepdim=True)

    print('phantom modality gap ', modality_gap(x, phantom_y))

    W = findW(x, phantom_y)

    aligned_x = x @ W.T

    aligned_x = aligned_x / torch.norm(aligned_x, p=2, dim=1, keepdim=True)

    print('modality gap between x and y after aligining x ', modality_gap(aligned_x, y))

    gaps.append(modality_gap(aligned_x, y))

plt.plot(np.arange(0, 1, 0.1), gaps)

plt.show()

    



    


