import torch

import sys
import os

# import pca from skleran
from sklearn.decomposition import PCA
import numpy as np
# import matplotlib plt
import matplotlib.pyplot as plt




# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# randomly generate points
np.random.seed(0)
torch.manual_seed(0)

n=1000
d=3

# generate random points

X = np.random.rand(n,d)

# normalize
X = X / np.linalg.norm(X, axis=1)[:, None]

# X = X.T

# find SVD
U, S, V = np.linalg.svd(X)

print('S ', S)



# find PCA
pca = PCA(n_components=min(X.shape[0], X.shape[1]))

X_pca = pca.fit(X)

print('PCA explained variance ', pca.explained_variance_)
print('PCA explained variance ratio ', pca.explained_variance_ratio_)
print('PCA singular values ', pca.singular_values_)
# print('PCA components ', pca.components_)
# print('PCA mean ', pca.mean_)
print('PCA noise variance ', pca.noise_variance_)
print('PCA n_components ', pca.n_components_)
print('PCA n_features ', pca.n_features_)
print('PCA n_samples ', pca.n_samples_)

n_buckets = 4

buckets = torch.chunk(torch.tensor(pca.explained_variance_ratio_), n_buckets)

if len(buckets) < n_buckets:
    buckets += tuple([torch.tensor([0.0]) for _ in range(n_buckets - len(buckets))])


bucket_sums = [torch.sum(bucket) for bucket in buckets]

print('bucket sums ', bucket_sums)



# visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2])
plt.show()


