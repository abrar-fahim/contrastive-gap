import torch

# make unit vector
a = torch.tensor([1, 2, 3], dtype=torch.float32)
a = torch.randn(512)
a = a / torch.norm(a)

b = torch.tensor([-2, -2, -5], dtype=torch.float32)
b = torch.randn(512)
b = b / torch.norm(b)

c = torch.tensor([3, 10, 2], dtype=torch.float32)
c = torch.randn(512)
c = c / torch.norm(c)

d = torch.tensor([10, 11, 12], dtype=torch.float32)
d = torch.randn(512)
d = d / torch.norm(d)

# some centroid
centroid_ab = torch.mean(torch.stack([a, b]), dim=0)
centroid_cd = torch.mean(torch.stack([c, d]), dim=0)

# euclidean distance between centroids of a,b and c,d
euclidean_distance_centroids = torch.norm(centroid_ab - centroid_cd)





# cosine similarity

cosine_similarity = a @ b

print(cosine_similarity) 

# euclidean distance
euclidean_distance = torch.norm(a - b)

print('euclidean distance', euclidean_distance)
print('sqrt((a-b)T(a-b))', torch.sqrt((a - b) @ (a - b)))

print('(a-b)T(a-b) = ', (a - b) @ (a - b))
print('2(1-xTy)', 2 * (1 - a @ b))



print('2(1 - cos(a,b))', 2 * (1 - cosine_similarity))
