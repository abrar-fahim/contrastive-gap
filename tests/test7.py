import torch




text_embeddings = []

for i in range(10):
    text_embedding = torch.randn(20, 512)
    text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
    text_embedding = text_embedding.mean(dim = 0)
    text_embedding /= text_embedding.norm()
    text_embeddings.append(text_embedding)


# print('text embeddings ', text_embeddings)

text_embeddings = torch.stack(text_embeddings, dim = 1)

print('new_vectors ', text_embeddings.shape)

print('text embeddings centroid ', text_embeddings.mean(dim = 1).shape)




