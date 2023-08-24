import numpy as np

import sklearn

from sklearn.metrics.pairwise import euclidean_distances


def text_similarity(text_features, similarity_type='cosine_similarity', pca_dims=None):

    # text_embeds = outputs.text_embeds
    # text_features = text_embeds.detach().cpu()

    # if pca_dims is not None:
    #     # do PCA dimensionality reduction on text embeddings
    #     pca = sklearn.decomposition.PCA(n_components=pca_dims)
    #     text_features = pca.fit_transform(text_features)

    if similarity_type == 'cosine_similarity':
        

        # text_features = np.array([[1,2,3,4], [100, 200, 300, 400]]) # for sanity testing cosine similarity concepts
        norms = np.linalg.norm(text_features, axis=1)

        # add dimension to norms 
        norms = np.expand_dims(norms, axis=1) # shape: (5,1)

        # norms matrix
        norms_matrix = np.matmul(norms, norms.T) # shape: (5,5)

        similarities = np.matmul(text_features, text_features.T) / norms_matrix
    elif similarity_type == 'euclidean_distance':
        # euclidean distance
        similarities = euclidean_distances(text_features, text_features)

    elif similarity_type == 'euclidean_similarity':

         # euclidean distance
        similarities = euclidean_distances(text_features, text_features)

        # normalize to [0,1]
        similarities = 1 - (similarities / np.max(similarities))

    else:
        raise ValueError('Invalid similarity type')


    return similarities

def image_similarity(image_features, similarity_type='cosine_similarity', pca_dims=None):

    # image_embeds = outputs.image_embeds
    # image_features = image_embeds.detach().cpu()

    # if pca_dims is not None:
    #     # do PCA dimensionality reduction on image embeddings
    #     pca = sklearn.decomposition.PCA(n_components=pca_dims)
    #     image_features = pca.fit_transform(image_features)

    if similarity_type == 'cosine_similarity':
        norms = np.linalg.norm(image_features, axis=1)

        # add dimension to norms
        norms = np.expand_dims(norms, axis=1) # shape: (5,1)

        # norms matrix
        norms_matrix = np.matmul(norms, norms.T) # shape: (5,5)

        similarities = np.matmul(image_features, image_features.T) / norms_matrix

    elif similarity_type == 'euclidean_distance':
        # euclidean distance
        similarities = euclidean_distances(image_features, image_features)

    elif similarity_type == 'euclidean_similarity':
        # euclidean distance
        similarities = euclidean_distances(image_features, image_features)

        

        # normalize to [0,1]
        similarities = 1 - (similarities / np.max(similarities))
    else:
        raise ValueError('Invalid similarity type')

    return similarities

def text_image_similarity(text_features, image_features, similarity_type='cosine_similarity', pca_dims=None):

    # text_embeds = outputs.text_embeds
    # image_embeds = outputs.image_embeds

    # text_features = text_embeds.detach().cpu()
    # image_features = image_embeds.detach().cpu()

    # if pca_dims is not None:
    #     # do PCA dimensionality reduction on text embeddings
    #     pca = sklearn.decomposition.PCA(n_components=pca_dims)
    #     text_features = pca.fit_transform(text_features)

    #     # do PCA dimensionality reduction on image embeddings
    #     pca = sklearn.decomposition.PCA(n_components=pca_dims)
    #     image_features = pca.fit_transform(image_features)

    if similarity_type == 'cosine_similarity':

        norms_text = np.linalg.norm(text_features, axis=1)
        norms_image = np.linalg.norm(image_features, axis=1)

        # add dimension to norms
        norms_text = np.expand_dims(norms_text, axis=1) # shape: (5,1)
        norms_image = np.expand_dims(norms_image, axis=1) # shape: (5,1)

        # norms matrix
        norms_matrix = np.matmul(norms_text, norms_image.T) # shape: (5,5)

        similarities = np.matmul(text_features, image_features.T) / norms_matrix
    elif similarity_type == 'euclidean_distance':
        # euclidean distance
        similarities = euclidean_distances(text_features, image_features)

    elif similarity_type == 'euclidean_similarity':

        # euclidean distance
        similarities = euclidean_distances(text_features, image_features)


        # normalize to [0,1]
        similarities = 1 - (similarities / np.max(similarities))
    else:
        raise ValueError('Invalid similarity type')

    return similarities