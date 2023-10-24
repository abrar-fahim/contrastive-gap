'''
Try projecting CLIP embeddings into lower dimensions to see if it improves distance results
'''
import json
from PIL import Image
import requests

import random
from archives2.CLIPWrapper import CLIPWrapper

from sklearn.decomposition import PCA

import numpy as np

similarity_measure = 'cosine_similarity'
# similarity_measure = 'euclidean_distance'

# first try to project texts into lower dimensions

# load MSCOCO dataset
f = open('datasets/mscoco/annotations/captions_val2014.json')

data = json.load(f)


# randomly sample at most n captions from dataset

random.seed(0)
n = 100

sampled_captions = []

sampled_caption_ids = []


for i in random.sample(range(len(data['annotations'])), n):

    # only append if caption id is not already in sampled_caption_ids

    if data['annotations'][i]['image_id'] not in sampled_caption_ids:

        sampled_captions.append(data['annotations'][i]['caption'])

        sampled_caption_ids.append(data['annotations'][i]['image_id'])


# get first image as placeholder for clip wrapper

# image = Image.open(requests.get(data['images'][0]['coco_url'], stream=True).raw)

# sampled_images = [image] # since processor expects a list of images


# get the corresponding image for each of the captions in sampled_captions

sampled_images = []

for image_id in sampled_caption_ids:
    for image in data['images']:
        if image['id'] == image_id:
            # print progress 
            if len(sampled_images) % 10 == 0:
                print(len(sampled_images), ' / ', len(sampled_caption_ids))
            image = Image.open(requests.get(image['coco_url'], stream=True).raw)
            sampled_images.append(image)
            break # break out of inner loop since we found the image we were looking for


# in sampled_images, ith image corresponds to ith caption in sampled_captions

print('NUM IMAGES SAMPLED ', len(sampled_images))
print('NUM CAPTIONS SAMPLED ', len(sampled_captions))

clip_model = CLIPWrapper(sampled_captions, sampled_images, similarity_measure)

# get text embeddings
text_embeddings = clip_model.get_text_embeddings() # shape: (n, 512)


# do PCA dimensionality reduction on text embeddings

pca = PCA(n_components=n)

text_embeddings_pca = pca.fit_transform(text_embeddings)

# # see variance explained by each component

# print('variance explained by each component ', pca.explained_variance_ratio_)

# # plot variance explained by each component

# import matplotlib.pyplot as plt

# plt.plot(pca.explained_variance_ratio_)

# plt.show()

# do PCA dimensionality reduction on image embeddings


# in image_embeddings_pca, ith image embedding corresponds to ith caption embedding in text_embeddings_pca, atleast, it should, maybe check LATER


pca_clip_model = CLIPWrapper(sampled_captions, sampled_images, similarity_measure, pca_dims=2*n)

# average similarity between two texts

print(f'average {similarity_measure} between two texts with pca dims {2*n}', pca_clip_model.get_average_text_text_similarity())

# average similarity between image and their corresponding caption

similarity_matrix = pca_clip_model.get_text_image_similarities()



print(f'average {similarity_measure} between image and their corresponding caption with pca dims {2*n}', np.mean(similarity_matrix.diagonal())) # this works since diagonal elements are similarities between image and their corresponding caption, since ith caption corresponds to ith image