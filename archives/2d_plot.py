'''
Take the combined CLIP embedding space, project it into 2D using PCA, then plot
'''

from PIL import Image
import requests
from archives2.CLIPWrapper import CLIPWrapper
import json
import random
import matplotlib.pyplot as plt

import numpy as np

dims = 2
n = 10

similarity_measure = 'cosine_similarity'

dim_reduction_technique = 'tsne'

f = open('datasets/mscoco/annotations/captions_val2014.json')

data = json.load(f)



# randomly sample at most n captions from dataset

random.seed(0)


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

clip_model = CLIPWrapper(sampled_captions, sampled_images, similarity_measure, dim_reduction_technique=dim_reduction_technique, dims=dims)

# get text embeddings
text_embeddings = clip_model.get_text_embeddings() # shape: (n, pca_dims)

# get image embeddings

image_embeddings = clip_model.get_image_embeddings() # shape: (n, pca_dims)

# 1. plot the embeddings TEXT VS IMAGES

# plt.scatter(text_embeddings[:,0], text_embeddings[:,1], color='red')

# plt.scatter(image_embeddings[:,0], image_embeddings[:,1], color='blue')

# plt.show()



# 2. plot the embeddings such that each image is plotted with its corresponding caption

# take ith color from color wheel
colors = plt.cm.gist_rainbow(np.linspace(0, 1, n))

for i in range(n):
    plt.scatter(text_embeddings[i,0], text_embeddings[i,1], color=colors[i])
    plt.scatter(image_embeddings[i,0], image_embeddings[i,1], color=colors[i])
plt.show()