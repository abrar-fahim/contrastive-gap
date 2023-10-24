from PIL import Image
import requests
from archives2.CLIPWrapper import CLIPWrapper
import json
import random
import matplotlib.pyplot as plt

from archives2.CLIPDataLoader import CLIPDataLoader

import numpy as np

similarity_measure = 'cosine_similarity'

dim_reduction_technique = 'pca'

dims = 2

n = 10

# load MSCOCO dataset

data = CLIPDataLoader(n, dataset='mscoco')

clip_model = CLIPWrapper(data.sampled_captions, data.sampled_images, similarity_type=similarity_measure, dim_reduction_technique=dim_reduction_technique, dims=dims)

# get text encoder outputs

clip_text_pooler_output = clip_model.get_text_encoder_outputs(both=True) # shape: (n, 512 or dims)


# get clip text embeddings

text_embeds = clip_model.get_text_embeddings(both=True) # shape: (n, 512 or dims)


# get image encoder outputs

image_encoder_outputs = clip_model.get_image_encoder_outputs(both=True) # shape: (n, 768 or dims)

# get clip image embeddings

image_embeds = clip_model.get_image_embeddings(both=True) # shape: (n, 768 or dims)

# plot embeddings and encoder outputs for both text and images

# plot text encoder outputs

plt.scatter(clip_text_pooler_output[:, 0], clip_text_pooler_output[:, 1], color='orange')

# plot text embeddings

plt.scatter(text_embeds[:, 0], text_embeds[:, 1], color='red') 

# plot image encoder outputs

# plt.scatter(image_encoder_outputs[:, 0], image_encoder_outputs[:, 1], c='lightblue')

# plot image embeddings

plt.scatter(image_embeds[:, 0], image_embeds[:, 1], color='blue')

plt.show()









