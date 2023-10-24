
from PIL import Image

import numpy as np
from evaluate import load


import matplotlib.pyplot as plt
from PIL import Image

from archives2.clip_caption import init, get_caption

import json

import torch

import clip

# import sklearn cosine similarity
from sklearn.metrics.pairwise import cosine_similarity





similarity_measure = 'cosine_similarity'

'''
get images and captions from mscoco dataset
'''

f = open('datasets/mscoco/annotations/captions_val2014.json')

data = json.load(f)


'''
1. Get image, get true caption, compute caption, measure similarity between true caption and generated caption

- Using BERTScore from huggingface for similarity for now

- Computing average f1 score for now
'''




# set np random seed
np.random.seed(0)

# set whether to use text or image embeddings from CLIP
use_image_embeddings = True

# setup CLIP and CLIPCaptionModel
clip_caption_model = init()


n_images = 100

n_display_images = 2 # n of images to display for sanity checking


# setup bertscore

bertscore = load("bertscore")


predicted_captions = []
true_captions = []

display_images = []

random_image_indices = np.random.randint(0, len(data['images']), n_images)

images = []

# load images and true captions

for index in random_image_indices:
    image = Image.open('datasets/mscoco/val2014/' + data['images'][index]['file_name'])
    images.append(image)

    for annotation in data['annotations']:
        if annotation['image_id'] == data['images'][index]['id']:
            true_captions.append(annotation['caption'])
            break

    if len(display_images) < n_display_images:
        display_images.append(image)


# get clip model
clip_model = clip_caption_model['clip_model']

preprocess = clip_caption_model['preprocess']

tokenizer = clip_caption_model['tokenizer']

device = clip_caption_model['device']

processed_images = [preprocess(image).to(device) for image in images]

processed_images = torch.stack(processed_images)

# convert images to tensors

cosine_similarities = []

with torch.no_grad():
    image_features = clip_model.encode_image(processed_images) # (n_images, 512)
    text_tokens = clip.tokenize(true_captions).to(device)
    text_features = clip_model.encode_text(text_tokens).to(device, dtype=torch.float32) # (n_captions, 512)


    # compute similarity between image and caption features

    if similarity_measure == 'cosine_similarity':

        # find cosine similarity between text features
        # cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        # similarity = cosine_similarity(text_features[0], text_features[1]).cpu().numpy()

        # compute pairwise cosine similarity between image and caption features using sklearn
        cosine_similarities = cosine_similarity(image_features.cpu(), text_features.cpu())

        # make a list of the diagonal elements of the cosine similarity matrix
        cosine_similarities = np.diag(cosine_similarities)



        # cosine_similarities.append(similarity)
# print average cosine similairtty
# print('average cosine similarity: ', np.mean(cosine_similarities))

print('cosine similarities shape: ', cosine_similarities.shape)
print('cosine similarities mean: ', np.mean(cosine_similarities))
print('cosine_similarities: ', cosine_similarities)




