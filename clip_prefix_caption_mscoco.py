
from PIL import Image

import numpy as np
from evaluate import load

import matplotlib.pyplot as plt
from PIL import Image

from clip_caption import init, get_caption

import json





similarity_measure = 'cosine_similarity'

'''
get images and captions from mscoco dataset
'''

f = open('datasets/mscoco/annotations/captions_val2014.json')

data = json.load(f)


# check to see if there are images without any captions
# for i, image in enumerate(data['images']):
#     # print progress
#     if i % 1000 == 0:
#         print('progress: ', i, '/', len(data['images']))
#     has_caption = False
    
#     for annotation in data['annotations']:
#         if annotation['image_id'] == image['id']:
#             has_caption = True
#             break
#     if not has_caption:
#         print('image without caption: ', image['id'])


# check if there are any images with < 5 captions
for i, image in enumerate(data['images']):
    # print progress
    if i % 1000 == 0:
        print('progress: ', i, '/', len(data['images']))
    n_captions = 0
    
    for annotation in data['annotations']:
        if annotation['image_id'] == image['id']:
            n_captions += 1
        if n_captions >= 5:
            break
    if n_captions < 5:
        print('image with less than 5 captions: ', image['id'])
exit()

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

# convert images to tensors


if use_image_embeddings:
    predicted_captions = get_caption(images, clip_caption_model, type='image')

else:
    predicted_captions = get_caption(true_captions, clip_caption_model, type='text')


print('prediction captions ', predicted_captions)


# compute bertscore

bertscore = bertscore.compute(predictions=predicted_captions, references=true_captions, model_type="distilbert-base-uncased")

print('bertscore: ', bertscore)

# find average precision and recall

precision = np.mean(bertscore['precision'])
recall = np.mean(bertscore['recall'])
# find harmonic mean of precision and recall
f1 = 2 * (precision * recall) / (precision + recall)

print('----  AVERAGES  ----')

print('precision: ', precision)
print('recall: ', recall)
print('f1: ', f1)

# display images and captions in one figure

fig = plt.figure(figsize=(10, 10))

for i in range(n_display_images):
    fig.add_subplot(1, n_display_images, i + 1)
    plt.imshow(display_images[i])
    plt.title('true caption: ' + true_captions[i] + '\n' + 'generated caption: ' + predicted_captions[i])
plt.show()





# plt.imshow(image)
# plt.title('true caption: ' + true_caption + '\n' + 'generated caption: ' + generated_caption)
# plt.show()

